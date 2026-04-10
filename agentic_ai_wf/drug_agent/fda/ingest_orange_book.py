#!/usr/bin/env python3
"""
Ingest FDA Orange Book data (Products, Patents, Exclusivity).
Downloads the Orange Book ZIP, merges the three files by NDA number,
and ingests into Qdrant as one document per product with patent/exclusivity overlay.

Usage:
    python ingest_orange_book.py                  # Full run
    python ingest_orange_book.py --limit 100      # Test with 100 products
    python ingest_orange_book.py --recreate       # Wipe and re-ingest
"""

import sys
import logging
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
)
from agentic_ai_wf.drug_agent.fda.fda_base import (
    download_orange_book, parse_tilde_file,
)

COLLECTION = "FDA_Orange_Book"
BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

NDA_TYPE_MAP = {"N": "NDA (innovator)", "A": "ANDA (generic)"}


def _merge_patent_exclusivity(patents_rows, excl_rows):
    """Build lookup dicts keyed by (nda_type + nda_number, product_number)."""
    patent_lookup = defaultdict(list)
    for row in patents_rows:
        key = (row.get("Appl_Type", "") + row.get("Appl_No", ""), row.get("Product_No", ""))
        patent_lookup[key].append({
            "patent_number": row.get("Patent_No", ""),
            "expiry": row.get("Patent_Expire_Date_Text", ""),
            "substance": row.get("Drug_Substance_Flag", "") == "Y",
            "product": row.get("Drug_Product_Flag", "") == "Y",
            "use_code": row.get("Patent_Use_Code", ""),
        })

    excl_lookup = defaultdict(list)
    for row in excl_rows:
        key = (row.get("Appl_Type", "") + row.get("Appl_No", ""), row.get("Product_No", ""))
        excl_lookup[key].append({
            "code": row.get("Exclusivity_Code", ""),
            "expiry": row.get("Exclusivity_Date", ""),
        })

    return patent_lookup, excl_lookup


def _product_text(prod, patents, exclusivities):
    nda_type_str = NDA_TYPE_MAP.get(prod.get("Appl_Type", ""), prod.get("Appl_Type", ""))
    parts = [
        f"FDA Orange Book: {prod.get('Trade_Name', 'Unknown')} ({prod.get('Ingredient', '')}).",
        f"{prod.get('Strength', '')} {prod.get('DF;Route', '')}.",
        f"Application {prod.get('Appl_Type', '')}{prod.get('Appl_No', '')} ({nda_type_str}).",
        f"Approved: {prod.get('Approval_Date', 'Unknown')}.",
        f"Applicant: {prod.get('Applicant_Full_Name', prod.get('Applicant', ''))}.",
        f"Category: {prod.get('Type', '')}.",
    ]

    te = prod.get("TE_Code", "")
    if te:
        parts.append(f"Therapeutic Equivalence: {te}.")

    rld = "Yes" if prod.get("RLD", "") == "RLD" else "No"
    rs = "Yes" if prod.get("RS", "") == "RS" else "No"
    parts.append(f"Reference Listed Drug: {rld}. Reference Standard: {rs}.")

    if patents:
        patent_strs = [
            f"{p['patent_number']} (expires {p['expiry']}, "
            f"{'substance' if p['substance'] else ''}{'/' if p['substance'] and p['product'] else ''}"
            f"{'product' if p['product'] else ''})"
            for p in patents
        ]
        parts.append(f"Patents: {'; '.join(patent_strs)}.")

    if exclusivities:
        excl_strs = [f"{e['code']} (expires {e['expiry']})" for e in exclusivities]
        parts.append(f"Market Exclusivity: {'; '.join(excl_strs)}.")

    return " ".join(parts)


def build_documents(files):
    products = parse_tilde_file(files["products"])
    patents_rows = parse_tilde_file(files["patents"]) if "patents" in files else []
    excl_rows = parse_tilde_file(files["exclusivity"]) if "exclusivity" in files else []

    patent_lookup, excl_lookup = _merge_patent_exclusivity(patents_rows, excl_rows)

    docs = []
    for prod in products:
        key = (prod.get("Appl_Type", "") + prod.get("Appl_No", ""), prod.get("Product_No", ""))
        patents = patent_lookup.get(key, [])
        exclusivities = excl_lookup.get(key, [])

        doc_id = f"ob_{prod.get('Appl_Type', '')}{prod.get('Appl_No', '')}_{prod.get('Product_No', '')}"

        docs.append({
            "id": doc_id,
            "entity_type": "orange_book_product",
            "ingredient": prod.get("Ingredient", ""),
            "trade_name": prod.get("Trade_Name", ""),
            "strength": prod.get("Strength", ""),
            "dosage_form_route": prod.get("DF;Route", ""),
            "nda_number": prod.get("Appl_No", ""),
            "nda_type": prod.get("Appl_Type", ""),
            "te_code": prod.get("TE_Code", ""),
            "approval_date": prod.get("Approval_Date", ""),
            "product_type": prod.get("Type", ""),
            "applicant": prod.get("Applicant_Full_Name", prod.get("Applicant", "")),
            "is_rld": prod.get("RLD", "") == "RLD",
            "is_rs": prod.get("RS", "") == "RS",
            "patent_count": len(patents),
            "exclusivity_count": len(exclusivities),
            "patents": patents,
            "exclusivities": exclusivities,
            "text_content": _product_text(prod, patents, exclusivities),
            "source": "FDA_OrangeBook",
        })
    return docs


def main():
    parser = argparse.ArgumentParser(description="Ingest FDA Orange Book")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    logger.info("Downloading and parsing Orange Book...")
    files = download_orange_book()
    docs = build_documents(files)
    if args.limit:
        docs = docs[:args.limit]

    logger.info(f"Ingesting {len(docs):,} Orange Book products...")
    start = datetime.now()
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        upsert_batch(client, embedder, COLLECTION, batch)
        if (i // BATCH_SIZE) % 20 == 0:
            logger.info(f"Progress: {min(i + BATCH_SIZE, len(docs)):,}/{len(docs):,}")

    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"DONE | {count:,} docs in collection | {elapsed / 60:.1f}min")


if __name__ == "__main__":
    main()
