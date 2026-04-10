#!/usr/bin/env python3
"""
Ingest FDA drug enforcement actions (recalls, market withdrawals).
Downloads bulk JSON (~17K records, 4 MB) and ingests into Qdrant.

Usage:
    python ingest_enforcement.py                  # Full run
    python ingest_enforcement.py --limit 100      # Test
    python ingest_enforcement.py --recreate       # Wipe and re-ingest
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
)
from agentic_ai_wf.drug_agent.fda.fda_base import stream_bulk_json

COLLECTION = "FDA_Enforcement"
BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_SEVERITY = {
    "Class I": "most serious — may cause or lead to death",
    "Class II": "may cause temporary or reversible adverse health consequences",
    "Class III": "not likely to cause adverse health consequences",
}


def build_doc(record):
    recall_num = record.get("recall_number", "")
    classification = record.get("classification", "")
    product_desc = record.get("product_description", "")
    reason = record.get("reason_for_recall", "")
    firm = record.get("recalling_firm", "")
    status = record.get("status", "")
    voluntary = record.get("voluntary_mandated", "")
    distribution = record.get("distribution_pattern", "")
    init_date = record.get("recall_initiation_date", "")
    city = record.get("city", "")
    state = record.get("state", "")
    country = record.get("country", "")

    of = record.get("openfda", {})
    brand_name = ", ".join(of.get("brand_name", []))
    generic_name = ", ".join(of.get("generic_name", []))

    severity = CLASSIFICATION_SEVERITY.get(classification, "")
    severity_str = f" ({severity})" if severity else ""

    text_parts = [
        f"FDA Drug Enforcement: {classification}{severity_str} recall.",
        f"Product: {product_desc[:400]}." if product_desc else "",
        f"Reason: {reason[:300]}." if reason else "",
        f"Recalling firm: {firm}." if firm else "",
        f"Distribution: {distribution[:200]}." if distribution else "",
        f"Status: {status}. {voluntary}." if status else "",
        f"Initiated: {init_date}." if init_date else "",
    ]
    if brand_name:
        text_parts.append(f"Brand: {brand_name}.")
    if generic_name:
        text_parts.append(f"Generic: {generic_name}.")

    return {
        "id": f"enforce_{recall_num}" if recall_num else f"enforce_{hash(product_desc)}",
        "entity_type": "enforcement",
        "recall_number": recall_num,
        "classification": classification,
        "product_description": product_desc[:500],
        "reason_for_recall": reason[:500],
        "recalling_firm": firm,
        "status": status,
        "voluntary_mandated": voluntary,
        "recall_initiation_date": init_date,
        "city": city,
        "state": state,
        "country": country,
        "brand_name": brand_name,
        "generic_name": generic_name,
        "text_content": " ".join(p for p in text_parts if p),
        "source": "FDA_Enforcement",
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest FDA Enforcement Actions")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    logger.info("Streaming FDA enforcement bulk download...")
    start = datetime.now()
    batch = []
    total = 0

    for record in stream_bulk_json("drug/enforcement"):
        doc = build_doc(record)
        batch.append(doc)
        total += 1

        if len(batch) >= BATCH_SIZE:
            upsert_batch(client, embedder, COLLECTION, batch)
            batch = []

        if total % 5000 == 0:
            logger.info(f"Processed {total:,} enforcement actions...")

        if args.limit and total >= args.limit:
            break

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"DONE | {count:,} docs in collection | {total:,} processed | {elapsed / 60:.1f}min")


if __name__ == "__main__":
    main()
