#!/usr/bin/env python3
"""
Ingest Drugs@FDA data — regulatory submission history for FDA-approved drugs.
Uses bulk JSON download (9 MB, 1 file) for efficiency.

Usage:
    python ingest_drugsfda.py                  # Full run (~28K applications)
    python ingest_drugsfda.py --limit 100      # Test with 100 records
    python ingest_drugsfda.py --recreate       # Wipe and re-ingest
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

COLLECTION = "FDA_DrugsFDA"
BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def _extract_openfda(record):
    of = record.get("openfda", {})
    return {
        "brand_name": ", ".join(of.get("brand_name", [])),
        "generic_name": ", ".join(of.get("generic_name", [])),
        "manufacturer": ", ".join(of.get("manufacturer_name", [])),
        "product_type": ", ".join(of.get("product_type", [])),
        "route": ", ".join(of.get("route", [])),
        "substance_name": ", ".join(of.get("substance_name", [])),
        "pharm_class_epc": ", ".join(of.get("pharm_class_epc", [])),
        "pharm_class_moa": ", ".join(of.get("pharm_class_moa", [])),
        "rxcui": of.get("rxcui", []),
        "unii": of.get("unii", []),
    }


def _submission_summary(submissions):
    """Distill submission list into a compact text summary."""
    if not submissions:
        return "", "", ""

    orig = None
    efficacy_suppls = []
    latest_date = ""

    for s in submissions:
        stype = s.get("submission_type", "")
        sdate = s.get("submission_status_date", "")
        if stype == "ORIG":
            orig = s
        if s.get("submission_class_code") == "EFFICACY":
            efficacy_suppls.append(sdate[:4] if sdate else "")
        if sdate > latest_date:
            latest_date = sdate

    orig_text = ""
    orig_date = ""
    if orig:
        orig_date = orig.get("submission_status_date", "")
        priority = orig.get("review_priority", "STANDARD")
        sclass = orig.get("submission_class_code_description", "")
        orig_text = f"Original approval {orig_date[:4] if orig_date else 'N/A'} ({sclass}, {priority} review)"

    efficacy_text = ""
    if efficacy_suppls:
        efficacy_text = f"{len(efficacy_suppls)} efficacy supplements ({', '.join(filter(None, efficacy_suppls))})"

    return orig_text, efficacy_text, orig_date


def _products_text(products):
    parts = []
    for p in (products or []):
        ingredients = [ing.get("name", "") for ing in p.get("active_ingredients", [])]
        parts.append(
            f"{p.get('brand_name', '')} "
            f"({', '.join(ingredients)}) "
            f"{p.get('dosage_form', '')} {p.get('route', '')} "
            f"[{p.get('marketing_status', '')}]"
        )
    return "; ".join(parts)


def build_doc(record):
    app_num = record.get("application_number", "")
    sponsor = record.get("sponsor_name", "")
    submissions = record.get("submissions", [])
    products = record.get("products", [])
    of = _extract_openfda(record)

    orig_text, efficacy_text, orig_date = _submission_summary(submissions)
    prod_text = _products_text(products)

    text_parts = [f"FDA-approved drug application {app_num}."]
    if of["brand_name"]:
        text_parts.append(f"{of['brand_name']} ({of['generic_name']}).")
    text_parts.append(f"Sponsor: {sponsor}.")
    if prod_text:
        text_parts.append(f"Products: {prod_text}.")
    if orig_text:
        text_parts.append(orig_text + ".")
    if efficacy_text:
        text_parts.append(efficacy_text + ".")
    text_parts.append(f"{len(submissions)} total submissions.")
    if of["pharm_class_epc"]:
        text_parts.append(f"Pharmacologic class: {of['pharm_class_epc']}.")
    if of["pharm_class_moa"]:
        text_parts.append(f"Mechanism: {of['pharm_class_moa']}.")

    active_ingredients = []
    for p in products or []:
        for ing in p.get("active_ingredients", []):
            name = ing.get("name", "")
            if name and name not in active_ingredients:
                active_ingredients.append(name)

    return {
        "id": f"drugsfda_{app_num}",
        "entity_type": "drugsfda_application",
        "application_number": app_num,
        "sponsor_name": sponsor,
        "brand_name": of["brand_name"],
        "generic_name": of["generic_name"],
        "manufacturer": of["manufacturer"],
        "product_type": of["product_type"],
        "route": of["route"],
        "active_ingredients": active_ingredients,
        "submission_count": len(submissions),
        "original_approval_date": orig_date,
        "pharm_class_epc": of["pharm_class_epc"],
        "pharm_class_moa": of["pharm_class_moa"],
        "text_content": " ".join(text_parts),
        "source": "FDA_DrugsFDA",
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Drugs@FDA")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    logger.info("Streaming Drugs@FDA bulk download...")
    start = datetime.now()
    batch = []
    total = 0

    for record in stream_bulk_json("drug/drugsfda"):
        doc = build_doc(record)
        batch.append(doc)
        total += 1

        if len(batch) >= BATCH_SIZE:
            upsert_batch(client, embedder, COLLECTION, batch)
            batch = []

        if total % 1000 == 0:
            logger.info(f"Processed {total:,} applications...")

        if args.limit and total >= args.limit:
            break

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"DONE | {count:,} docs in collection | {total:,} processed | {elapsed / 60:.1f}min")


if __name__ == "__main__":
    main()
