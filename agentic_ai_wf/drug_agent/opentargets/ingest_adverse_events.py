#!/usr/bin/env python3
"""
Ingest adverse events (FAERS) and drug warnings from Open Targets.
Reads drug IDs from the existing OpenTargets_drugs_enriched collection,
fetches paginated AE data + warnings per drug, and ingests into a new collection.

Usage:
    python ingest_adverse_events.py                    # Full run (3,230 drugs)
    python ingest_adverse_events.py --limit 50         # Test with 50 drugs
    python ingest_adverse_events.py --recreate         # Wipe and re-ingest
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, gql_query,
    upsert_batch, load_drug_ids_from_qdrant, load_checkpoint, save_checkpoint,
)

COLLECTION = "OpenTargets_adverse_events"
CHECKPOINT_LABEL = "adverse_events"
BATCH_SIZE = 100
AE_PAGE_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# First page fetches AEs + warnings together; subsequent pages only AEs
QUERY_FIRST_PAGE = """
query($id: String!, $size: Int!) {
  drug(chemblId: $id) {
    adverseEvents(page: {index: 0, size: $size}) {
      count criticalValue
      rows { name count logLR meddraCode }
    }
    drugWarnings {
      toxicityClass description warningType country year
      efoTerm efoId
    }
  }
}"""

QUERY_AE_PAGE = """
query($id: String!, $page: Int!, $size: Int!) {
  drug(chemblId: $id) {
    adverseEvents(page: {index: $page, size: $size}) {
      rows { name count logLR meddraCode }
    }
  }
}"""


def _ae_text(drug_name: str, drug_type: str, max_phase: float, ae: dict, critical_value: float) -> str:
    sig = "statistically significant" if ae["logLR"] > critical_value else "below significance threshold"
    phase_str = f", Phase {max_phase:.0f}" if max_phase else ""
    return (
        f"Adverse drug reaction for {drug_name} ({drug_type}{phase_str}): {ae['name']}. "
        f"{ae['count']:,} FDA FAERS reports, logLR={ae['logLR']:.1f} ({sig}). "
        f"MedDRA: {ae['meddraCode']}."
    )


def _warning_text(drug_name: str, w: dict) -> str:
    parts = [f"Drug safety warning for {drug_name}"]
    if w.get("warningType"):
        parts.append(f"({w['warningType']})")
    parts.append(f": {w.get('toxicityClass', 'unspecified toxicity')}.")
    if w.get("description"):
        parts.append(w["description"][:200])
    if w.get("efoTerm"):
        parts.append(f"Related condition: {w['efoTerm']}.")
    if w.get("country"):
        parts.append(f"Country: {w['country']}.")
    return " ".join(parts)


def fetch_drug_ae(session, drug_id, drug_name, drug_type, max_phase):
    """Yield all AE and warning documents for one drug."""
    data = gql_query(session, QUERY_FIRST_PAGE, {"id": drug_id, "size": AE_PAGE_SIZE})
    if not data:
        return
    drug = data.get("drug")
    if not drug:
        return

    ae_data = drug.get("adverseEvents") or {}
    total_ae = ae_data.get("count", 0)
    critical = ae_data.get("criticalValue", 0)

    # First page AEs
    for ae in ae_data.get("rows") or []:
        if ae["logLR"] <= 0:
            continue
        yield {
            "id": f"{drug_id}_ae_{ae['meddraCode']}",
            "entity_type": "adverse_event",
            "drug_id": drug_id, "drug_name": drug_name, "drug_type": drug_type,
            "max_phase": max_phase,
            "event_name": ae["name"], "meddra_code": ae["meddraCode"],
            "report_count": ae["count"], "log_lr": ae["logLR"],
            "critical_value": critical,
            "text_content": _ae_text(drug_name, drug_type, max_phase, ae, critical),
            "source": "OpenTargets_FAERS",
        }

    # Paginate remaining AEs
    pages_needed = (total_ae // AE_PAGE_SIZE) + (1 if total_ae % AE_PAGE_SIZE else 0)
    for page_idx in range(1, pages_needed):
        page_data = gql_query(session, QUERY_AE_PAGE,
                              {"id": drug_id, "page": page_idx, "size": AE_PAGE_SIZE})
        if not page_data:
            continue
        for ae in ((page_data.get("drug") or {}).get("adverseEvents") or {}).get("rows") or []:
            if ae["logLR"] <= 0:
                continue
            yield {
                "id": f"{drug_id}_ae_{ae['meddraCode']}",
                "entity_type": "adverse_event",
                "drug_id": drug_id, "drug_name": drug_name, "drug_type": drug_type,
                "max_phase": max_phase,
                "event_name": ae["name"], "meddra_code": ae["meddraCode"],
                "report_count": ae["count"], "log_lr": ae["logLR"],
                "critical_value": critical,
                "text_content": _ae_text(drug_name, drug_type, max_phase, ae, critical),
                "source": "OpenTargets_FAERS",
            }

    # Drug warnings
    for w in drug.get("drugWarnings") or []:
        yield {
            "id": f"{drug_id}_warn_{w.get('toxicityClass', 'unk')}_{w.get('warningType', '')}",
            "entity_type": "drug_warning",
            "drug_id": drug_id, "drug_name": drug_name, "drug_type": drug_type,
            "max_phase": max_phase,
            "toxicity_class": w.get("toxicityClass", ""),
            "warning_type": w.get("warningType", ""),
            "description": (w.get("description") or "")[:500],
            "country": w.get("country", ""),
            "year": w.get("year"),
            "efo_term": w.get("efoTerm", ""),
            "efo_id": w.get("efoId", ""),
            "text_content": _warning_text(drug_name, w),
            "source": "OpenTargets_DrugWarnings",
        }


def main():
    parser = argparse.ArgumentParser(description="Ingest Open Targets adverse events")
    parser.add_argument("--limit", type=int, default=0, help="Max drugs to process (0=all)")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    drugs = load_drug_ids_from_qdrant(client)
    if args.limit:
        drugs = drugs[:args.limit]

    processed = set() if args.no_resume or args.recreate else load_checkpoint(CHECKPOINT_LABEL)
    remaining = [(d, n, t, p) for d, n, t, p in drugs if d not in processed]
    logger.info(f"Processing {len(remaining):,} drugs ({len(processed):,} already done)")

    session = requests.Session()
    session.headers["Content-Type"] = "application/json"
    batch, stats = [], {"ae": 0, "warn": 0, "drugs_done": len(processed)}
    start = datetime.now()

    for i, (drug_id, drug_name, drug_type, max_phase) in enumerate(remaining):
        for doc in fetch_drug_ae(session, drug_id, drug_name, drug_type, max_phase):
            batch.append(doc)
            stats["ae" if doc["entity_type"] == "adverse_event" else "warn"] += 1

            if len(batch) >= BATCH_SIZE:
                upsert_batch(client, embedder, COLLECTION, batch)
                batch = []

        processed.add(drug_id)
        stats["drugs_done"] += 1

        if stats["drugs_done"] % 50 == 0:
            save_checkpoint(CHECKPOINT_LABEL, processed)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"Drugs: {stats['drugs_done']:,}/{len(drugs):,} | "
                f"AEs: {stats['ae']:,} | Warnings: {stats['warn']:,} | "
                f"{elapsed/60:.1f}min"
            )

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    save_checkpoint(CHECKPOINT_LABEL, processed)
    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()

    logger.info("=" * 50)
    logger.info(f"DONE | AEs: {stats['ae']:,} | Warnings: {stats['warn']:,} | "
                f"Collection: {count:,} | {elapsed/60:.1f}min")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
