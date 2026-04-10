#!/usr/bin/env python3
"""
Ingest FDA FAERS adverse event data via aggregated count API.
For each drug in OpenTargets_drugs_enriched, queries openFDA for:
  - Top 25 adverse reactions (count API)
  - Seriousness breakdown
  - Outcome distribution
Produces one summary doc + up to 25 reaction docs per drug.

Usage:
    python ingest_faers.py                    # Full run (~3,230 drugs)
    python ingest_faers.py --limit 50         # Test with 50 drugs
    python ingest_faers.py --recreate         # Wipe and re-ingest
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
    load_drug_ids_from_qdrant,
)
from agentic_ai_wf.drug_agent.fda.fda_base import (
    openfda_api_query, load_fda_checkpoint, save_fda_checkpoint,
)

COLLECTION = "FDA_FAERS"
CHECKPOINT_LABEL = "fda_faers"
BATCH_SIZE = 100
TOP_N_REACTIONS = 25

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Search fields to try in order for matching a drug
SEARCH_FIELDS = [
    "patient.drug.openfda.generic_name.exact",
    "patient.drug.openfda.brand_name.exact",
    "patient.drug.medicinalproduct",
]


def _build_search(drug_name: str, field: str) -> str:
    safe = drug_name.upper().replace('"', "")
    return f'{field}:"{safe}"'


def _query_drug_faers(session, drug_name):
    """Try multiple search strategies to find FAERS data for a drug."""
    for field in SEARCH_FIELDS:
        search = _build_search(drug_name, field)

        reactions_resp = openfda_api_query(
            session, "drug/event", search=search,
            count_field="patient.reaction.reactionmeddrapt.exact",
        )
        if reactions_resp and reactions_resp.get("results"):
            serious_resp = openfda_api_query(
                session, "drug/event", search=search,
                count_field="serious",
            )
            outcome_resp = openfda_api_query(
                session, "drug/event", search=search,
                count_field="patient.reaction.reactionoutcome",
            )
            return reactions_resp, serious_resp, outcome_resp, search

    return None, None, None, None


def _parse_seriousness(resp):
    if not resp or not resp.get("results"):
        return 0, 0, 0
    total, serious = 0, 0
    for r in resp["results"]:
        total += r.get("count", 0)
        if r.get("term") == 1:
            serious = r.get("count", 0)
    return total, serious, (serious / total * 100) if total else 0


def _parse_outcomes(resp):
    """Parse outcome counts: 1=recovered, 2=recovering, 3=not recovered, 4=sequelae, 5=fatal, 6=unknown."""
    OUTCOME_MAP = {1: "recovered", 2: "recovering", 3: "not_recovered", 4: "sequelae", 5: "fatal", 6: "unknown"}
    outcomes = {v: 0 for v in OUTCOME_MAP.values()}
    if not resp or not resp.get("results"):
        return outcomes
    for r in resp["results"]:
        key = OUTCOME_MAP.get(r.get("term"), "unknown")
        outcomes[key] = r.get("count", 0)
    return outcomes


def fetch_drug_faers(session, drug_id, drug_name, drug_type, max_phase):
    """Yield summary + top-N reaction docs for one drug from FDA FAERS."""
    reactions_resp, serious_resp, outcome_resp, _ = _query_drug_faers(session, drug_name)

    if not reactions_resp or not reactions_resp.get("results"):
        return

    reactions = reactions_resp["results"][:TOP_N_REACTIONS]
    total_reports, serious_count, serious_pct = _parse_seriousness(serious_resp)
    outcomes = _parse_outcomes(outcome_resp)

    total_reaction_count = sum(r.get("count", 0) for r in reactions_resp["results"])
    fatal_pct = (outcomes["fatal"] / total_reports * 100) if total_reports else 0

    phase_str = f", Phase {max_phase:.0f}" if max_phase else ""
    reaction_list = "; ".join(
        f"{i + 1}. {r['term']} ({r['count']:,})" for i, r in enumerate(reactions)
    )

    yield {
        "id": f"faers_summary_{drug_id}",
        "entity_type": "faers_summary",
        "drug_id": drug_id,
        "drug_name": drug_name,
        "drug_type": drug_type,
        "max_phase": max_phase,
        "total_reports": total_reports,
        "serious_count": serious_count,
        "serious_pct": round(serious_pct, 1),
        "fatal_count": outcomes["fatal"],
        "fatal_pct": round(fatal_pct, 1),
        "outcomes": outcomes,
        "top_reaction_count": len(reactions),
        "text_content": (
            f"FDA FAERS safety profile for {drug_name} ({drug_type}{phase_str}): "
            f"{total_reports:,} adverse event reports. "
            f"Seriousness: {serious_pct:.0f}% serious ({serious_count:,} reports). "
            f"Outcomes: {outcomes['fatal']:,} fatal ({fatal_pct:.1f}%), "
            f"{outcomes['recovered']:,} recovered, {outcomes['not_recovered']:,} not recovered. "
            f"Top {len(reactions)} reported reactions: {reaction_list}."
        ),
        "source": "FDA_FAERS",
    }

    for r in reactions:
        term = r["term"]
        count = r["count"]
        pct = (count / total_reaction_count * 100) if total_reaction_count else 0

        yield {
            "id": f"faers_{drug_id}_{term.replace(' ', '_')[:60]}",
            "entity_type": "faers_reaction",
            "drug_id": drug_id,
            "drug_name": drug_name,
            "drug_type": drug_type,
            "max_phase": max_phase,
            "reaction_term": term,
            "reaction_count": count,
            "total_reports": total_reports,
            "serious_pct": round(serious_pct, 1),
            "text_content": (
                f"FAERS adverse event for {drug_name} ({drug_type}{phase_str}): "
                f"{term} reported {count:,} times in {total_reports:,} total reports ({pct:.1f}%). "
                f"{drug_name} has {serious_pct:.0f}% serious rate and {fatal_pct:.1f}% fatal outcome rate."
            ),
            "source": "FDA_FAERS",
        }


def main():
    parser = argparse.ArgumentParser(description="Ingest FDA FAERS aggregated data")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    drugs = load_drug_ids_from_qdrant(client)
    if args.limit:
        drugs = drugs[:args.limit]

    processed = set() if args.no_resume or args.recreate else load_fda_checkpoint(CHECKPOINT_LABEL)
    remaining = [(d, n, t, p) for d, n, t, p in drugs if d not in processed]
    logger.info(f"Processing {len(remaining):,} drugs ({len(processed):,} already done)")

    session = requests.Session()
    batch = []
    stats = {"summaries": 0, "reactions": 0, "drugs_done": len(processed), "no_data": 0}
    start = datetime.now()

    for drug_id, drug_name, drug_type, max_phase in remaining:
        found = False
        for doc in fetch_drug_faers(session, drug_id, drug_name, drug_type, max_phase):
            batch.append(doc)
            found = True
            if doc["entity_type"] == "faers_summary":
                stats["summaries"] += 1
            else:
                stats["reactions"] += 1

            if len(batch) >= BATCH_SIZE:
                upsert_batch(client, embedder, COLLECTION, batch)
                batch = []

        if not found:
            stats["no_data"] += 1

        processed.add(drug_id)
        stats["drugs_done"] += 1

        if stats["drugs_done"] % 50 == 0:
            save_fda_checkpoint(CHECKPOINT_LABEL, processed)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"Drugs: {stats['drugs_done']:,}/{len(drugs):,} | "
                f"Summaries: {stats['summaries']:,} | Reactions: {stats['reactions']:,} | "
                f"No data: {stats['no_data']:,} | {elapsed / 60:.1f}min"
            )

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    save_fda_checkpoint(CHECKPOINT_LABEL, processed)
    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()

    logger.info("=" * 60)
    logger.info(
        f"DONE | Summaries: {stats['summaries']:,} | Reactions: {stats['reactions']:,} | "
        f"No data: {stats['no_data']:,} | Collection: {count:,} | {elapsed / 60:.1f}min"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
