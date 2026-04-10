#!/usr/bin/env python3
"""
Ingest ClinicalTrials.gov Phase 2-4 trial results — outcome measures,
statistical analyses, serious adverse events, and participant flow.

Only trials with posted results (hasResults=true) are fetched.
ALL serious AEs are kept; non-serious AEs truncated to top 20 by frequency.

Usage:
    python -m agentic_ai_wf.drug_agent.clinicaltrials.ingest_results
    python -m ... --limit 20          # test batch
    python -m ... --recreate          # wipe and re-ingest
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
)
from agentic_ai_wf.drug_agent.clinicaltrials.ct_base import (
    safe_get, iterate_studies, extract_drug_interventions,
    extract_conditions, phase_list_to_numeric, phase_list_to_str,
    chunk_text, load_ct_checkpoint, save_ct_checkpoint,
)
import requests

COLLECTION = "ClinicalTrials_results"
BATCH_SIZE = 100
CHECKPOINT_LABEL = "ct_results"
CHECKPOINT_INTERVAL = 200
MAX_NON_SERIOUS_AES = 20

FILTER_ADVANCED = (
    "AREA[StudyType]INTERVENTIONAL AND "
    "(AREA[InterventionType]DRUG OR AREA[InterventionType]BIOLOGICAL) AND "
    "AREA[Phase](PHASE2 OR PHASE3 OR PHASE4) AND "
    "AREA[HasResults]true"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


# ── Outcome measure extraction ──────────────────────────────────────

def _extract_outcome_text(outcome: Dict, groups: List[Dict]) -> str:
    """Build a readable summary string for a single outcome measure."""
    title = outcome.get("title", "Untitled")
    om_type = outcome.get("type", "")
    unit = outcome.get("unitOfMeasure", "")
    timeframe = outcome.get("timeFrame", "")
    param_type = outcome.get("paramType", "")

    group_map = {g.get("id", ""): g.get("title", f"Group {i}")
                 for i, g in enumerate(outcome.get("groups", []))}

    # Measurements: flatten classes → categories → measurements
    value_parts = []
    for cls in outcome.get("classes", [])[:3]:
        for cat in cls.get("categories", [])[:3]:
            for m in cat.get("measurements", [])[:6]:
                gid = m.get("groupId", "")
                val = m.get("value", "")
                spread = m.get("spread", "")
                arm = group_map.get(gid, gid)
                entry = f"{arm}={val}"
                if spread:
                    entry += f"±{spread}"
                value_parts.append(entry)

    values_str = ", ".join(value_parts[:8]) if value_parts else "N/A"

    # Statistical analyses
    analysis_parts = []
    for a in outcome.get("analyses", [])[:3]:
        p_val = a.get("pValue", "")
        method = a.get("statisticalMethod", "")
        est_type = a.get("paramType", "")
        est_val = a.get("paramValue", "")
        ci_lo = a.get("ciLowerLimit", "")
        ci_hi = a.get("ciUpperLimit", "")
        part = ""
        if method:
            part += method
        if p_val:
            part += f", p={p_val}"
        if est_type and est_val:
            part += f", {est_type}={est_val}"
        if ci_lo and ci_hi:
            part += f" (95% CI [{ci_lo}, {ci_hi}])"
        if part:
            analysis_parts.append(part)

    analysis_str = "; ".join(analysis_parts) if analysis_parts else ""

    line = f"{om_type}: {title}"
    if param_type:
        line += f" ({param_type})"
    line += f" — {values_str} ({unit})"
    if timeframe:
        line += f". Timeframe: {timeframe}"
    if analysis_str:
        line += f". Analysis: {analysis_str}"
    return line


# ── Adverse events extraction ───────────────────────────────────────

def _extract_ae_text(ae_module: Dict) -> tuple:
    """Return (serious_ae_text, other_ae_text, num_serious, num_other, p_values_from_ae)."""
    timeframe = ae_module.get("timeFrame", "")
    header = f"Adverse events timeframe: {timeframe}. " if timeframe else ""

    # Event groups (arm descriptions)
    event_groups = ae_module.get("eventGroups", [])
    group_map = {g.get("id", ""): g.get("title", "") for g in event_groups}

    # Serious AEs — keep ALL
    serious = ae_module.get("seriousEvents", [])
    serious_lines = []
    for ev in serious:
        term = ev.get("term", "")
        organ = ev.get("organSystem", "")
        stat_parts = []
        for s in ev.get("stats", []):
            arm = group_map.get(s.get("groupId", ""), "")
            affected = s.get("numAffected", 0)
            at_risk = s.get("numAtRisk", 0)
            if arm and at_risk:
                stat_parts.append(f"{arm}: {affected}/{at_risk}")
        stats_str = "; ".join(stat_parts)
        line = f"{term} ({organ}): {stats_str}" if stats_str else f"{term} ({organ})"
        serious_lines.append(line)

    # Non-serious AEs — sort by total affected, keep top N
    other_events = ae_module.get("otherEvents", [])
    scored_other = []
    for ev in other_events:
        total_affected = sum(s.get("numAffected", 0) for s in ev.get("stats", []))
        scored_other.append((total_affected, ev))
    scored_other.sort(key=lambda x: x[0], reverse=True)

    other_lines = []
    for _, ev in scored_other[:MAX_NON_SERIOUS_AES]:
        term = ev.get("term", "")
        total = sum(s.get("numAffected", 0) for s in ev.get("stats", []))
        other_lines.append(f"{term} ({total} affected)")

    serious_text = ""
    if serious_lines:
        serious_text = f"{header}SERIOUS ADVERSE EVENTS ({len(serious_lines)}): " + "; ".join(serious_lines)

    other_text = ""
    if other_lines:
        other_text = (
            f"OTHER ADVERSE EVENTS (top {len(other_lines)} of {len(other_events)} by frequency): "
            + "; ".join(other_lines)
        )

    return serious_text, other_text, len(serious), len(other_events)


# ── Document builder ────────────────────────────────────────────────

def _build_results_doc(study: Dict) -> Optional[Dict]:
    proto = study.get("protocolSection", {})
    results = study.get("resultsSection", {})
    if not results:
        return None

    ident = proto.get("identificationModule", {})
    nct_id = ident.get("nctId", "")
    if not nct_id:
        return None

    drug_interventions = extract_drug_interventions(study)
    if not drug_interventions:
        return None

    conditions = extract_conditions(study)
    phases = safe_get(proto, "designModule", "phases", default=[])
    phase_str = phase_list_to_str(phases)
    phase_num = phase_list_to_numeric(phases)

    brief_title = ident.get("briefTitle", "")
    sponsor = safe_get(proto, "sponsorCollaboratorsModule", "leadSponsor", "name", default="")
    sponsor_class = safe_get(proto, "sponsorCollaboratorsModule", "leadSponsor", "class", default="")
    enrollment = safe_get(proto, "designModule", "enrollmentInfo", "count", default=0)
    drug_names = [iv["name"] for iv in drug_interventions]

    # Outcome measures
    om_module = results.get("outcomeMeasuresModule", {})
    all_outcomes = om_module.get("outcomeMeasures", [])

    primary = [o for o in all_outcomes if o.get("type") == "PRIMARY"]
    secondary = [o for o in all_outcomes if o.get("type") == "SECONDARY"][:5]

    outcome_texts = []
    p_values = []

    for o in primary + secondary:
        outcome_texts.append(_extract_outcome_text(o, o.get("groups", [])))
        for a in o.get("analyses", []):
            pv = a.get("pValue", "")
            if pv:
                p_values.append(pv)

    primary_titles = [o.get("title", "") for o in primary if o.get("title")]

    # Adverse events
    ae_module = results.get("adverseEventsModule", {})
    serious_text, other_text, n_serious, n_other = "", "", 0, 0
    if ae_module:
        serious_text, other_text, n_serious, n_other = _extract_ae_text(ae_module)

    # Participant flow summary
    flow_module = results.get("participantFlowModule", {})
    flow_text = ""
    if flow_module:
        periods = flow_module.get("periods", [])
        if periods:
            milestones = periods[0].get("milestones", [])
            parts = []
            for ms in milestones[:3]:
                ms_type = ms.get("type", "")
                total = 0
                for a in ms.get("achievements", []):
                    val = a.get("numSubjects", 0)
                    try:
                        total += int(val)
                    except (ValueError, TypeError):
                        pass
                if ms_type and total:
                    parts.append(f"{ms_type}: {total}")
            if parts:
                flow_text = "Participant flow: " + ", ".join(parts) + "."

    # Assemble text_content
    cond_str = ", ".join(conditions[:8]) if conditions else "Not specified"
    drug_str = ", ".join(drug_names[:8])

    sections = [
        f"Clinical trial results for {nct_id}: {brief_title}.",
        f"Phase {phase_str}. Conditions: {cond_str}. Drugs: {drug_str}.",
        f"Sponsor: {sponsor}. Enrollment: {enrollment}.",
    ]
    if outcome_texts:
        sections.append("OUTCOMES: " + " | ".join(outcome_texts))
    if serious_text:
        sections.append(serious_text)
    if other_text:
        sections.append(other_text)
    if flow_text:
        sections.append(flow_text)

    text_content = " ".join(sections)

    return {
        "id": f"ct_results_{nct_id}",
        "entity_type": "clinical_trial_results",
        "nct_id": nct_id,
        "brief_title": brief_title,
        "phase": phase_str,
        "phase_numeric": phase_num,
        "conditions": conditions,
        "drug_names": drug_names,
        "sponsor": sponsor,
        "sponsor_class": sponsor_class,
        "enrollment": enrollment,
        "has_primary_outcomes": len(primary) > 0,
        "num_primary_outcomes": len(primary),
        "num_serious_aes": n_serious,
        "num_other_aes": n_other,
        "primary_outcome_titles": primary_titles,
        "p_values": p_values,
        "source": "ClinicalTrials.gov",
        "text_content": text_content,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest ClinicalTrials.gov results data")
    parser.add_argument("--limit", type=int, default=0, help="Max studies (0=all)")
    parser.add_argument("--recreate", action="store_true", help="Wipe collection first")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    processed = set()
    if not (args.no_resume or args.recreate):
        processed = load_ct_checkpoint(CHECKPOINT_LABEL)

    session = requests.Session()
    batch: list = []
    stats = {"ingested": 0, "skipped_checkpoint": 0, "skipped_no_results": 0, "chunked": 0}
    start = datetime.now()

    for study in iterate_studies(session, FILTER_ADVANCED):
        nct_id = safe_get(study, "protocolSection", "identificationModule", "nctId", default="")
        if not nct_id:
            continue
        if nct_id in processed:
            stats["skipped_checkpoint"] += 1
            continue

        doc = _build_results_doc(study)
        if not doc:
            stats["skipped_no_results"] += 1
            processed.add(nct_id)
            continue

        chunks = chunk_text(doc["text_content"])
        if len(chunks) == 1:
            batch.append(doc)
        else:
            stats["chunked"] += 1
            for ci, chunk in enumerate(chunks):
                chunked_doc = {**doc}
                chunked_doc["id"] = f"{doc['id']}_c{ci}"
                chunked_doc["text_content"] = chunk
                chunked_doc["chunk_index"] = ci
                chunked_doc["total_chunks"] = len(chunks)
                batch.append(chunked_doc)

        stats["ingested"] += 1
        processed.add(nct_id)

        if len(batch) >= BATCH_SIZE:
            upsert_batch(client, embedder, COLLECTION, batch)
            batch = []

        if stats["ingested"] % CHECKPOINT_INTERVAL == 0:
            save_ct_checkpoint(CHECKPOINT_LABEL, processed)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"Results: {stats['ingested']:,} | "
                f"Skipped (checkpoint): {stats['skipped_checkpoint']:,} | "
                f"Skipped (no results): {stats['skipped_no_results']:,} | "
                f"Chunked: {stats['chunked']:,} | "
                f"{elapsed / 60:.1f} min"
            )

        if args.limit and stats["ingested"] >= args.limit:
            break

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    save_ct_checkpoint(CHECKPOINT_LABEL, processed)
    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()

    logger.info("=" * 60)
    logger.info(
        f"DONE | Ingested: {stats['ingested']:,} | No-results: {stats['skipped_no_results']:,} | "
        f"Chunked: {stats['chunked']:,} | Collection points: {count:,} | "
        f"{elapsed / 60:.1f} min"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
