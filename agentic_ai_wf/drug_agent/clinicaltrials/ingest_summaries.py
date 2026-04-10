#!/usr/bin/env python3
"""
Ingest ClinicalTrials.gov Phase 2-4 drug/biological trial protocol summaries.

Fetches full study records via API v2, extracts protocol data client-side,
and embeds one document per study into the ClinicalTrials_summaries collection.

Usage:
    python -m agentic_ai_wf.drug_agent.clinicaltrials.ingest_summaries
    python -m ... --limit 50          # test batch
    python -m ... --recreate          # wipe and re-ingest
    python -m ... --no-resume         # ignore checkpoint
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
)
from agentic_ai_wf.drug_agent.clinicaltrials.ct_base import (
    safe_get, iterate_studies, extract_drug_interventions,
    extract_conditions, extract_conditions_mesh, extract_interventions_mesh,
    phase_list_to_numeric, phase_list_to_str, chunk_text,
    load_ct_checkpoint, save_ct_checkpoint,
)
import requests

COLLECTION = "ClinicalTrials_summaries"
BATCH_SIZE = 100
CHECKPOINT_LABEL = "ct_summaries"
CHECKPOINT_INTERVAL = 500

FILTER_ADVANCED = (
    "AREA[StudyType]INTERVENTIONAL AND "
    "(AREA[InterventionType]DRUG OR AREA[InterventionType]BIOLOGICAL) AND "
    "AREA[Phase](PHASE2 OR PHASE3 OR PHASE4)"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def _build_summary_doc(study: Dict) -> Optional[Dict]:
    """Extract protocol fields from a full study record and build a single document."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    desc = proto.get("descriptionModule", {})
    elig = proto.get("eligibilityModule", {})

    nct_id = ident.get("nctId", "")
    if not nct_id:
        return None

    drug_interventions = extract_drug_interventions(study)
    if not drug_interventions:
        return None

    conditions = extract_conditions(study)
    phases = design.get("phases", [])
    phase_str = phase_list_to_str(phases)
    phase_num = phase_list_to_numeric(phases)

    brief_title = ident.get("briefTitle", "")
    official_title = ident.get("officialTitle", "")
    overall_status = status.get("overallStatus", "")

    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    lead = sponsor_mod.get("leadSponsor", {})
    sponsor_name = lead.get("name", "")
    sponsor_class = lead.get("class", "")

    enrollment_info = design.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count", 0)
    enrollment_type = enrollment_info.get("type", "")

    start_date = safe_get(status, "startDateStruct", "date", default="")
    completion_date = safe_get(status, "completionDateStruct", "date", default="")

    brief_summary = (desc.get("briefSummary") or "")[:1500]

    primary_outcomes = [
        {"measure": o.get("measure", ""), "description": (o.get("description") or "")[:200],
         "timeframe": o.get("timeFrame", "")}
        for o in safe_get(proto, "outcomesModule", "primaryOutcomes", default=[])
    ]
    secondary_outcomes = [
        {"measure": o.get("measure", ""), "timeframe": o.get("timeFrame", "")}
        for o in safe_get(proto, "outcomesModule", "secondaryOutcomes", default=[])[:5]
    ]

    design_info = design.get("designInfo", {})
    study_design = {
        "allocation": design_info.get("allocation", ""),
        "purpose": design_info.get("primaryPurpose", ""),
        "masking": safe_get(design_info, "maskingInfo", "masking", default=""),
        "model": design_info.get("interventionModel", ""),
    }

    drug_names = [iv["name"] for iv in drug_interventions]
    conditions_mesh = extract_conditions_mesh(study)
    interventions_mesh = extract_interventions_mesh(study)

    is_fda = safe_get(proto, "oversightModule", "isFdaRegulatedDrug", default=False)
    has_results = study.get("hasResults", False)
    why_stopped = status.get("whyStopped", None)

    # Build text_content
    cond_str = ", ".join(conditions[:10]) if conditions else "Not specified"
    drug_str = "; ".join(
        f"{iv['name']} ({iv['description'][:150]})" if iv['description'] else iv['name']
        for iv in drug_interventions[:8]
    )
    outcome_str = "; ".join(
        f"{o['measure']} ({o['timeframe']})" for o in primary_outcomes if o['measure']
    )
    sec_outcome_str = "; ".join(o['measure'] for o in secondary_outcomes if o['measure'])

    design_parts = [v for v in study_design.values() if v]
    design_str = ", ".join(design_parts) if design_parts else "Not specified"

    age_range = f"{elig.get('minimumAge', 'N/A')} – {elig.get('maximumAge', 'N/A')}"
    eligibility_snippet = (elig.get("eligibilityCriteria") or "")[:500]

    text_content = (
        f"Clinical trial {nct_id}: {official_title or brief_title}. "
        f"Phase {phase_str}. Status: {overall_status}. "
        f"Sponsor: {sponsor_name} ({sponsor_class}). "
        f"Enrollment: {enrollment} {enrollment_type} participants. "
        f"Conditions studied: {cond_str}. "
        f"Drug interventions: {drug_str}. "
        f"Study design: {design_str}. "
        f"Brief summary: {brief_summary}. "
        f"Primary outcomes: {outcome_str}. "
        f"Secondary outcomes: {sec_outcome_str}. "
        f"Eligibility: {age_range}. {eligibility_snippet}"
    )

    return {
        "id": f"ct_summary_{nct_id}",
        "entity_type": "clinical_trial_summary",
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": overall_status,
        "phase": phase_str,
        "phase_numeric": phase_num,
        "start_date": start_date,
        "completion_date": completion_date,
        "enrollment": enrollment,
        "enrollment_type": enrollment_type,
        "sponsor": sponsor_name,
        "sponsor_class": sponsor_class,
        "conditions": conditions,
        "conditions_mesh": conditions_mesh,
        "interventions": drug_interventions,
        "interventions_mesh": interventions_mesh,
        "drug_names": drug_names,
        "primary_outcomes": primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "design": study_design,
        "is_fda_regulated": is_fda,
        "has_results": has_results,
        "why_stopped": why_stopped,
        "source": "ClinicalTrials.gov",
        "text_content": text_content,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest ClinicalTrials.gov protocol summaries")
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
    stats = {"ingested": 0, "skipped_checkpoint": 0, "skipped_no_drugs": 0, "chunked": 0}
    start = datetime.now()

    for study in iterate_studies(session, FILTER_ADVANCED):
        nct_id = safe_get(study, "protocolSection", "identificationModule", "nctId", default="")
        if not nct_id:
            continue
        if nct_id in processed:
            stats["skipped_checkpoint"] += 1
            continue

        doc = _build_summary_doc(study)
        if not doc:
            stats["skipped_no_drugs"] += 1
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
                f"Studies: {stats['ingested']:,} | "
                f"Checkpoint-skipped: {stats['skipped_checkpoint']:,} | "
                f"No-drugs-skipped: {stats['skipped_no_drugs']:,} | "
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
        f"DONE | Ingested: {stats['ingested']:,} | No-drugs: {stats['skipped_no_drugs']:,} | "
        f"Chunked: {stats['chunked']:,} | Collection points: {count:,} | "
        f"{elapsed / 60:.1f} min"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
