#!/usr/bin/env python3
"""
Ingest pharmacogenomics data from Open Targets.
Dual-mode: fetches PGx from the drug endpoint (3,230 drugs) AND target endpoint
(~50 key pharmacogenes), deduplicates, and ingests into a new collection.

Usage:
    python ingest_pharmacogenomics.py                  # Full run
    python ingest_pharmacogenomics.py --limit 50       # Test with 50 drugs
    python ingest_pharmacogenomics.py --recreate       # Wipe and re-ingest
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

COLLECTION = "OpenTargets_pharmacogenomics"
CHECKPOINT_LABEL = "pharmacogenomics"
BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

QUERY_DRUG_PGX = """
query($id: String!) {
  drug(chemblId: $id) {
    pharmacogenomics {
      variantRsId variantId genotype genotypeId
      pgxCategory phenotypeText evidenceLevel
      datasourceId isDirectTarget targetFromSourceId
      genotypeAnnotationText literature
    }
  }
}"""

QUERY_TARGET_PGX = """
query($id: String!) {
  target(ensemblId: $id) {
    approvedSymbol
    pharmacogenomics {
      variantRsId variantId genotype genotypeId
      pgxCategory phenotypeText evidenceLevel
      datasourceId isDirectTarget targetFromSourceId
      genotypeAnnotationText literature
      drugs { drugId drugFromSource }
    }
    safetyLiabilities {
      event eventId datasource literature
      effects { direction dosing }
      biosamples { tissueLabel cellLabel }
    }
  }
}"""

# Key pharmacogenes — dynamically discovered targets with high PGx relevance
PHARMACOGENES = {
    "ENSG00000100197": "CYP2D6",   "ENSG00000165841": "CYP2C19",
    "ENSG00000138115": "CYP2C9",   "ENSG00000160868": "CYP3A4",
    "ENSG00000188641": "DPYD",     "ENSG00000137364": "TPMT",
    "ENSG00000241635": "UGT1A1",   "ENSG00000167397": "VKORC1",
    "ENSG00000196344": "ADH7",     "ENSG00000255027": "HLA-A",
    "ENSG00000234745": "HLA-B",    "ENSG00000204568": "MRPS36",
    "ENSG00000108846": "ABCC3",    "ENSG00000085563": "ABCB1",
    "ENSG00000170006": "NUDT15",   "ENSG00000196502": "IFNL4",
    "ENSG00000163235": "TGFA",     "ENSG00000141736": "ERBB2",
    "ENSG00000091831": "ESR1",     "ENSG00000171862": "PTEN",
    "ENSG00000012048": "BRCA1",    "ENSG00000139618": "BRCA2",
    "ENSG00000157764": "BRAF",     "ENSG00000133703": "KRAS",
    "ENSG00000168078": "PBX4",     "ENSG00000196218": "RYR1",
    "ENSG00000117400": "MPL",      "ENSG00000105397": "TYK2",
    "ENSG00000162434": "JAK1",     "ENSG00000096968": "JAK2",
}


def _dedup_key(drug_id: str, variant: str, genotype: str, category: str) -> str:
    return f"{drug_id}|{variant}|{genotype}|{category}"


def _pgx_text(entry: dict) -> str:
    parts = [f"Pharmacogenomics: {entry['gene_symbol']} variant {entry['variant_rs_id']}"]
    if entry.get("genotype"):
        parts.append(f"(genotype {entry['genotype']})")
    parts.append(f"affects {entry['drug_name']} {entry['pgx_category']}.")
    if entry.get("phenotype_text"):
        parts.append(f"Clinical effect: {entry['phenotype_text']}.")
    parts.append(f"Evidence: {entry['evidence_level']}.")
    if entry.get("is_direct_target"):
        parts.append(f"{entry['gene_symbol']} is a direct drug target.")
    if entry.get("datasource"):
        parts.append(f"Source: {entry['datasource']}.")
    return " ".join(parts)


def _safety_text(gene: str, s: dict) -> str:
    parts = [f"Target safety liability for {gene}: {s.get('event', '')}"]
    for eff in s.get("effects") or []:
        if eff.get("direction"):
            parts.append(f"({eff['direction']}, {eff.get('dosing', '')} dosing)")
    if s.get("datasource"):
        parts.append(f"Source: {s['datasource']}.")
    tissues = [b.get("tissueLabel", "") for b in (s.get("biosamples") or []) if b.get("tissueLabel")]
    if tissues:
        parts.append(f"Tissues: {', '.join(tissues[:5])}.")
    return " ".join(parts)


def fetch_drug_pgx(session, drug_id, drug_name, drug_type, max_phase):
    """Yield PGx documents from the drug endpoint."""
    data = gql_query(session, QUERY_DRUG_PGX, {"id": drug_id})
    if not data:
        return
    for p in (data.get("drug") or {}).get("pharmacogenomics") or []:
        gene = p.get("targetFromSourceId", "")
        entry = {
            "id": _dedup_key(drug_id, p.get("variantRsId", ""), p.get("genotype", ""), p.get("pgxCategory", "")),
            "entity_type": "pharmacogenomics",
            "variant_rs_id": p.get("variantRsId", ""),
            "variant_id": p.get("variantId", ""),
            "genotype": p.get("genotype", ""),
            "genotype_id": p.get("genotypeId", ""),
            "pgx_category": p.get("pgxCategory", ""),
            "phenotype_text": p.get("phenotypeText", ""),
            "evidence_level": p.get("evidenceLevel", ""),
            "datasource": p.get("datasourceId", ""),
            "is_direct_target": p.get("isDirectTarget", False),
            "gene_symbol": gene,
            "drug_id": drug_id,
            "drug_name": drug_name,
            "drug_type": drug_type,
            "max_phase": max_phase,
            "genotype_annotation": p.get("genotypeAnnotationText", ""),
            "literature": p.get("literature") or [],
            "source": "OpenTargets_PGx_drug",
        }
        entry["text_content"] = _pgx_text(entry)
        yield entry


def fetch_target_pgx(session, ensembl_id, gene_symbol):
    """Yield PGx + safety docs from the target endpoint."""
    data = gql_query(session, QUERY_TARGET_PGX, {"id": ensembl_id})
    if not data:
        return
    target = data.get("target")
    if not target:
        return

    for p in target.get("pharmacogenomics") or []:
        for drug_ref in p.get("drugs") or [{"drugId": "", "drugFromSource": ""}]:
            drug_id = drug_ref.get("drugId", "")
            drug_name = drug_ref.get("drugFromSource", "")
            entry = {
                "id": _dedup_key(drug_id, p.get("variantRsId", ""), p.get("genotype", ""), p.get("pgxCategory", "")),
                "entity_type": "pharmacogenomics",
                "variant_rs_id": p.get("variantRsId", ""),
                "variant_id": p.get("variantId", ""),
                "genotype": p.get("genotype", ""),
                "genotype_id": p.get("genotypeId", ""),
                "pgx_category": p.get("pgxCategory", ""),
                "phenotype_text": p.get("phenotypeText", ""),
                "evidence_level": p.get("evidenceLevel", ""),
                "datasource": p.get("datasourceId", ""),
                "is_direct_target": p.get("isDirectTarget", False),
                "gene_symbol": gene_symbol,
                "drug_id": drug_id,
                "drug_name": drug_name,
                "drug_type": "",
                "genotype_annotation": p.get("genotypeAnnotationText", ""),
                "literature": p.get("literature") or [],
                "source": "OpenTargets_PGx_target",
            }
            entry["text_content"] = _pgx_text(entry)
            yield entry

    # Safety liabilities bundled with the same API call
    for s in target.get("safetyLiabilities") or []:
        yield {
            "id": f"safety_{ensembl_id}_{s.get('eventId', s.get('event', ''))}",
            "entity_type": "target_safety",
            "gene_symbol": gene_symbol,
            "ensembl_id": ensembl_id,
            "event": s.get("event", ""),
            "event_id": s.get("eventId", ""),
            "datasource": s.get("datasource", ""),
            "literature": s.get("literature") or [],
            "effects": s.get("effects") or [],
            "biosamples": [b.get("tissueLabel", "") for b in (s.get("biosamples") or [])],
            "text_content": _safety_text(gene_symbol, s),
            "source": "OpenTargets_TargetSafety",
        }


def main():
    parser = argparse.ArgumentParser(description="Ingest Open Targets pharmacogenomics")
    parser.add_argument("--limit", type=int, default=0, help="Max drugs to process (0=all)")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    drugs = load_drug_ids_from_qdrant(client)
    if args.limit:
        drugs = drugs[:args.limit]

    processed = set() if args.no_resume or args.recreate else load_checkpoint(CHECKPOINT_LABEL)
    remaining = [(d, n, t, p) for d, n, t, p in drugs if d not in processed]
    logger.info(f"Drug-side PGx: {len(remaining):,} remaining ({len(processed):,} done)")

    session = requests.Session()
    session.headers["Content-Type"] = "application/json"
    seen_keys, batch = set(), []
    stats = {"pgx": 0, "safety": 0, "drugs_done": len(processed), "skipped_dup": 0}
    start = datetime.now()

    # Phase 1: Drug-centric PGx
    logger.info("=" * 50)
    logger.info("PHASE 1: Drug-centric pharmacogenomics")
    logger.info("=" * 50)

    for drug_id, drug_name, drug_type, max_phase in remaining:
        for doc in fetch_drug_pgx(session, drug_id, drug_name, drug_type, max_phase):
            if doc["id"] in seen_keys:
                stats["skipped_dup"] += 1
                continue
            seen_keys.add(doc["id"])
            batch.append(doc)
            stats["pgx"] += 1

            if len(batch) >= BATCH_SIZE:
                upsert_batch(client, embedder, COLLECTION, batch)
                batch = []

        processed.add(drug_id)
        stats["drugs_done"] += 1

        if stats["drugs_done"] % 100 == 0:
            save_checkpoint(CHECKPOINT_LABEL, processed)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"Drugs: {stats['drugs_done']:,}/{len(drugs):,} | "
                f"PGx: {stats['pgx']:,} | Dups skipped: {stats['skipped_dup']:,} | "
                f"{elapsed/60:.1f}min"
            )

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)
        batch = []

    save_checkpoint(CHECKPOINT_LABEL, processed)
    logger.info(f"Phase 1 complete: {stats['pgx']:,} PGx entries from drugs")

    # Phase 2: Target-centric PGx + Safety
    logger.info("=" * 50)
    logger.info(f"PHASE 2: Target-centric PGx ({len(PHARMACOGENES)} pharmacogenes)")
    logger.info("=" * 50)

    for ensembl_id, gene_symbol in PHARMACOGENES.items():
        for doc in fetch_target_pgx(session, ensembl_id, gene_symbol):
            if doc["id"] in seen_keys:
                stats["skipped_dup"] += 1
                continue
            seen_keys.add(doc["id"])
            batch.append(doc)
            if doc["entity_type"] == "target_safety":
                stats["safety"] += 1
            else:
                stats["pgx"] += 1

            if len(batch) >= BATCH_SIZE:
                upsert_batch(client, embedder, COLLECTION, batch)
                batch = []

        logger.info(f"  {gene_symbol}: PGx={stats['pgx']:,} Safety={stats['safety']:,}")

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()

    logger.info("=" * 50)
    logger.info(
        f"DONE | PGx: {stats['pgx']:,} | Safety: {stats['safety']:,} | "
        f"Dups: {stats['skipped_dup']:,} | Collection: {count:,} | {elapsed/60:.1f}min"
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
