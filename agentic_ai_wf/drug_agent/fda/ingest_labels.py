#!/usr/bin/env python3
"""
Ingest FDA drug labels (SPL) — clinical narrative sections.
Downloads bulk JSON (254K labels, 1.7 GB), filters to prescription drugs,
splits into section-level documents with text chunking for long sections.

Usage:
    python ingest_labels.py                      # Full run
    python ingest_labels.py --limit 1000         # Test with 1000 labels
    python ingest_labels.py --recreate           # Wipe and re-ingest
    python ingest_labels.py --all-types          # Include OTC + other label types
"""

import re
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
)
from agentic_ai_wf.drug_agent.fda.fda_base import (
    stream_bulk_json, get_bulk_download_urls, load_fda_checkpoint,
    save_fda_checkpoint,
)
import requests, zipfile, io, json as _json
from agentic_ai_wf.drug_agent.fda.fda_base import _BROWSER_HEADERS

COLLECTION = "FDA_Drug_Labels"
BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# High-value clinical sections to extract and embed
TARGET_SECTIONS = [
    "indications_and_usage",
    "mechanism_of_action",
    "adverse_reactions",
    "contraindications",
    "drug_interactions",
    "boxed_warning",
    "clinical_pharmacology",
    "warnings_and_cautions",
    "warnings",
    "pharmacogenomics",
]

SECTION_DISPLAY = {
    "indications_and_usage": "Indications and Usage",
    "mechanism_of_action": "Mechanism of Action",
    "adverse_reactions": "Adverse Reactions",
    "contraindications": "Contraindications",
    "drug_interactions": "Drug Interactions",
    "boxed_warning": "Boxed Warning",
    "clinical_pharmacology": "Clinical Pharmacology",
    "warnings_and_cautions": "Warnings and Cautions",
    "warnings": "Warnings",
    "pharmacogenomics": "Pharmacogenomics",
}

CHUNK_SIZE = 3500
CHUNK_OVERLAP = 500


def _clean_html(text):
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _extract_openfda(record):
    of = record.get("openfda", {})
    return {
        "brand_name": ", ".join(of.get("brand_name", [])),
        "generic_name": ", ".join(of.get("generic_name", [])),
        "manufacturer": ", ".join(of.get("manufacturer_name", [])),
        "product_type": ", ".join(of.get("product_type", [])),
        "route": ", ".join(of.get("route", [])),
        "application_number": ", ".join(of.get("application_number", [])),
        "pharm_class_epc": ", ".join(of.get("pharm_class_epc", [])),
        "pharm_class_moa": ", ".join(of.get("pharm_class_moa", [])),
        "spl_id": ", ".join(of.get("spl_id", [])),
    }


def build_docs(record, include_all_types=False):
    """Yield section-level documents from one label record."""
    of = _extract_openfda(record)

    if not include_all_types and "HUMAN PRESCRIPTION DRUG" not in of["product_type"].upper():
        return

    spl_id = record.get("id", of["spl_id"] or "unknown")
    drug_label = of["brand_name"] or of["generic_name"] or "Unknown"

    for section_key in TARGET_SECTIONS:
        raw_values = record.get(section_key, [])
        if not raw_values:
            continue

        raw_text = " ".join(raw_values) if isinstance(raw_values, list) else str(raw_values)
        clean = _clean_html(raw_text)
        if len(clean) < 20:
            continue

        section_title = SECTION_DISPLAY.get(section_key, section_key)
        chunks = _chunk_text(clean)

        for ci, chunk in enumerate(chunks):
            chunk_suffix = f"_c{ci}" if len(chunks) > 1 else ""
            doc_id = f"label_{spl_id}_{section_key}{chunk_suffix}"

            yield {
                "id": doc_id,
                "entity_type": "drug_label_section",
                "spl_id": spl_id,
                "brand_name": of["brand_name"],
                "generic_name": of["generic_name"],
                "manufacturer": of["manufacturer"],
                "application_number": of["application_number"],
                "product_type": of["product_type"],
                "route": of["route"],
                "section_name": section_key,
                "section_title": section_title,
                "section_length": len(clean),
                "chunk_index": ci,
                "total_chunks": len(chunks),
                "pharm_class_epc": of["pharm_class_epc"],
                "pharm_class_moa": of["pharm_class_moa"],
                "text_content": (
                    f"FDA Drug Label — {section_title} for {drug_label} "
                    f"({of['generic_name']}): {chunk}"
                ),
                "source": "FDA_DrugLabels",
            }


def main():
    parser = argparse.ArgumentParser(description="Ingest FDA Drug Labels")
    parser.add_argument("--limit", type=int, default=0, help="Max labels to process (0=all)")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--all-types", action="store_true", help="Include OTC and other label types")
    args = parser.parse_args()

    client = get_qdrant()
    embedder = get_embedder()
    ensure_collection(client, COLLECTION, args.recreate)

    # File-level checkpointing: skip already-processed bulk files on resume
    CP_LABEL = "labels_files"
    if args.recreate:
        save_fda_checkpoint(CP_LABEL, set())
    completed_files = load_fda_checkpoint(CP_LABEL)

    urls = get_bulk_download_urls("drug/label")
    logger.info(f"Streaming FDA drug labels from {len(urls)} bulk files (~1.7 GB)...")
    start = datetime.now()
    batch = []
    total_docs = 0
    labels_processed = 0
    labels_with_sections = 0
    hit_limit = False

    for i, url in enumerate(urls):
        file_key = url.split("/")[-1]
        if file_key in completed_files:
            logger.info(f"  Skipping bulk file {i+1}/{len(urls)} (checkpointed): {file_key}")
            continue

        logger.info(f"Downloading bulk file {i+1}/{len(urls)}: {file_key}")
        try:
            resp = requests.get(url, timeout=300, stream=True, headers=_BROWSER_HEADERS)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    if not name.endswith(".json"):
                        continue
                    data = _json.loads(zf.read(name))
                    for record in data.get("results", []):
                        had_docs = False
                        for doc in build_docs(record, args.all_types):
                            batch.append(doc)
                            total_docs += 1
                            had_docs = True

                            if len(batch) >= BATCH_SIZE:
                                upsert_batch(client, embedder, COLLECTION, batch)
                                batch = []

                        labels_processed += 1
                        if had_docs:
                            labels_with_sections += 1

                        if labels_processed % 5000 == 0:
                            elapsed = (datetime.now() - start).total_seconds()
                            logger.info(
                                f"Labels: {labels_processed:,} | With sections: {labels_with_sections:,} | "
                                f"Docs: {total_docs:,} | {elapsed / 60:.1f}min"
                            )

                        if args.limit and labels_processed >= args.limit:
                            hit_limit = True
                            break
                    if hit_limit:
                        break
        except Exception as e:
            logger.error(f"Failed on bulk file {url}: {e}")
            continue

        # Checkpoint this file as fully processed
        if not hit_limit:
            completed_files.add(file_key)
            save_fda_checkpoint(CP_LABEL, completed_files)
            logger.info(f"  Checkpointed file {i+1}/{len(urls)} — {total_docs:,} docs so far")

        if hit_limit:
            break

    if batch:
        upsert_batch(client, embedder, COLLECTION, batch)

    count = client.count(COLLECTION).count
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=" * 60)
    logger.info(
        f"DONE | Labels: {labels_processed:,} | With sections: {labels_with_sections:,} | "
        f"Docs: {total_docs:,} | Collection: {count:,} | {elapsed / 60:.1f}min"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
