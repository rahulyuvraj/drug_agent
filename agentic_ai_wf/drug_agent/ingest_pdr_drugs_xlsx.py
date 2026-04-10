#!/usr/bin/env python3
"""
PDR Drugs XLSX → Qdrant Ingestion
===================================

Reads Sub11_PDR_Drugs_Data XLSX, parses complex target formats (gene+HSA,
bare HSA, molecular targets), explodes gene targets into individual docs,
and upserts to a new PDR_Drugs_Data collection.

Two doc_types:
  pdr_drug_target    — one doc per extracted gene symbol (exact-match filterable)
  pdr_drug_reference — one doc per drug for non-gene targets + null targets

Usage:
    python ingest_pdr_drugs_xlsx.py --input "Sub11_PDR_Drugs_Data_v010_Final_Updated(pathwys targets corrected) (1).xlsx"
    python ingest_pdr_drugs_xlsx.py --input "Sub11_..." --dry-run
"""

import os
import re
import hashlib
import logging
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Target / pathway / AR parsers ────────────────────────────────────────────

# Gene symbol with HSA ID — handles NR1B (RAR), BRAF*, [HSA_VAR:...] prefix
GENE_HSA_PATTERN = re.compile(
    r'([A-Z][A-Z0-9*]*(?:\s*\([^)]+\))?)\s*'
    r'(?:\[HSA_VAR:[^\]]+\]\s*)?'
    r'\[HSA[:\s]+([\d\s]+)\]'
)

KEGG_PATTERN = re.compile(r'(hsa\d+)\([^)]+\)\s+([^,]+)')


def extract_gene_targets(target_str):
    """Parse gene symbols with HSA IDs from the complex targets column."""
    if pd.isna(target_str):
        return []
    matches = GENE_HSA_PATTERN.findall(str(target_str))
    results = []
    for gene_raw, hsa_raw in matches:
        gene = gene_raw.split('(')[0].strip().rstrip('*')
        hsa_ids = hsa_raw.strip().split()
        if gene:
            results.append((gene, hsa_ids))
    return results


def extract_pathways(pathway_str):
    """Parse KEGG pathway IDs and names."""
    if pd.isna(pathway_str) or str(pathway_str).strip() == "Not Found":
        return [], []
    matches = KEGG_PATTERN.findall(str(pathway_str))
    ids = [m[0].strip() for m in matches]
    names = [m[1].strip() for m in matches]
    return ids, names


def parse_adverse_reactions(ar_str):
    """Severity-bucketed adverse reaction parsing."""
    tiers = {"severe": [], "moderate": [], "mild": []}
    if pd.isna(ar_str):
        return tiers
    current = "unknown"
    for line in str(ar_str).split('\n'):
        line = line.strip()
        low = line.lower()
        if low in tiers:
            current = low
        elif '/' in line and current in tiers:
            reaction = line.split('/')[0].strip()
            if reaction:
                tiers[current].append(reaction)
    return tiers


# ── Data loading ─────────────────────────────────────────────────────────────

COLUMN_MAP = {
    "Drug Name": "drug_name",
    "Cancer-Status": "cancer_status",
    "PDR - Link": "pdr_link",
    "Mechanism Of Action": "mechanism_of_action",
    "Route of Administration": "route",
    "indication": "indication",
    "drug-interactions(Details)": "drug_interactions",
    "Adverse Reactions": "adverse_reactions_raw",
    "targets": "targets_raw",
    "pathways": "pathways_raw",
    "efficacy": "efficacy",
}


def _safe_str(val):
    return str(val).strip() if pd.notna(val) else ""


def load_and_explode(input_path: Path, sheet: str = "Sheet1") -> list:
    """Read Excel, parse targets/pathways/AR, explode gene targets."""
    df = pd.read_excel(input_path, sheet_name=sheet, dtype=str)
    df = df.dropna(subset=["Drug Name"])

    docs = []
    for _, row in df.iterrows():
        drug = _safe_str(row.get("Drug Name"))
        if not drug:
            continue

        cancer_raw = _safe_str(row.get("Cancer-Status"))
        is_anticancer = cancer_raw.lower() == "anticancer"

        moa = _safe_str(row.get("Mechanism Of Action"))
        indication = _safe_str(row.get("indication"))
        route = _safe_str(row.get("Route of Administration"))
        drug_int = _safe_str(row.get("drug-interactions(Details)"))
        ar_raw = _safe_str(row.get("Adverse Reactions"))
        targets_raw = row.get("targets")
        pathways_raw = row.get("pathways")
        efficacy = _safe_str(row.get("efficacy"))
        pdr_link = _safe_str(row.get("PDR - Link"))

        ar = parse_adverse_reactions(row.get("Adverse Reactions"))
        pathway_ids, pathway_names = extract_pathways(pathways_raw)
        gene_targets = extract_gene_targets(targets_raw)

        base = {
            "data_source": "PDR_Drugs_Data",
            "drug_name": drug,
            "drug_name_lower": drug.lower(),
            "is_anticancer": is_anticancer,
            "pdr_link": pdr_link,
            "mechanism_of_action": moa,
            "route": route,
            "indication": indication,
            "drug_interactions": drug_int,
            "adverse_reactions_raw": ar_raw,
            "severe_reactions": ar["severe"],
            "moderate_reactions": ar["moderate"],
            "mild_reactions": ar["mild"],
            "all_targets_raw": _safe_str(targets_raw),
            "pathway_ids": pathway_ids,
            "pathway_names": pathway_names,
            "efficacy": efficacy,
        }

        if gene_targets:
            for gene, hsa_ids in gene_targets:
                docs.append({
                    **base,
                    "doc_type": "pdr_drug_target",
                    "gene_symbol": gene,
                    "hsa_ids": hsa_ids,
                })
        else:
            target_str = _safe_str(targets_raw)
            docs.append({
                **base,
                "doc_type": "pdr_drug_reference",
                "target": target_str,
            })

    target_count = sum(1 for d in docs if d["doc_type"] == "pdr_drug_target")
    ref_count = sum(1 for d in docs if d["doc_type"] == "pdr_drug_reference")
    logger.info(f"Loaded {len(df)} rows → {len(docs)} docs "
                f"(pdr_drug_target={target_count}, pdr_drug_reference={ref_count})")
    return docs


def build_text_content(doc: dict) -> str:
    target = doc.get("gene_symbol") or doc.get("target") or ""
    moa = (doc.get("mechanism_of_action") or "")[:500]
    ind = (doc.get("indication") or "")[:400]
    eff = (doc.get("efficacy") or "")[:200]
    route = doc.get("route") or ""
    return (
        f"{doc['drug_name']} targets {target}. "
        f"Mechanism: {moa}. "
        f"Indicated for: {ind}. "
        f"Efficacy: {eff}. "
        f"Route: {route}."
    )[:2000]


def generate_point_id(doc: dict) -> str:
    drug = doc["drug_name_lower"]
    if doc["doc_type"] == "pdr_drug_target":
        key = f"{drug}|{doc['gene_symbol'].lower()}"
    else:
        key = f"{drug}|_ref"
    return hashlib.md5(key.encode()).hexdigest()


# ── Qdrant setup ─────────────────────────────────────────────────────────────

def create_collection(client, collection_name: str, vector_size: int = 768):
    from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        logger.info(f"Collection already exists: {collection_name}")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f"Created collection: {collection_name}")

    for field in ("doc_type", "drug_name_lower", "gene_symbol", "route", "pathway_ids"):
        client.create_payload_index(collection_name, field, PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "is_anticancer", PayloadSchemaType.BOOL)
    logger.info(f"Created 6 payload indexes on {collection_name}")


# ── Upsert ───────────────────────────────────────────────────────────────────

def upsert_worker(client, collection_name: str, points: list, max_retries: int = 3) -> int:
    if not points:
        return 0
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return len(points)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Upsert retry {attempt + 1}/{max_retries}: {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"Upsert failed after {max_retries} attempts: {e}")
                raise
    return 0


# ── Main ingestion ───────────────────────────────────────────────────────────

def ingest(docs: list, embedder, client, collection_name: str,
           batch_size: int = 512, embed_chunk_size: int = 5000):
    from qdrant_client.models import PointStruct

    total_upserted = 0
    executor = ThreadPoolExecutor(max_workers=4)
    pending = []
    pbar = tqdm(total=len(docs), desc="Ingesting", unit="docs")

    for chunk_start in range(0, len(docs), embed_chunk_size):
        chunk = docs[chunk_start: chunk_start + embed_chunk_size]
        texts = [build_text_content(d) for d in chunk]

        result = embedder.embed_texts(texts, show_progress=True)
        embeddings = result.embeddings

        for batch_start in range(0, len(chunk), batch_size):
            batch_docs = chunk[batch_start: batch_start + batch_size]
            batch_embs = embeddings[batch_start: batch_start + batch_size]

            points = [
                PointStruct(
                    id=generate_point_id(d),
                    vector=emb,
                    payload={**d, "text_content": build_text_content(d)},
                )
                for d, emb in zip(batch_docs, batch_embs)
            ]
            pending.append(executor.submit(upsert_worker, client, collection_name, points))

            done = [f for f in pending if f.done()]
            for f in done:
                total_upserted += f.result()
                pending.remove(f)
            pbar.update(len(batch_docs))

    for f in as_completed(pending):
        total_upserted += f.result()

    pbar.close()
    executor.shutdown(wait=True)
    logger.info(f"Upserted {total_upserted:,} points to {collection_name}")
    return total_upserted


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest PDR Drugs XLSX → Qdrant PDR_Drugs_Data")
    parser.add_argument("--input", required=True, help="Path to Sub11 PDR Drugs XLSX")
    parser.add_argument("--collection", default="PDR_Drugs_Data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embed-chunk", type=int, default=5000,
                        help="Texts per embed_texts() call (OOM guard)")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path

    docs = load_and_explode(input_path, args.sheet)
    logger.info(f"Total documents: {len(docs):,}")

    if args.dry_run:
        targets = [d for d in docs if d["doc_type"] == "pdr_drug_target"]
        refs = [d for d in docs if d["doc_type"] == "pdr_drug_reference"]
        samples = (targets[:3] + refs[:2]) if targets else docs[:5]
        for i, d in enumerate(samples):
            print(f"\n--- Sample {i + 1} ({d['doc_type']}) ---")
            print(f"  text_content: {build_text_content(d)}")
            print(f"  point_id:     {generate_point_id(d)}")
            for k, v in d.items():
                val = str(v)[:120] + "..." if len(str(v)) > 120 else v
                print(f"  {k}: {val}")
        print(f"\n  Total docs: {len(docs):,}")
        print(f"  pdr_drug_target: {len(targets):,}")
        print(f"  pdr_drug_reference: {len(refs):,}")

        ids = [generate_point_id(d) for d in docs]
        print(f"  Unique point IDs: {len(set(ids)):,} / {len(ids):,}")
        return

    from embedding.embedder import PubMedBERTEmbedder

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "basic_auth_qdrant", Path(__file__).parent / "storage" / "basic_auth_qdrant.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    get_qdrant_client_from_env = _mod.get_qdrant_client_from_env

    client, _, err = get_qdrant_client_from_env()
    if err:
        logger.error(f"Qdrant connection failed: {err}")
        return

    embedder = PubMedBERTEmbedder(device=args.device)
    create_collection(client, args.collection)
    ingest(docs, embedder, client, args.collection, args.batch_size, args.embed_chunk)

    count = client.get_collection(args.collection).points_count
    logger.info(f"Final points_count in {args.collection}: {count:,}")


if __name__ == "__main__":
    main()
