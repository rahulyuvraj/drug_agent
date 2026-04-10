#!/usr/bin/env python3
"""
DrugPath XLSX → Qdrant Ingestion
=================================

Reads drugpath_data_with_symbols.xlsx, explodes multi-gene rows into one
document per gene symbol, embeds with PubMedBERT in memory-safe chunks,
and upserts to a new DrugPath_KEGG collection.

Usage:
    python ingest_drugpath_xlsx.py --input drugpath_data_with_symbols.xlsx
    python ingest_drugpath_xlsx.py --input drugpath_data_with_symbols.xlsx --dry-run
"""

import os
import hashlib
import logging
import time
import argparse
from itertools import zip_longest
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_explode(input_path: Path, sheet: str = "Sheet1") -> list:
    """Read Excel and emit one doc per individual gene symbol."""
    df = pd.read_excel(input_path, sheet_name=sheet, dtype={"GeneID": str})
    df = df.dropna(subset=["GeneSymbol"])
    df = df[df["FDR"].apply(lambda v: pd.notna(v) and float(v) <= 1.0)]

    docs = []
    for _, row in df.iterrows():
        drug = str(row["Drug"]).strip()
        pathway_id = str(row["PathwayID"]).strip()
        pathway_name = str(row["Pathway Name"]).strip()
        raw_genes = str(row["GeneSymbol"]).strip()
        raw_ids = str(row["GeneID"]).strip()
        gene_count = int(row["Gene counts"]) if pd.notna(row["Gene counts"]) else 0
        effect_direction = 1 if str(row["Type"]).strip() == "+" else -1
        p_value = float(row["p-value"]) if pd.notna(row["p-value"]) else 1.0
        fdr = float(row["FDR"]) if pd.notna(row["FDR"]) else 1.0

        gene_symbols = raw_genes.split("_")
        gene_ids = raw_ids.split("_")

        for gene_sym, gene_id in zip_longest(gene_symbols, gene_ids, fillvalue=""):
            gene_sym = gene_sym.strip()
            if not gene_sym:
                continue
            docs.append({
                "doc_type": "drugpath_kegg",
                "drug_name": drug,
                "drug_name_lower": drug.lower(),
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "gene_symbol": gene_sym,
                "gene_id": gene_id.strip(),
                "gene_count": gene_count,
                "effect_direction": effect_direction,
                "p_value": p_value,
                "fdr": fdr,
                "parent_gene_symbols": raw_genes,
                "data_source": "DrugPath_KEGG",
            })

    logger.info(f"Exploded {len(df)} rows → {len(docs)} gene-level documents")
    return docs


def build_text_content(doc: dict) -> str:
    action = "suppresses" if doc["effect_direction"] == -1 else "activates"
    return (
        f"{doc['drug_name']} {action} {doc['gene_symbol']} through "
        f"{doc['pathway_name']} pathway (KEGG:{doc['pathway_id']}, "
        f"p={doc['p_value']:.2e}, FDR={doc['fdr']:.2e})"
    )


def generate_point_id(drug: str, pathway_id: str, gene_symbol: str) -> str:
    return hashlib.md5(f"{drug}|{pathway_id}|{gene_symbol}".encode()).hexdigest()


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

    keyword_fields = ["doc_type", "gene_symbol", "drug_name_lower", "pathway_id", "pathway_name"]
    for field in keyword_fields:
        client.create_payload_index(collection_name, field, PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "effect_direction", PayloadSchemaType.INTEGER)
    client.create_payload_index(collection_name, "fdr", PayloadSchemaType.FLOAT)
    client.create_payload_index(collection_name, "p_value", PayloadSchemaType.FLOAT)
    logger.info(f"Created payload indexes on {collection_name}")


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
        chunk = docs[chunk_start : chunk_start + embed_chunk_size]
        texts = [build_text_content(d) for d in chunk]

        result = embedder.embed_texts(texts, show_progress=True)
        embeddings = result.embeddings

        for batch_start in range(0, len(chunk), batch_size):
            batch_docs = chunk[batch_start : batch_start + batch_size]
            batch_embs = embeddings[batch_start : batch_start + batch_size]

            points = [
                PointStruct(
                    id=generate_point_id(d["drug_name"], d["pathway_id"], d["gene_symbol"]),
                    vector=emb,
                    payload={**d, "text_content": build_text_content(d)},
                )
                for d, emb in zip(batch_docs, batch_embs)
            ]
            pending.append(executor.submit(upsert_worker, client, collection_name, points))

            # Collect completed futures
            done = [f for f in pending if f.done()]
            for f in done:
                total_upserted += f.result()
                pending.remove(f)
            pbar.update(len(batch_docs))

    # Drain remaining futures
    for f in as_completed(pending):
        total_upserted += f.result()

    pbar.close()
    executor.shutdown(wait=True)
    logger.info(f"Upserted {total_upserted:,} points to {collection_name}")
    return total_upserted


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest DrugPath XLSX → Qdrant DrugPath_KEGG")
    parser.add_argument("--input", required=True, help="Path to drugpath_data_with_symbols.xlsx")
    parser.add_argument("--collection", default="DrugPath_KEGG")
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
        for i, d in enumerate(docs[:5]):
            print(f"\n--- Sample {i + 1} ---")
            print(f"  text_content: {build_text_content(d)}")
            print(f"  point_id:     {generate_point_id(d['drug_name'], d['pathway_id'], d['gene_symbol'])}")
            for k, v in d.items():
                print(f"  {k}: {v}")
        print(f"\n  Total docs: {len(docs):,}")
        return

    from embedding.embedder import PubMedBERTEmbedder

    # Direct module load to avoid storage/__init__.py relative-import chain
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
