"""Shared infrastructure for Open Targets ingestion pipelines."""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import requests
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, OptimizersConfigDiff
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"
QDRANT_URL = os.getenv("QDRANT_URL", "https://vector.f420.ai")
QDRANT_USER = os.getenv("QDRANT_USERNAME", "admin")
QDRANT_PASS = os.getenv("QDRANT_PASSWORD", "4-2i!CW~5ic+")
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
VECTOR_DIM = 768


def get_qdrant() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, port=443, timeout=120, prefer_grpc=False, https=True, check_compatibility=False)
    auth = httpx.BasicAuth(QDRANT_USER, QDRANT_PASS)
    http = httpx.Client(auth=auth, timeout=120.0, limits=httpx.Limits(max_connections=50, max_keepalive_connections=30))
    for name in ["collections_api", "points_api", "service_api", "search_api"]:
        api = getattr(client._client.http, name, None)
        if api and hasattr(api, "api_client"):
            api.api_client._client = http
    return client


def get_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading PubMedBERT on {device}")
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def ensure_collection(client: QdrantClient, name: str, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if recreate and name in existing:
        client.delete_collection(name)
        existing.remove(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=10000, memmap_threshold=50000),
        )
        logger.info(f"Created collection: {name}")
    count = client.count(name).count
    logger.info(f"Collection '{name}': {count:,} points")
    return count


def gql_query(session: requests.Session, query: str, variables: Dict = None,
              max_retries: int = 5, delay: float = 0.4) -> Dict:
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            r = session.post(OPENTARGETS_API, json={"query": query, "variables": variables or {}}, timeout=30)
            r.raise_for_status()
            return r.json().get("data", {})
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (429, 502, 503, 504):
                wait = delay * (attempt + 1) * 3
                logger.warning(f"API {e.response.status_code}, retry in {wait:.0f}s")
                time.sleep(wait)
            else:
                logger.error(f"HTTP {e.response.status_code}: {e}")
                break
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    return {}


def upsert_batch(client: QdrantClient, embedder: SentenceTransformer,
                 collection: str, docs: List[Dict]):
    if not docs:
        return
    texts = [d["text_content"] for d in docs]
    vectors = embedder.encode(texts, show_progress_bar=False)
    points = [
        PointStruct(
            id=int(hashlib.md5(d["id"].encode()).hexdigest()[:15], 16),
            vector=v.tolist(), payload=d,
        )
        for d, v in zip(docs, vectors)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)


def load_drug_ids_from_qdrant(client: QdrantClient,
                               collection: str = "OpenTargets_drugs_enriched") -> List[Tuple[str, str, str, float]]:
    """Pull all (chembl_id, drug_name, drug_type, max_phase) from the enriched drugs collection."""
    drugs = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection, limit=256, offset=offset, with_payload=True,
        )
        for r in results:
            p = r.payload
            if p.get("id", "").startswith("CHEMBL"):
                drugs.append((p["id"], p.get("name", ""), p.get("drug_type", ""),
                              p.get("max_phase", 0)))
        if offset is None:
            break
    logger.info(f"Loaded {len(drugs):,} drug IDs from {collection}")
    return drugs


# Checkpoint helpers
def _ckpt_path(label: str) -> Path:
    return Path(__file__).parent / f".checkpoint_{label}.json"


def load_checkpoint(label: str) -> set:
    p = _ckpt_path(label)
    if p.exists():
        data = json.loads(p.read_text())
        logger.info(f"Resuming from checkpoint: {len(data):,} drugs already processed")
        return set(data)
    return set()


def save_checkpoint(label: str, processed: set):
    _ckpt_path(label).write_text(json.dumps(sorted(processed)))
