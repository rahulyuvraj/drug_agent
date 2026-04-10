#!/usr/bin/env python3
"""
Open Targets Robust Ingestion
=============================
Uses the robust fetcher with smaller batches and better error handling.

Usage:
    python ingest_robust.py                     # Default (~20K docs)
    python ingest_robust.py --targets 20000     # Custom limits
    python ingest_robust.py --quick             # Quick test (~1K docs)
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

load_dotenv(SCRIPT_DIR.parent / ".env")

# Logging
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"ingest_robust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "OpenTargets_data"


@dataclass
class Stats:
    total: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = 0
    
    def elapsed(self) -> str:
        return str(timedelta(seconds=int(time.time() - self.start_time)))
    
    def rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0


def get_qdrant_client():
    """Get Qdrant client with Basic Auth."""
    from qdrant_client import QdrantClient
    import httpx
    
    url = os.getenv("QDRANT_URL", "https://vector.f420.ai")
    username = os.getenv("QDRANT_USERNAME", "admin")
    password = os.getenv("QDRANT_PASSWORD", "4-2i!CW~5ic+")
    
    client = QdrantClient(url=url, port=443, timeout=120, prefer_grpc=False, https=True)
    
    if username and password:
        auth = httpx.BasicAuth(username, password)
        custom_http = httpx.Client(auth=auth, timeout=120.0)
        for api_name in ['collections_api', 'points_api', 'service_api', 'search_api']:
            api = getattr(client._client.http, api_name, None)
            if api and hasattr(api, 'api_client'):
                api.api_client._client = custom_http
    
    return client


def get_embedder():
    """Load PubMedBERT embedder."""
    logger.info("Loading PubMedBERT embedder...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    device = "CUDA" if model.device.type == "cuda" else "CPU"
    logger.info(f"Embedder loaded ({device})")
    return model


def ensure_collection(client, collection_name: str, recreate: bool = False):
    """Create or verify collection exists."""
    from qdrant_client.http import models
    
    collections = [c.name for c in client.get_collections().collections]
    
    if recreate and collection_name in collections:
        logger.info(f"Deleting collection: {collection_name}")
        client.delete_collection(collection_name)
        collections.remove(collection_name)
    
    if collection_name not in collections:
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )
    
    info = client.get_collection(collection_name)
    logger.info(f"Collection '{collection_name}': {info.points_count:,} points")
    return info.points_count


def ingest(
    target_limit: int = 10000,
    disease_limit: int = 5000,
    drug_limit: int = 5000,
    assoc_per_disease: int = 20,
    batch_size: int = 50,
    recreate: bool = False
):
    """Main ingestion function."""
    from qdrant_client.http import models
    from robust_fetcher import RobustFetcher
    
    logger.info("=" * 60)
    logger.info("  OPEN TARGETS ROBUST INGESTION")
    logger.info("=" * 60)
    logger.info(f"  Target Limit: {target_limit:,}")
    logger.info(f"  Disease Limit: {disease_limit:,}")
    logger.info(f"  Drug Limit: {drug_limit:,}")
    logger.info(f"  Assoc/Disease: {assoc_per_disease}")
    logger.info("=" * 60)
    
    # Initialize
    client = get_qdrant_client()
    embedder = get_embedder()
    fetcher = RobustFetcher()
    
    # Setup collection
    point_id = ensure_collection(client, COLLECTION_NAME, recreate)
    
    # Track stats
    stats = Stats(start_time=time.time())
    type_counts = {}
    
    # Process in batches
    batch = []
    
    logger.info("\nStarting fetch and ingestion...\n")
    
    for doc in fetcher.fetch_all(
        target_limit=target_limit,
        disease_limit=disease_limit,
        drug_limit=drug_limit,
        assoc_per_disease=assoc_per_disease
    ):
        batch.append(doc)
        entity_type = doc.get("entity_type", "unknown")
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        if len(batch) >= batch_size:
            try:
                texts = [d.get("text_content", str(d)) for d in batch]
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
                
                points = []
                for i, d in enumerate(batch):
                    d["ingested_at"] = datetime.now().isoformat()
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embeddings[i],
                            payload=d
                        )
                    )
                    point_id += 1
                
                client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                stats.successful += len(batch)
                
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                stats.failed += len(batch)
            
            stats.processed += len(batch)
            
            if stats.processed % 500 == 0:
                logger.info(
                    f"Progress: {stats.processed:,} | "
                    f"Rate: {stats.rate():.1f}/s | "
                    f"T:{type_counts.get('target',0):,} "
                    f"D:{type_counts.get('disease',0):,} "
                    f"Dr:{type_counts.get('drug',0):,} "
                    f"A:{type_counts.get('association',0):,}"
                )
            
            batch = []
    
    # Final batch
    if batch:
        try:
            texts = [d.get("text_content", str(d)) for d in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
            
            points = []
            for i, d in enumerate(batch):
                d["ingested_at"] = datetime.now().isoformat()
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[i],
                        payload=d
                    )
                )
                point_id += 1
            
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            stats.successful += len(batch)
        except Exception as e:
            logger.error(f"Final batch failed: {e}")
            stats.failed += len(batch)
        stats.processed += len(batch)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Processed: {stats.processed:,}")
    logger.info(f"  Successful: {stats.successful:,}")
    logger.info(f"  Failed: {stats.failed:,}")
    logger.info(f"  Duration: {stats.elapsed()}")
    logger.info(f"  Rate: {stats.rate():.1f}/s")
    logger.info("")
    for t, c in sorted(type_counts.items()):
        logger.info(f"    {t}: {c:,}")
    
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"\n  Collection now has {info.points_count:,} points")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=10000)
    parser.add_argument("--diseases", type=int, default=5000)
    parser.add_argument("--drugs", type=int, default=5000)
    parser.add_argument("--assoc-per-disease", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    if args.quick:
        ingest(
            target_limit=500,
            disease_limit=200,
            drug_limit=200,
            assoc_per_disease=10,
            batch_size=args.batch_size,
            recreate=args.recreate
        )
    else:
        ingest(
            target_limit=args.targets,
            disease_limit=args.diseases,
            drug_limit=args.drugs,
            assoc_per_disease=args.assoc_per_disease,
            batch_size=args.batch_size,
            recreate=args.recreate
        )


if __name__ == "__main__":
    main()
