#!/usr/bin/env python3
"""
Ingest Enriched Drug Data to Qdrant
===================================
Ingests detailed drug data with mechanisms, targets, and indications
for effective drug recommendations.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import httpx
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
    OptimizersConfigDiff, PayloadSchemaType
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentic_ai_wf.drug_agent.opentargets.drug_enrichment_fetcher import (
    DrugEnrichmentFetcher, FetchConfig
)

# Configuration
QDRANT_HOST = "https://vector.f420.ai"
QDRANT_USERNAME = "admin"
QDRANT_PASSWORD = "4-2i!CW~5ic+"
COLLECTION_NAME = "OpenTargets_drugs_enriched"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
VECTOR_SIZE = 768
BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DrugEnrichmentIngester:
    def __init__(self):
        logger.info("Initializing Drug Enrichment Ingester...")
        
        # Qdrant client with Basic Auth
        self.client = QdrantClient(
            url=QDRANT_HOST,
            port=443,
            timeout=120,
            prefer_grpc=False,
            https=True
        )
        
        # Apply Basic Auth to all HTTP clients
        auth = httpx.BasicAuth(QDRANT_USERNAME, QDRANT_PASSWORD)
        custom_http = httpx.Client(auth=auth, timeout=120.0)
        for api_name in ['collections_api', 'points_api', 'service_api', 'search_api']:
            api = getattr(self.client._client.http, api_name, None)
            if api and hasattr(api, 'api_client'):
                api.api_client._client = custom_http
        
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}")
        
        # Embedding model with GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model on {device}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        
        # Fetcher
        self.fetcher = DrugEnrichmentFetcher(FetchConfig())
        
        # Stats
        self.stats = {
            "drugs": 0,
            "associations": 0,
            "total": 0,
            "errors": 0
        }
    
    def setup_collection(self, recreate: bool = False):
        """Create or recreate the collection."""
        collections = [c.name for c in self.client.get_collections().collections]
        
        if COLLECTION_NAME in collections:
            if recreate:
                logger.warning(f"Deleting existing collection: {COLLECTION_NAME}")
                self.client.delete_collection(COLLECTION_NAME)
            else:
                logger.info(f"Collection {COLLECTION_NAME} exists with {self.client.count(COLLECTION_NAME).count} points")
                return
        
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10000,
                memmap_threshold=50000
            )
        )
        
        # Create payload indexes for filtering
        for field in ["entity_type", "drug_type", "max_phase", "mechanism_targets", "linked_diseases"]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                logger.warning(f"Could not create index for {field}: {e}")
    
    def ingest_batch(self, documents: list):
        """Ingest a batch of documents."""
        if not documents:
            return
        
        texts = [doc["text_content"] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = abs(hash(doc["id"])) % (10**15)
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=doc
            ))
        
        self.client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    
    def run(self, drug_limit: int = 5000, association_limit: int = 10000, recreate: bool = False):
        """Run the ingestion process."""
        start_time = datetime.now()
        logger.info(f"Starting enriched drug ingestion at {start_time}")
        logger.info(f"Limits: {drug_limit:,} drugs, {association_limit:,} associations")
        
        self.setup_collection(recreate)
        
        batch = []
        
        for doc in self.fetcher.fetch_all(drug_limit, association_limit):
            entity_type = doc.get("entity_type", "unknown")
            
            if entity_type == "drug_enriched":
                self.stats["drugs"] += 1
            elif entity_type == "drug_indication":
                self.stats["associations"] += 1
            
            self.stats["total"] += 1
            batch.append(doc)
            
            if len(batch) >= BATCH_SIZE:
                try:
                    self.ingest_batch(batch)
                    batch = []
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = self.stats["total"] / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: Dr:{self.stats['drugs']:,} | Assoc:{self.stats['associations']:,} | "
                        f"Total:{self.stats['total']:,} | Rate:{rate:.1f}/sec"
                    )
                except Exception as e:
                    logger.error(f"Batch ingest failed: {e}")
                    self.stats["errors"] += 1
        
        # Final batch
        if batch:
            try:
                self.ingest_batch(batch)
            except Exception as e:
                logger.error(f"Final batch failed: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 50)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Enriched Drugs: {self.stats['drugs']:,}")
        logger.info(f"Drug-Disease Assoc: {self.stats['associations']:,}")
        logger.info(f"Total Documents: {self.stats['total']:,}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info(f"Rate: {self.stats['total'] / elapsed:.1f} docs/sec")
        
        # Verify
        count = self.client.count(COLLECTION_NAME).count
        logger.info(f"Collection {COLLECTION_NAME} now has {count:,} points")


def main():
    parser = argparse.ArgumentParser(description="Ingest enriched drug data to Qdrant")
    parser.add_argument("--drugs", type=int, default=5000, help="Max drugs to fetch")
    parser.add_argument("--associations", type=int, default=10000, help="Max drug-disease associations")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    args = parser.parse_args()
    
    ingester = DrugEnrichmentIngester()
    ingester.run(
        drug_limit=args.drugs,
        association_limit=args.associations,
        recreate=args.recreate
    )


if __name__ == "__main__":
    main()
