#!/usr/bin/env python3
"""
Full Data Ingestion Script for Drug Discovery Agent
====================================================
Ingests ALL gene JSON files from GeneALaCart dataset into Qdrant.

Features:
- Progress logging with percentage complete
- Checkpoint/resume support
- Batch processing for efficiency
- Error handling and retry logic
- Detailed statistics
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import traceback

# Setup logging
LOG_FILE = Path(__file__).parent / "logs" / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Checkpoint file for resume support
CHECKPOINT_FILE = Path(__file__).parent / "ingestion_checkpoint.json"


@dataclass
class IngestionStats:
    """Track ingestion statistics."""
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = 0
    last_checkpoint: int = 0
    
    def percent_complete(self) -> float:
        if self.total_files == 0:
            return 0
        return (self.processed / self.total_files) * 100
    
    def elapsed_time(self) -> str:
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
    
    def estimated_remaining(self) -> str:
        if self.processed == 0:
            return "calculating..."
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed
        remaining = (self.total_files - self.processed) / rate if rate > 0 else 0
        return str(timedelta(seconds=int(remaining)))
    
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
    collection = os.getenv("QDRANT_COLLECTION", "Drug_agent")
    
    logger.info(f"Connecting to Qdrant: {url}")
    
    client = QdrantClient(
        url=url,
        port=443,
        timeout=120,
        prefer_grpc=False,
        https=True,
    )
    
    # Patch with Basic Auth
    auth = httpx.BasicAuth(username, password)
    custom_http = httpx.Client(auth=auth, timeout=120.0)
    
    http_apis = client._client.http
    for api_name in ['collections_api', 'points_api', 'service_api', 'search_api', 
                     'snapshots_api', 'indexes_api', 'aliases_api', 'distributed_api', 'beta_api']:
        api = getattr(http_apis, api_name, None)
        if api and hasattr(api, 'api_client'):
            api.api_client._client = custom_http
    
    return client, collection


def get_embedder():
    """Load PubMedBERT embedder."""
    logger.info("Loading PubMedBERT embedder...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    logger.info("Embedder loaded successfully!")
    return model


def parse_gene_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a gene JSON file and extract relevant data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gene_symbol = file_path.stem
        
        # Extract drugs
        drugs = []
        for field in ["UnifiedDrugs", "UnifiedCompounds", "Compounds"]:
            if field in data and data[field]:
                for drug in data[field][:15]:
                    if isinstance(drug, dict):
                        name = drug.get("Name") or drug.get("DrugName") or drug.get("CompoundName", "")
                        if name and name not in [d["name"] for d in drugs]:
                            drugs.append({"name": name, "type": drug.get("Type", "")})
        
        # Extract diseases
        diseases = []
        for field in ["MalaCardsDisorders", "MalaCardsInferredDisorders", "UniProtDisorders"]:
            if field in data and data[field]:
                for disease in data[field][:15]:
                    if isinstance(disease, dict):
                        name = disease.get("Name") or disease.get("DiseaseName", "")
                        if name and name not in [d["name"] for d in diseases]:
                            diseases.append({"name": name, "score": disease.get("Score", 0)})
        
        # Extract phenotypes
        phenotypes = []
        if "HumanPhenotypeOntology" in data and data["HumanPhenotypeOntology"]:
            for hp in data["HumanPhenotypeOntology"][:10]:
                if isinstance(hp, dict):
                    name = hp.get("Name") or hp.get("Term", "")
                    if name:
                        phenotypes.append(name)
        if "GWASPhenotypes" in data and data["GWASPhenotypes"]:
            for gwas in data["GWASPhenotypes"][:10]:
                if isinstance(gwas, dict):
                    name = gwas.get("Phenotype") or gwas.get("Trait", "")
                    if name and name not in phenotypes:
                        phenotypes.append(name)
        
        # Extract pathways
        pathways = []
        for field in ["Pathways", "SuperPathway"]:
            if field in data and data[field]:
                for pw in data[field][:10]:
                    if isinstance(pw, dict):
                        name = pw.get("Name") or pw.get("PathwayName", "")
                        if name and name not in pathways:
                            pathways.append(name)
        
        # Extract functions
        functions = []
        for field in ["MolecularFunctions", "BiologicalProcesses", "MolecularFunctionDescriptions"]:
            if field in data and data[field]:
                for f in data[field][:5]:
                    if isinstance(f, dict):
                        name = f.get("Name") or f.get("Term", "")
                        if name:
                            functions.append(name)
                    elif isinstance(f, str):
                        functions.append(f)
        
        # Extract summary
        summary = ""
        if "Summaries" in data and data["Summaries"]:
            for s in data["Summaries"]:
                if isinstance(s, dict) and s.get("Summary"):
                    summary = s["Summary"]
                    break
        
        # Extract gene info
        gene_info = {}
        if "Gene" in data and data["Gene"]:
            g = data["Gene"][0] if isinstance(data["Gene"], list) else data["Gene"]
            if isinstance(g, dict):
                gene_info = {
                    "name": g.get("Name", ""),
                    "category": g.get("Category", ""),
                    "description": g.get("Description", ""),
                }
        
        # Build text content for embedding
        text_parts = [f"Gene: {gene_symbol}"]
        
        if gene_info.get("name"):
            text_parts.append(f"Full name: {gene_info['name']}")
        if gene_info.get("description"):
            text_parts.append(f"Description: {gene_info['description'][:200]}")
        if summary:
            text_parts.append(f"Summary: {summary[:400]}")
        if drugs:
            drug_names = [d["name"] for d in drugs if d["name"]][:10]
            if drug_names:
                text_parts.append(f"Drug interactions: {', '.join(drug_names)}")
        if diseases:
            disease_names = [d["name"] for d in diseases if d["name"]][:10]
            if disease_names:
                text_parts.append(f"Disease associations: {', '.join(disease_names)}")
        if phenotypes:
            text_parts.append(f"Phenotypes: {', '.join(phenotypes[:8])}")
        if pathways:
            text_parts.append(f"Pathways: {', '.join(pathways[:5])}")
        if functions:
            text_parts.append(f"Functions: {', '.join(functions[:5])}")
        
        text_content = " | ".join(text_parts)
        
        return {
            "gene_symbol": gene_symbol,
            "text_content": text_content,
            "drugs": drugs,
            "diseases": diseases,
            "phenotypes": phenotypes,
            "pathways": pathways,
            "summary": summary[:500] if summary else "",
            "doc_type": "gene_data",
        }
    except Exception as e:
        return None


def save_checkpoint(stats: IngestionStats, processed_files: List[str]):
    """Save checkpoint for resume support."""
    checkpoint = {
        "processed": stats.processed,
        "successful": stats.successful,
        "failed": stats.failed,
        "last_file_index": stats.processed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def print_progress_bar(stats: IngestionStats):
    """Print a visual progress bar."""
    percent = stats.percent_complete()
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    logger.info(f"")
    logger.info(f"╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"║  INGESTION PROGRESS                                              ║")
    logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
    logger.info(f"║  [{bar}] {percent:6.2f}%  ║")
    logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
    logger.info(f"║  Processed: {stats.processed:>10,} / {stats.total_files:<10,}                       ║")
    logger.info(f"║  Successful: {stats.successful:>9,}  |  Failed: {stats.failed:<9,}                  ║")
    logger.info(f"║  Rate: {stats.rate():>6.1f} docs/sec                                        ║")
    logger.info(f"║  Elapsed: {stats.elapsed_time():<12}  |  ETA: {stats.estimated_remaining():<12}       ║")
    logger.info(f"╚══════════════════════════════════════════════════════════════════╝")
    logger.info(f"")


def ingest_full_dataset(
    json_dir: str,
    batch_size: int = 100,
    checkpoint_interval: int = 1000,
    resume: bool = True,
    recreate_collection: bool = False,
):
    """
    Ingest the full GeneALaCart dataset into Qdrant.
    
    Args:
        json_dir: Path to the GeneALaCart-AllGenes directory
        batch_size: Number of documents per batch
        checkpoint_interval: Save checkpoint every N documents
        resume: Whether to resume from checkpoint
        recreate_collection: Whether to delete and recreate collection
    """
    from qdrant_client.http import models
    
    logger.info("=" * 70)
    logger.info("  FULL DATASET INGESTION - Drug Discovery Agent")
    logger.info("=" * 70)
    logger.info(f"  Source: {json_dir}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Checkpoint Interval: {checkpoint_interval}")
    logger.info("=" * 70)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
    
    # Get client and embedder
    client, collection_name = get_qdrant_client()
    embedder = get_embedder()
    
    # Verify connection
    logger.info("Verifying Qdrant connection...")
    collections = client.get_collections()
    logger.info(f"Connected! Existing collections: {[c.name for c in collections.collections]}")
    
    # Handle collection
    collection_exists = any(c.name == collection_name for c in collections.collections)
    
    if recreate_collection and collection_exists:
        logger.info(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        collection_exists = False
    
    if not collection_exists:
        logger.info(f"Creating collection: {collection_name} (768 dimensions, cosine)")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        info = client.get_collection(collection_name)
        logger.info(f"Using existing collection: {collection_name} ({info.points_count:,} points)")
    
    # Find all JSON files
    logger.info(f"Scanning directory: {json_dir}")
    json_dir_path = Path(json_dir)
    
    if not json_dir_path.exists():
        logger.error(f"Directory not found: {json_dir}")
        return
    
    json_files = sorted(list(json_dir_path.rglob("*.json")))
    total_files = len(json_files)
    logger.info(f"Found {total_files:,} JSON files to process")
    
    # Check for resume
    start_index = 0
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            start_index = checkpoint.get("last_file_index", 0)
            logger.info(f"Resuming from checkpoint: {start_index:,} files already processed")
    
    # Initialize stats
    stats = IngestionStats(
        total_files=total_files,
        processed=start_index,
        successful=start_index,  # Assume previous were successful
        start_time=time.time(),
    )
    
    # Process files
    logger.info("")
    logger.info("Starting ingestion...")
    logger.info("")
    
    batch_docs = []
    batch_ids = []
    
    try:
        for idx, file_path in enumerate(json_files[start_index:], start=start_index):
            # Parse file
            doc = parse_gene_json(file_path)
            
            if doc:
                batch_docs.append(doc)
                batch_ids.append(idx)
            else:
                stats.failed += 1
            
            # Process batch
            if len(batch_docs) >= batch_size:
                try:
                    # Generate embeddings
                    texts = [d["text_content"] for d in batch_docs]
                    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
                    
                    # Upload to Qdrant
                    points = [
                        models.PointStruct(
                            id=batch_ids[i],
                            vector=embeddings[i],
                            payload=batch_docs[i]
                        )
                        for i in range(len(batch_docs))
                    ]
                    
                    client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True
                    )
                    
                    stats.successful += len(batch_docs)
                    
                except Exception as e:
                    logger.error(f"Batch upload failed: {e}")
                    stats.failed += len(batch_docs)
                
                stats.processed = idx + 1
                batch_docs = []
                batch_ids = []
                
                # Progress logging
                if stats.processed % 500 == 0:
                    print_progress_bar(stats)
                
                # Checkpoint
                if stats.processed % checkpoint_interval == 0:
                    save_checkpoint(stats, [])
                    logger.info(f"Checkpoint saved at {stats.processed:,} files")
        
        # Process remaining batch
        if batch_docs:
            try:
                texts = [d["text_content"] for d in batch_docs]
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
                
                points = [
                    models.PointStruct(
                        id=batch_ids[i],
                        vector=embeddings[i],
                        payload=batch_docs[i]
                    )
                    for i in range(len(batch_docs))
                ]
                
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                
                stats.successful += len(batch_docs)
                stats.processed = total_files
                
            except Exception as e:
                logger.error(f"Final batch upload failed: {e}")
                stats.failed += len(batch_docs)
        
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user!")
        save_checkpoint(stats, [])
        logger.info(f"Checkpoint saved. Resume with --resume flag.")
    
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        logger.error(traceback.format_exc())
        save_checkpoint(stats, [])
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("  INGESTION COMPLETE")
    logger.info("=" * 70)
    print_progress_bar(stats)
    
    # Verify final count
    info = client.get_collection(collection_name)
    logger.info(f"")
    logger.info(f"Final collection status:")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Total Points: {info.points_count:,}")
    logger.info(f"  Status: {info.status}")
    logger.info(f"")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 70)
    
    # Clean up checkpoint on successful completion
    if stats.processed >= stats.total_files and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed (ingestion complete)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest full GeneALaCart dataset into Qdrant")
    parser.add_argument(
        "json_dir",
        nargs="?",
        default="/mnt/c/Users/7wraa/Downloads/Ayass-v5.26-neo4j/GeneALaCart-AllGenes",
        help="Path to GeneALaCart-AllGenes directory"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size (default: 100)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint interval (default: 1000)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume from checkpoint")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    
    args = parser.parse_args()
    
    ingest_full_dataset(
        json_dir=args.json_dir,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        recreate_collection=args.recreate,
    )
