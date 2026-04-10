#!/usr/bin/env python3
"""
Simple Ingestion Script for Drug Agent
Ingests gene JSON data into self-hosted Qdrant with Basic Auth
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

def get_qdrant_client():
    """Get Qdrant client with Basic Auth support."""
    from qdrant_client import QdrantClient
    import httpx
    
    url = os.getenv("QDRANT_URL")
    username = os.getenv("QDRANT_USERNAME")
    password = os.getenv("QDRANT_PASSWORD")
    collection = os.getenv("QDRANT_COLLECTION", "Drug_agent")
    
    print(f"Connecting to: {url}")
    print(f"Collection: {collection}")
    
    # Create client with port 443 for nginx
    client = QdrantClient(
        url=url,
        port=443,
        timeout=120,
        prefer_grpc=False,
        https=True,
        check_compatibility=False,
    )
    
    # Patch with Basic Auth
    auth = httpx.BasicAuth(username, password)
    custom_http = httpx.Client(auth=auth, timeout=120.0)
    
    http_apis = client._client.http
    for api_name in ['collections_api', 'points_api', 'service_api', 'snapshots_api', 
                     'indexes_api', 'search_api', 'aliases_api', 'distributed_api', 'beta_api']:
        api = getattr(http_apis, api_name, None)
        if api and hasattr(api, 'api_client'):
            api.api_client._client = custom_http
    
    return client, collection


def get_embedder():
    """Load PubMedBERT embedder."""
    print("Loading PubMedBERT embedder (this may take a minute)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    print("Embedder loaded!")
    return model


def parse_gene_json(file_path: Path) -> Dict[str, Any]:
    """Parse a gene JSON file and extract relevant data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gene_symbol = file_path.stem  # Filename without extension
        
        # Extract drug-related information (UnifiedDrugs and Compounds)
        drugs = []
        for field in ["UnifiedDrugs", "UnifiedCompounds", "Compounds"]:
            if field in data and data[field]:
                for drug in data[field][:15]:
                    if isinstance(drug, dict):
                        name = drug.get("Name") or drug.get("DrugName") or drug.get("CompoundName", "")
                        if name and name not in [d["name"] for d in drugs]:
                            drugs.append({
                                "name": name,
                                "type": drug.get("Type", ""),
                            })
        
        # Extract disease associations (MalaCardsDisorders, UniProtDisorders, HumanPhenotypeOntology, GWASPhenotypes)
        diseases = []
        for field in ["MalaCardsDisorders", "MalaCardsInferredDisorders", "UniProtDisorders"]:
            if field in data and data[field]:
                for disease in data[field][:15]:
                    if isinstance(disease, dict):
                        name = disease.get("Name") or disease.get("DiseaseName", "")
                        if name and name not in [d["name"] for d in diseases]:
                            diseases.append({
                                "name": name,
                                "score": disease.get("Score", 0),
                            })
        
        # Extract phenotypes (HPO and GWAS)
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
        
        # Extract biological processes and molecular functions
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
        
        # Build rich text content for embedding
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
        logger.error(f"Error parsing {file_path}: {e}")
        return None


def ingest_data(json_dir: str, max_files: int = 40000, batch_size: int = 100):
    """Main ingestion function."""
    from qdrant_client.http import models
    
    print("=" * 60)
    print("DRUG AGENT - DATA INGESTION")
    print("=" * 60)
    
    # Get client and embedder
    client, collection_name = get_qdrant_client()
    embedder = get_embedder()
    
    # Verify connection
    print("\nVerifying Qdrant connection...")
    collections = client.get_collections()
    print(f"Connected! Collections: {[c.name for c in collections.collections]}")
    
    # Check/create collection
    collection_exists = any(c.name == collection_name for c in collections.collections)
    if not collection_exists:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,  # PubMedBERT dimension
                distance=models.Distance.COSINE,
            ),
        )
    
    # Find all JSON files
    print(f"\nScanning directory: {json_dir}")
    json_dir_path = Path(json_dir)
    json_files = list(json_dir_path.rglob("*.json"))[:max_files]
    total_files = len(json_files)
    print(f"Found {total_files} JSON files to process")
    
    # Process in batches
    start_time = time.time()
    processed = 0
    errors = 0
    
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = json_files[batch_start:batch_end]
        
        # Parse files
        documents = []
        for file_path in batch_files:
            doc = parse_gene_json(file_path)
            if doc:
                documents.append(doc)
        
        if not documents:
            continue
        
        # Generate embeddings
        texts = [doc["text_content"] for doc in documents]
        try:
            embeddings = embedder.encode(texts, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            errors += len(documents)
            continue
        
        # Prepare points for Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = batch_start + i
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "gene_symbol": doc["gene_symbol"],
                    "text_content": doc["text_content"],
                    "drugs": doc["drugs"],
                    "diseases": doc["diseases"],
                    "doc_type": "gene_data",
                }
            ))
        
        # Upsert to Qdrant
        try:
            client.upsert(collection_name=collection_name, points=points)
            processed += len(points)
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            errors += len(points)
            continue
        
        # Progress
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_files - batch_end) / rate if rate > 0 else 0
        print(f"Progress: {batch_end}/{total_files} ({100*batch_end/total_files:.1f}%) | "
              f"Rate: {rate:.1f} docs/s | ETA: {eta/60:.1f} min")
    
    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE!")
    print(f"  Documents processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)
    
    # Verify
    info = client.get_collection(collection_name)
    print(f"\nCollection '{collection_name}' now has {info.points_count} points")


if __name__ == "__main__":
    JSON_DIR = "/mnt/c/Users/7wraa/Downloads/Ayass-v5.26-neo4j/GeneALaCart-AllGenes"
    MAX_FILES = 40000  # ~1GB
    BATCH_SIZE = 100
    
    ingest_data(JSON_DIR, MAX_FILES, BATCH_SIZE)
