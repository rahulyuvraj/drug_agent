"""
run_ingestion.py - One-time script to build the knowledge base
Run this ONCE to ingest all gene JSON files into Qdrant Cloud.

Usage:
    cd drug_agent
    python run_ingestion.py
"""

import sys
import time
from pathlib import Path

# Add PARENT directory to path so 'drug_agent' is importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from drug_agent import DrugDiscoveryAgent


def main():
    print("=" * 60)
    print("DRUG DISCOVERY AGENT - KNOWLEDGE BASE INGESTION")
    print("=" * 60)
    
    # ===== CONFIGURATION =====
    # UPDATE THIS PATH to your gene JSON files location
    JSON_DIRECTORY = "/mnt/c/Users/7wraa/Downloads/Ayass-v5.26-neo4j/GeneALaCart-AllGenes"
    
    # Set to True for fresh start, False to add to existing
    RECREATE_COLLECTION = False  # Keep existing collection, add data
    
    # Batch size (reduce if memory issues)
    BATCH_SIZE = 100
    
    # Limit to ~1GB of data (~40K files out of 443K)
    MAX_FILES = 40000
    # ==========================
    
    # Step 1: Initialize agent
    print("\n[1/4] Initializing agent...")
    try:
        agent = DrugDiscoveryAgent()
    except Exception as e:
        print(f"ERROR: Failed to initialize agent: {e}")
        print("\nCheck your .env file has:")
        print("  QDRANT_URL=https://your-cluster.cloud.qdrant.io")
        print("  QDRANT_API_KEY=your-api-key")
        return
    
    # Verify connection
    health = agent.health_check()
    print(f"  - Qdrant Connected: {health.get('qdrant_connected', False)}")
    
    if not health.get('qdrant_connected'):
        print("\nERROR: Cannot connect to Qdrant Cloud!")
        print("Check your .env file: QDRANT_URL and QDRANT_API_KEY")
        return
    
    # Verify directory exists
    if not Path(JSON_DIRECTORY).exists():
        print(f"\nERROR: Directory not found: {JSON_DIRECTORY}")
        print("Update JSON_DIRECTORY in this script to your gene files location.")
        return
    
    # Step 2: Run ingestion
    print(f"\n[2/4] Starting ingestion from: {JSON_DIRECTORY}")
    print(f"  - Processing up to {MAX_FILES} files (~1GB of data)")
    print("  - This may take 30-60 minutes")
    print("  - Progress will be shown below\n")
    
    start_time = time.time()
    
    result = agent.ingest_gene_data(
        json_directory=JSON_DIRECTORY,
        recreate_collection=RECREATE_COLLECTION,
        batch_size=BATCH_SIZE,
        max_files=MAX_FILES,
    )
    
    elapsed = time.time() - start_time
    
    # Step 3: Show results
    print(f"\n[3/4] Ingestion Complete!")
    print(f"  - Success: {result.success}")
    print(f"  - Files Processed: {result.total_files_processed}")
    print(f"  - Documents Created: {result.total_documents_created}")
    print(f"  - Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    if result.documents_by_type:
        print("\n  Documents by Type:")
        for doc_type, count in result.documents_by_type.items():
            print(f"    - {doc_type}: {count}")
    
    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"    - {err}")
    
    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for warn in result.warnings[:5]:
            print(f"    - {warn}")
    
    # Step 4: Verify
    print("\n[4/4] Verifying knowledge base...")
    try:
        stats = agent.get_knowledge_base_stats()
        print(f"  - Total Documents: {stats.total_documents}")
        print(f"  - Collection Status: {stats.collection_status}")
    except Exception as e:
        print(f"  - Could not verify: {e}")
    
    print("\n" + "=" * 60)
    if result.success:
        print("SUCCESS! Knowledge base ready for queries.")
        print("Next step: python run_query.py")
    else:
        print("INGESTION FAILED. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()