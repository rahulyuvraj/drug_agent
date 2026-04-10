"""
run_query.py - Query the drug discovery agent for recommendations
Run AFTER ingestion is complete.

Usage:
    cd drug_agent
    python run_query.py
"""

import sys
from pathlib import Path

# Add PARENT directory to path so 'drug_agent' is importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from drug_agent import DrugDiscoveryAgent, DrugAgentInput
from drug_agent.models.data_models import GeneMapping, PathwayMapping


def main():
    print("=" * 60)
    print("DRUG DISCOVERY AGENT - QUERY")
    print("=" * 60)
    
    # ===== CONFIGURATION =====
    # UPDATE THESE with your actual analysis results
    
    DISEASE_NAME = "Breast Cancer"
    
    # Top DEGs from your DEG analysis (copy from your DEG output)
    TOP_GENES = [
        {"gene": "ERBB2", "log2fc": 4.25, "direction": "up", "score": 10},
        {"gene": "ESR1", "log2fc": -3.6, "direction": "down", "score": 9},
        {"gene": "BRCA2", "log2fc": 1.67, "direction": "up", "score": 8},
        {"gene": "PIK3CA", "log2fc": 2.1, "direction": "up", "score": 7},
        {"gene": "MKI67", "log2fc": 3.2, "direction": "up", "score": 6},
        {"gene": "TOP2A", "log2fc": 2.8, "direction": "up", "score": 5},
        {"gene": "CCND1", "log2fc": 1.9, "direction": "up", "score": 4},
    ]
    
    # Top pathways from your Pathway analysis
    TOP_PATHWAYS = [
        {"name": "PI3K-AKT Signaling Pathway", "regulation": "Up", "p_value": 1e-9},
        {"name": "Cell Cycle", "regulation": "Up", "p_value": 1e-8},
        {"name": "Apoptosis", "regulation": "Down", "p_value": 1e-7},
        {"name": "DNA Repair", "regulation": "Up", "p_value": 1e-6},
    ]
    # ==========================
    
    # Initialize agent
    print("\n[1/3] Initializing agent...")
    try:
        agent = DrugDiscoveryAgent()
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    # Check if knowledge base exists
    health = agent.health_check()
    if health.get('documents_count', 0) == 0:
        print("\nWARNING: Knowledge base is empty!")
        print("Run 'python run_ingestion.py' first.")
        return None
    
    print(f"  - Knowledge base: {health.get('documents_count', 0)} documents")
    
    # Build input data
    print("\n[2/3] Building query...")
    
    gene_mappings = [
        GeneMapping(
            gene=g["gene"],
            log2fc=g["log2fc"],
            observed_direction=g["direction"],
            composite_score=g["score"]
        )
        for g in TOP_GENES
    ]
    
    pathway_mappings = [
        PathwayMapping(
            pathway_name=p["name"],
            regulation=p["regulation"],
            p_value=p["p_value"]
        )
        for p in TOP_PATHWAYS
    ]
    
    input_data = DrugAgentInput(
        disease_name=DISEASE_NAME,
        gene_mappings=gene_mappings,
        pathway_mappings=pathway_mappings,
    )
    
    print(f"  - Disease: {DISEASE_NAME}")
    print(f"  - Genes: {len(gene_mappings)}")
    print(f"  - Pathways: {len(pathway_mappings)}")
    
    # Generate recommendations
    print("\n[3/3] Generating recommendations...")
    output = agent.generate_recommendations(input_data)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"DRUG RECOMMENDATIONS FOR: {DISEASE_NAME}")
    print(f"{'='*60}")
    
    print(f"\nSummary:")
    print(f"  - Total drugs found: {output.total_drugs_found}")
    print(f"  - FDA Approved: {output.fda_approved_count}")
    print(f"  - With Gene Match: {output.drugs_with_gene_match}")
    
    print(f"\n{'='*60}")
    print("TOP RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    for i, rec in enumerate(output.drug_recommendations[:10], 1):
        print(f"{i}. {rec.drug_name}")
        print(f"   Score: {rec.composite_score:.3f}")
        if rec.target_genes:
            print(f"   Targets: {', '.join(rec.target_genes[:3])}")
        if rec.approval_status:
            print(f"   Status: {rec.approval_status}")
        if rec.patient_gene_match:
            print(f"   Patient Gene Match: {', '.join(rec.patient_gene_match)}")
        if rec.mechanism_of_action:
            print(f"   Mechanism: {rec.mechanism_of_action[:80]}...")
        print()
    
    print(f"{'='*60}")
    print("THERAPEUTIC SUMMARY")
    print(f"{'='*60}")
    print(output.therapeutic_summary)
    
    return output


if __name__ == "__main__":
    output = main()