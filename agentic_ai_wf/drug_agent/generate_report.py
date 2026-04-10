"""
generate_report.py - Generate report section from drug recommendations
Run AFTER run_query.py to create report-ready output.

Usage:
    cd drug_agent
    python generate_report.py
"""

import sys
import json
from pathlib import Path

# Add PARENT directory to path so 'drug_agent' is importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from drug_agent import DrugDiscoveryAgent, DrugAgentInput
from drug_agent.models.data_models import GeneMapping, PathwayMapping


def main():
    print("=" * 60)
    print("DRUG DISCOVERY AGENT - REPORT GENERATION")
    print("=" * 60)
    
    # ===== CONFIGURATION =====
    DISEASE_NAME = "Breast Cancer"
    SECTION_NUMBER = "5"  # Section number in your report
    OUTPUT_DIR = "./output"  # Where to save reports
    
    # Same gene/pathway data as run_query.py
    TOP_GENES = [
        {"gene": "ERBB2", "log2fc": 4.25, "direction": "up", "score": 10},
        {"gene": "ESR1", "log2fc": -3.6, "direction": "down", "score": 9},
        {"gene": "BRCA2", "log2fc": 1.67, "direction": "up", "score": 8},
        {"gene": "PIK3CA", "log2fc": 2.1, "direction": "up", "score": 7},
        {"gene": "MKI67", "log2fc": 3.2, "direction": "up", "score": 6},
    ]
    
    TOP_PATHWAYS = [
        {"name": "PI3K-AKT Signaling Pathway", "regulation": "Up", "p_value": 1e-9},
        {"name": "Cell Cycle", "regulation": "Up", "p_value": 1e-8},
        {"name": "Apoptosis", "regulation": "Down", "p_value": 1e-7},
    ]
    # ==========================
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Initialize agent
    print("\n[1/4] Initializing agent...")
    agent = DrugDiscoveryAgent()
    
    # Build input
    print("[2/4] Building query...")
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
    
    # Generate recommendations
    print("[3/4] Generating recommendations...")
    output = agent.generate_recommendations(input_data)
    
    print(f"  - Found {output.total_drugs_found} drugs")
    
    # Generate reports
    print("[4/4] Generating report outputs...")
    
    # 1. Markdown report section
    md_report = agent.generate_report_section(output, section_number=SECTION_NUMBER)
    md_file = output_path / "drug_recommendations.md"
    with open(md_file, "w") as f:
        f.write(md_report)
    print(f"  - Saved: {md_file}")
    
    # 2. JSON data for DOCX integration
    report_data = agent.generate_report_data(output)
    json_file = output_path / "drug_recommendations.json"
    with open(json_file, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"  - Saved: {json_file}")
    
    # 3. Summary text
    summary_file = output_path / "therapeutic_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Disease: {DISEASE_NAME}\n")
        f.write(f"Date: {Path(__file__).stat().st_mtime}\n")
        f.write(f"\n{'='*60}\n")
        f.write("THERAPEUTIC SUMMARY\n")
        f.write(f"{'='*60}\n\n")
        f.write(output.therapeutic_summary)
        f.write(f"\n\n{'='*60}\n")
        f.write("TOP 5 RECOMMENDATIONS\n")
        f.write(f"{'='*60}\n\n")
        for i, rec in enumerate(output.drug_recommendations[:5], 1):
            f.write(f"{i}. {rec.drug_name} (Score: {rec.composite_score:.3f})\n")
            f.write(f"   Status: {rec.approval_status or 'N/A'}\n")
            f.write(f"   Targets: {', '.join(rec.target_genes) if rec.target_genes else 'N/A'}\n\n")
    print(f"  - Saved: {summary_file}")
    
    # 4. CSV for easy viewing
    csv_file = output_path / "drug_recommendations.csv"
    with open(csv_file, "w") as f:
        f.write("Rank,Drug,Score,Status,Targets,Gene_Match,Evidence_Level\n")
        for i, rec in enumerate(output.drug_recommendations, 1):
            targets = "|".join(rec.target_genes) if rec.target_genes else ""
            gene_match = "|".join(rec.patient_gene_match) if rec.patient_gene_match else ""
            status = (rec.approval_status or "").replace(",", ";")
            f.write(f'{i},"{rec.drug_name}",{rec.composite_score:.3f},"{status}","{targets}","{gene_match}","{rec.evidence_level or ""}"\n')
    print(f"  - Saved: {csv_file}")
    
    print(f"\n{'='*60}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput files in: {output_path.absolute()}")
    print("  - drug_recommendations.md   (Markdown for report)")
    print("  - drug_recommendations.json (Data for DOCX)")
    print("  - drug_recommendations.csv  (Spreadsheet view)")
    print("  - therapeutic_summary.txt   (Quick summary)")
    
    return output


if __name__ == "__main__":
    output = main()