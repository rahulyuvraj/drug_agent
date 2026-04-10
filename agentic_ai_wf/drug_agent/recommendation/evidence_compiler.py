"""
Evidence Compiler Module - Aggregates and formats evidence for drug recommendations.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from ..models.data_models import (
    DrugRecommendation, DrugAgentInput, DrugAgentOutput,
    GeneDrugAssociation, PathwayDrugAssociation,
)

logger = logging.getLogger(__name__)


@dataclass
class DrugEvidence:
    """Compiled evidence for a drug."""
    drug_name: str
    target_genes: List[str] = field(default_factory=list)
    mechanism: str = ""
    approval_status: str = ""
    evidence_level: str = ""
    indications: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    clinical_notes: List[str] = field(default_factory=list)


class EvidenceCompiler:
    """Compiles and formats evidence from multiple sources."""
    
    def __init__(self):
        self.evidence_cache: Dict[str, DrugEvidence] = {}
    
    def compile_evidence(
        self,
        recommendations: List[DrugRecommendation],
        input_data: DrugAgentInput,
    ) -> DrugAgentOutput:
        """
        Compile all evidence into a structured output.
        
        Args:
            recommendations: List of drug recommendations.
            input_data: Original input data.
            
        Returns:
            Complete DrugAgentOutput with all compiled data.
        """
        # Build gene-drug table
        gene_drug_table = self._build_gene_drug_table(recommendations, input_data)
        
        # Build pathway-drug table
        pathway_drug_table = self._build_pathway_drug_table(recommendations, input_data)
        
        # Calculate statistics
        fda_approved_count = sum(
            1 for r in recommendations 
            if "approved" in r.approval_status.lower()
        )
        drugs_with_gene_match = sum(
            1 for r in recommendations 
            if r.patient_gene_match
        )
        
        # Generate therapeutic summary
        therapeutic_summary = self._generate_therapeutic_summary(
            recommendations, input_data
        )
        
        # Generate evidence narrative
        evidence_narrative = self._generate_evidence_narrative(
            recommendations, input_data
        )
        
        output = DrugAgentOutput(
            drug_recommendations=recommendations,
            total_drugs_found=len(recommendations),
            drugs_with_gene_match=drugs_with_gene_match,
            drugs_with_pathway_match=len([r for r in recommendations if r.target_pathways]),
            fda_approved_count=fda_approved_count,
            gene_drug_table=gene_drug_table,
            pathway_drug_table=pathway_drug_table,
            therapeutic_summary=therapeutic_summary,
            evidence_narrative=evidence_narrative,
            disease_queried=input_data.disease_name,
            genes_queried=input_data.get_top_genes(20),
            pathways_queried=input_data.get_top_pathways(10),
        )
        
        return output
    
    def _build_gene_drug_table(
        self,
        recommendations: List[DrugRecommendation],
        input_data: DrugAgentInput,
    ) -> List[GeneDrugAssociation]:
        """Build gene to drug association table."""
        # Map genes to drugs
        gene_to_drugs: Dict[str, List[str]] = {}
        
        for rec in recommendations:
            for gene in rec.target_genes:
                if gene not in gene_to_drugs:
                    gene_to_drugs[gene] = []
                if rec.drug_name not in gene_to_drugs[gene]:
                    gene_to_drugs[gene].append(rec.drug_name)
        
        # Create associations for patient genes
        associations = []
        for gene_mapping in input_data.gene_mappings[:20]:  # Top 20 genes
            gene = gene_mapping.gene
            drugs = gene_to_drugs.get(gene, [])
            
            if drugs or gene_mapping.composite_score > 5:  # Include high-priority genes
                # Format expression change
                lfc = gene_mapping.log2fc
                expression = f"{'+' if lfc > 0 else ''}{lfc:.2f} LFC"
                
                direction = "Up" if gene_mapping.observed_direction == "up" else "Down"
                
                # Add note for downregulated targets
                notes = ""
                if direction == "Down" and drugs:
                    notes = "Target downregulation may affect drug response"
                
                assoc = GeneDrugAssociation(
                    gene_symbol=gene,
                    expression_change=expression,
                    direction=direction,
                    associated_drugs=drugs[:5],  # Top 5 drugs
                    notes=notes,
                )
                associations.append(assoc)
        
        return associations
    
    def _build_pathway_drug_table(
        self,
        recommendations: List[DrugRecommendation],
        input_data: DrugAgentInput,
    ) -> List[PathwayDrugAssociation]:
        """Build pathway to drug association table."""
        # Map pathways to drugs
        pathway_to_drugs: Dict[str, List[str]] = {}
        pathway_to_genes: Dict[str, List[str]] = {}
        
        for rec in recommendations:
            for pathway in rec.target_pathways:
                if pathway:
                    if pathway not in pathway_to_drugs:
                        pathway_to_drugs[pathway] = []
                        pathway_to_genes[pathway] = []
                    if rec.drug_name not in pathway_to_drugs[pathway]:
                        pathway_to_drugs[pathway].append(rec.drug_name)
                    for gene in rec.target_genes:
                        if gene not in pathway_to_genes[pathway]:
                            pathway_to_genes[pathway].append(gene)
        
        # Create associations for patient pathways
        associations = []
        for pathway_mapping in input_data.pathway_mappings[:10]:
            pathway = pathway_mapping.pathway_name
            drugs = pathway_to_drugs.get(pathway, [])
            
            # Check if any pathway gene is a drug target
            pathway_genes = pathway_mapping.input_genes
            targeted_genes = [g for g in pathway_genes if g in pathway_to_genes.get(pathway, [])]
            
            if drugs:
                assoc = PathwayDrugAssociation(
                    pathway_name=pathway,
                    regulation=pathway_mapping.regulation,
                    targeting_drugs=drugs[:5],
                    member_genes_targeted=targeted_genes[:5],
                )
                associations.append(assoc)
        
        return associations
    
    def _generate_therapeutic_summary(
        self,
        recommendations: List[DrugRecommendation],
        input_data: DrugAgentInput,
    ) -> str:
        """Generate a therapeutic summary paragraph."""
        if not recommendations:
            return (
                f"No specific therapeutic agents were identified for {input_data.disease_name} "
                f"based on the current transcriptomic profile."
            )
        
        # Count categories
        fda_count = sum(1 for r in recommendations if "approved" in r.approval_status.lower())
        gene_match_count = sum(1 for r in recommendations if r.patient_gene_match)
        
        disease = input_data.disease_name
        total = len(recommendations)
        
        # Build summary
        summary_parts = [
            f"Based on the transcriptomic profile of this patient with {disease}, "
            f"{total} potential therapeutic agents were identified."
        ]
        
        if fda_count > 0:
            summary_parts.append(
                f"Of these, {fda_count} are FDA-approved agents with established efficacy."
            )
        
        if gene_match_count > 0:
            summary_parts.append(
                f"{gene_match_count} drugs directly target genes showing significant "
                f"differential expression in this patient's profile."
            )
        
        # Highlight top recommendations
        top_drugs = [r.drug_name for r in recommendations[:3]]
        if top_drugs:
            summary_parts.append(
                f"Top recommendations include {', '.join(top_drugs)}."
            )
        
        return " ".join(summary_parts)
    
    def _generate_evidence_narrative(
        self,
        recommendations: List[DrugRecommendation],
        input_data: DrugAgentInput,
    ) -> str:
        """Generate an evidence narrative."""
        if not recommendations:
            return ""
        
        narratives = []
        
        for rec in recommendations[:5]:  # Top 5 drugs
            parts = [f"**{rec.drug_name}**"]
            
            if rec.target_genes:
                parts.append(f"targets {', '.join(rec.target_genes)}")
            
            if rec.mechanism_of_action:
                parts.append(f"via {rec.mechanism_of_action}")
            
            if rec.approval_status:
                parts.append(f"({rec.approval_status})")
            
            if rec.patient_gene_match:
                genes = rec.patient_gene_match
                parts.append(
                    f". Patient evidence: {', '.join(genes)} differentially expressed"
                )
            
            narratives.append(" ".join(parts) + ".")
        
        return "\n\n".join(narratives)
    
    def format_for_report(
        self,
        output: DrugAgentOutput,
        include_tables: bool = True,
    ) -> Dict[str, any]:
        """Format output for report integration."""
        report_data = {
            "therapeutic_summary": output.therapeutic_summary,
            "recommendations": [
                {
                    "rank": i + 1,
                    "drug_name": r.drug_name,
                    "targets": ", ".join(r.target_genes),
                    "status": r.approval_status,
                    "evidence": r.evidence_level,
                    "gene_match": ", ".join(r.patient_gene_match),
                    "score": f"{r.composite_score:.3f}",
                }
                for i, r in enumerate(output.drug_recommendations)
            ],
            "statistics": {
                "total_drugs": output.total_drugs_found,
                "fda_approved": output.fda_approved_count,
                "with_gene_match": output.drugs_with_gene_match,
            },
        }
        
        if include_tables:
            report_data["gene_drug_table"] = [
                {
                    "gene": g.gene_symbol,
                    "expression": g.expression_change,
                    "direction": g.direction,
                    "drugs": ", ".join(g.associated_drugs),
                }
                for g in output.gene_drug_table
            ]
            report_data["pathway_drug_table"] = [
                {
                    "pathway": p.pathway_name,
                    "regulation": p.regulation,
                    "drugs": ", ".join(p.targeting_drugs),
                }
                for p in output.pathway_drug_table
            ]
        
        return report_data
