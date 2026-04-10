"""
Report Section Generator Module - Generates report content for drug recommendations.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..models.data_models import DrugAgentOutput, DrugRecommendation

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section of the report."""
    title: str
    content: str
    tables: List[Dict] = None
    subsections: List["ReportSection"] = None


class ReportSectionGenerator:
    """Generates formatted report sections for drug recommendations."""
    
    SECTION_TITLE = "Therapeutic Recommendations"
    
    def __init__(self, section_number: str = "X"):
        """
        Initialize report generator.
        
        Args:
            section_number: Section number for the report (e.g., "5" or "X").
        """
        self.section_number = section_number
    
    def generate_full_section(self, output: DrugAgentOutput) -> str:
        """
        Generate the complete therapeutic recommendations section.
        
        Args:
            output: Drug agent output with all recommendations.
            
        Returns:
            Formatted section text.
        """
        sections = []
        
        # Section header
        sections.append(f"## {self.section_number}. {self.SECTION_TITLE}\n")
        
        # Summary subsection
        sections.append(f"### {self.section_number}.1 Summary\n")
        sections.append(output.therapeutic_summary + "\n")
        
        # Prioritized recommendations table
        sections.append(f"### {self.section_number}.2 Prioritized Drug Recommendations\n")
        sections.append(self._generate_recommendations_table(output.drug_recommendations))
        
        # Gene-Drug associations
        if output.gene_drug_table:
            sections.append(f"### {self.section_number}.3 Gene-Drug Associations\n")
            sections.append(self._generate_gene_drug_table(output))
        
        # Pathway-Drug associations
        if output.pathway_drug_table:
            sections.append(f"### {self.section_number}.4 Pathway-Drug Associations\n")
            sections.append(self._generate_pathway_drug_table(output))
        
        # Evidence summary for top drugs
        sections.append(f"### {self.section_number}.5 Evidence Summary\n")
        sections.append(self._generate_evidence_details(output.drug_recommendations[:5]))
        
        # Important considerations
        sections.append(f"### {self.section_number}.6 Important Considerations\n")
        sections.append(self._generate_considerations())
        
        return "\n".join(sections)
    
    def _generate_recommendations_table(
        self,
        recommendations: List[DrugRecommendation],
    ) -> str:
        """Generate the main recommendations table."""
        if not recommendations:
            return "_No drug recommendations available._\n"
        
        # Table header
        lines = [
            "| Rank | Drug Name | Target(s) | Status | Evidence | Patient Gene Match |",
            "|------|-----------|-----------|--------|----------|-------------------|",
        ]
        
        for i, rec in enumerate(recommendations[:15], 1):
            targets = ", ".join(rec.target_genes[:3]) if rec.target_genes else "-"
            status = rec.approval_status or "-"
            evidence = rec.evidence_level or "-"
            gene_match = ", ".join(rec.patient_gene_match[:2]) if rec.patient_gene_match else "-"
            
            # Truncate long fields
            if len(status) > 20:
                status = status[:17] + "..."
            if len(gene_match) > 20:
                gene_match = gene_match[:17] + "..."
            
            lines.append(
                f"| {i} | {rec.drug_name} | {targets} | {status} | {evidence} | {gene_match} |"
            )
        
        return "\n".join(lines) + "\n"
    
    def _generate_gene_drug_table(self, output: DrugAgentOutput) -> str:
        """Generate gene-drug association table."""
        if not output.gene_drug_table:
            return ""
        
        lines = [
            "| Gene | Expression | Direction | Associated Drugs |",
            "|------|------------|-----------|------------------|",
        ]
        
        for assoc in output.gene_drug_table[:15]:
            drugs = ", ".join(assoc.associated_drugs[:3]) if assoc.associated_drugs else "-"
            note = f" *{assoc.notes}*" if assoc.notes else ""
            
            lines.append(
                f"| {assoc.gene_symbol} | {assoc.expression_change} | {assoc.direction} | {drugs}{note} |"
            )
        
        return "\n".join(lines) + "\n"
    
    def _generate_pathway_drug_table(self, output: DrugAgentOutput) -> str:
        """Generate pathway-drug association table."""
        if not output.pathway_drug_table:
            return ""
        
        lines = [
            "| Pathway | Regulation | Drugs Targeting Pathway |",
            "|---------|------------|------------------------|",
        ]
        
        for assoc in output.pathway_drug_table[:10]:
            drugs = ", ".join(assoc.targeting_drugs[:3]) if assoc.targeting_drugs else "-"
            
            # Truncate long pathway names
            pathway = assoc.pathway_name
            if len(pathway) > 40:
                pathway = pathway[:37] + "..."
            
            lines.append(f"| {pathway} | {assoc.regulation} | {drugs} |")
        
        return "\n".join(lines) + "\n"
    
    def _generate_evidence_details(
        self,
        recommendations: List[DrugRecommendation],
    ) -> str:
        """Generate detailed evidence for top recommendations."""
        if not recommendations:
            return ""
        
        details = []
        
        for rec in recommendations:
            drug_detail = [f"**{rec.drug_name}**"]
            
            if rec.drug_aliases:
                drug_detail.append(f"  - Also known as: {', '.join(rec.drug_aliases[:3])}")
            
            if rec.target_genes:
                drug_detail.append(f"  - Target: {', '.join(rec.target_genes)}")
            
            if rec.patient_gene_match:
                gene = rec.patient_gene_match[0]
                drug_detail.append(f"  - Patient Evidence: {gene} is differentially expressed")
            
            if rec.evidence_level:
                drug_detail.append(f"  - Evidence Level: {rec.evidence_level}")
            
            if rec.approval_status:
                drug_detail.append(f"  - Approval Status: {rec.approval_status}")
            
            if rec.mechanism_of_action:
                drug_detail.append(f"  - Mechanism: {rec.mechanism_of_action}")
            
            if rec.confirmation_tests:
                drug_detail.append(f"  - Confirmation Required: {', '.join(rec.confirmation_tests)}")
            
            details.append("\n".join(drug_detail))
        
        return "\n\n".join(details) + "\n"
    
    def _generate_considerations(self) -> str:
        """Generate standard clinical considerations."""
        considerations = [
            "• All recommendations require confirmatory testing as specified",
            "• RNA expression patterns are suggestive and do not determine eligibility",
            "• Clinical decisions should integrate pathology, staging, and patient factors",
            "• Consult current NCCN guidelines for treatment sequencing",
            "• Drug interactions and contraindications must be evaluated",
            "• This analysis is for research purposes and requires clinical validation",
        ]
        return "\n".join(considerations) + "\n"
    
    def generate_summary_only(self, output: DrugAgentOutput) -> str:
        """Generate just the summary section."""
        return output.therapeutic_summary
    
    def generate_for_docx(self, output: DrugAgentOutput) -> Dict:
        """
        Generate structured data for DOCX report generation.
        
        Returns:
            Dictionary with structured report data.
        """
        return {
            "section_title": self.SECTION_TITLE,
            "section_number": self.section_number,
            "summary": output.therapeutic_summary,
            "recommendations_table": {
                "headers": ["Rank", "Drug Name", "Target(s)", "Status", "Evidence", "Patient Gene Match"],
                "rows": [
                    [
                        str(i + 1),
                        rec.drug_name,
                        ", ".join(rec.target_genes[:2]),
                        rec.approval_status or "-",
                        rec.evidence_level or "-",
                        ", ".join(rec.patient_gene_match[:2]) if rec.patient_gene_match else "-",
                    ]
                    for i, rec in enumerate(output.drug_recommendations[:15])
                ],
            },
            "gene_drug_table": {
                "headers": ["Gene", "Expression", "Direction", "Associated Drugs"],
                "rows": [
                    [g.gene_symbol, g.expression_change, g.direction, ", ".join(g.associated_drugs[:3])]
                    for g in output.gene_drug_table[:15]
                ],
            },
            "pathway_drug_table": {
                "headers": ["Pathway", "Regulation", "Drugs Targeting Pathway"],
                "rows": [
                    [p.pathway_name[:40], p.regulation, ", ".join(p.targeting_drugs[:3])]
                    for p in output.pathway_drug_table[:10]
                ],
            },
            "evidence_details": [
                {
                    "drug_name": rec.drug_name,
                    "targets": rec.target_genes,
                    "mechanism": rec.mechanism_of_action,
                    "status": rec.approval_status,
                    "evidence": rec.evidence_level,
                    "patient_match": rec.patient_gene_match,
                    "confirmation": rec.confirmation_tests,
                }
                for rec in output.drug_recommendations[:5]
            ],
            "considerations": [
                "All recommendations require confirmatory testing as specified",
                "RNA expression patterns are suggestive and do not determine eligibility",
                "Clinical decisions should integrate pathology, staging, and patient factors",
                "Consult current NCCN guidelines for treatment sequencing",
            ],
            "statistics": {
                "total_drugs": output.total_drugs_found,
                "fda_approved": output.fda_approved_count,
                "with_gene_match": output.drugs_with_gene_match,
            },
        }
