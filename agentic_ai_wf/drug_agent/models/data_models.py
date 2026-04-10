"""Data Models for Drug Discovery Agent."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class DocumentType(str, Enum):
    GENE_DRUG = "gene_drug"
    DISEASE_DRUG = "disease_drug"
    PATHWAY_DRUG = "pathway_drug"
    GENE_DISEASE_CONTEXT = "gene_disease_context"


class EvidenceLevel(str, Enum):
    LEVEL_1A = "Level 1A"
    LEVEL_1B = "Level 1B"
    LEVEL_2A = "Level 2A"
    LEVEL_2B = "Level 2B"
    LEVEL_3 = "Level 3"
    LEVEL_4 = "Level 4"


class ApprovalStatus(str, Enum):
    FDA_APPROVED = "FDA Approved"
    EMA_APPROVED = "EMA Approved"
    PHASE_3 = "Phase III"
    PHASE_2 = "Phase II"
    PHASE_1 = "Phase I"
    PRECLINICAL = "Preclinical"
    UNKNOWN = "Unknown"


@dataclass
class GeneMapping:
    """Gene information from DEG Agent."""
    gene: str
    log2fc: float = 0.0
    adj_p_value: float = 1.0
    observed_direction: str = "unknown"
    category: str = ""
    therapeutic_target: bool = False
    composite_score: float = 0.0
    aliases: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneMapping":
        direction = data.get('observed_direction', data.get('Patient_LFC_Trend', 'unknown'))
        if isinstance(direction, str):
            direction = direction.lower()
            if direction in ['up', 'upregulated']:
                direction = 'up'
            elif direction in ['down', 'downregulated']:
                direction = 'down'
        return cls(
            gene=data.get('gene', data.get('Gene', '')),
            log2fc=float(data.get('log2fc', data.get('Patient_LFC_mean', 0)) or 0),
            adj_p_value=float(data.get('adj_p_value', 1.0) or 1.0),
            observed_direction=direction,
            category=data.get('category', ''),
            therapeutic_target=data.get('therapeutic_target', False),
            composite_score=float(data.get('composite_score', 0) or 0),
            aliases=data.get('aliases', []),
        )


@dataclass
class PathwayMapping:
    """Pathway information from Pathway Agent."""
    pathway_name: str
    pathway_id: str = ""
    pathway_source: str = ""
    p_value: float = 1.0
    fdr: float = 1.0
    regulation: str = ""
    input_genes: List[str] = field(default_factory=list)
    clinical_relevance: str = ""
    functional_relevance: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathwayMapping":
        input_genes = data.get('input_genes', data.get('Input_Genes', data.get('Pathway_Associated_Genes', [])))
        if isinstance(input_genes, str):
            input_genes = [g.strip() for g in input_genes.split(',') if g.strip()]
        return cls(
            pathway_name=data.get('pathway_name', data.get('Pathway_Name', '')),
            pathway_id=data.get('pathway_id', data.get('Pathway_ID', '')),
            pathway_source=data.get('pathway_source', data.get('Pathway_Source', '')),
            p_value=float(data.get('p_value', data.get('P_Value', 1.0)) or 1.0),
            fdr=float(data.get('fdr', data.get('FDR', 1.0)) or 1.0),
            regulation=data.get('regulation', data.get('Regulation', '')),
            input_genes=input_genes,
            clinical_relevance=data.get('clinical_relevance', data.get('Clinical_Relevance', '')),
            functional_relevance=data.get('functional_relevance', data.get('Functional_Relevance', '')),
        )


@dataclass
class DrugAgentInput:
    """Input for Drug Discovery Agent."""
    disease_name: str
    gene_mappings: List[GeneMapping] = field(default_factory=list)
    pathway_mappings: List[PathwayMapping] = field(default_factory=list)
    xcell_findings: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_pipeline_data(cls, disease_name: str, gene_mappings: List[Dict],
                           pathway_mappings: List[Dict], xcell_findings: Optional[Dict] = None) -> "DrugAgentInput":
        return cls(
            disease_name=disease_name,
            gene_mappings=[GeneMapping.from_dict(g) for g in gene_mappings],
            pathway_mappings=[PathwayMapping.from_dict(p) for p in pathway_mappings],
            xcell_findings=xcell_findings,
        )
    
    def get_top_genes(self, n: int = 20) -> List[str]:
        sorted_genes = sorted(self.gene_mappings, key=lambda g: (g.composite_score, abs(g.log2fc)), reverse=True)
        return [g.gene for g in sorted_genes[:n]]
    
    def get_upregulated_genes(self) -> List[str]:
        return [g.gene for g in self.gene_mappings if g.observed_direction == "up"]
    
    def get_downregulated_genes(self) -> List[str]:
        return [g.gene for g in self.gene_mappings if g.observed_direction == "down"]
    
    def get_gene_directions(self) -> Dict[str, str]:
        return {g.gene: g.observed_direction for g in self.gene_mappings}
    
    def get_top_pathways(self, n: int = 10) -> List[str]:
        sorted_pathways = sorted(self.pathway_mappings, key=lambda p: p.p_value)
        return [p.pathway_name for p in sorted_pathways[:n]]


@dataclass
class DrugInfo:
    """Drug information from JSON files."""
    drug_name: str
    drug_aliases: List[str] = field(default_factory=list)
    drug_type: str = ""
    mechanism_of_action: str = ""
    approval_status: str = ""
    indications: List[str] = field(default_factory=list)
    target_genes: List[str] = field(default_factory=list)
    evidence_level: str = ""
    source: str = ""


@dataclass
class DiseaseInfo:
    """Disease information from JSON files."""
    disease_name: str
    disease_aliases: List[str] = field(default_factory=list)
    disease_category: str = ""
    associated_genes: List[str] = field(default_factory=list)
    source: str = ""


@dataclass
class PathwayInfo:
    """Pathway information from JSON files."""
    pathway_name: str
    pathway_id: str = ""
    pathway_source: str = ""
    pathway_genes: List[str] = field(default_factory=list)


@dataclass
class VectorDocument:
    """Document for vector storage."""
    doc_id: str
    doc_type: DocumentType
    text_content: str
    embedding: Optional[List[float]] = None
    gene_symbol: Optional[str] = None
    gene_aliases: List[str] = field(default_factory=list)
    drug_name: Optional[str] = None
    drug_aliases: List[str] = field(default_factory=list)
    disease_name: Optional[str] = None
    disease_aliases: List[str] = field(default_factory=list)
    pathway_name: Optional[str] = None
    pathway_genes: List[str] = field(default_factory=list)
    evidence_level: str = ""
    approval_status: str = ""
    mechanism_of_action: str = ""
    indications: List[str] = field(default_factory=list)
    source: str = ""
    source_file: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_payload(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value if isinstance(self.doc_type, DocumentType) else self.doc_type,
            "text_content": self.text_content,
            "gene_symbol": self.gene_symbol,
            "gene_aliases": self.gene_aliases,
            "drug_name": self.drug_name,
            "drug_aliases": self.drug_aliases,
            "disease_name": self.disease_name,
            "disease_aliases": self.disease_aliases,
            "pathway_name": self.pathway_name,
            "pathway_genes": self.pathway_genes,
            "evidence_level": self.evidence_level,
            "approval_status": self.approval_status,
            "mechanism_of_action": self.mechanism_of_action,
            "indications": self.indications,
            "source": self.source,
            "source_file": self.source_file,
            "created_at": self.created_at,
        }


@dataclass
class DrugRecommendation:
    """Drug recommendation with evidence."""
    drug_name: str
    drug_aliases: List[str] = field(default_factory=list)
    drug_type: str = ""
    target_genes: List[str] = field(default_factory=list)
    target_pathways: List[str] = field(default_factory=list)
    mechanism_of_action: str = ""
    approval_status: str = ""
    indication_match: str = ""
    evidence_level: str = ""
    patient_gene_match: List[str] = field(default_factory=list)
    patient_pathway_match: List[str] = field(default_factory=list)
    expression_concordance: str = ""
    relevance_score: float = 0.0
    gene_match_score: float = 0.0
    evidence_score: float = 0.0
    approval_score: float = 0.0
    composite_score: float = 0.0
    evidence_summary: str = ""
    evidence_sources: List[str] = field(default_factory=list)
    confirmation_tests: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_name": self.drug_name,
            "drug_aliases": self.drug_aliases,
            "drug_type": self.drug_type,
            "target_genes": self.target_genes,
            "target_pathways": self.target_pathways,
            "mechanism_of_action": self.mechanism_of_action,
            "approval_status": self.approval_status,
            "evidence_level": self.evidence_level,
            "patient_gene_match": self.patient_gene_match,
            "composite_score": self.composite_score,
            "evidence_summary": self.evidence_summary,
        }


@dataclass
class GeneDrugAssociation:
    """Gene to drug association for tables."""
    gene_symbol: str
    expression_change: str = ""
    direction: str = ""
    associated_drugs: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PathwayDrugAssociation:
    """Pathway to drug association for tables."""
    pathway_name: str
    regulation: str = ""
    targeting_drugs: List[str] = field(default_factory=list)
    member_genes_targeted: List[str] = field(default_factory=list)


@dataclass
class DrugAgentOutput:
    """Complete output from Drug Discovery Agent."""
    drug_recommendations: List[DrugRecommendation] = field(default_factory=list)
    total_drugs_found: int = 0
    drugs_with_gene_match: int = 0
    drugs_with_pathway_match: int = 0
    fda_approved_count: int = 0
    gene_drug_table: List[GeneDrugAssociation] = field(default_factory=list)
    pathway_drug_table: List[PathwayDrugAssociation] = field(default_factory=list)
    therapeutic_summary: str = ""
    evidence_narrative: str = ""
    disease_queried: str = ""
    genes_queried: List[str] = field(default_factory=list)
    pathways_queried: List[str] = field(default_factory=list)
    query_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_recommendations": [d.to_dict() for d in self.drug_recommendations],
            "total_drugs_found": self.total_drugs_found,
            "drugs_with_gene_match": self.drugs_with_gene_match,
            "fda_approved_count": self.fda_approved_count,
            "therapeutic_summary": self.therapeutic_summary,
            "disease_queried": self.disease_queried,
        }


@dataclass
class IngestionResult:
    """Result of data ingestion."""
    success: bool
    total_files_processed: int = 0
    total_documents_created: int = 0
    documents_by_type: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass 
class KnowledgeBaseStats:
    """Statistics about the knowledge base."""
    total_documents: int = 0
    documents_by_type: Dict[str, int] = field(default_factory=dict)
    unique_genes: int = 0
    unique_drugs: int = 0
    unique_diseases: int = 0
    unique_pathways: int = 0
    last_updated: Optional[str] = None
    collection_status: str = "unknown"
