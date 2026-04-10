"""
Data Normalizer Module (Dynamic)
================================

Normalizes data extracted from gene JSON files.
Fully dynamic - learns mappings from the knowledge base itself.
No hardcoded disease, drug, or gene information.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NormalizedDrug:
    """Normalized drug information."""
    drug_name: str = ""
    drug_aliases: List[str] = field(default_factory=list)
    drug_type: str = ""
    mechanism: str = ""
    approval_status: str = ""
    indications: List[str] = field(default_factory=list)
    evidence_level: str = ""
    sources: List[str] = field(default_factory=list)


@dataclass
class NormalizedDisease:
    """Normalized disease information."""
    canonical_name: str = ""
    disease_aliases: List[str] = field(default_factory=list)
    category: str = ""
    associated_genes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass
class NormalizedPathway:
    """Normalized pathway information."""
    pathway_name: str = ""
    pathway_id: str = ""
    source: str = ""
    genes: List[str] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)


class DataNormalizer:
    """
    Normalizes extracted gene data for consistent storage and retrieval.
    
    This normalizer is fully dynamic and does not contain any hardcoded
    mappings. It learns from the data during ingestion and can be enhanced
    with external mapping files if needed.
    """
    
    def __init__(
        self,
        disease_mappings: Optional[Dict[str, str]] = None,
        drug_mappings: Optional[Dict[str, str]] = None,
        gene_mappings: Optional[Dict[str, str]] = None,
    ):
        self.disease_mappings = disease_mappings or {}
        self.drug_mappings = drug_mappings or {}
        self.gene_mappings = gene_mappings or {}
        
        self._learned_disease_aliases: Dict[str, Set[str]] = defaultdict(set)
        self._learned_drug_aliases: Dict[str, Set[str]] = defaultdict(set)
        self._learned_gene_aliases: Dict[str, Set[str]] = defaultdict(set)
        
        self.seen_diseases: Set[str] = set()
        self.seen_drugs: Set[str] = set()
        self.seen_genes: Set[str] = set()
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def normalize_disease_name(self, name: str) -> str:
        """Normalize a disease name."""
        if not name:
            return ""
        cleaned = self.normalize_text(name)
        lookup_key = cleaned.lower()
        if lookup_key in self.disease_mappings:
            normalized = self.disease_mappings[lookup_key]
        else:
            normalized = cleaned.title()
        self.seen_diseases.add(normalized)
        return normalized
    
    def normalize_drug_name(self, name: str) -> str:
        """Normalize a drug name."""
        if not name:
            return ""
        cleaned = self.normalize_text(name)
        lookup_key = cleaned.lower()
        if lookup_key in self.drug_mappings:
            normalized = self.drug_mappings[lookup_key]
        else:
            normalized = cleaned.title()
        self.seen_drugs.add(normalized)
        return normalized
    
    def normalize_gene_symbol(self, symbol: str) -> str:
        """Normalize a gene symbol."""
        if not symbol:
            return ""
        cleaned = symbol.strip().upper()
        if cleaned in self.gene_mappings:
            normalized = self.gene_mappings[cleaned]
        else:
            normalized = cleaned
        self.seen_genes.add(normalized)
        return normalized
    
    # def normalize_gene_data(self, parsed_data: Any) -> Dict[str, Any]:
    #     """
    #     Normalize all data from a parsed gene file.
        
    #     Args:
    #         parsed_data: ParsedGeneData object from json_parser.
            
    #     Returns:
    #         Dictionary with normalized drugs, diseases, pathways.
    #     """
    #     gene_symbol = self.normalize_gene_symbol(parsed_data.gene_symbol)
    #     gene_aliases = [a.upper() for a in parsed_data.gene_aliases if a]
        
    #     # Normalize drugs
    #     drugs = []
    #     for drug_data in parsed_data.drugs:
    #         drug = self._normalize_drug(drug_data)
    #         if drug.drug_name:
    #             drugs.append(drug)
        
    #     # Normalize diseases
    #     diseases = []
    #     for disease_data in parsed_data.diseases:
    #         disease = self._normalize_disease(disease_data, gene_symbol)
    #         if disease.canonical_name:
    #             diseases.append(disease)
        
    #     # Normalize pathways
    #     pathways = []
    #     for pathway_data in parsed_data.pathways:
    #         pathway = self._normalize_pathway(pathway_data, gene_symbol)
    #         if pathway.pathway_name:
    #             pathways.append(pathway)
        
    #     return {
    #         "gene_symbol": gene_symbol,
    #         "gene_aliases": gene_aliases,
    #         "drugs": drugs,
    #         "diseases": diseases,
    #         "pathways": pathways,
    #     }
    
    def normalize_gene_data(self, parsed_data: Any) -> Dict[str, Any]:
        """
        Normalize all data from a parsed gene file.
        
        Args:
            parsed_data: ParsedGeneData object from json_parser.
            
        Returns:
            Dictionary with normalized drugs, diseases, pathways.
        """
        gene_symbol = self.normalize_gene_symbol(parsed_data.gene_symbol)
        gene_aliases = [a.upper() for a in parsed_data.gene_aliases if a]
        
        # Normalize drugs - use get_all_drugs() method
        drugs = []
        for drug_data in parsed_data.get_all_drugs():
            drug = self._normalize_drug(drug_data)
            if drug.drug_name:
                drugs.append(drug)
        
        # Normalize diseases - use get_all_diseases() method
        diseases = []
        for disease_data in parsed_data.get_all_diseases():
            disease = self._normalize_disease(disease_data, gene_symbol)
            if disease.canonical_name:
                diseases.append(disease)
        
        # Normalize pathways
        pathways = []
        for pathway_data in parsed_data.pathways:
            pathway = self._normalize_pathway(pathway_data, gene_symbol)
            if pathway.pathway_name:
                pathways.append(pathway)
        
        return {
            "gene_symbol": gene_symbol,
            "gene_aliases": gene_aliases,
            "drugs": drugs,
            "diseases": diseases,
            "pathways": pathways,
        }
    def _normalize_drug(self, drug_data: Dict) -> NormalizedDrug:
        """Normalize a single drug entry."""
        drug_name = drug_data.get("drug_name") or drug_data.get("DrugName") or drug_data.get("Name") or ""
        drug_name = self.normalize_drug_name(drug_name)
        
        drug_type = drug_data.get("drug_type") or drug_data.get("DrugType") or drug_data.get("Type") or ""
        mechanism = drug_data.get("mechanism") or drug_data.get("Mechanism") or drug_data.get("MechanismOfAction") or ""
        approval_status = drug_data.get("approval_status") or drug_data.get("ApprovalStatus") or drug_data.get("Status") or ""
        evidence_level = drug_data.get("evidence_level") or drug_data.get("EvidenceLevel") or ""
        
        indications = drug_data.get("indications") or drug_data.get("Indications") or []
        if isinstance(indications, str):
            indications = [indications]
        
        source = drug_data.get("source") or drug_data.get("Source") or ""
        sources = [source] if source else []
        
        aliases = drug_data.get("aliases") or drug_data.get("Aliases") or []
        
        return NormalizedDrug(
            drug_name=drug_name,
            drug_aliases=aliases,
            drug_type=drug_type,
            mechanism=mechanism,
            approval_status=approval_status,
            indications=indications,
            evidence_level=evidence_level,
            sources=sources,
        )
    
    def _normalize_disease(self, disease_data: Dict, gene_symbol: str) -> NormalizedDisease:
        """Normalize a single disease entry."""
        disease_name = disease_data.get("disease_name") or disease_data.get("DiseaseName") or disease_data.get("Name") or ""
        canonical_name = self.normalize_disease_name(disease_name)
        
        aliases = disease_data.get("aliases") or disease_data.get("Aliases") or []
        category = disease_data.get("category") or disease_data.get("Category") or ""
        source = disease_data.get("source") or disease_data.get("Source") or ""
        
        return NormalizedDisease(
            canonical_name=canonical_name,
            disease_aliases=aliases,
            category=category,
            associated_genes=[gene_symbol] if gene_symbol else [],
            sources=[source] if source else [],
        )
    
    def _normalize_pathway(self, pathway_data: Dict, gene_symbol: str) -> NormalizedPathway:
        """Normalize a single pathway entry."""
        pathway_name = pathway_data.get("pathway_name") or pathway_data.get("PathwayName") or pathway_data.get("Name") or ""
        pathway_id = pathway_data.get("pathway_id") or pathway_data.get("PathwayID") or pathway_data.get("ID") or ""
        source = pathway_data.get("source") or pathway_data.get("Source") or ""
        
        genes = pathway_data.get("genes") or pathway_data.get("Genes") or []
        if gene_symbol and gene_symbol not in genes:
            genes = [gene_symbol] + genes
        
        diseases = pathway_data.get("diseases") or pathway_data.get("Diseases") or []
        
        return NormalizedPathway(
            pathway_name=pathway_name,
            pathway_id=pathway_id,
            source=source,
            genes=genes,
            diseases=diseases,
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get normalization statistics."""
        return {
            "unique_diseases": len(self.seen_diseases),
            "unique_drugs": len(self.seen_drugs),
            "unique_genes": len(self.seen_genes),
        }