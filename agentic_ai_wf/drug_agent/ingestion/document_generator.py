"""
Document Generator Module - Generates vectorizable documents from normalized gene data.
"""

import logging
from typing import Dict, List, Optional, Any, Generator

from .json_parser import ParsedGeneData
from .data_normalizer import DataNormalizer
from ..models.data_models import DocumentType, VectorDocument

def generate_doc_id(*parts) -> str:
    """Generate a unique document ID from parts."""
    import hashlib
    combined = "_".join(str(p) for p in parts if p)
    return hashlib.md5(combined.encode()).hexdigest()[:16]

def truncate_text(text: str, max_length: int = 512) -> str:
    """Truncate text to max length."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length].rsplit(" ", 1)[0] + "..."
logger = logging.getLogger(__name__)


class DocumentGenerator:
    """Generates vectorizable documents from gene data."""
    
    MAX_TEXT_LENGTH = 512
    
    def __init__(self, normalizer: Optional[DataNormalizer] = None):
        self.normalizer = normalizer or DataNormalizer()
        self.stats = {"gene_drug_docs": 0, "disease_drug_docs": 0, "pathway_drug_docs": 0, "context_docs": 0}
    
    def generate_documents(self, parsed_data: ParsedGeneData) -> Generator[Dict[str, Any], None, None]:
        """Generate all document types from parsed gene data."""
        normalized = self.normalizer.normalize_gene_data(parsed_data)
        gene_symbol = normalized["gene_symbol"]
        gene_aliases = normalized["gene_aliases"]
        
        for drug in normalized["drugs"]:
            doc = self._create_gene_drug_document(gene_symbol, gene_aliases, drug, parsed_data.file_path)
            if doc:
                self.stats["gene_drug_docs"] += 1
                yield doc
        
        for disease in normalized["diseases"]:
            doc = self._create_disease_drug_document(disease, gene_symbol, normalized["drugs"], parsed_data.file_path)
            if doc:
                self.stats["disease_drug_docs"] += 1
                yield doc
        
        for pathway in normalized["pathways"]:
            doc = self._create_pathway_drug_document(pathway, gene_symbol, normalized["drugs"], parsed_data.file_path)
            if doc:
                self.stats["pathway_drug_docs"] += 1
                yield doc
    
    def _create_gene_drug_document(self, gene_symbol, gene_aliases, drug, source_file):
        if not drug.drug_name:
            return None
        doc_id = generate_doc_id("GD", gene_symbol, drug.drug_name)
        text_parts = [f"{gene_symbol} is targeted by {drug.drug_name}."]
        if drug.drug_type:
            text_parts.append(f"{drug.drug_name} is a {drug.drug_type}.")
        if drug.mechanism:
            text_parts.append(f"Mechanism: {drug.mechanism}.")
        if drug.indications:
            text_parts.append(f"Indicated for: {', '.join(drug.indications[:5])}.")
        text_content = truncate_text(" ".join(text_parts), self.MAX_TEXT_LENGTH)
        return {
            "doc_id": doc_id, "doc_type": DocumentType.GENE_DRUG.value,
            "text_content": text_content, "source_file": source_file,
            "gene_symbol": gene_symbol, "gene_aliases": gene_aliases,
            "drug_name": drug.drug_name, "drug_aliases": drug.drug_aliases,
            "drug_type": drug.drug_type, "mechanism_of_action": drug.mechanism,
            "approval_status": drug.approval_status, "indication_diseases": drug.indications,
            "evidence_level": drug.evidence_level, "data_source": ", ".join(drug.sources),
        }
    
    def _create_disease_drug_document(self, disease, gene_symbol, drugs, source_file):
        if not disease.canonical_name:
            return None
        relevant_drugs = [{"drug_name": d.drug_name, "target_gene": gene_symbol} for d in drugs if d.drug_name]
        doc_id = generate_doc_id("DD", disease.canonical_name, gene_symbol)
        text_parts = [f"Treatment options for {disease.canonical_name}.", f"Associated gene: {gene_symbol}."]
        if relevant_drugs:
            text_parts.append(f"Drugs: {', '.join([d['drug_name'] for d in relevant_drugs[:5]])}.")
        text_content = truncate_text(" ".join(text_parts), self.MAX_TEXT_LENGTH)
        return {
            "doc_id": doc_id, "doc_type": DocumentType.DISEASE_DRUG.value,
            "text_content": text_content, "source_file": source_file,
            "disease_name": disease.canonical_name, "disease_aliases": disease.disease_aliases,
            "disease_category": disease.category, "associated_genes": disease.associated_genes,
            "approved_drugs": relevant_drugs, "data_source": ", ".join(disease.sources),
        }
    
    def _create_pathway_drug_document(self, pathway, gene_symbol, drugs, source_file):
        if not pathway.pathway_name:
            return None
        targeting_drugs = [{"drug_name": d.drug_name, "target_gene": gene_symbol} for d in drugs if d.drug_name]
        doc_id = generate_doc_id("PD", pathway.pathway_name, pathway.pathway_id)
        text_parts = [f"The {pathway.pathway_name} pathway includes gene {gene_symbol}."]
        if targeting_drugs:
            text_parts.append(f"Drugs targeting this pathway: {', '.join([d['drug_name'] for d in targeting_drugs[:5]])}.")
        text_content = truncate_text(" ".join(text_parts), self.MAX_TEXT_LENGTH)
        return {
            "doc_id": doc_id, "doc_type": DocumentType.PATHWAY_DRUG.value,
            "text_content": text_content, "source_file": source_file,
            "pathway_name": pathway.pathway_name, "pathway_id": pathway.pathway_id,
            "pathway_source": pathway.source, "pathway_genes": pathway.genes,
            "targeting_drugs": targeting_drugs, "disease_relevance": pathway.diseases,
        }
    
    def get_statistics(self) -> Dict[str, int]:
        return dict(self.stats)
