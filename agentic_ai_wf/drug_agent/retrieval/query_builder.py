"""
Query Builder Module (Dynamic)
==============================

Builds optimized queries for drug discovery retrieval.
Fully dynamic - no hardcoded disease, drug, or gene information.
"""

import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from ..models.data_models import DrugAgentInput

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """A search query with its configuration."""
    text: str
    query_type: str  # disease, gene, pathway, combined, mechanism
    weight: float = 1.0
    filters: Dict[str, any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class QueryBuilder:
    """
    Builds search queries for drug discovery retrieval.
    
    Fully dynamic - queries are built from input data without
    any hardcoded disease, drug, or gene information.
    
    Query types:
    1. Disease-focused: Find drugs for the disease
    2. Gene-focused: Find drugs targeting specific genes
    3. Pathway-focused: Find drugs targeting pathway components
    4. Mechanism-focused: Find drugs based on gene regulation
    """
    
    # Query templates (generic, not disease-specific)
    TEMPLATES = {
        "disease": "{disease} treatment therapy drug targeted therapeutic",
        "gene": "{genes} targeted therapy inhibitor drug treatment molecular",
        "pathway": "{pathway} pathway inhibitor drug therapy signaling",
        "combined": "{disease} {genes} treatment drug therapy targeted",
        "mechanism": "{gene} {direction} therapy drug treatment",
        "general": "{terms} drug therapy treatment inhibitor targeted",
    }
    
    def __init__(
        self,
        alias_resolver: Optional[Callable[[str], List[str]]] = None,
        enable_dynamic_expansion: bool = True,
    ):
        """
        Initialize query builder.
        
        Args:
            alias_resolver: Optional function to resolve aliases dynamically
                           (disease/gene/drug name) -> list of aliases
            enable_dynamic_expansion: Enable query expansion from knowledge base
        """
        self.alias_resolver = alias_resolver
        self.enable_dynamic_expansion = enable_dynamic_expansion
    
    def build_queries(
        self,
        input_data: DrugAgentInput,
        max_genes: int = 10,
        max_pathways: int = 5,
    ) -> List[Query]:
        """
        Build a set of queries from input data.
        
        Args:
            input_data: Drug agent input with disease, genes, pathways.
            max_genes: Maximum genes to include in query.
            max_pathways: Maximum pathways to query.
            
        Returns:
            List of Query objects.
        """
        queries = []
        
        disease = input_data.disease_name
        top_genes = input_data.get_top_genes(max_genes)
        top_pathways = input_data.get_top_pathways(max_pathways)
        gene_directions = input_data.get_gene_directions()
        
        # 1. Disease-focused query (weight: 0.35)
        disease_query = self._build_disease_query(disease)
        disease_query.weight = 0.35
        queries.append(disease_query)
        
        # 2. Gene-focused query (weight: 0.35)
        if top_genes:
            gene_query = self._build_gene_query(top_genes[:5], disease)
            gene_query.weight = 0.35
            queries.append(gene_query)
        
        # 3. Pathway-focused queries (weight: 0.2 total)
        if top_pathways:
            pathway_weight = 0.2 / min(len(top_pathways), 3)
            for pathway in top_pathways[:3]:
                pathway_query = self._build_pathway_query(pathway, disease)
                pathway_query.weight = pathway_weight
                queries.append(pathway_query)
        
        # 4. Gene-specific mechanism queries for high-priority targets
        upregulated = input_data.get_upregulated_genes()[:3]
        downregulated = input_data.get_downregulated_genes()[:3]
        
        for gene in upregulated:
            mech_query = self._build_mechanism_query(gene, "upregulated", disease)
            mech_query.weight = 0.1
            queries.append(mech_query)
        
        for gene in downregulated:
            mech_query = self._build_mechanism_query(gene, "downregulated", disease)
            mech_query.weight = 0.05
            queries.append(mech_query)
        
        logger.info(f"Built {len(queries)} queries for disease={disease}, "
                   f"genes={len(top_genes)}, pathways={len(top_pathways)}")
        
        return queries
    
    def _build_disease_query(self, disease: str) -> Query:
        """Build a disease-focused query."""
        text = self.TEMPLATES["disease"].format(disease=disease)
        
        # Add aliases if resolver is available
        if self.alias_resolver and self.enable_dynamic_expansion:
            aliases = self.alias_resolver(disease)
            if aliases:
                text += " " + " ".join(aliases[:3])
        
        return Query(
            text=text,
            query_type="disease",
            filters={},  # No filter - rely on semantic search
        )
    
    def _build_gene_query(self, genes: List[str], disease: str) -> Query:
        """Build a gene-focused query."""
        genes_str = " ".join(genes)
        text = self.TEMPLATES["gene"].format(genes=genes_str)
        text += f" {disease}"
        
        return Query(
            text=text,
            query_type="gene",
            filters={"gene_symbol": genes} if genes else {},
        )
    
    def _build_pathway_query(self, pathway: str, disease: str) -> Query:
        """Build a pathway-focused query."""
        # Clean pathway name for query
        pathway_clean = pathway.replace("_", " ").replace("-", " ")
        text = self.TEMPLATES["pathway"].format(pathway=pathway_clean)
        text += f" {disease}"
        
        return Query(
            text=text,
            query_type="pathway",
            filters={},
        )
    
    def _build_mechanism_query(self, gene: str, direction: str, disease: str) -> Query:
        """Build a mechanism-specific query."""
        # Build query based on regulation direction
        if direction == "upregulated":
            mechanism_terms = "inhibitor antagonist blocker suppressor"
        else:
            mechanism_terms = "activator agonist enhancer"
        
        text = f"{gene} {direction} {mechanism_terms} {disease} therapy drug treatment"
        
        return Query(
            text=text,
            query_type="mechanism",
            filters={"gene_symbol": gene},
        )
    
    def build_custom_query(
        self,
        query_text: str,
        query_type: str = "custom",
        weight: float = 1.0,
        filters: Optional[Dict] = None,
    ) -> Query:
        """Build a custom query."""
        return Query(
            text=query_text,
            query_type=query_type,
            weight=weight,
            filters=filters or {},
        )
    
    def build_simple_query(
        self,
        disease: str,
        genes: Optional[List[str]] = None,
    ) -> Query:
        """Build a simple combined query."""
        terms = [disease]
        if genes:
            terms.extend(genes[:5])
        
        text = self.TEMPLATES["general"].format(terms=" ".join(terms))
        
        return Query(
            text=text,
            query_type="simple",
            weight=1.0,
            filters={},
        )
    
    def expand_query(self, query: Query, expansions: List[str]) -> Query:
        """Expand query with additional terms."""
        if expansions:
            query.text = query.text + " " + " ".join(expansions)
        return query
    
    def set_alias_resolver(self, resolver: Callable[[str], List[str]]):
        """Set the alias resolver function."""
        self.alias_resolver = resolver
