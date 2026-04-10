"""
Hybrid Search Module
====================

Performs hybrid search combining multiple queries with result fusion.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..storage.qdrant_client import QdrantStorage, SearchResult
from ..embedding.embedder import PubMedBERTEmbedder
from .query_builder import Query, QueryBuilder

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Result after fusion from multiple queries."""
    doc_id: str
    drug_name: str
    score: float  # Fused score
    payload: Dict
    query_scores: Dict[str, float] = field(default_factory=dict)  # Scores from each query
    rank_contributions: Dict[str, int] = field(default_factory=dict)  # Ranks from each query

    @property
    def gene_symbol(self) -> Optional[str]:
        return self.payload.get("gene_symbol")

    @property
    def disease_name(self) -> Optional[str]:
        return self.payload.get("disease_name")

    @property
    def pathway_name(self) -> Optional[str]:
        return self.payload.get("pathway_name")

    @property
    def evidence_level(self) -> str:
        return self.payload.get("evidence_level", "")

    @property
    def approval_status(self) -> str:
        return self.payload.get("approval_status", "")

    @property
    def mechanism_of_action(self) -> str:
        return self.payload.get("mechanism_of_action", "")

    @property
    def text_content(self) -> str:
        return self.payload.get("text_content", "")


def _expand_gene_doc_to_drugs(result: SearchResult) -> List[Tuple[str, Dict]]:
    """
    Expand a gene-centric document into individual drug entries.

    Gene-centric docs have payload like:
        {"gene_symbol": "TNF", "drugs": [{"name": "Etanercept", "type": "..."}, ...],
         "diseases": [...], "pathways": [...]}

    Returns list of (drug_name, enriched_payload) tuples.
    """
    payload = result.payload or {}
    drugs = payload.get("drugs", [])
    if not drugs:
        return []

    gene_symbol = payload.get("gene_symbol", "")
    diseases = payload.get("diseases", [])
    pathways = payload.get("pathways", [])

    # Build disease/indication list from diseases array
    disease_names = []
    for d in diseases:
        if isinstance(d, dict):
            disease_names.append(d.get("name", ""))
        elif isinstance(d, str):
            disease_names.append(d)
    disease_names = [d for d in disease_names if d]

    # Build pathway list
    pathway_names = []
    for p in pathways:
        if isinstance(p, dict):
            pathway_names.append(p.get("name", ""))
        elif isinstance(p, str):
            pathway_names.append(p)
    pathway_names = [p for p in pathway_names if p]

    entries = []
    for drug_info in drugs:
        if isinstance(drug_info, dict):
            drug_name = drug_info.get("name", "")
            drug_type = drug_info.get("type", "")
        elif isinstance(drug_info, str):
            drug_name = drug_info
            drug_type = ""
        else:
            continue

        if not drug_name:
            continue

        # Create an enriched payload for this specific drug
        enriched = {
            "drug_name": drug_name,
            "drug_type": drug_type,
            "gene_symbol": gene_symbol,
            "gene_aliases": payload.get("gene_aliases", []),
            "indications": disease_names,
            "pathway_names": pathway_names,
            "pathway_genes": [],
            "text_content": payload.get("text_content", ""),
            "source": payload.get("source", "GeneALaCart"),
            "doc_type": payload.get("doc_type", "gene_data"),
        }
        entries.append((drug_name, enriched))

    return entries


class HybridSearcher:
    """
    Performs hybrid search with result fusion.

    Executes multiple queries in parallel and fuses results using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        storage: QdrantStorage,
        embedder: PubMedBERTEmbedder,
        rrf_k: int = 60,
        default_top_k: int = 50,
    ):
        self.storage = storage
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.default_top_k = default_top_k

    def search(
        self,
        queries: List[Query],
        top_k: int = None,
        min_score: float = 0.3,
    ) -> List[FusedResult]:
        """
        Execute hybrid search with multiple queries.

        Handles both drug-centric documents (with top-level drug_name)
        and gene-centric documents (with drugs array) by expanding the
        latter into per-drug entries before fusion.
        """
        top_k = top_k or self.default_top_k

        # Generate embeddings for all queries
        query_texts = [q.text for q in queries]
        embeddings_result = self.embedder.embed_texts(query_texts, show_progress=False)

        for query, embedding in zip(queries, embeddings_result.embeddings):
            query.embedding = embedding

        # Execute searches and collect per-drug results
        # Key = drug_name (lowercased), Value = list of (query, rank, payload, score)
        all_results: Dict[str, List[Tuple[Query, int, Dict, float]]] = defaultdict(list)

        for query in queries:
            results = self.storage.search(
                query_vector=query.embedding,
                top_k=self.default_top_k,
                filter_conditions=query.filters if query.filters else None,
                score_threshold=min_score,
            )

            for rank, result in enumerate(results, 1):
                payload = result.payload or {}

                # Check if this is a drug-centric doc (has drug_name)
                if payload.get("drug_name"):
                    key = payload["drug_name"].lower()
                    all_results[key].append((query, rank, payload, result.score))

                # Gene-centric doc: expand drugs array into individual entries
                elif payload.get("drugs"):
                    drug_entries = _expand_gene_doc_to_drugs(result)
                    for drug_name, enriched_payload in drug_entries:
                        key = drug_name.lower()
                        all_results[key].append((query, rank, enriched_payload, result.score))

                # Fallback: use doc_id as key
                else:
                    key = result.doc_id
                    all_results[key].append((query, rank, payload, result.score))

        # Fuse results using RRF
        fused_scores: Dict[str, float] = defaultdict(float)
        all_payloads: Dict[str, List[Tuple[Dict, float]]] = defaultdict(list)
        query_details: Dict[str, Dict[str, any]] = defaultdict(lambda: {"scores": {}, "ranks": {}})

        for key, query_results in all_results.items():
            for query, rank, payload, score in query_results:
                # RRF formula: 1 / (k + rank)
                rrf_score = query.weight / (self.rrf_k + rank)
                fused_scores[key] += rrf_score

                # Track query-level details
                query_details[key]["scores"][query.query_type] = score
                query_details[key]["ranks"][query.query_type] = rank

                # Collect all payloads for merging
                all_payloads[key].append((payload, score))

        # Create fused results with merged payloads
        fused_results = []
        for key in fused_scores:
            if key not in all_payloads:
                continue

            payloads_with_scores = all_payloads[key]

            # Start with the highest-scoring payload as the base
            payloads_with_scores.sort(key=lambda x: x[1], reverse=True)
            merged_payload = dict(payloads_with_scores[0][0])

            # Merge gene_symbol and indications from all payloads
            all_gene_symbols = set()
            all_indications = set()
            for p, _ in payloads_with_scores:
                gs = p.get("gene_symbol")
                if gs:
                    all_gene_symbols.add(gs)
                for ind in p.get("indications", []):
                    if ind:
                        all_indications.add(ind)

            # Store all target genes so ranker can match against patient genes
            merged_payload["all_target_genes"] = list(all_gene_symbols)
            if all_indications:
                merged_payload["indications"] = list(all_indications)

            drug_name = merged_payload.get("drug_name", key)
            fused = FusedResult(
                doc_id=key,
                drug_name=drug_name,
                score=fused_scores[key],
                payload=merged_payload,
                query_scores=query_details[key]["scores"],
                rank_contributions=query_details[key]["ranks"],
            )
            fused_results.append(fused)

        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        # Deduplicate by drug name
        seen_drugs: Set[str] = set()
        deduplicated = []
        for result in fused_results:
            drug_key = result.drug_name.lower() if result.drug_name else result.doc_id
            if drug_key not in seen_drugs:
                seen_drugs.add(drug_key)
                deduplicated.append(result)

        logger.info(f"Hybrid search: {len(queries)} queries -> {len(all_results)} unique drugs -> {len(deduplicated)} deduplicated")

        return deduplicated[:top_k]
    
    def search_single(
        self,
        query_text: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Execute a single query search.
        
        Args:
            query_text: Query text.
            top_k: Number of results.
            filters: Optional filter conditions.
            
        Returns:
            List of SearchResult objects.
        """
        embedding = self.embedder.embed_text(query_text)
        
        return self.storage.search(
            query_vector=embedding,
            top_k=top_k,
            filter_conditions=filters,
        )
    
    def search_by_gene(
        self,
        gene_symbol: str,
        disease: Optional[str] = None,
        top_k: int = 20,
    ) -> List[SearchResult]:
        """Search for drugs targeting a specific gene."""
        query_text = f"{gene_symbol} drug therapy inhibitor treatment"
        if disease:
            query_text += f" {disease}"
        
        filters = {"gene_symbol": gene_symbol}
        if disease:
            filters["disease_name"] = disease
        
        return self.search_single(query_text, top_k, filters)
    
    def search_by_pathway(
        self,
        pathway_name: str,
        disease: Optional[str] = None,
        top_k: int = 20,
    ) -> List[SearchResult]:
        """Search for drugs targeting a pathway."""
        query_text = f"{pathway_name} pathway inhibitor drug therapy"
        if disease:
            query_text += f" {disease}"
        
        return self.search_single(query_text, top_k)
