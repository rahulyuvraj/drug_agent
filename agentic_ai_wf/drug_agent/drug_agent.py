"""
Drug Discovery Agent
====================

A disease-agnostic, RAG-powered drug recommendation system.
Designed for integration with the Reporting Pipeline Agent.
"""

import logging
import time
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .config.settings import Settings, load_settings, get_settings
from .models.data_models import (
    DrugAgentInput, DrugAgentOutput, DrugRecommendation,
    GeneMapping, PathwayMapping,
    IngestionResult, KnowledgeBaseStats,
)
from .ingestion.json_parser import JSONParser
from .ingestion.data_normalizer import DataNormalizer
from .ingestion.document_generator import DocumentGenerator
from .embedding.embedder import PubMedBERTEmbedder
from .storage.qdrant_client import QdrantStorage
from .retrieval.query_builder import QueryBuilder
from .retrieval.hybrid_search import HybridSearcher
from .recommendation.drug_ranker import DrugRanker
from .recommendation.evidence_compiler import EvidenceCompiler
from .recommendation.report_generator import ReportSectionGenerator
from .utils.disease_mapper import DiseaseMapper
from .utils.gene_resolver import GeneResolver

logger = logging.getLogger(__name__)


# =============================================================================
# Inter-Agent Communication Models
# =============================================================================

@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication."""
    message_id: str
    source_agent: str
    target_agent: str
    message_type: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(**data)


@dataclass
class AgentResponse:
    """Standard response format from the agent."""
    success: bool
    message_id: str
    correlation_id: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Main Agent Class
# =============================================================================

class DrugDiscoveryAgent:
    """
    Drug Discovery Agent - Main orchestration class.
    """
    
    AGENT_ID = "drug_agent"
    VERSION = "1.0.0"
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        settings: Optional[Settings] = None,
        auto_connect: bool = True,
    ):
        if settings:
            self.settings = settings
        elif config_path:
            self.settings = load_settings(config_path)
        else:
            self.settings = get_settings()
        
        errors = self.settings.validate()
        if errors:
            for error in errors:
                logger.warning(f"Configuration warning: {error}")
        
        self._storage: Optional[QdrantStorage] = None
        self._embedder: Optional[PubMedBERTEmbedder] = None
        self._searcher: Optional[HybridSearcher] = None
        self._ranker: Optional[DrugRanker] = None
        self._compiler: Optional[EvidenceCompiler] = None
        self._report_generator: Optional[ReportSectionGenerator] = None
        self._query_builder: Optional[QueryBuilder] = None
        self._disease_mapper: Optional[DiseaseMapper] = None
        self._gene_resolver: Optional[GeneResolver] = None
        self._normalizer: Optional[DataNormalizer] = None
        
        self._message_handlers: Dict[str, Callable] = {}
        self._register_handlers()
        
        if auto_connect:
            try:
                self._init_storage()
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant Cloud: {e}")
        
        logger.info(f"DrugDiscoveryAgent v{self.VERSION} initialized")
    
    # =========================================================================
    # Properties (Lazy Loading)
    # =========================================================================
    
    @property
    def storage(self) -> QdrantStorage:
        if self._storage is None:
            self._init_storage()
        return self._storage
    
    @property
    def embedder(self) -> PubMedBERTEmbedder:
        if self._embedder is None:
            self._embedder = PubMedBERTEmbedder(
                model_name=self.settings.embedding.model_name,
                device=self.settings.embedding.get_device(),
                cache_enabled=self.settings.embedding.cache_enabled,
                cache_directory=self.settings.embedding.cache_directory,
                batch_size=self.settings.embedding.batch_size,
            )
        return self._embedder
    
    @property
    def searcher(self) -> HybridSearcher:
        if self._searcher is None:
            self._searcher = HybridSearcher(
                storage=self.storage,
                embedder=self.embedder,
                rrf_k=self.settings.retrieval.rrf_k,
                default_top_k=self.settings.retrieval.default_top_k,
            )
        return self._searcher
    
    @property
    def ranker(self) -> DrugRanker:
        if self._ranker is None:
            self._ranker = DrugRanker(config=self.settings.ranking)
        return self._ranker
    
    @property
    def compiler(self) -> EvidenceCompiler:
        if self._compiler is None:
            self._compiler = EvidenceCompiler()
        return self._compiler
    
    @property
    def report_generator(self) -> ReportSectionGenerator:
        if self._report_generator is None:
            self._report_generator = ReportSectionGenerator()
        return self._report_generator
    
    @property
    def query_builder(self) -> QueryBuilder:
        if self._query_builder is None:
            self._query_builder = QueryBuilder(
                alias_resolver=self._resolve_aliases,
                enable_dynamic_expansion=self.settings.retrieval.dynamic_expansion,
            )
        return self._query_builder
    
    @property
    def disease_mapper(self) -> DiseaseMapper:
        if self._disease_mapper is None:
            self._disease_mapper = DiseaseMapper()
        return self._disease_mapper
    
    @property
    def gene_resolver(self) -> GeneResolver:
        if self._gene_resolver is None:
            self._gene_resolver = GeneResolver()
        return self._gene_resolver
    
    @property
    def normalizer(self) -> DataNormalizer:
        if self._normalizer is None:
            self._normalizer = DataNormalizer()
        return self._normalizer
    
    def _init_storage(self):
        """Initialize Qdrant Cloud storage."""
        self._storage = QdrantStorage(
            url=self.settings.qdrant.url,
            port=self.settings.qdrant.port,
            api_key=self.settings.qdrant.api_key,
            username=self.settings.qdrant.username,
            password=self.settings.qdrant.password,
            collection_name=self.settings.qdrant.collection_name,
            timeout=self.settings.qdrant.timeout_seconds,
            use_https=self.settings.qdrant.use_https,
            prefer_grpc=self.settings.qdrant.prefer_grpc,
        )
    
    def _resolve_aliases(self, term: str) -> List[str]:
        """Resolve aliases for query expansion."""
        aliases = []
        aliases.extend(self.disease_mapper.get_aliases(term))
        aliases.extend(self.gene_resolver.get_aliases(term))
        return aliases[:5]
    
    # =========================================================================
    # Ingestion Methods
    # =========================================================================
    
    def ingest_gene_data(
        self,
        json_directory: str,
        recreate_collection: bool = False,
        batch_size: int = None,
        max_files: int = None,
        progress_callback: Optional[Callable] = None,
    ) -> IngestionResult:
        """
        Ingest gene JSON files into the knowledge base.
        
        Args:
            json_directory: Path to directory with gene JSON files.
            recreate_collection: Delete and recreate collection.
            batch_size: Override default batch size.
            max_files: Maximum files to process (for testing).
            progress_callback: Callback(stage, current, total).
            
        Returns:
            IngestionResult with statistics.
        """
        start_time = time.time()
        batch_size = batch_size or self.settings.ingestion.batch_size
        max_files = max_files or 500  # Default limit for testing
        
        logger.info(f"Starting ingestion from: {json_directory}")
        
        try:
            # Create collection
            self.storage.create_collection(recreate=recreate_collection)
            
            # Initialize components
            # FIX: JSONParser doesn't take directory in __init__, only skip_empty
            parser = JSONParser(skip_empty=self.settings.ingestion.skip_empty_files)
            doc_generator = DocumentGenerator(self.normalizer)
            
            # Parse and generate documents
            all_documents: List[Dict[str, Any]] = []
            files_processed = 0
            
            print(f"Parsing files from: {json_directory}")
            
            # FIX: Pass directory to parse_directory(), not __init__
            for gene_data in parser.parse_directory(json_directory):
                files_processed += 1
                
                # Learn aliases from data
                self.gene_resolver.learn_from_data(
                    gene_data.gene_symbol,
                    gene_data.gene_aliases
                )
                
                # Generate documents
                for doc in doc_generator.generate_documents(gene_data):
                    all_documents.append(doc)
                
                if files_processed % 100 == 0:
                    print(f"  Processed {files_processed} files, {len(all_documents)} documents...")
                
                if progress_callback and files_processed % 100 == 0:
                    progress_callback("parsing", files_processed, 0)
                
                # Stop after max_files
                if files_processed >= max_files:
                    print(f"  Stopping at {max_files} files (limit reached)")
                    break
            
            print(f"Total: {files_processed} files, {len(all_documents)} documents")
            
            if not all_documents:
                return IngestionResult(
                    success=True,
                    total_files_processed=files_processed,
                    total_documents_created=0,
                    warnings=["No documents generated"],
                    duration_seconds=time.time() - start_time,
                )
            
            # Generate embeddings
            print(f"Generating embeddings for {len(all_documents)} documents...")
            
            if progress_callback:
                progress_callback("embedding", 0, len(all_documents))
            
            texts = [doc["text_content"] for doc in all_documents]
            embedding_result = self.embedder.embed_texts(texts, show_progress=True)
            
            print(f"Generated {len(embedding_result.embeddings)} embeddings")
            
            for doc, embedding in zip(all_documents, embedding_result.embeddings):
                doc["embedding"] = embedding
            
            # Upload to Qdrant Cloud
            print(f"Uploading {len(all_documents)} documents to Qdrant Cloud...")
            
            if progress_callback:
                progress_callback("uploading", 0, len(all_documents))
            
            upserted = self.storage.upsert_documents(all_documents, batch_size=batch_size)
            
            print(f"Uploaded {upserted} documents successfully!")
            
            # Get stats
            doc_stats = doc_generator.get_statistics()
            
            return IngestionResult(
                success=True,
                total_files_processed=files_processed,
                total_documents_created=upserted,
                documents_by_type={
                    "gene_drug": doc_stats.get("gene_drug_docs", 0),
                    "disease_drug": doc_stats.get("disease_drug_docs", 0),
                    "pathway_drug": doc_stats.get("pathway_drug_docs", 0),
                    "gene_disease_context": doc_stats.get("context_docs", 0),
                },
                duration_seconds=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return IngestionResult(
                success=False,
                errors=[str(e)],
                duration_seconds=time.time() - start_time,
            )
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def generate_recommendations(
        self,
        input_data: DrugAgentInput,
        max_recommendations: int = None,
    ) -> DrugAgentOutput:
        """Generate drug recommendations for the given input."""
        max_recommendations = max_recommendations or self.settings.output.max_recommendations
        
        normalized_disease = self.disease_mapper.normalize(input_data.disease_name)
        input_data.disease_name = normalized_disease
        
        logger.info(f"Generating recommendations for: {normalized_disease}")
        
        queries = self.query_builder.build_queries(input_data)
        
        fused_results = self.searcher.search(
            queries=queries,
            top_k=self.settings.retrieval.default_top_k,
            min_score=self.settings.retrieval.min_relevance_score,
        )
        
        logger.info(f"Retrieved {len(fused_results)} results")
        
        recommendations = self.ranker.rank_results(
            results=fused_results,
            input_data=input_data,
            max_results=max_recommendations,
        )
        
        output = self.compiler.compile_evidence(recommendations, input_data)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return output
    
    def query_drugs_for_disease(
        self,
        disease_name: str,
        top_genes: List[str],
        top_pathways: List[str] = None,
        gene_directions: Dict[str, str] = None,
    ) -> DrugAgentOutput:
        """Simplified query interface."""
        gene_mappings = []
        for i, gene in enumerate(top_genes):
            direction = gene_directions.get(gene, "up") if gene_directions else "up"
            mapping = GeneMapping(
                gene=gene,
                observed_direction=direction,
                composite_score=len(top_genes) - i,
            )
            gene_mappings.append(mapping)
        
        pathway_mappings = []
        if top_pathways:
            for i, pathway in enumerate(top_pathways):
                mapping = PathwayMapping(
                    pathway_name=pathway,
                    p_value=0.001 * (i + 1),
                )
                pathway_mappings.append(mapping)
        
        input_data = DrugAgentInput(
            disease_name=disease_name,
            gene_mappings=gene_mappings,
            pathway_mappings=pathway_mappings,
        )
        
        return self.generate_recommendations(input_data)
    
    # =========================================================================
    # Report Generation
    # =========================================================================
    
    def generate_report_section(
        self,
        output: DrugAgentOutput,
        section_number: str = "X",
    ) -> str:
        """Generate formatted report section."""
        self.report_generator.section_number = section_number
        return self.report_generator.generate_full_section(output)
    
    def generate_report_data(self, output: DrugAgentOutput) -> Dict[str, Any]:
        """Generate structured data for DOCX integration."""
        return self.report_generator.generate_for_docx(output)
    
    # =========================================================================
    # Inter-Agent Communication
    # =========================================================================
    
    def _register_handlers(self):
        """Register message handlers."""
        self._message_handlers = {
            "get_recommendations": self._handle_get_recommendations,
            "ingest_data": self._handle_ingest_data,
            "health_check": self._handle_health_check,
            "get_stats": self._handle_get_stats,
            "query_gene": self._handle_query_gene,
            "query_disease": self._handle_query_disease,
        }
    
    def handle_message(self, message: AgentMessage) -> AgentResponse:
        """Handle incoming message from another agent."""
        logger.info(f"Received message: {message.action} from {message.source_agent}")
        
        handler = self._message_handlers.get(message.action)
        
        if not handler:
            return AgentResponse(
                success=False,
                message_id=message.message_id,
                correlation_id=message.correlation_id,
                error=f"Unknown action: {message.action}",
            )
        
        try:
            result = handler(message.payload)
            return AgentResponse(
                success=True,
                message_id=message.message_id,
                correlation_id=message.correlation_id,
                data=result,
            )
        except Exception as e:
            logger.error(f"Handler error: {e}")
            return AgentResponse(
                success=False,
                message_id=message.message_id,
                correlation_id=message.correlation_id,
                error=str(e),
            )
    
    def _handle_get_recommendations(self, payload: Dict) -> Dict:
        input_data = DrugAgentInput.from_pipeline_data(
            disease_name=payload.get("disease_name", ""),
            gene_mappings=payload.get("gene_mappings", []),
            pathway_mappings=payload.get("pathway_mappings", []),
            xcell_findings=payload.get("xcell_findings"),
        )
        output = self.generate_recommendations(input_data)
        return output.to_dict()
    
    def _handle_ingest_data(self, payload: Dict) -> Dict:
        result = self.ingest_gene_data(
            json_directory=payload.get("json_directory", ""),
            recreate_collection=payload.get("recreate", False),
        )
        return {
            "success": result.success,
            "files_processed": result.total_files_processed,
            "documents_created": result.total_documents_created,
        }
    
    def _handle_health_check(self, payload: Dict) -> Dict:
        return self.health_check()
    
    def _handle_get_stats(self, payload: Dict) -> Dict:
        stats = self.get_knowledge_base_stats()
        return {
            "total_documents": stats.total_documents,
            "documents_by_type": stats.documents_by_type,
            "status": stats.collection_status,
        }
    
    def _handle_query_gene(self, payload: Dict) -> Dict:
        gene = payload.get("gene", "")
        results = self.storage.get_by_gene(gene)
        return {
            "gene": gene,
            "results_count": len(results),
            "drugs": list(set(r.drug_name for r in results if r.drug_name)),
        }
    
    def _handle_query_disease(self, payload: Dict) -> Dict:
        disease = payload.get("disease", "")
        results = self.storage.get_by_disease(disease)
        return {
            "disease": disease,
            "results_count": len(results),
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_knowledge_base_stats(self) -> KnowledgeBaseStats:
        return self.storage.get_stats()
    
    def health_check(self) -> Dict[str, Any]:
        status = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "qdrant_connected": False,
            "collection_exists": False,
            "embedder_loaded": False,
            "documents_count": 0,
        }
        
        try:
            status["qdrant_connected"] = self.storage.health_check()
            
            if status["qdrant_connected"]:
                info = self.storage.get_collection_info()
                status["collection_exists"] = info.get("status") != "error"
                status["documents_count"] = info.get("points_count", 0)
            
            _ = self.embedder.get_dimension()
            status["embedder_loaded"] = True
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "capabilities": [
                "drug_recommendations",
                "gene_drug_mapping",
                "pathway_drug_mapping",
                "disease_drug_mapping",
                "knowledge_base_search",
            ],
            "supported_actions": list(self._message_handlers.keys()),
        }
    
    def close(self):
        logger.info("Closing DrugDiscoveryAgent")


# =============================================================================
# Factory Functions
# =============================================================================

def create_agent(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = "drug_knowledge_base",
    embedding_model: str = "NeuML/pubmedbert-base-embeddings",
    device: str = "auto",
) -> DrugDiscoveryAgent:
    settings = Settings()
    settings.qdrant.url = qdrant_url
    settings.qdrant.api_key = qdrant_api_key
    settings.qdrant.collection_name = collection_name
    settings.embedding.model_name = embedding_model
    settings.embedding.device = device
    
    return DrugDiscoveryAgent(settings=settings)


def create_agent_from_env() -> DrugDiscoveryAgent:
    return DrugDiscoveryAgent()
