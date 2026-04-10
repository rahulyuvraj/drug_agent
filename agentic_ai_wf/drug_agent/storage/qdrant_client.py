"""
Qdrant Cloud Storage Module
===========================

Handles all interactions with Qdrant Cloud vector database.
Fully dynamic - no hardcoded data.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

from ..models.data_models import DocumentType, KnowledgeBaseStats

logger = logging.getLogger(__name__)


class SearchResult:
    """Result from a vector search."""
    
    def __init__(self, doc_id: str, score: float, payload: Dict[str, Any]):
        self.doc_id = doc_id
        self.score = score
        self.payload = payload
    
    @property
    def text_content(self) -> str:
        return self.payload.get("text_content", "")
    
    @property
    def doc_type(self) -> str:
        return self.payload.get("doc_type", "")
    
    @property
    def gene_symbol(self) -> Optional[str]:
        return self.payload.get("gene_symbol")
    
    @property
    def drug_name(self) -> Optional[str]:
        return self.payload.get("drug_name")
    
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


class QdrantStorage:
    """
    Qdrant Cloud vector database storage handler.
    Supports both cloud, local, and self-hosted (with Basic Auth) deployments.
    """
    
    VECTOR_SIZE = 768  # PubMedBERT dimension
    
    def __init__(
        self,
        url: Optional[str] = None,
        port: int = 6333,
        api_key: Optional[str] = None,
        username: Optional[str] = None,  # HTTP Basic Auth
        password: Optional[str] = None,  # HTTP Basic Auth
        collection_name: str = "drug_knowledge_base",
        timeout: int = 60,
        use_https: bool = True,
        prefer_grpc: bool = True,
    ):
        self.url = url
        self.port = port
        self.api_key = api_key
        self.username = username
        self.password = password
        self.collection_name = collection_name
        self.timeout = timeout
        self.use_https = use_https
        self.prefer_grpc = prefer_grpc
        
        self._client = None
        logger.info(f"QdrantStorage initialized for collection: {collection_name}")
    
    @property
    def client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            self._connect()
        return self._client
    
    def _connect(self):
        """Connect to Qdrant (Cloud, self-hosted with Basic Auth, or local)."""
        try:
            from qdrant_client import QdrantClient
            import httpx
            
            # Option 1: Self-hosted with HTTP Basic Auth (nginx proxy)
            if self.username and self.password:
                if self.url and not self.url.startswith("http"):
                    url = f"https://{self.url}"
                else:
                    url = self.url
                
                # Use port 443 for HTTPS behind nginx
                port = 443 if url and url.startswith("https://") else self.port
                
                logger.info(f"Connecting to self-hosted Qdrant with Basic Auth: {url}:{port}")
                self._client = QdrantClient(
                    url=url,
                    port=port,
                    timeout=self.timeout,
                    prefer_grpc=False,  # gRPC doesn't support Basic Auth
                    https=True,
                    check_compatibility=False,
                )
                
                # Patch the internal httpx client with Basic Auth
                auth = httpx.BasicAuth(self.username, self.password)
                custom_http_client = httpx.Client(auth=auth, timeout=float(self.timeout))
                
                # Patch ALL api endpoints
                http_apis = self._client._client.http
                api_names = [
                    'collections_api', 'points_api', 'service_api', 
                    'snapshots_api', 'indexes_api', 'search_api',
                    'aliases_api', 'distributed_api', 'beta_api'
                ]
                for api_name in api_names:
                    api = getattr(http_apis, api_name, None)
                    if api and hasattr(api, 'api_client'):
                        api.api_client._client = custom_http_client
                        
            # Option 2: Qdrant Cloud with API key
            elif self.api_key:
                if self.url and not self.url.startswith("http"):
                    url = f"https://{self.url}"
                else:
                    url = self.url
                
                logger.info(f"Connecting to Qdrant Cloud: {url}")
                self._client = QdrantClient(
                    url=url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    prefer_grpc=self.prefer_grpc,
                )
                
            # Option 3: Local Qdrant (no auth)
            else:
                logger.info(f"Connecting to local Qdrant: {self.url}:{self.port}")
                self._client = QdrantClient(
                    host=self.url or "localhost",
                    port=self.port,
                    timeout=self.timeout,
                )
            
            logger.info("Connected to Qdrant successfully")
            
        except ImportError:
            logger.error("qdrant-client not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self, recreate: bool = False) -> bool:
        """Create the collection with proper configuration."""
        try:
            from qdrant_client.http import models
            
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                exists = False
            
            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                    ),
                )
                self._create_indexes()
                logger.info(f"Collection {self.collection_name} created")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def _create_indexes(self):
        """Create payload indexes for filtering."""
        try:
            from qdrant_client.http import models
            
            index_fields = ["doc_type", "gene_symbol", "drug_name", "disease_name", "pathway_name"]
            
            for field in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    logger.debug(f"Index {field} may already exist: {e}")
            
            logger.info("Payload indexes created")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def upsert_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Upsert documents to the collection."""
        try:
            from qdrant_client.http import models
            
            total_upserted = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                points = []
                for doc in batch:
                    embedding = doc.get("embedding")
                    if embedding is None:
                        continue
                    
                    doc_id = doc.get("doc_id", str(uuid.uuid4()))
                    
                    # Create payload without embedding
                    payload = {k: v for k, v in doc.items() if k != "embedding"}
                    
                    point = models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)),
                        vector=embedding,
                        payload=payload,
                    )
                    points.append(point)  # THIS WAS THE MISSING LINE!
                
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    total_upserted += len(points)
                
                if (i + batch_size) % 1000 == 0:
                    logger.info(f"Upserted {total_upserted} documents...")
            
            logger.info(f"Total documents upserted: {total_upserted}")
            return total_upserted
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 50,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        try:
            from qdrant_client.http import models

            query_filter = None
            if filter_conditions:
                must_conditions = []

                for field, value in filter_conditions.items():
                    if value is None:
                        continue

                    if isinstance(value, list):
                        must_conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchAny(any=value),
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchValue(value=value),
                            )
                        )

                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)

            # Use query_points (newer API) with fallback to search (older API)
            if hasattr(self.client, 'query_points'):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
                return [
                    SearchResult(
                        doc_id=str(r.id),
                        score=r.score,
                        payload=r.payload or {},
                    )
                    for r in response.points
                ]
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                )
                return [
                    SearchResult(
                        doc_id=str(r.id),
                        score=r.score,
                        payload=r.payload or {},
                    )
                    for r in results
                ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text(
        self,
        query_vector: List[float],
        text_filter: str,
        field: str = "text_content",
        top_k: int = 50,
    ) -> List[SearchResult]:
        """Search with text matching."""
        return self.search(query_vector=query_vector, top_k=top_k)
    
    def get_by_gene(self, gene_symbol: str, top_k: int = 100) -> List[SearchResult]:
        """Get all documents for a gene."""
        try:
            from qdrant_client.http import models
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="gene_symbol",
                            match=models.MatchValue(value=gene_symbol),
                        )
                    ]
                ),
                limit=top_k,
            )
            
            return [
                SearchResult(doc_id=str(r.id), score=1.0, payload=r.payload or {})
                for r in results[0]
            ]
        except Exception as e:
            logger.error(f"Get by gene failed: {e}")
            return []
    
    def get_by_disease(self, disease_name: str, top_k: int = 100) -> List[SearchResult]:
        """Get all documents for a disease."""
        try:
            from qdrant_client.http import models
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="disease_name",
                            match=models.MatchValue(value=disease_name),
                        )
                    ]
                ),
                limit=top_k,
            )
            
            return [
                SearchResult(doc_id=str(r.id), score=1.0, payload=r.payload or {})
                for r in results[0]
            ]
        except Exception as e:
            logger.error(f"Get by disease failed: {e}")
            return []
    
    def get_unique_values(self, field: str, limit: int = 1000) -> List[str]:
        """Get unique values for a field."""
        try:
            unique_values = set()
            offset = None
            
            while len(unique_values) < limit:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=[field],
                )
                
                if not results:
                    break
                
                for r in results:
                    if r.payload and field in r.payload:
                        value = r.payload[field]
                        if value:
                            unique_values.add(value)
                
                if offset is None:
                    break
            
            return list(unique_values)[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get unique values: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": getattr(info, 'vectors_count', None) or getattr(info, 'indexed_vectors_count', 0),
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as e:
            return {"name": self.collection_name, "status": "error", "error": str(e)}
    
    def get_stats(self) -> KnowledgeBaseStats:
        """Get knowledge base statistics."""
        try:
            from qdrant_client.http import models
            
            info = self.client.get_collection(self.collection_name)
            
            docs_by_type = {}
            for doc_type in DocumentType:
                count = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_type",
                                match=models.MatchValue(value=doc_type.value),
                            )
                        ]
                    ),
                )
                docs_by_type[doc_type.value] = count.count
            
            return KnowledgeBaseStats(
                total_documents=info.points_count or 0,
                documents_by_type=docs_by_type,
                collection_status=info.status.value if info.status else "unknown",
            )
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return KnowledgeBaseStats(collection_status="error")
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return False
