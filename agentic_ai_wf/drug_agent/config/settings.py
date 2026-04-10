"""
Configuration Settings Module - Supports Qdrant Cloud and dynamic configuration.
"""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Load .env file
try:
    from dotenv import load_dotenv
    # Load from .env file in project root or current directory
    env_paths = [
        Path(__file__).parent.parent / ".env",
        Path(".env"),
        Path(__file__).parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Qdrant Cloud configuration."""
    url: Optional[str] = None
    port: int = 6333
    api_key: Optional[str] = None
    username: Optional[str] = None  # HTTP Basic Auth
    password: Optional[str] = None  # HTTP Basic Auth
    collection_name: str = "drug_knowledge_base"
    timeout_seconds: int = 60
    use_https: bool = True
    prefer_grpc: bool = True
    
    def __post_init__(self):
        self.url = os.getenv("QDRANT_URL", self.url)
        self.api_key = os.getenv("QDRANT_API_KEY", self.api_key)
        self.username = os.getenv("QDRANT_USERNAME", self.username)
        self.password = os.getenv("QDRANT_PASSWORD", self.password)
        if os.getenv("QDRANT_COLLECTION"):
            self.collection_name = os.getenv("QDRANT_COLLECTION")
        # Auto-set port to 443 for HTTPS if not explicitly set
        if self.url and self.url.startswith("https://") and self.username and self.password:
            self.port = 443
    
    def get_connection_url(self) -> str:
        if not self.url:
            raise ValueError("QDRANT_URL must be set")
        if self.url.startswith("http"):
            return self.url
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://{self.url}"
    
    def is_cloud(self) -> bool:
        return self.url is not None and ("cloud.qdrant.io" in str(self.url) or self.api_key is not None)


@dataclass
class EmbeddingConfig:
    """HuggingFace embedding model configuration."""
    model_name: str = "NeuML/pubmedbert-base-embeddings"
    device: str = "auto"
    batch_size: int = 32
    max_sequence_length: int = 512
    cache_enabled: bool = True
    cache_directory: str = "./embedding_cache"
    hf_token: Optional[str] = None
    
    def __post_init__(self):
        self.hf_token = os.getenv("HF_TOKEN", self.hf_token)
    
    def get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


@dataclass
class IngestionConfig:
    json_directory: str = "/mnt/c/Users/7wraa/Downloads/Ayass-v5.26/GeneALaCart-AllGenes"
    batch_size: int = 1000
    skip_empty_files: bool = True
    checkpoint_enabled: bool = True
    checkpoint_directory: str = "./checkpoints"
    max_workers: int = 4


@dataclass
class RetrievalConfig:
    default_top_k: int = 50
    final_top_k: int = 20
    min_relevance_score: float = 0.3
    rrf_k: int = 60
    dynamic_expansion: bool = True


@dataclass
class RankingWeights:
    relevance: float = 0.30
    gene_match: float = 0.35
    evidence: float = 0.20
    approval_status: float = 0.15


@dataclass
class RankingConfig:
    weights: RankingWeights = field(default_factory=RankingWeights)


@dataclass
class OutputConfig:
    max_recommendations: int = 30
    include_evidence_details: bool = True


@dataclass
class AgentConfig:
    """Agent communication configuration."""
    agent_id: str = "drug_agent"
    version: str = "1.0.0"
    enable_messaging: bool = True
    response_timeout: int = 120


@dataclass
class Settings:
    """Main settings container."""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        path = Path(config_path)
        if not path.exists():
            return cls()
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Settings":
        s = cls()
        for section in ['qdrant', 'embedding', 'ingestion', 'retrieval', 'output', 'agent']:
            if section in d:
                obj = getattr(s, section)
                for k, v in d[section].items():
                    # Only set from config if value is not None AND env var hasn't set it
                    if hasattr(obj, k) and v is not None:
                        # For qdrant section, check if env var was already loaded
                        if section == 'qdrant' and k == 'collection_name':
                            # Only override if QDRANT_COLLECTION env var is not set
                            if not os.getenv("QDRANT_COLLECTION"):
                                setattr(obj, k, v)
                        elif section == 'qdrant' and k in ['url', 'api_key', 'username', 'password']:
                            # Skip - these are loaded from env vars in __post_init__
                            pass
                        else:
                            setattr(obj, k, v)
        if 'ranking' in d and 'weights' in d['ranking']:
            for k, v in d['ranking']['weights'].items():
                if hasattr(s.ranking.weights, k):
                    setattr(s.ranking.weights, k, v)
        return s
    
    def validate(self) -> List[str]:
        errors = []
        if not self.qdrant.url:
            errors.append("QDRANT_URL is required")
        if self.qdrant.is_cloud() and not self.qdrant.api_key:
            errors.append("QDRANT_API_KEY is required for Qdrant Cloud")
        return errors


def load_settings(config_path: Optional[str] = None) -> Settings:
    if config_path:
        return Settings.from_yaml(config_path)
    for path in [Path(__file__).parent / "drug_agent_config.yaml", Path("config/drug_agent_config.yaml")]:
        if path.exists():
            return Settings.from_yaml(str(path))
    return Settings()


_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings

def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings

def reset_settings() -> None:
    global _settings
    _settings = None
