"""
Drug Discovery Agent - A disease-agnostic RAG-powered drug recommendation system.

Features:
- Fully dynamic: No hardcoded disease/drug/gene information
- Qdrant Cloud: Scalable vector storage
- HuggingFace PubMed-BERT: Biomedical embeddings
- Inter-agent communication: Standardized interfaces
"""

from .drug_agent import (
    DrugDiscoveryAgent,
    AgentMessage,
    AgentResponse,
    create_agent,
    create_agent_from_env,
)
from .models.data_models import (
    DrugAgentInput,
    DrugAgentOutput,
    DrugRecommendation,
    GeneMapping,
    PathwayMapping,
)

__version__ = "1.0.0"
__all__ = [
    "DrugDiscoveryAgent",
    "DrugAgentInput",
    "DrugAgentOutput",
    "DrugRecommendation",
    "GeneMapping",
    "PathwayMapping",
    "AgentMessage",
    "AgentResponse",
    "create_agent",
    "create_agent_from_env",
]
