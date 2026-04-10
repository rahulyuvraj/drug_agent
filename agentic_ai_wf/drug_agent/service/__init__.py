"""Drug Agent Service — universal drug intelligence API for all agents."""

from .drug_agent_service import DrugAgentService, get_service
from .schemas import (
    DrugQueryRequest, DrugQueryResponse, QueryType, ScoringConfig,
    GeneContext, PathwayContext, BiomarkerContext, TMEContext, MolecularSignatures,
    DrugCandidate, DrugIdentity, TargetEvidence, TrialEvidence,
    SafetyProfile, ScoreBreakdown,
    _GENE_TO_PROTEINS, _PROTEIN_TO_GENE,
)

__all__ = [
    "DrugAgentService", "get_service",
    "DrugQueryRequest", "DrugQueryResponse", "QueryType", "ScoringConfig",
    "GeneContext", "PathwayContext", "BiomarkerContext", "TMEContext", "MolecularSignatures",
    "DrugCandidate", "DrugIdentity", "TargetEvidence", "TrialEvidence",
    "SafetyProfile", "ScoreBreakdown",
    "_GENE_TO_PROTEINS", "_PROTEIN_TO_GENE",
]
