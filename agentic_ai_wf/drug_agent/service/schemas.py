"""Service contract — input/output dataclasses for the Drug Agent Service."""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from enum import Enum

try:
    from agentic_ai_wf.reporting_pipeline_agent.core_types import (
        DRUG_HIGH_PRIORITY_THRESHOLD, DRUG_MODERATE_PRIORITY_THRESHOLD,
    )
except ImportError:
    DRUG_HIGH_PRIORITY_THRESHOLD, DRUG_MODERATE_PRIORITY_THRESHOLD = 55, 30


class QueryType(Enum):
    FULL_RECOMMENDATION = "full_recommendation"
    VALIDATE_DRUG = "validate_drug"
    CHECK_CONTRAINDICATION = "check_contraindication"
    SAFETY_PROFILE = "safety_profile"
    DRUG_DETAILS = "drug_details"


# ── Request Context ──────────────────────────────────────────────────────────

@dataclass
class GeneContext:
    gene_symbol: str
    log2fc: float
    adj_p_value: float
    direction: str                                    # "up" | "down"
    role: Optional[str] = None                        # "pathogenic" | "protective" | "therapeutic_target" | "immune_modulator"
    composite_score: Optional[float] = None
    evidence_stratum: Optional[str] = None            # "known_driver" | "ppi_connected" | "expression_significant" | "novel_candidate"
    causal_tier: Optional[str] = None

@dataclass
class PathwayContext:
    pathway_name: str
    direction: str
    fdr: float
    gene_count: int
    category: Optional[str] = None
    key_genes: Optional[List[str]] = None
    disease_relevance: Optional[str] = None           # ME-validated therapeutic implication

@dataclass
class BiomarkerContext:
    biomarker_name: str
    status: str                                       # "positive" | "negative" | "not_assessed" | "suggestive"
    supporting_genes: Optional[List[str]] = None
    biomarker_type: Optional[str] = None              # "A" (RNA-assessable) | "B" (orthogonal test required)

@dataclass
class TMEContext:
    highly_enriched_cells: List[str] = field(default_factory=list)
    moderately_enriched_cells: List[str] = field(default_factory=list)
    immune_infiltration_level: str = "unknown"

@dataclass
class MolecularSignatures:
    proliferation: Optional[float] = None
    apoptosis: Optional[float] = None
    dna_repair: Optional[float] = None
    inflammation: Optional[float] = None
    immune_activation: Optional[float] = None


@dataclass
class ScoringConfig:
    target_direction_weight: float = 18.0
    target_magnitude_weight: float = 12.0
    clinical_regulatory_weight: float = 25.0
    ot_weight: float = 15.0
    pathway_weight: float = 15.0
    safety_max_penalty: float = -30.0
    signature_bonus_max: float = 8.0
    target_exact_match_weight: float = 15.0
    # Fix 2: tier-weighted contra multiplier
    apply_contra_multipliers: bool = True
    contra_tier_multipliers: Dict = field(default_factory=lambda: {1: 0.0, 2: 0.25, 3: 0.75})
    # Fix 5: SOC multi-signal composite
    use_soc_composite: bool = True
    soc_signal_weights: Dict = field(default_factory=lambda: {
        "indication_sim": 0.40, "pharm_class_sim": 0.25, "clinical_depth": 0.35,
    })
    # Gene evidence stratum multipliers (known_driver=full credit → novel_candidate=half)
    stratum_multipliers: Dict = field(default_factory=lambda: {
        "known_driver": 1.0, "ppi_connected": 0.85,
        "expression_significant": 0.65, "novel_candidate": 0.5,
    })
    # Downstream effector analysis thresholds
    min_effectors_concordant: int = 2
    effector_credit_fraction: float = 0.6
    # Stage 5 post-score filtering thresholds
    base_noise_threshold: float = 10.0
    high_clinical_noise_threshold: float = 5.0
    high_clinical_score_cutoff: float = 15.0
    # Pathway drug class discovery
    pathway_drug_class_map: Dict = field(default_factory=lambda: {
        "JAK-STAT": ["tofacitinib", "baricitinib", "ruxolitinib", "upadacitinib", "filgotinib", "abrocitinib", "deucravacitinib"],
        "PI3K-AKT-mTOR": ["everolimus", "temsirolimus", "sirolimus", "idelalisib", "copanlisib", "alpelisib", "duvelisib"],
        "MAPK-ERK": ["trametinib", "binimetinib", "selumetinib", "cobimetinib", "dabrafenib", "vemurafenib", "encorafenib"],
        "NF-kB": ["bortezomib", "carfilzomib", "ixazomib", "thalidomide", "lenalidomide", "pomalidomide"],
        "Wnt-beta-catenin": ["celecoxib", "sulindac", "pyrvinium"],
        "Hedgehog": ["vismodegib", "sonidegib", "glasdegib"],
        "Notch": ["nirogacestat", "crenigacestat"],
        "TGF-beta": ["pirfenidone", "nintedanib", "fresolimumab"],
        "VEGF": ["bevacizumab", "ramucirumab", "aflibercept", "axitinib", "lenvatinib", "cabozantinib"],
        "HER2-ERBB": ["trastuzumab", "pertuzumab", "lapatinib", "neratinib", "tucatinib", "margetuximab"],
        "EGFR": ["erlotinib", "gefitinib", "afatinib", "osimertinib", "cetuximab", "panitumumab"],
        "CDK-cell-cycle": ["palbociclib", "ribociclib", "abemaciclib", "trilaciclib"],
        "BCR-ABL": ["imatinib", "dasatinib", "nilotinib", "bosutinib", "ponatinib", "asciminib"],
        "BTK": ["ibrutinib", "acalabrutinib", "zanubrutinib", "pirtobrutinib"],
        "BCL2-apoptosis": ["venetoclax", "navitoclax", "obatoclax"],
        "PD1-PDL1": ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "avelumab", "cemiplimab"],
        "CTLA4": ["ipilimumab", "tremelimumab"],
        "TNF": ["infliximab", "adalimumab", "etanercept", "certolizumab", "golimumab"],
        "IL6": ["tocilizumab", "sarilumab", "siltuximab"],
        "IL17": ["secukinumab", "ixekizumab", "brodalumab", "bimekizumab"],
        "IL23": ["guselkumab", "risankizumab", "tildrakizumab"],
        "IL4-IL13": ["dupilumab", "tralokinumab", "lebrikizumab"],
        "Complement": ["eculizumab", "ravulizumab", "avacopan", "iptacopan"],
        "PARP": ["olaparib", "rucaparib", "niraparib", "talazoparib"],
    })
    pathway_marker_genes: Dict = field(default_factory=lambda: {
        "JAK1": "JAK-STAT", "JAK2": "JAK-STAT", "JAK3": "JAK-STAT",
        "STAT1": "JAK-STAT", "STAT3": "JAK-STAT", "STAT4": "JAK-STAT",
        "TYK2": "JAK-STAT",
        "PIK3CA": "PI3K-AKT-mTOR", "PIK3R1": "PI3K-AKT-mTOR",
        "AKT1": "PI3K-AKT-mTOR", "MTOR": "PI3K-AKT-mTOR", "PTEN": "PI3K-AKT-mTOR",
        "BRAF": "MAPK-ERK", "KRAS": "MAPK-ERK", "NRAS": "MAPK-ERK",
        "MAP2K1": "MAPK-ERK", "MAPK1": "MAPK-ERK", "MAPK3": "MAPK-ERK",
        "NFKB1": "NF-kB", "RELA": "NF-kB", "IKBKB": "NF-kB",
        "CTNNB1": "Wnt-beta-catenin", "APC": "Wnt-beta-catenin", "WNT1": "Wnt-beta-catenin",
        "SMO": "Hedgehog", "GLI1": "Hedgehog", "PTCH1": "Hedgehog",
        "NOTCH1": "Notch", "NOTCH2": "Notch", "HES1": "Notch",
        "TGFB1": "TGF-beta", "SMAD2": "TGF-beta", "SMAD3": "TGF-beta",
        "VEGFA": "VEGF", "KDR": "VEGF", "FLT1": "VEGF",
        "ERBB2": "HER2-ERBB", "ERBB3": "HER2-ERBB", "ERBB4": "HER2-ERBB",
        "EGFR": "EGFR",
        "CDK4": "CDK-cell-cycle", "CDK6": "CDK-cell-cycle", "RB1": "CDK-cell-cycle",
        "BCR": "BCR-ABL", "ABL1": "BCR-ABL",
        "BTK": "BTK", "PLCG2": "BTK",
        "BCL2": "BCL2-apoptosis", "BAX": "BCL2-apoptosis", "MCL1": "BCL2-apoptosis",
        "PDCD1": "PD1-PDL1", "CD274": "PD1-PDL1",
        "CTLA4": "CTLA4",
        "TNF": "TNF", "TNFRSF1A": "TNF",
        "IL6": "IL6", "IL6R": "IL6",
        "IL17A": "IL17", "IL17F": "IL17", "IL17RA": "IL17",
        "IL23A": "IL23", "IL12B": "IL23",
        "IL4": "IL4-IL13", "IL13": "IL4-IL13", "IL4R": "IL4-IL13",
        "C3": "Complement", "C5": "Complement",
        "PARP1": "PARP", "BRCA1": "PARP", "BRCA2": "PARP",
    })

# ── Gene ↔ Protein Name Aliases ─────────────────────────────────────────────
# Bidirectional: HGNC gene symbol ↔ common protein/surface-marker names used
# in FDA labels, ChEMBL, and clinical literature.  Curated for drug-target
# genes where the protein name differs from the HGNC symbol.
_GENE_TO_PROTEINS: Dict[str, List[str]] = {
    # Surface markers / cluster of differentiation
    "MS4A1": ["CD20"], "CD19": ["B4"], "CD22": ["SIGLEC2"],
    "CD33": ["SIGLEC3"], "CD38": ["CYCLIC-ADP-RIBOSE"], "CD52": ["CAMPATH-1"],
    "TNFRSF17": ["BCMA"], "TNFRSF8": ["CD30"], "TNFRSF10B": ["DR5", "TRAIL-R2"],
    "SLAMF7": ["CS1", "CD319"], "FCGR3A": ["CD16A"],
    # Immune checkpoints
    "PDCD1": ["PD-1", "PD1", "CD279"], "CD274": ["PD-L1", "PDL1", "B7-H1"],
    "CTLA4": ["CD152"], "LAG3": ["CD223"], "HAVCR2": ["TIM-3", "TIM3"],
    "TIGIT": ["VSTM3"],
    # Growth factor receptors
    "ERBB2": ["HER2", "NEU"], "EGFR": ["HER1", "ERBB1"],
    "ERBB3": ["HER3"], "ERBB4": ["HER4"],
    "KDR": ["VEGFR2", "FLK1"], "FLT1": ["VEGFR1"], "FLT4": ["VEGFR3"],
    "FGFR1": ["CD331"], "FGFR2": ["CD332"], "FGFR3": ["CD333"],
    "MET": ["HGFR", "C-MET"], "IGF1R": ["CD221"],
    "IL2RA": ["CD25"], "IL6R": ["CD126"],
    # Cytokines / ligands — FDA labels use these names
    "TNF": ["TNF-ALPHA", "TNFA"], "VEGFA": ["VEGF"],
    "IL6": ["IL-6", "INTERLEUKIN-6"], "IL17A": ["IL-17", "IL-17A"],
    "IL4": ["IL-4", "INTERLEUKIN-4"], "IL13": ["IL-13"],
    "IL23A": ["IL-23"], "IL5": ["IL-5", "INTERLEUKIN-5"],
    "IL1B": ["IL-1B", "IL-1-BETA"], "IL2": ["IL-2"],
    "IFNG": ["IFN-GAMMA", "INTERFERON-GAMMA"],
    "TSLP": ["THYMIC-STROMAL-LYMPHOPOIETIN"],
    # Signalling kinases
    "BTK": ["BRUTON-TYROSINE-KINASE"], "ABL1": ["C-ABL"],
    "ALK": ["CD246"], "ROS1": ["MCF3"], "RET": ["CRET"],
    "BRAF": ["B-RAF"], "NTRK1": ["TRKA"], "NTRK2": ["TRKB"],
    # Intracellular targets
    "BCL2": ["BCL-2"], "HDAC1": ["HDAC"], "MTOR": ["FRAP1"],
    "PARP1": ["PARP"], "CDK4": ["CDK4/6"],
    # Complement
    "C5": ["COMPLEMENT-C5"],
}
_PROTEIN_TO_GENE: Dict[str, str] = {
    alias.upper(): gene
    for gene, aliases in _GENE_TO_PROTEINS.items()
    for alias in aliases
}


@dataclass
class DrugQueryRequest:
    disease: str
    query_type: QueryType = QueryType.FULL_RECOMMENDATION
    genes: List[GeneContext] = field(default_factory=list)
    pathways: List[PathwayContext] = field(default_factory=list)
    biomarkers: List[BiomarkerContext] = field(default_factory=list)
    tme: Optional[TMEContext] = None
    signatures: Optional[MolecularSignatures] = None
    drug_name: Optional[str] = None
    max_results: int = 30
    include_safety: bool = True
    include_trials: bool = True
    scoring_config: Optional[ScoringConfig] = None
    disease_context: Optional[str] = None             # Pipeline ME synthesis for OT fallback
    disease_aliases: List[str] = field(default_factory=list)
    all_patient_genes: List[GeneContext] = field(default_factory=list)  # Full DEG list for scoring (discovery uses `genes`)
    signature_scores: Optional[Dict] = None            # Full pathway signature scores (e.g., ifn, inflammation)
    is_gene_inferred_disease: bool = False             # True when disease was inferred from a gene symbol query
    is_target_only_query: bool = False                  # True when query is purely target-centric (no disease context)

    def get_upregulated_targets(self) -> List[GeneContext]:
        return [g for g in self.genes if g.direction == "up"
                and g.role in ("pathogenic", "therapeutic_target", "immune_modulator", None)]

    def get_downregulated_genes(self) -> List[GeneContext]:
        return [g for g in self.genes if g.direction == "down"]

    def get_downregulated_genes_significant(self) -> List[GeneContext]:
        """Downregulated genes meeting clinical significance threshold (|log2FC| >= 0.58)."""
        from agentic_ai_wf.reporting_pipeline_agent.core_types import DEG_LOG2FC_THRESHOLD
        return [g for g in self.genes
                if g.direction == "down" and abs(g.log2fc) >= DEG_LOG2FC_THRESHOLD]


# ── Response Evidence ────────────────────────────────────────────────────────

@dataclass
class DrugIdentity:
    drug_name: str
    chembl_id: Optional[str] = None
    drug_type: Optional[str] = None
    max_phase: Optional[int] = None
    first_approval: Optional[int] = None
    is_fda_approved: bool = False
    brand_names: List[str] = field(default_factory=list)
    patent_count: int = 0
    exclusivity_count: int = 0
    generics_available: bool = False
    pharm_class_moa: Optional[str] = None
    pharm_class_epc: Optional[str] = None
    indication_text: Optional[str] = None
    approved_indications: List[Dict[str, Any]] = field(default_factory=list)
    withdrawn: bool = False
    genetic_eligibility_required: bool = False
    genetic_eligibility_detail: str = ""

@dataclass
class TargetEvidence:
    gene_symbol: str
    action_type: str
    mechanism_of_action: Optional[str] = None
    fda_moa_narrative: Optional[str] = None
    patient_gene_log2fc: Optional[float] = None
    patient_gene_direction: Optional[str] = None
    ot_association_score: Optional[float] = None
    related_patient_gene: Optional[str] = None        # Pathway co-member from patient DEGs
    related_gene_log2fc: Optional[float] = None
    related_gene_direction: Optional[str] = None
    related_gene_source: Optional[str] = None         # "pathway" | "knowledge_graph"
    downstream_effector_genes: Optional[List[str]] = None   # All dysregulated pathway members
    downstream_pathway: Optional[str] = None
    known_effectors: Optional[List[str]] = None            # KG-resolved functionally related genes

@dataclass
class TrialEvidence:
    total_trials: int = 0
    highest_phase: Optional[float] = None
    completed_trials: int = 0
    trials_with_results: int = 0
    best_p_value: Optional[float] = None
    total_enrollment: int = 0
    top_trials: List[Dict] = field(default_factory=list)
    stopped_for_safety: bool = False

@dataclass
class SafetyProfile:
    boxed_warnings: List[str] = field(default_factory=list)
    top_adverse_events: List[Dict] = field(default_factory=list)
    serious_ratio: Optional[float] = None
    fatal_ratio: Optional[float] = None
    contraindications: List[str] = field(default_factory=list)
    pgx_warnings: List[Dict] = field(default_factory=list)
    recall_history: List[Dict] = field(default_factory=list)

@dataclass
class ScoreBreakdown:
    target_direction_match: float = 0.0
    target_magnitude_match: float = 0.0
    clinical_regulatory_score: float = 0.0
    ot_association_score: float = 0.0
    pathway_concordance: float = 0.0
    safety_penalty: float = 0.0
    disease_indication_bonus: float = 0.0
    signature_bonus: float = 0.0
    target_exact_match_bonus: float = 0.0
    gene_evidence_quality: float = 1.0                 # Stratum multiplier of best-matched gene (1.0=driver, 0.5=novel)
    composite_score: float = 0.0
    pipeline_evidence_used: bool = False               # True when OT fallback used pipeline context
    disease_relevant: bool = True                       # False when drug targets genes but lacks disease-treatment evidence
    tier_reasoning: str = ""                             # Q1→Q2→Q3 decision-tree explanation of tier placement

    def calculate(self):
        self.composite_score = max(0.0, min(100.0,
            self.target_direction_match
            + self.target_magnitude_match
            + self.clinical_regulatory_score
            + self.ot_association_score
            + self.pathway_concordance
            + self.safety_penalty
            + self.disease_indication_bonus
            + self.signature_bonus
            + self.target_exact_match_bonus
        ))

@dataclass
class ContraindicationEntry:
    tier: int                                         # 1=Avoid, 2=Contraindicated, 3=Use With Caution
    reason: str
    source: str                                       # "gene_based" | "biomarker" | "disease_ae" | "trial_stopped" | "withdrawn"
    gene_symbol: Optional[str] = None
    log2fc: Optional[float] = None

    @property
    def label(self) -> str:
        return {1: "Avoid", 2: "Contraindicated", 3: "Use With Caution"}.get(self.tier, "Unknown")

@dataclass
class DrugCandidate:
    identity: DrugIdentity
    targets: List[TargetEvidence] = field(default_factory=list)
    trial_evidence: Optional[TrialEvidence] = None
    safety: Optional[SafetyProfile] = None
    score: Optional[ScoreBreakdown] = None
    contraindication_flags: List[str] = field(default_factory=list)
    contraindication_entries: List[ContraindicationEntry] = field(default_factory=list)
    caution_notes: List[ContraindicationEntry] = field(default_factory=list)
    is_soc_candidate: bool = False
    soc_confidence: float = 0.0
    soc_advisory_notes: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    discovery_paths: List[str] = field(default_factory=list)
    validation_caveat: str = ""
    merged_from: List[str] = field(default_factory=list)
    inn: str = ""
    is_adc: bool = False
    is_diagnostic: bool = False
    pathway_class_match: str = ""
    enriched_pathways: List[str] = field(default_factory=list)
    match_group: str = ""                                    # "target" | "disease" | "" (no grouping)

@dataclass
class DrugQueryResponse:
    success: bool
    disease: str
    query_type: str
    recommendations: List[DrugCandidate] = field(default_factory=list)
    contraindicated: List[DrugCandidate] = field(default_factory=list)
    gene_targeted_only: List[DrugCandidate] = field(default_factory=list)
    targets_without_drugs: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def high_priority(self) -> List[DrugCandidate]:
        return [r for r in self.recommendations if r.score and r.score.composite_score >= DRUG_HIGH_PRIORITY_THRESHOLD]

    @property
    def moderate_priority(self) -> List[DrugCandidate]:
        return [r for r in self.recommendations
                if r.score and DRUG_MODERATE_PRIORITY_THRESHOLD <= r.score.composite_score < DRUG_HIGH_PRIORITY_THRESHOLD]

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)
