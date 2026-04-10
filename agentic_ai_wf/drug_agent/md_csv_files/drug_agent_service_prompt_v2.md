# Copilot Prompt — Drug Agent Service Layer (Week 1-2)

---

## TASK

Create a `service/` package inside `drug_agent/` that exposes the Drug Agent as a **callable Python service** for upstream agents (reporting pipeline, clinical assessment, etc.). Currently, the only way to query all 14 Qdrant collections is through `app_multi.py` (Streamlit UI) which does flat parallel semantic search with no cross-collection reasoning. The new service must do **orchestrated multi-collection retrieval** with structured input/output, cross-collection drug entity resolution, and composite evidence-based scoring.

**Read this entire document before writing any code. Provide your strategy first.**

---

## EXISTING CODEBASE — WHAT EXISTS AND WHAT DOESN'T

### Infrastructure You MUST Reuse (do NOT rewrite)

**`opentargets/ot_base.py`** (136 lines) — Shared infrastructure used by ALL ingestion scripts:
```python
# These functions are your Qdrant + embedding entry points:
from drug_agent.opentargets.ot_base import get_qdrant, get_embedder

qdrant_client = get_qdrant()          # Returns QdrantClient with Basic Auth httpx patching
embedder = get_embedder()             # Returns PubMedBERT SentenceTransformer on CUDA
ensure_collection(client, name)       # Creates collection if missing (768-dim, Cosine)
upsert_batch(client, name, points)    # Batch upsert with retry
```

**`embedding/embedder.py`** (185 lines) — `PubMedBERTEmbedder` class with `embed_texts()` and `embed_single()`. Has file-based cache. Returns 768-dim vectors.

**`storage/basic_auth_qdrant.py`** (132 lines) — `create_qdrant_client_with_basic_auth()` — the httpx monkey-patch for Basic Auth. Used internally by `ot_base.get_qdrant()`.

**`.env` file** — Qdrant credentials:
```
QDRANT_URL=https://vector.f420.ai
QDRANT_USERNAME=admin
QDRANT_PASSWORD=<in .env>
```

### Existing Code You Should UNDERSTAND But NOT Depend On

**`drug_agent.py`** (618 lines) — `DrugDiscoveryAgent` class. This was the original orchestrator that only queries the `Drug_agent` collection (443K GeneALaCart gene docs). It has `query()` and `get_recommendations()` but these only use `retrieval/query_builder.py` → `retrieval/hybrid_search.py` → `recommendation/drug_ranker.py` against ONE collection. **Your service replaces this query path with a 14-collection orchestrated pipeline.** Do not modify this file.

**`models/data_models.py`** (322 lines) — Existing dataclasses:
```python
@dataclass
class GeneMapping:
    gene_symbol: str          # e.g., "ERBB2"
    log2fc: float             # e.g., 4.25
    p_value: float            # e.g., 0.05
    direction: str            # "up" or "down"

@dataclass
class PathwayMapping:
    pathway_name: str         # e.g., "Regulation of Cell Cycle"
    enrichment_score: float   # FDR value
    direction: str            # "up" or "down"
    gene_count: int           # Genes in pathway

@dataclass
class DrugAgentInput:
    disease_name: str
    gene_mappings: List[GeneMapping]
    pathway_mappings: List[PathwayMapping]
    xcell_findings: Optional[Dict] = None

@dataclass
class DrugAgentOutput:
    recommendations: List[DrugRecommendation]
    metadata: Dict
    # ... etc.

@dataclass
class DrugRecommendation:
    drug_name: str
    score: float
    evidence_summary: str
    genes: List[str]
    pathways: List[str]
    approval_status: str
    confidence: str
```

These models are **too limited** for the 14-collection service. They lack biomarker context, TME data, molecular signatures, clinical trial evidence, safety profiles, and composite scoring breakdowns. **Create new schemas in `service/schemas.py` that extend these concepts.** The existing models remain untouched for backward compatibility.

**`app_multi.py`** (720 lines) — The Streamlit UI. Key pattern to understand:
```python
# It defines all 14 collections in a dict:
COLLECTIONS = {
    "Drug_agent": {"description": "Gene-Drug-Disease-Pathway relationships", ...},
    "ChEMBL_drugs": {"description": "Drug compounds", ...},
    "Raw_csv_KG": {"description": "Knowledge Graph triples", ...},
    # ... all 14
}

# Search pattern (same query to all collections):
def search_collection(collection_name, query_vector, top_k):
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

# Runs searches in parallel via ThreadPoolExecutor
# Displays results in per-collection tabs with entity-specific formatting
```

Your service should NOT import from `app_multi.py`. Instead, replicate its Qdrant query capability in `collection_router.py` but with **structured, targeted queries** per collection instead of the same semantic search everywhere.

### What's In Each Collection (payload fields that matter for routing)

| Collection | Docs | Key Queryable Fields |
|---|---|---|
| `Drug_agent` | 443K | `doc_type` (gene_drug/disease_drug/pathway_drug), `gene_symbol`, `drug_name`, `disease_name`, `pathway_name` |
| `ChEMBL_drugs` | 4K | `molecule_name`, `target_gene`, `action_type`, `mechanism_of_action`, `max_phase`, `molecule_type` |
| `Raw_csv_KG` | 8.1M | `subject`, `predicate`, `object`, `source` |
| `OpenTargets_data` | 43K | `entity_type` (target/disease/drug/association), `gene_symbol`/`disease_name`/`drug_name`, `overall_score` |
| `OpenTargets_drugs_enriched` | 3.2K | `drug_name`, `drug_id`, `drug_type`, `max_phase`, `mechanisms`, `linked_targets` (Ensembl IDs), `linked_diseases`, `indication_phases` |
| `OpenTargets_adverse_events` | 61K | `entity_type` (adverse_event/drug_warning), `drug_name`, `event_name`, `logLR`, `count`, `toxicity_class` |
| `OpenTargets_pharmacogenomics` | 16K | `drug_name`, `gene_symbol`, `variant_rs_id`, `pgx_category`, `phenotype_text`, `evidence_level`, `is_direct_target` |
| `FDA_Orange_Book` | 48K | `ingredient`, `trade_name`, `nda_number`, `approval_date` |
| `FDA_DrugsFDA` | 29K | `brand_name`, `generic_name`, `pharm_class_moa`, `pharm_class_epc`, `application_number` |
| `FDA_FAERS` | 52K | `entity_type` (faers_summary/faers_reaction), `drug_name`, `serious_ratio`, `fatal_ratio`, `reaction_term`, `reaction_count` |
| `FDA_Drug_Labels` | 426K | `brand_name`, `generic_name`, `section_type` (indication/mechanism_of_action/adverse_reaction/contraindication/boxed_warning/drug_interaction/clinical_pharmacology/warning), `pharm_class_moa`, `text_content` |
| `FDA_Enforcement` | 17K | `classification`, `reason_for_recall`, `recalling_firm`, `status` |
| `ClinicalTrials_summaries` | 146K | `nct_id`, `phase`, `phase_numeric`, `overall_status`, `conditions`, `drug_names`, `enrollment`, `sponsor` |
| `ClinicalTrials_results` | 116K | `nct_id`, `phase`, `phase_numeric`, `conditions`, `drug_names`, `has_primary_outcomes`, `num_serious_aes`, `p_values`, `primary_outcome_titles` |

---

## WHAT YOU'RE BUILDING

### New Files

```
drug_agent/
├── service/                          ← NEW PACKAGE
│   ├── __init__.py                   ← Exports DrugAgentService + all schemas
│   ├── schemas.py                    ← Input/output dataclasses (the service contract)
│   ├── drug_agent_service.py         ← Main orchestrator (entry point for callers)
│   ├── collection_router.py          ← Targeted queries to specific collections
│   ├── result_aggregator.py          ← Cross-collection drug entity resolution + merging
│   └── drug_scorer.py                ← Composite evidence-based scoring (0-100)
```

**Do NOT modify any existing files.** The service is additive.

---

## FILE 1: `service/schemas.py`

Define the service contract. Two main classes: `DrugQueryRequest` (what callers send) and `DrugQueryResponse` (what they get back).

### `DrugQueryRequest`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class QueryType(Enum):
    FULL_RECOMMENDATION = "full_recommendation"
    VALIDATE_DRUG = "validate_drug"
    CHECK_CONTRAINDICATION = "check_contraindication"
    SAFETY_PROFILE = "safety_profile"
    DRUG_DETAILS = "drug_details"

@dataclass
class GeneContext:
    gene_symbol: str                              # "ERBB2"
    log2fc: float                                 # 4.25
    adj_p_value: float                            # 0.05
    direction: str                                # "up" or "down"
    role: Optional[str] = None                    # "oncogenic_driver" | "tumor_suppressor" | "therapeutic_target" | "immune_regulator"
    composite_score: Optional[float] = None       # From reporting pipeline gene prioritization (0-1)

@dataclass
class PathwayContext:
    pathway_name: str                             # "Regulation of Cell Cycle"
    direction: str                                # "up" or "down"
    fdr: float                                    # 7.59e-10
    gene_count: int                               # 42
    category: Optional[str] = None                # "Cell Cycle" | "Immune System" | "DNA Repair" | etc.
    key_genes: Optional[List[str]] = None         # ["CDC6", "BRIP1", "NCAPG"]

@dataclass
class BiomarkerContext:
    biomarker_name: str                           # "HER2" | "ER" | "PR" | "PD-L1" | "BRCA_mutation"
    status: str                                   # "positive" | "negative" | "not_assessed" | "suggestive"
    supporting_genes: Optional[List[str]] = None  # ["ERBB2"]

@dataclass
class TMEContext:
    highly_enriched_cells: List[str] = field(default_factory=list)       # ["MSC", "Osteoblast"]
    moderately_enriched_cells: List[str] = field(default_factory=list)   # ["CD4+ memory T-cells", "Th1 cells"]
    immune_infiltration_level: str = "unknown"                           # "high" | "moderate" | "low" | "unknown"

@dataclass
class MolecularSignatures:
    proliferation: Optional[float] = None         # 0.0-1.0
    apoptosis: Optional[float] = None
    dna_repair: Optional[float] = None
    inflammation: Optional[float] = None
    immune_activation: Optional[float] = None

@dataclass
class DrugQueryRequest:
    disease: str                                                         # Required always
    query_type: QueryType = QueryType.FULL_RECOMMENDATION
    # Molecular context (for FULL_RECOMMENDATION)
    genes: List[GeneContext] = field(default_factory=list)
    pathways: List[PathwayContext] = field(default_factory=list)
    biomarkers: List[BiomarkerContext] = field(default_factory=list)
    tme: Optional[TMEContext] = None
    signatures: Optional[MolecularSignatures] = None
    # For single-drug queries
    drug_name: Optional[str] = None
    # Config
    max_results: int = 15
    include_safety: bool = True
    include_trials: bool = True

    # Helpers
    def get_upregulated_targets(self) -> List[GeneContext]:
        return [g for g in self.genes if g.direction == "up" and g.role in ("oncogenic_driver", "therapeutic_target", None)]

    def get_downregulated_genes(self) -> List[GeneContext]:
        return [g for g in self.genes if g.direction == "down"]
```

### `DrugQueryResponse`

```python
@dataclass
class DrugIdentity:
    drug_name: str
    chembl_id: Optional[str] = None
    drug_type: Optional[str] = None               # "Small molecule" | "Antibody" | etc.
    max_phase: Optional[int] = None               # 1-4
    first_approval: Optional[int] = None
    is_fda_approved: bool = False
    brand_names: List[str] = field(default_factory=list)

@dataclass
class TargetEvidence:
    gene_symbol: str
    action_type: str                              # "INHIBITOR" | "ANTAGONIST" | "AGONIST" | etc.
    mechanism_of_action: Optional[str] = None     # Short from ChEMBL
    fda_moa_narrative: Optional[str] = None       # Rich from FDA_Drug_Labels section_type=mechanism_of_action
    patient_gene_log2fc: Optional[float] = None
    patient_gene_direction: Optional[str] = None
    ot_association_score: Optional[float] = None  # OpenTargets target-disease score (0-1)

@dataclass
class TrialEvidence:
    total_trials: int = 0
    highest_phase: Optional[float] = None
    completed_trials: int = 0
    trials_with_results: int = 0
    best_p_value: Optional[float] = None
    total_enrollment: int = 0
    top_trials: List[Dict] = field(default_factory=list)  # [{nct_id, title, phase, status}]

@dataclass
class SafetyProfile:
    boxed_warnings: List[str] = field(default_factory=list)
    top_adverse_events: List[Dict] = field(default_factory=list)     # [{event_name, logLR, count}]
    serious_ratio: Optional[float] = None
    fatal_ratio: Optional[float] = None
    contraindications: List[str] = field(default_factory=list)
    pgx_warnings: List[Dict] = field(default_factory=list)           # [{gene, variant, phenotype}]

@dataclass
class ScoreBreakdown:
    target_expression_match: float = 0.0          # 0-25
    clinical_trial_evidence: float = 0.0          # 0-25
    ot_association_score: float = 0.0             # 0-15
    fda_regulatory_status: float = 0.0            # 0-15
    pathway_concordance: float = 0.0              # 0-10
    safety_penalty: float = 0.0                   # 0 to -20
    composite_score: float = 0.0                  # 0-100

    def calculate(self):
        self.composite_score = max(0.0, min(100.0,
            self.target_expression_match +
            self.clinical_trial_evidence +
            self.ot_association_score +
            self.fda_regulatory_status +
            self.pathway_concordance +
            self.safety_penalty
        ))

@dataclass
class DrugCandidate:
    identity: DrugIdentity
    targets: List[TargetEvidence] = field(default_factory=list)
    trial_evidence: Optional[TrialEvidence] = None
    safety: Optional[SafetyProfile] = None
    score: Optional[ScoreBreakdown] = None
    contraindication_flags: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)         # Which collections contributed

@dataclass
class DrugQueryResponse:
    success: bool
    disease: str
    query_type: str
    recommendations: List[DrugCandidate] = field(default_factory=list)
    contraindicated: List[DrugCandidate] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)                      # timing, collections_queried, candidate_count
    errors: List[str] = field(default_factory=list)

    @property
    def high_priority(self) -> List[DrugCandidate]:
        return [r for r in self.recommendations if r.score and r.score.composite_score >= 70]

    @property
    def moderate_priority(self) -> List[DrugCandidate]:
        return [r for r in self.recommendations if r.score and 40 <= r.score.composite_score < 70]

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)
```

---

## FILE 2: `service/collection_router.py`

Routes queries to the correct collections. Uses `ot_base.get_qdrant()` and `ot_base.get_embedder()` for infrastructure.

### Qdrant Query Patterns

```python
# Semantic search (used for most queries):
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.query_points(
    collection_name="ChEMBL_drugs",
    query=embedder.encode("ERBB2 inhibitor breast cancer"),  # 768-dim vector
    limit=10,
    with_payload=True,
    query_filter=Filter(must=[
        FieldCondition(key="target_gene", match=MatchValue(value="ERBB2"))
    ])
)
# Each result has: result.id, result.score (cosine similarity), result.payload (dict)

# Scroll (for loading all docs of a type, e.g., gene_symbol→Ensembl cache):
points, next_offset = client.scroll(
    collection_name="OpenTargets_data",
    scroll_filter=Filter(must=[
        FieldCondition(key="entity_type", match=MatchValue(value="target"))
    ]),
    limit=1000,
    with_payload=["gene_symbol"],  # Only fetch needed fields
    offset=next_offset
)
```

### Methods to Implement

```python
class CollectionRouter:
    def __init__(self):
        self.client = get_qdrant()           # from ot_base
        self.embedder = get_embedder()       # from ot_base
        self._ensembl_cache = {}             # gene_symbol → Ensembl ID (loaded from OpenTargets_data targets)
        self._available_collections = set()  # Populated at init — skip unavailable collections gracefully
        self._init_available_collections()
        self._load_ensembl_cache()

    def _init_available_collections(self):
        """Check which of the 14 collections exist. Store names of available ones.
        Log warnings for missing collections but do NOT crash."""

    def _load_ensembl_cache(self):
        """Scroll OpenTargets_data, filter entity_type=target, build gene_symbol→ensembl_id dict.
        ~9,500 entries. Cache in memory. Skip if OpenTargets_data unavailable."""

    def _embed(self, text: str) -> list:
        """Embed text using PubMedBERT. Use embedder.encode() directly."""

    def _search(self, collection: str, query_text: str, top_k: int = 10, filter: Optional[Filter] = None) -> List[Dict]:
        """Wrapper: embed query_text → query_points → return list of {score, payload} dicts.
        Skip silently if collection not in self._available_collections."""

    # === STAGE 1: CANDIDATE DISCOVERY ===

    def find_drugs_for_target(self, gene_symbol: str, top_k: int = 10) -> List[Dict]:
        """Find drugs targeting a specific gene.
        Collections queried:
          1. ChEMBL_drugs — filter: target_gene={gene_symbol}. Falls back to semantic search "{gene_symbol} inhibitor drug" if filter returns <3 results.
          2. OpenTargets_drugs_enriched — semantic search "{gene_symbol} drug mechanism". Post-filter: check linked_targets contains ensembl_cache[gene_symbol].
          3. FDA_Drug_Labels — filter: section_type=mechanism_of_action, semantic: "{gene_symbol}"
        Returns: [{drug_name, source, action_type, mechanism, phase, score}] deduplicated by normalized drug name."""

    def find_drugs_for_disease(self, disease_name: str, top_k: int = 10) -> List[Dict]:
        """Find drugs associated with a disease.
        Collections queried:
          1. ClinicalTrials_summaries — semantic: "{disease_name} drug treatment". Extract drug_names from payload.
          2. OpenTargets_drugs_enriched — semantic: "{disease_name}". Check linked_diseases.
          3. FDA_Drug_Labels — filter: section_type=indication, semantic: "{disease_name}"
        Returns: [{drug_name, source, indication, phase, score}]"""

    # === STAGE 2: EVIDENCE ENRICHMENT (per candidate drug) ===

    def get_drug_identity(self, drug_name: str) -> Dict:
        """Merge identity from: ChEMBL_drugs (molecule_name, chembl_id, type, phase) + FDA_DrugsFDA (brand, generic, pharm_class) + OpenTargets_drugs_enriched (mechanisms, indication_phases)."""

    def get_drug_targets(self, drug_name: str) -> List[Dict]:
        """Get target genes from: ChEMBL_drugs (target_gene, action_type, mechanism) + FDA_Drug_Labels section_type=mechanism_of_action (rich narrative)."""

    def get_target_disease_score(self, gene_symbol: str, disease_name: str) -> Optional[float]:
        """Query OpenTargets_data for entity_type=association, semantic: "{gene_symbol} {disease_name}". Return overall_score (0-1) if found."""

    def get_indication_status(self, drug_name: str, disease_name: str) -> Dict:
        """Is this drug approved for this disease?
        1. FDA_Drug_Labels section_type=indication, search "{drug_name} {disease_name}"
        2. OpenTargets_drugs_enriched — check indication_phases dict for disease match
        Returns: {is_approved, highest_phase_for_indication, indication_text}"""

    def get_trial_evidence(self, drug_name: str, disease_name: str) -> Dict:
        """Clinical trial data from:
        1. ClinicalTrials_results — semantic: "{drug_name} {disease_name}", filter has_primary_outcomes=true preferred
        2. ClinicalTrials_summaries — semantic: "{drug_name} {disease_name}", aggregate by phase/status
        Returns: {total_trials, highest_phase, best_p_value, enrollment, top_trials}"""

    def get_safety_profile(self, drug_name: str) -> Dict:
        """Aggregate safety from:
        1. FDA_Drug_Labels sections: boxed_warning, adverse_reaction, contraindication
        2. OpenTargets_adverse_events — semantic: "{drug_name}", sort by logLR
        3. FDA_FAERS — semantic: "{drug_name}", get faers_summary (serious/fatal ratios)
        4. OpenTargets_pharmacogenomics — semantic: "{drug_name}", filter pgx_category in (efficacy, toxicity)
        Returns: {boxed_warnings, top_aes, serious_ratio, fatal_ratio, contraindications, pgx_warnings}"""

    def check_contraindication(self, drug_name: str, gene_symbol: str, gene_direction: str) -> Dict:
        """Check if drug's action on target conflicts with patient's gene expression.
        1. Get action_type from ChEMBL_drugs (INHIBITOR/AGONIST/etc.)
        2. Logic: INHIBITOR + target DOWNREGULATED → contraindicated (targeting suppressed target)
                  AGONIST + target UPREGULATED → may be reinforcing oncogene (context-dependent)
        Returns: {is_contraindicated, reason, severity}"""
```

---

## FILE 3: `service/result_aggregator.py`

Merges results from multiple collections into unified drug records. Key challenge: the same drug has different names across collections.

```python
class ResultAggregator:
    def normalize_drug_name(self, name: str) -> str:
        """Normalize for matching.
        1. UPPER case
        2. Strip parentheticals: 'trastuzumab (Herceptin®)' → 'TRASTUZUMAB'
        3. Strip suffixes: ' [EPC]', ' [MoA]', ' (Drug Compound)'
        4. Strip non-alphanumeric except hyphens
        Known patterns from your data:
          - ClinicalTrials drug_names often have brand in parens: 'trastuzumab (Herceptin®)'
          - FDA_DrugsFDA uses UPPER
          - ChEMBL_drugs uses UPPER
          - OpenTargets uses mixed case"""

    def merge_candidates(self, discovery_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Input: keyed by query source (e.g., "target_ERBB2", "disease", "pathway_cell_cycle").
        Output: Deduplicated list of unique drugs with all sources merged.
        Group by normalized drug name. For each group: merge all source collections, targets, mechanisms. Keep highest similarity score."""

    def build_candidate(self, drug_name: str, identity: Dict, targets: List[Dict],
                        indication: Dict, trials: Dict, safety: Dict) -> DrugCandidate:
        """Assemble a DrugCandidate from all evidence dicts returned by CollectionRouter methods."""
```

---

## FILE 4: `service/drug_scorer.py`

```python
class DrugScorer:
    def score(self, candidate: DrugCandidate, request: DrugQueryRequest) -> ScoreBreakdown:
        s = ScoreBreakdown()
        s.target_expression_match = self._target_match(candidate, request)      # 0-25
        s.clinical_trial_evidence = self._trial_evidence(candidate)             # 0-25
        s.ot_association_score = self._ot_score(candidate)                      # 0-15
        s.fda_regulatory_status = self._regulatory(candidate, request)          # 0-15
        s.pathway_concordance = self._pathway_match(candidate, request)         # 0-10
        s.safety_penalty = self._safety_penalty(candidate)                      # 0 to -20
        s.calculate()
        return s

    def _target_match(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
        """0-25 points.
        For each target in c.targets, find matching gene in r.genes.
        - INHIBITOR targeting UPREGULATED gene: 15 base + bonus by |log2fc| (max 25)
        - INHIBITOR targeting DOWNREGULATED gene: 0 (wrong direction)
        - AGONIST targeting DOWNREGULATED tumor suppressor: 15 (restoration)
        Take best score across all targets."""

    def _trial_evidence(self, c: DrugCandidate) -> float:
        """0-25 points.
        Phase 4 approved for this indication = 25
        Phase 3 completed + positive p-value = 20
        Phase 3 completed (no p-value) = 17
        Phase 3 recruiting = 15
        Phase 2 + positive results = 12
        Phase 2 = 8
        Phase 1 = 4
        No trials = 0
        Bonus +2 if multiple Phase 3 trials, +1 if enrollment > 500."""

    def _ot_score(self, c: DrugCandidate) -> float:
        """0-15 points. Best ot_association_score across targets × 15."""

    def _regulatory(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
        """0-15 points.
        FDA-approved for THIS disease = 15
        Approved for related disease = 10
        Phase 3 not yet approved = 5
        Investigational = 0."""

    def _pathway_match(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
        """0-10 points.
        Map drug mechanism keywords to pathway categories:
          HER2/ERBB2 → Signal Transduction
          PARP → DNA Repair
          CDK/cell cycle → Cell Cycle
          PD-1/PD-L1/immune → Immune System
          mTOR/PI3K → PI3K-AKT-mTOR
        Check if category matches any upregulated pathway in request.
        5 points per match, max 10."""

    def _safety_penalty(self, c: DrugCandidate) -> float:
        """0 to -20. Boxed warning: -5 each (max -10). serious_ratio > 0.5: -3. fatal_ratio > 0.05: -5. PGx toxicity: -2."""
```

---

## FILE 5: `service/drug_agent_service.py`

```python
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DrugAgentService:
    """
    Callable drug recommendation service.

    Usage:
        from drug_agent.service import DrugAgentService, DrugQueryRequest, QueryType, GeneContext

        svc = DrugAgentService()
        req = DrugQueryRequest(
            disease="Breast Cancer",
            genes=[GeneContext("ERBB2", 4.25, 0.05, "up", role="oncogenic_driver")],
        )
        resp = svc.query(req)
        print(resp.high_priority)
    """

    def __init__(self):
        """
        1. Initialize CollectionRouter (this connects to Qdrant, loads embedder, checks collections)
        2. Initialize ResultAggregator
        3. Initialize DrugScorer
        """

    def query(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Route to handler by query_type."""

    def _full_recommendation(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """
        STAGE 1 — CANDIDATE DISCOVERY
        For each gene in request.get_upregulated_targets():
            router.find_drugs_for_target(gene.gene_symbol)
        router.find_drugs_for_disease(request.disease)
        Merge via aggregator.merge_candidates()

        STAGE 2 — EVIDENCE ENRICHMENT (for each unique candidate, cap at ~30)
        For each candidate:
            router.get_drug_identity(name)
            router.get_drug_targets(name)
            For each target gene: router.get_target_disease_score(gene, disease)
            router.get_indication_status(name, disease)
            if request.include_trials: router.get_trial_evidence(name, disease)
            if request.include_safety: router.get_safety_profile(name)
            aggregator.build_candidate(...)

        STAGE 3 — CONTRAINDICATION CHECK
        For each downregulated gene in request:
            For each candidate targeting that gene:
                router.check_contraindication(drug, gene, direction)
                If contraindicated → move to contraindicated list

        STAGE 4 — SCORING
        For each remaining candidate:
            scorer.score(candidate, request)

        STAGE 5 — SORT AND RETURN
        Sort by composite_score desc, take top max_results.
        Populate metadata with timing and stats.
        """

    def _validate_drug(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Enrich and score a single drug (request.drug_name) against the molecular profile."""

    def _check_contraindication(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Quick check: is request.drug_name contraindicated given request.genes?"""

    def _safety_profile(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Get full safety data for request.drug_name."""

    def _drug_details(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Get all available info about request.drug_name from all collections."""

    def health_check(self) -> Dict:
        """Return {available_collections, total_points, qdrant_status}."""
```

---

## FILE 6: `service/__init__.py`

```python
from .drug_agent_service import DrugAgentService
from .schemas import (
    DrugQueryRequest, DrugQueryResponse, QueryType,
    GeneContext, PathwayContext, BiomarkerContext, TMEContext, MolecularSignatures,
    DrugCandidate, DrugIdentity, TargetEvidence, TrialEvidence, SafetyProfile, ScoreBreakdown,
)

__all__ = [
    "DrugAgentService",
    "DrugQueryRequest", "DrugQueryResponse", "QueryType",
    "GeneContext", "PathwayContext", "BiomarkerContext", "TMEContext", "MolecularSignatures",
    "DrugCandidate", "DrugIdentity", "TargetEvidence", "TrialEvidence", "SafetyProfile", "ScoreBreakdown",
]
```

---

## TEST FILE: `service/test_service.py`

Test with the breast cancer profile from the report. This is your acceptance criteria:

```python
def test_breast_cancer_her2_positive():
    svc = DrugAgentService()
    req = DrugQueryRequest(
        disease="Breast Cancer",
        query_type=QueryType.FULL_RECOMMENDATION,
        genes=[
            GeneContext("ERBB2", 4.25, 0.05, "up", role="oncogenic_driver"),
            GeneContext("S100A7", 7.60, 0.05, "up", role="oncogenic_driver"),
            GeneContext("BRCA2", 1.67, 0.05, "up", role="tumor_suppressor"),
            GeneContext("FGFR4", 2.03, 0.05, "up", role="therapeutic_target"),
            GeneContext("CDC6", 1.50, 0.05, "up", role="therapeutic_target"),
            GeneContext("ESR1", -3.60, 0.05, "down", role="therapeutic_target"),
            GeneContext("MUC4", -4.25, 0.05, "down"),
            GeneContext("IL6", -3.67, 0.05, "down", role="immune_regulator"),
            GeneContext("BCL2", -1.50, 0.05, "down"),
        ],
        pathways=[
            PathwayContext("Regulation of Cell Cycle", "up", 7.59e-10, 42, category="Cell Cycle"),
            PathwayContext("Defective HRR due to BRCA2", "up", 7.51e-8, 10, category="DNA Repair"),
            PathwayContext("Immune system process", "up", 4.02e-8, 54, category="Immune System"),
            PathwayContext("PI3K-Akt signaling pathway", "down", 3.61e-8, 20, category="Signal Transduction"),
        ],
        biomarkers=[
            BiomarkerContext("HER2", "positive", ["ERBB2"]),
            BiomarkerContext("ER", "negative", ["ESR1"]),
            BiomarkerContext("PR", "negative", ["PGR"]),
        ],
        signatures=MolecularSignatures(proliferation=0.60, apoptosis=0.64, dna_repair=0.60, inflammation=0.68, immune_activation=0.60),
    )

    resp = svc.query(req)

    assert resp.success, f"Failed: {resp.errors}"

    # Trastuzumab: ERBB2 target upregulated + Phase 4 approved for HER2+ BC → should score >= 70
    tras = next((r for r in resp.recommendations if "TRASTUZUMAB" in r.identity.drug_name.upper()), None)
    assert tras is not None, "Trastuzumab missing from recommendations"
    assert tras.score.composite_score >= 70, f"Trastuzumab too low: {tras.score.composite_score}"

    # Tamoxifen/Anastrozole: ESR1 downregulated → should be contraindicated
    contra_names = [c.identity.drug_name.upper() for c in resp.contraindicated]
    assert any("TAMOXIFEN" in n for n in contra_names), "Tamoxifen should be contraindicated (ESR1 down)"

    # Venetoclax: BCL2 downregulated → should be contraindicated
    assert any("VENETOCLAX" in n for n in contra_names), "Venetoclax should be contraindicated (BCL2 down)"

    print(f"✅ {len(resp.recommendations)} drugs recommended, {len(resp.contraindicated)} contraindicated")
    print(f"   High priority: {[r.identity.drug_name for r in resp.high_priority]}")
    print(f"   Trastuzumab: {tras.score.composite_score}/100")
    for r in resp.high_priority[:5]:
        s = r.score
        print(f"   {r.identity.drug_name}: target={s.target_expression_match} trial={s.clinical_trial_evidence} ot={s.ot_association_score} fda={s.fda_regulatory_status} path={s.pathway_concordance} safety={s.safety_penalty} → {s.composite_score}")


if __name__ == "__main__":
    test_breast_cancer_her2_positive()
```

---

## CONSTRAINTS

- Python 3.10+, type hints on all signatures
- Synchronous only — no async/await (matches entire codebase)
- Use `ot_base.get_qdrant()` and `ot_base.get_embedder()` — do NOT create new Qdrant/embedder instances
- Only existing dependencies: `qdrant-client`, `sentence-transformers`, `python-dotenv`, `httpx`
- Use `logging` module for all operational logs (query timing, collection availability, scoring)
- Do NOT modify any existing files
- Do NOT hardcode drug names, gene names, or disease names in scoring logic
- Handle missing collections gracefully — if a collection doesn't exist, skip it and reduce evidence, don't crash

---

## STRATEGY REQUEST

Before writing code, answer these questions:

1. **File order** — Which file will you implement first and why?
2. **Qdrant client** — `get_qdrant()` returns a shared client. Will you call it once in `CollectionRouter.__init__` and pass references, or call it per-method?
3. **Embedding** — `get_embedder()` loads PubMedBERT (768-dim). How will you handle embedding caching for repeated queries within a single `_full_recommendation` call? (e.g., "ERBB2 inhibitor" searched in 3 collections)
4. **Drug name normalization** — ChEMBL uses "TRASTUZUMAB", ClinicalTrials uses "trastuzumab (Herceptin®)", FDA uses "HERCEPTIN". Walk me through your normalization strategy.
5. **Query volume** — A full recommendation with 5 upregulated targets could generate 50+ Qdrant queries across all stages. How will you manage latency?
6. **Missing data** — If ClinicalTrials collections are empty or a drug has no trial data, how does scoring degrade gracefully (not just return 0)?
7. **Ensembl cache** — The gene→Ensembl mapping loads ~9,500 entries at init. What's your plan if OpenTargets_data is unavailable?
8. **Testing** — How will you verify the scoring produces Trastuzumab ≥70 for the test case without hitting live Qdrant?

**Provide this strategy first. Do not write code until I approve.**
