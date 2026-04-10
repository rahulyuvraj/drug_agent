# Drug Agent Gap Closure Specification

## Problem Statement

The drug agent pipeline currently returns **16 drug candidates** for an SLE (Systemic Lupus Erythematosus) causal linkage query, while an equivalent analysis using k-dense (web-based AI co-scientist) returns **32 candidates**. This document specifies the architectural changes needed to close this gap.

### Current Agent Output (16 drugs)

| Drug | Score | Phase | FDA | Discovery |
|------|-------|-------|-----|-----------|
| ANIFROLUMAB-FNIA | 67.5 | 4 | Yes | gene, disease |
| BELIMUMAB | 54.5 | 4 | Yes | gene, disease |
| BARICITINIB | 50.9 | 4 | Yes | gene |
| Hydroxychloroquine | 50.7 | 4 | Yes | gene, disease |
| VOCLOSPORIN | 49.1 | 4 | Yes | gene, disease |
| PRAVASTATIN | 41.1 | 4 | Yes | gene |
| Prednisolone | 39.7 | 4 | Yes | disease |
| PREDNISONE | 35.7 | 4 | Yes | disease |
| Rinvoq | 39.2 | — | Yes | gene |
| METHYLPREDNISOLONE | 29.7 | 4 | Yes | disease |
| AZATHIOPRINE | 29.3 | 4 | Yes | disease |
| ATORVASTATIN | 23.6 | 4 | Yes | gene |
| RITUXIMAB | 17.4 | 4 | Yes | disease |
| CALCIFEDIOL | 15.6 | 4 | Yes | gene |
| ROSUVASTATIN | 14.7 | 4 | Yes | gene, disease |
| Mycophenolate | 14.6 | 4 | Yes | gene, disease |

### Missing Candidates (found by k-dense but not by agent)

| Drug | Target Gene | Mechanism | Phase | Why Missed |
|------|-------------|-----------|-------|------------|
| Deucravacitinib | TYK2 | Selective TYK2 inhibitor (JH2 domain) | Phase 3 SLE | ClinicalTrials collection gap |
| Obinutuzumab | CD20 | Anti-CD20 Type II mAb (ADCC) | FDA 2025 | FDA collection outdated |
| Telitacicept | BAFF/APRIL | Dual BLyS/APRIL inhibitor (TACI-Fc) | Phase 3 SLE | ClinicalTrials gap |
| Atacicept | BAFF/APRIL | TACI-Fc fusion (BAFF + APRIL) | Phase 2b SLE | ClinicalTrials gap |
| Dapirolizumab pegol | CD40L | Anti-CD40L PEGylated Fab | Phase 3 SLE | ClinicalTrials gap |
| Iscalimab | CD40 | Anti-CD40 mAb | Phase 2 LN | ClinicalTrials gap |
| Iberdomide (CC-220) | IKZF3 | Cereblon modulator → IKZF1/IKZF3 degradation | Phase 2 SLE | ChEMBL gene-drug not indexed |
| Lenalidomide | IKZF3 | Cereblon E3 ligase → IKZF3 ubiquitination | FDA (myeloma) | ChEMBL gene-drug not indexed |
| RWJ-445380 | CTSS | Selective cathepsin S inhibitor | Phase 2 | ChEMBL gene-drug not indexed |
| RO5459072 | CTSS | Selective cathepsin S inhibitor | Phase 2 | ChEMBL gene-drug not indexed |
| ASP1617 | CTSS | Potent cathepsin S inhibitor | Phase 1 | ChEMBL gene-drug not indexed |
| Tipifarnib | HRAS | Farnesyltransferase inhibitor | Investigational | Gene-to-drug mapping missing |
| Lonafarnib | HRAS | Farnesyltransferase inhibitor | FDA (progeria) | Gene-to-drug mapping missing |
| Tofacitinib | JAK1/JAK3 | Pan-JAK inhibitor | FDA (RA/UC) | Pathway traversal depth |
| Clozapine | DRD4 | Non-selective dopamine receptor antagonist | FDA (schizophrenia) | Gene-drug link missing |
| L-745870 | DRD4 | Selective DRD4 antagonist | Research tool | Gene-drug link missing |
| Lifitegrast | ICAM1 | LFA-1 antagonist → blocks ICAM1 adhesion | FDA (dry eye) | Indirect target not traversed |
| Efalizumab | ICAM1 | Anti-LFA-1 → blocks LFA-1/ICAM1 | Withdrawn (PML) | Indirect target not traversed |
| Eculizumab | C4B→C5 | Anti-C5 mAb → terminal complement blockade | FDA (PNH) | Pathway hop missing (C4B→C5) |
| Ravulizumab | C4B→C5 | Long-acting anti-C5 mAb | FDA (PNH), Phase 2 LN | Pathway hop missing |
| Avacopan | C3AR1 | Oral C5aR antagonist | FDA (ANCA vasculitis) | Tier 2 gene under-queried |
| Baminercept | LTBR | LTβR-Fc fusion protein | Phase 2 (Sjögren) | Gene-drug link missing |
| Alemtuzumab | CD52 | Anti-CD52 mAb → lymphocyte depletion | FDA (MS) | Gene-drug link missing |
| Sirolimus | HRAS→mTOR | mTOR inhibitor | FDA (transplant) | Pathway hop missing (HRAS→mTOR) |
| Low-dose IL-2 | IL12RB2 | Recombinant IL-2 → Treg expansion | Phase 2 SLE | Biological therapy not indexed |
| BIIB023 | TNFSF12 | Anti-TWEAK neutralizing antibody | Phase 2 LN | Tier 2 gene-drug gap |
| FT819 | CD19 CAR-T | iPSC-derived anti-CD19 CAR-T | Investigational | Novel modality not in DB |

---

## Root Cause Analysis

### Root Cause 1: `max_results=15` Hard Cap

**Location:** `config/drug_agent_config.yaml` → `output.max_recommendations: 15`

**Impact:** Even if the pipeline discovers 25+ valid candidates, only 15 are returned. The remaining are silently discarded at Stage 5 of `drug_agent_service.py`.

**Evidence:** The agent found 16 drugs (1 extra likely from gene_targeted_only), suggesting the pipeline may have had more candidates that were capped.

---

### Root Cause 2: Qdrant Collection Data Staleness

**Location:** All 15 Qdrant collections (ingested at build time, never refreshed)

**Impact:** The pipeline is entirely closed-loop — it can only discover drugs that were pre-indexed into Qdrant during initial data loading. If a gene-drug relationship, clinical trial, or FDA approval was not captured during ingestion, it is invisible to the pipeline forever.

**Specific collection gaps identified:**

| Collection | Missing Data | Drugs Lost |
|------------|-------------|------------|
| `ChEMBL_drugs` | CTSS inhibitors (RWJ-445380, RO5459072, ASP1617), HRAS farnesylation inhibitors (tipifarnib, lonafarnib), DRD4 antagonists (clozapine, L-745870), IKZF3 degraders (iberdomide, lenalidomide), LTBR biologics (baminercept), CD52 mAbs (alemtuzumab) | 13 drugs |
| `ClinicalTrials_summaries` | Active SLE pipeline trials: deucravacitinib Phase 3 (POETYK SLE-1/SLE-2), telitacicept Phase 3, dapirolizumab pegol Phase 3, iscalimab Phase 2, atacicept Phase 2b, low-dose IL-2 Phase 2 (LUPIL-2), BIIB023 Phase 2 | 7 drugs |
| `FDA_Drug_Labels` | Obinutuzumab 2025 lupus nephritis approval | 1 drug |
| `Drug_agent` (gene_drug) | Many Tier 1 causal genes (CTSS, DRD4, HRAS, ICAM1, LTBR, C4B) lack gene→drug document entries | ~10 drugs |

---

### Root Cause 3: No Pathway-Hop Drug Discovery

**Location:** `service/drug_agent_service.py` — Stage 1 (Candidate Discovery)

**Impact:** The agent queries drugs for the **exact causal gene** only. It does not traverse pathway edges to find drugs targeting functionally adjacent genes in the same signaling cascade.

**Examples of missed pathway hops:**

```
C4B (Tier 1 causal gene)
  └→ Complement cascade → C5 (druggable target)
       └→ Eculizumab, Ravulizumab (MISSED)

HRAS (Tier 1 causal gene)
  └→ MAPK → mTOR pathway (druggable target)
       └→ Sirolimus (MISSED)

ICAM1 (Tier 1 causal gene)
  └→ LFA-1/ICAM1 interaction → LFA-1 (druggable target)
       └→ Lifitegrast (MISSED)

GRB2 (Tier 1 causal gene)
  └→ JAK-STAT pathway → JAK1/JAK3 (druggable targets)
       └→ Tofacitinib (MISSED — only baricitinib found via JAK1/JAK2)
```

**Current code behavior:** The `_score_downstream_effector()` method in `drug_scorer.py` gives 60% credit for drugs targeting pathway neighbors, but this only applies during **scoring** (Stage 4). The **discovery** step (Stage 1) never queries for drugs targeting pathway neighbors — it only finds them if they happen to appear in disease-based or pathway-based search results.

---

### Root Cause 4: Causal Tier Not Mapped to Evidence Stratum

**Location:** `service/drug_scorer.py` — `gene_evidence_quality` multiplier

**Impact:** Genes from the causal linkage CSV have strong MR+eQTL+GWAS evidence (Tier 1 = full causal chain), but the scoring system assigns evidence strata based on `log2fc` magnitude and role annotations from the `GeneContext` input. Many causal genes have `log2fc = 0` in this dataset (because the CSV captures causal evidence, not patient-specific expression), causing them to receive `novel_candidate` stratum (0.5x multiplier) instead of `known_driver` (1.0x).

**Scoring multiplier table:**

| Stratum | Multiplier | When Assigned |
|---------|------------|---------------|
| `known_driver` | 1.0 | role="pathogenic" or role="driver" |
| `ppi_connected` | 0.85 | Gene found via PPI network |
| `expression_significant` | 0.65 | |log2fc| ≥ 0.58 |
| `novel_candidate` | 0.50 | Default / low evidence |

**Result:** Tier 1 causal genes with `log2fc = 0` get 0.5x multiplier on their target_direction score (18 × 0.5 = 9 instead of 18), which combined with modest clinical scores pushes drugs below the `composite_score ≥ 10` noise filter.

---

### Root Cause 5: Novel Modalities Not Represented in Collections

**Location:** `chembl/config/chembl_config.yaml` → `molecule_types`

**Current configuration:**
```yaml
molecule_types: [Small molecule, Antibody, Protein, Oligosaccharide, Enzyme]
```

**Missing modality types:**

| Modality | Missing Drugs | ChEMBL Type |
|----------|--------------|-------------|
| Cell therapy | FT819 (CAR-T) | Cell therapy |
| Protein degrader | Iberdomide, Lenalidomide (cereblon modulators) | Small molecule (but mechanism class missing) |
| Cytokine therapy | Low-dose IL-2 | Protein |
| Dual-target biologic | Telitacicept (TACI-Fc) | Protein |

---

## Implementation Specification

### Phase 1: Quick Wins (Priority: CRITICAL)

Estimated effort: 1-2 days. Closes ~40% of the gap.

#### 1.1 Increase max_results

**File:** `config/drug_agent_config.yaml`

**Change:**
```yaml
# BEFORE
output:
  max_recommendations: 15

# AFTER
output:
  max_recommendations: 30
```

**Rationale:** The pipeline may already discover 20+ drugs but discard them at Stage 5. Increasing the cap surfaces drugs that are currently being silently dropped.

**Testing:** Re-run the SLE query and verify more results appear. Compare scores of drugs at positions 16-30 to ensure they are above noise threshold.

---

#### 1.2 Map Causal Linkage Tier to Evidence Stratum

**File:** `service/drug_agent_service.py`

**Where:** Early in the `query()` method, after parsing the `DrugQueryRequest` and before Stage 1.

**Logic:** When processing `GeneContext` objects from the input, check if the gene has causal tier metadata. If so, override the evidence stratum assignment:

```python
# Add this mapping function
def _map_causal_tier_to_stratum(gene_context: GeneContext) -> str:
    """
    Map causal linkage tier from input CSV to evidence stratum.

    Tier 1 (Full Causal Chain: MR+eQTL+Disease+Pathway) → known_driver (1.0x)
    Tier 2 (Strong Causal: MR+eQTL+Disease) → known_driver (1.0x)
    Tier 3+ or unknown → use existing log2fc-based logic
    """
    causal_tier = getattr(gene_context, 'causal_tier', None)
    if causal_tier is None:
        # Fall back to existing stratum assignment logic
        return None

    tier_lower = str(causal_tier).lower()

    if 'tier 1' in tier_lower or 'tier 2' in tier_lower:
        return 'known_driver'
    elif 'tier 3' in tier_lower:
        return 'expression_significant'
    else:
        return None  # Use existing logic
```

**Schema change needed:** Add `causal_tier` field to `GeneContext` in `service/schemas.py`:

```python
class GeneContext(BaseModel):
    symbol: str
    log2fc: float
    pvalue: float
    direction: str  # "up" or "down"
    role: Optional[str] = None
    evidence_stratum: Optional[str] = None
    causal_tier: Optional[str] = None  # NEW FIELD
    # ... existing fields
```

**Impact:** Tier 1/2 genes with `log2fc = 0` will now receive full 1.0x scoring multiplier instead of 0.5x, preventing their drugs from falling below the noise threshold.

---

#### 1.3 Lower Noise Filter for High-Phase Drugs

**File:** `service/drug_agent_service.py` — Stage 5 filtering

**Where:** The noise filter that requires `composite_score ≥ 10`.

**Change:** For drugs with `clinical_regulatory > 15` (Phase 3+ with trial data), reduce the noise threshold:

```python
# BEFORE
if candidate.score_breakdown.composite_score < 10:
    continue  # Filter as noise

# AFTER
clinical_score = candidate.score_breakdown.clinical_regulatory
noise_threshold = 5 if clinical_score > 15 else 10
if candidate.score_breakdown.composite_score < noise_threshold:
    continue  # Filter as noise
```

**Rationale:** A Phase 3 drug with strong clinical evidence but weak gene-level scoring (due to indirect targets or missing log2fc data) should not be discarded. Clinical regulatory evidence alone is sufficient to justify inclusion.

---

#### 1.4 Add Causal Tier to Input Pipeline

**File:** The Streamlit app or API endpoint that constructs the `DrugQueryRequest` from the CSV.

**Change:** When parsing `sle_dag_causal_linkage.csv`, map the `causal_linkage_tier` column to `GeneContext.causal_tier`:

```python
# When building GeneContext from CSV rows:
gene = GeneContext(
    symbol=row['id'],
    log2fc=float(row['expression_log2fc']),
    pvalue=float(row.get('mr_pval', 1.0)),
    direction=row['expression_trend'].lower(),
    role=row.get('therapeutic_recommendation', None),
    causal_tier=row['causal_linkage_tier'],  # NEW: pass tier through
    evidence_stratum=None,  # Will be set by _map_causal_tier_to_stratum
)
```

---

### Phase 2: Live API Fallback Layer (Priority: HIGH)

Estimated effort: 3-5 days. Closes ~35% of the gap.

#### 2.1 ChEMBL Live Gene-to-Drug Lookup

**New file:** `service/live_api/chembl_live.py`

**Purpose:** When Qdrant's `ChEMBL_drugs` collection returns fewer than 5 hits for a gene target, query the ChEMBL REST API in real-time to find additional drug-target relationships.

**API flow:**

```
Step 1: Search for target
GET https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={gene_symbol}&format=json
→ Extract target_chembl_id where target_type = "SINGLE PROTEIN" and organism = "Homo sapiens"

Step 2: Get mechanisms for target
GET https://www.ebi.ac.uk/chembl/api/data/mechanism.json?target_chembl_id={id}&format=json
→ Extract molecule_chembl_id, mechanism_of_action, action_type

Step 3: Get molecule details
GET https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_chembl_id}.json
→ Extract pref_name, max_phase, first_approval, molecule_type
```

**Implementation:**

```python
import httpx
from typing import List, Dict, Optional
import asyncio
from functools import lru_cache

class ChEMBLLiveClient:
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, max_retries: int = 3, rate_limit_per_sec: int = 10):
        self.client = httpx.Client(timeout=30.0)
        self.max_retries = max_retries
        self._request_count = 0

    def find_drugs_for_gene(self, gene_symbol: str) -> List[Dict]:
        """
        Query ChEMBL REST API to find all drugs targeting a gene.

        Returns list of dicts:
        {
            "drug_name": str,
            "chembl_id": str,
            "mechanism_of_action": str,
            "action_type": str,  # INHIBITOR, ANTAGONIST, AGONIST, etc.
            "max_phase": int,    # 0-4
            "first_approval": int or None,
            "molecule_type": str,
            "source": "chembl_live_api"
        }
        """
        # Step 1: Find target
        targets = self._search_target(gene_symbol)
        if not targets:
            return []

        results = []
        seen_molecules = set()

        for target in targets:
            target_id = target.get("target_chembl_id")
            if not target_id:
                continue

            # Step 2: Get mechanisms
            mechanisms = self._get_mechanisms(target_id)

            for mech in mechanisms:
                mol_id = mech.get("molecule_chembl_id")
                if not mol_id or mol_id in seen_molecules:
                    continue
                seen_molecules.add(mol_id)

                # Step 3: Get molecule details
                molecule = self._get_molecule(mol_id)
                if not molecule:
                    continue

                pref_name = molecule.get("pref_name", mol_id)
                max_phase = molecule.get("max_phase", 0)

                # Skip Phase 0 / preclinical unless they have mechanism data
                if max_phase == 0 and not mech.get("mechanism_of_action"):
                    continue

                results.append({
                    "drug_name": pref_name,
                    "chembl_id": mol_id,
                    "mechanism_of_action": mech.get("mechanism_of_action", ""),
                    "action_type": mech.get("action_type", "UNKNOWN"),
                    "max_phase": max_phase,
                    "first_approval": molecule.get("first_approval"),
                    "molecule_type": molecule.get("molecule_type", "Unknown"),
                    "source": "chembl_live_api",
                    "target_gene": gene_symbol,
                })

        return results

    def _search_target(self, gene_symbol: str) -> List[Dict]:
        """Search ChEMBL for protein targets matching gene symbol."""
        url = f"{self.BASE_URL}/target/search.json"
        params = {
            "q": gene_symbol,
            "format": "json",
            "limit": 10,
        }
        resp = self._get(url, params)
        if not resp:
            return []

        targets = resp.get("targets", [])
        # Filter to human single-protein targets
        return [
            t for t in targets
            if t.get("organism") == "Homo sapiens"
            and t.get("target_type") == "SINGLE PROTEIN"
            and gene_symbol.upper() in [
                comp.get("accession", "").upper()
                for comp in t.get("target_components", [])
            ] + [gene_symbol.upper()]
        ]

    def _get_mechanisms(self, target_chembl_id: str) -> List[Dict]:
        """Get drug mechanisms for a target."""
        url = f"{self.BASE_URL}/mechanism.json"
        params = {
            "target_chembl_id": target_chembl_id,
            "format": "json",
            "limit": 100,
        }
        resp = self._get(url, params)
        return resp.get("mechanisms", []) if resp else []

    def _get_molecule(self, molecule_chembl_id: str) -> Optional[Dict]:
        """Get molecule details."""
        url = f"{self.BASE_URL}/molecule/{molecule_chembl_id}.json"
        return self._get(url)

    def _get(self, url: str, params: dict = None) -> Optional[Dict]:
        """HTTP GET with retry logic."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.get(url, params=params)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    import time
                    time.sleep(1.0 * (attempt + 1))
                    continue
            except httpx.RequestError:
                continue
        return None
```

**Integration point in `collection_router.py`:**

```python
def find_drugs_for_target(self, gene_symbol: str, top_k: int = 10) -> List[DrugHit]:
    # Existing Qdrant search
    qdrant_results = self._search_qdrant_for_gene(gene_symbol, top_k)

    # NEW: If Qdrant returns fewer than 5 hits, query ChEMBL live
    if len(qdrant_results) < 5:
        live_client = ChEMBLLiveClient()
        live_results = live_client.find_drugs_for_gene(gene_symbol)

        # Convert to DrugHit format and merge
        for lr in live_results:
            if lr["drug_name"] not in {r.drug_name for r in qdrant_results}:
                qdrant_results.append(DrugHit(
                    drug_name=lr["drug_name"],
                    chembl_id=lr["chembl_id"],
                    score=0.45,  # Default relevance for live API hits
                    source="chembl_live_api",
                    mechanism_of_action=lr["mechanism_of_action"],
                    action_type=lr["action_type"],
                    max_phase=lr["max_phase"],
                    target_gene=gene_symbol,
                ))

    return qdrant_results
```

**Drugs recovered:** CTSS inhibitors (RWJ-445380, RO5459072, ASP1617), HRAS inhibitors (tipifarnib, lonafarnib), DRD4 antagonists (clozapine), IKZF3 degraders (iberdomide, lenalidomide), LTBR biologics (baminercept), CD52 mAbs (alemtuzumab).

---

#### 2.2 ClinicalTrials.gov Live Disease Search

**New file:** `service/live_api/clinicaltrials_live.py`

**Purpose:** Query ClinicalTrials.gov API v2 for active Phase 2/3 trials for the target disease. This catches pipeline drugs not yet in the Qdrant `ClinicalTrials_summaries` collection.

**API endpoint:** `https://clinicaltrials.gov/api/v2/studies`

**Implementation:**

```python
import httpx
from typing import List, Dict
import re

class ClinicalTrialsLiveClient:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self):
        self.client = httpx.Client(timeout=30.0)

    def find_trials_for_disease(
        self,
        disease: str,
        disease_aliases: List[str] = None,
        phases: List[str] = None,
        statuses: List[str] = None,
        max_results: int = 50,
    ) -> List[Dict]:
        """
        Search ClinicalTrials.gov for drug interventional trials.

        Returns list of dicts:
        {
            "nct_id": str,
            "title": str,
            "phase": str,
            "status": str,
            "conditions": List[str],
            "interventions": List[str],  # Drug names
            "enrollment": int,
            "start_date": str,
            "source": "clinicaltrials_live_api"
        }
        """
        if phases is None:
            phases = ["PHASE2", "PHASE3", "PHASE4"]
        if statuses is None:
            statuses = [
                "RECRUITING",
                "ACTIVE_NOT_RECRUITING",
                "COMPLETED",
                "ENROLLING_BY_INVITATION",
            ]

        all_terms = [disease] + (disease_aliases or [])
        all_results = []
        seen_nct = set()

        for term in all_terms[:5]:  # Limit alias expansion
            params = {
                "query.cond": term,
                "query.intr": "DRUG",  # Only drug interventions
                "filter.phase": ",".join(phases),
                "filter.overallStatus": ",".join(statuses),
                "pageSize": min(max_results, 100),
                "fields": "NCTId,BriefTitle,Phase,OverallStatus,"
                          "Condition,InterventionName,InterventionType,"
                          "EnrollmentCount,StartDate",
                "format": "json",
            }

            try:
                resp = self.client.get(self.BASE_URL, params=params)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                studies = data.get("studies", [])

                for study in studies:
                    protocol = study.get("protocolSection", {})
                    id_module = protocol.get("identificationModule", {})
                    nct_id = id_module.get("nctId", "")

                    if nct_id in seen_nct:
                        continue
                    seen_nct.add(nct_id)

                    # Extract interventions (drugs only)
                    arms_module = protocol.get("armsInterventionsModule", {})
                    interventions = arms_module.get("interventions", [])
                    drug_names = [
                        intv.get("name", "")
                        for intv in interventions
                        if intv.get("type", "").upper() in ("DRUG", "BIOLOGICAL")
                    ]

                    if not drug_names:
                        continue

                    design_module = protocol.get("designModule", {})
                    status_module = protocol.get("statusModule", {})
                    conditions_module = protocol.get("conditionsModule", {})

                    all_results.append({
                        "nct_id": nct_id,
                        "title": id_module.get("briefTitle", ""),
                        "phase": ",".join(design_module.get("phases", [])),
                        "status": status_module.get("overallStatus", ""),
                        "conditions": conditions_module.get("conditions", []),
                        "interventions": drug_names,
                        "enrollment": protocol.get("designModule", {})
                                             .get("enrollmentInfo", {})
                                             .get("count", 0),
                        "start_date": status_module.get("startDateStruct", {})
                                                   .get("date", ""),
                        "source": "clinicaltrials_live_api",
                    })
            except httpx.RequestError:
                continue

        return all_results

    def extract_unique_drugs(self, trials: List[Dict]) -> List[Dict]:
        """
        Extract unique drug names from trial results with their highest phase.

        Returns list of dicts:
        {
            "drug_name": str,
            "max_phase": int,
            "trial_count": int,
            "total_enrollment": int,
            "nct_ids": List[str],
            "source": "clinicaltrials_live_api"
        }
        """
        drug_map = {}

        for trial in trials:
            phase_num = self._phase_to_int(trial["phase"])

            for drug_name in trial["interventions"]:
                normalized = self._normalize_drug_name(drug_name)
                if normalized in ("placebo", "saline", "sham", ""):
                    continue

                if normalized not in drug_map:
                    drug_map[normalized] = {
                        "drug_name": drug_name,  # Keep original casing
                        "max_phase": phase_num,
                        "trial_count": 0,
                        "total_enrollment": 0,
                        "nct_ids": [],
                    }

                entry = drug_map[normalized]
                entry["max_phase"] = max(entry["max_phase"], phase_num)
                entry["trial_count"] += 1
                entry["total_enrollment"] += trial.get("enrollment", 0)
                entry["nct_ids"].append(trial["nct_id"])

        return [
            {**v, "source": "clinicaltrials_live_api"}
            for v in sorted(
                drug_map.values(),
                key=lambda x: (x["max_phase"], x["trial_count"]),
                reverse=True,
            )
        ]

    def _phase_to_int(self, phase_str: str) -> int:
        if "PHASE4" in phase_str or "Phase 4" in phase_str:
            return 4
        if "PHASE3" in phase_str or "Phase 3" in phase_str:
            return 3
        if "PHASE2" in phase_str or "Phase 2" in phase_str:
            return 2
        if "PHASE1" in phase_str or "Phase 1" in phase_str:
            return 1
        return 0

    def _normalize_drug_name(self, name: str) -> str:
        name = re.sub(r'\s*\(.*?\)', '', name)
        name = re.sub(r'\s+(sodium|hydrochloride|sulfate|mesylate|maleate)$',
                       '', name, flags=re.IGNORECASE)
        return name.strip().lower()
```

**Integration point in `drug_agent_service.py` Stage 1:**

```python
# After existing disease-based discovery
from service.live_api.clinicaltrials_live import ClinicalTrialsLiveClient

ct_client = ClinicalTrialsLiveClient()
live_trials = ct_client.find_trials_for_disease(
    disease=request.disease,
    disease_aliases=disease_aliases,
    phases=["PHASE2", "PHASE3"],
)
live_drugs = ct_client.extract_unique_drugs(live_trials)

# Merge live drugs into discovery results
for drug_info in live_drugs:
    normalized = normalize_drug_name(drug_info["drug_name"])
    if normalized not in seen_drugs:
        discovery_results[f"live_ct_{normalized}"] = [DrugHit(
            drug_name=drug_info["drug_name"],
            score=0.50,  # Default relevance for live CT hits
            source="clinicaltrials_live_api",
            max_phase=drug_info["max_phase"],
            trial_count=drug_info["trial_count"],
        )]
        seen_drugs.add(normalized)
```

**Drugs recovered:** Deucravacitinib, telitacicept, atacicept, dapirolizumab pegol, iscalimab, low-dose IL-2, BIIB023.

---

#### 2.3 OpenTargets GraphQL Live Lookup

**New file:** `service/live_api/opentargets_live.py`

**Purpose:** Query OpenTargets Platform GraphQL API for known drugs targeting a specific gene. Catches drugs that were not indexed in the Qdrant `OpenTargets_drugs_enriched` collection.

**API endpoint:** `https://api.platform.opentargets.org/api/v4/graphql`

**Implementation:**

```python
import httpx
from typing import List, Dict, Optional

class OpenTargetsLiveClient:
    GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self):
        self.client = httpx.Client(timeout=30.0)

    def find_drugs_for_gene(self, gene_symbol: str) -> List[Dict]:
        """
        Query OpenTargets for all drugs targeting a gene.

        Step 1: Resolve gene symbol → Ensembl ID
        Step 2: Query knownDrugs for that target

        Returns list of dicts:
        {
            "drug_name": str,
            "drug_id": str (ChEMBL ID),
            "mechanism_of_action": str,
            "action_type": str,
            "phase": int,
            "indication": str,
            "source": "opentargets_live_api"
        }
        """
        # Step 1: Search for gene
        ensembl_id = self._resolve_gene(gene_symbol)
        if not ensembl_id:
            return []

        # Step 2: Query known drugs
        query = """
        query KnownDrugs($ensemblId: String!, $size: Int!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            knownDrugs(size: $size) {
              uniqueDrugs
              rows {
                drug {
                  id
                  name
                  drugType
                  maximumClinicalTrialPhase
                  hasBeenWithdrawn
                  isApproved
                }
                mechanismOfAction
                actionType
                phase
                disease {
                  id
                  name
                }
                urls {
                  niceName
                  url
                }
              }
            }
          }
        }
        """

        variables = {
            "ensemblId": ensembl_id,
            "size": 100,
        }

        try:
            resp = self.client.post(
                self.GRAPHQL_URL,
                json={"query": query, "variables": variables},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            target = data.get("data", {}).get("target")
            if not target or not target.get("knownDrugs"):
                return []

            results = []
            seen_drugs = set()

            for row in target["knownDrugs"]["rows"]:
                drug = row.get("drug", {})
                drug_name = drug.get("name", "")
                drug_id = drug.get("id", "")

                if drug_name.lower() in seen_drugs:
                    continue
                seen_drugs.add(drug_name.lower())

                results.append({
                    "drug_name": drug_name,
                    "drug_id": drug_id,
                    "mechanism_of_action": row.get("mechanismOfAction", ""),
                    "action_type": row.get("actionType", "UNKNOWN"),
                    "phase": row.get("phase") or drug.get("maximumClinicalTrialPhase", 0),
                    "is_approved": drug.get("isApproved", False),
                    "withdrawn": drug.get("hasBeenWithdrawn", False),
                    "drug_type": drug.get("drugType", ""),
                    "indication": (row.get("disease") or {}).get("name", ""),
                    "target_gene": gene_symbol,
                    "source": "opentargets_live_api",
                })

            return results
        except httpx.RequestError:
            return []

    def _resolve_gene(self, gene_symbol: str) -> Optional[str]:
        """Resolve gene symbol to Ensembl ID via OpenTargets search."""
        query = """
        query GeneSearch($queryString: String!) {
          search(queryString: $queryString, entityNames: ["target"], page: {size: 5, index: 0}) {
            hits {
              id
              entity
              name
              description
            }
          }
        }
        """

        try:
            resp = self.client.post(
                self.GRAPHQL_URL,
                json={"query": query, "variables": {"queryString": gene_symbol}},
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            hits = data.get("data", {}).get("search", {}).get("hits", [])

            for hit in hits:
                if hit.get("entity") == "target":
                    # Verify the gene symbol matches
                    if gene_symbol.upper() in hit.get("name", "").upper():
                        return hit["id"]

            # Fallback: return first target hit
            for hit in hits:
                if hit.get("entity") == "target":
                    return hit["id"]

            return None
        except httpx.RequestError:
            return None
```

**Integration:** Same pattern as ChEMBL live — trigger when Qdrant returns <5 hits for a gene.

---

#### 2.4 Fallback Orchestration

**File:** `service/collection_router.py` — modify `find_drugs_for_target()`

**Logic:** Orchestrate all three live API fallbacks with a unified trigger:

```python
LIVE_API_THRESHOLD = 5  # Trigger live APIs when Qdrant returns fewer hits

def find_drugs_for_target(self, gene_symbol: str, top_k: int = 10) -> List[DrugHit]:
    # 1. Existing Qdrant search (primary)
    qdrant_results = self._search_qdrant_for_gene(gene_symbol, top_k)

    # 2. Live API fallback (when Qdrant coverage is thin)
    if len(qdrant_results) < LIVE_API_THRESHOLD:
        live_hits = self._query_live_apis_for_gene(gene_symbol)

        existing_names = {self._normalize(r.drug_name) for r in qdrant_results}
        for hit in live_hits:
            if self._normalize(hit.drug_name) not in existing_names:
                qdrant_results.append(hit)
                existing_names.add(self._normalize(hit.drug_name))

    return qdrant_results

def _query_live_apis_for_gene(self, gene_symbol: str) -> List[DrugHit]:
    """Query ChEMBL + OpenTargets live APIs in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    from service.live_api.chembl_live import ChEMBLLiveClient
    from service.live_api.opentargets_live import OpenTargetsLiveClient

    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        chembl_future = executor.submit(
            ChEMBLLiveClient().find_drugs_for_gene, gene_symbol
        )
        ot_future = executor.submit(
            OpenTargetsLiveClient().find_drugs_for_gene, gene_symbol
        )

        # Convert ChEMBL results
        for hit in chembl_future.result():
            results.append(DrugHit(
                drug_name=hit["drug_name"],
                chembl_id=hit.get("chembl_id"),
                score=0.45,
                source="chembl_live_api",
                mechanism_of_action=hit.get("mechanism_of_action"),
                action_type=hit.get("action_type"),
                max_phase=hit.get("max_phase", 0),
                target_gene=gene_symbol,
            ))

        # Convert OpenTargets results
        for hit in ot_future.result():
            results.append(DrugHit(
                drug_name=hit["drug_name"],
                chembl_id=hit.get("drug_id"),
                score=0.45,
                source="opentargets_live_api",
                mechanism_of_action=hit.get("mechanism_of_action"),
                action_type=hit.get("action_type"),
                max_phase=hit.get("phase", 0),
                target_gene=gene_symbol,
            ))

    return results
```

---

### Phase 3: Pathway-Hop Drug Discovery (Priority: HIGH)

Estimated effort: 2-3 days. Closes ~20% of the gap.

#### 3.1 Add Pathway Neighbor Drug Search to Stage 1

**File:** `service/drug_agent_service.py` — Stage 1 (Candidate Discovery)

**Current behavior:** For each causal gene, query `find_drugs_for_target(gene)` only.

**New behavior:** After querying the direct gene, also query drugs for the gene's top pathway neighbors.

**Implementation:**

```python
# Add this method to DrugAgentService

def _discover_pathway_hop_drugs(
    self,
    gene_symbol: str,
    gene_direction: str,
    causal_tier: str,
    max_neighbors: int = 5,
) -> Dict[str, List[DrugHit]]:
    """
    Find drugs targeting pathway neighbors of a causal gene.

    Strategy:
    1. Query Raw_csv_KG for PPI/pathway edges from gene_symbol
    2. For top N neighbors, query find_drugs_for_target()
    3. Tag results as pathway_hop with ppi_connected stratum

    This catches cases like:
    - C4B → complement cascade → C5 → eculizumab
    - HRAS → MAPK → mTOR → sirolimus
    - ICAM1 → LFA-1/ICAM1 → LFA-1 → lifitegrast
    """
    # Step 1: Get pathway neighbors from knowledge graph
    neighbors = self.router.get_functionally_related_genes(gene_symbol)

    if not neighbors:
        return {}

    # Step 2: Prioritize neighbors by druggability
    # Known druggable gene families get priority
    druggable_prefixes = {
        'JAK', 'TYK', 'STAT', 'IL', 'TNF', 'CD', 'HLA',
        'EGFR', 'VEGF', 'mTOR', 'MAPK', 'BTK', 'SYK',
        'CTLA', 'PD', 'IFNAR', 'CSF', 'CXCR', 'CCR',
    }

    def druggability_score(gene: str) -> int:
        for prefix in druggable_prefixes:
            if gene.upper().startswith(prefix):
                return 2
        return 1

    sorted_neighbors = sorted(
        neighbors[:max_neighbors * 2],
        key=lambda g: druggability_score(g),
        reverse=True,
    )[:max_neighbors]

    # Step 3: Query drugs for each neighbor
    hop_results = {}
    for neighbor_gene in sorted_neighbors:
        if neighbor_gene == gene_symbol:
            continue

        hits = self.router.find_drugs_for_target(neighbor_gene, top_k=5)

        if hits:
            # Tag as pathway hop
            for hit in hits:
                hit.source = f"pathway_hop:{gene_symbol}→{neighbor_gene}"
                hit.evidence_stratum = "ppi_connected"
                hit.discovery_gene = gene_symbol  # Original causal gene
                hit.hop_gene = neighbor_gene       # Actual drug target

            hop_results[f"hop_{gene_symbol}_{neighbor_gene}"] = hits

    return hop_results
```

**Integration in Stage 1:**

```python
# After direct gene queries, add pathway hop discovery
# Only for Tier 1/2 genes (high-confidence causal chain)
for gene_ctx in request.genes:
    tier = getattr(gene_ctx, 'causal_tier', '')
    if 'tier 1' in str(tier).lower() or 'tier 2' in str(tier).lower():
        hop_drugs = self._discover_pathway_hop_drugs(
            gene_symbol=gene_ctx.symbol,
            gene_direction=gene_ctx.direction,
            causal_tier=tier,
            max_neighbors=3,  # Top 3 neighbors per gene
        )
        discovery_results.update(hop_drugs)
```

**Drugs recovered:** Eculizumab, ravulizumab (C4B→C5), sirolimus (HRAS→mTOR), lifitegrast (ICAM1→LFA1), tofacitinib (GRB2→JAK1/JAK3).

---

#### 3.2 Adjust Pathway-Hop Scoring

**File:** `service/drug_scorer.py` — `_score_target_direction()`

**Change:** Currently, downstream effectors get 60% credit. For pathway-hop discoveries where the hop gene is itself a well-characterized drug target, increase to 85%:

```python
# In _score_target_direction(), when evaluating downstream effector match:

# BEFORE
effector_credit = 0.60

# AFTER
# If the hop gene is in a known druggable family or has ≥3 drugs in ChEMBL,
# give higher credit since the drug-target relationship is well-validated
if hop_gene_is_established_target:
    effector_credit = 0.85  # High confidence indirect target
else:
    effector_credit = 0.60  # Standard indirect target
```

---

### Phase 4: Data Refresh & Expansion (Priority: MEDIUM)

Estimated effort: 2-3 days. Closes remaining ~5% of gap.

#### 4.1 Expand ChEMBL Molecule Types

**File:** `chembl/config/chembl_config.yaml`

**Change:**
```yaml
# BEFORE
molecule_types: [Small molecule, Antibody, Protein, Oligosaccharide, Enzyme]

# AFTER
molecule_types:
  - Small molecule
  - Antibody
  - Protein
  - Oligosaccharide
  - Enzyme
  - Cell therapy
  - Unknown
  - Oligonucleotide
```

**Action:** Re-run ChEMBL ingestion pipeline after this config change.

---

#### 4.2 Refresh ClinicalTrials Collection

**Action:** Re-download ClinicalTrials.gov data with the following filters:
- Conditions: SLE, Lupus, Lupus Nephritis
- Date filter: studies updated after 2023-01-01
- Phases: Phase 1, 2, 3, 4
- Status: Recruiting, Active, Completed

**Key trials to ensure are indexed:**

| NCT ID | Drug | Phase | Status |
|--------|------|-------|--------|
| NCT05617677 | Deucravacitinib | Phase 3 | Active |
| NCT05089045 | Deucravacitinib | Phase 3 | Active |
| NCT04082416 | Telitacicept | Phase 3 | Completed |
| NCT04294667 | Dapirolizumab pegol | Phase 3 | Active |
| NCT03610516 | Iberdomide | Phase 2 | Completed |
| NCT05765006 | Iscalimab | Phase 2 | Active |
| NCT03707781 | Low-dose IL-2 | Phase 2 | Completed |
| NCT04306926 | BIIB023 | Phase 2 | Completed |
| NCT05765006 | Obinutuzumab | Phase 3 | Active |

---

#### 4.3 Refresh FDA Drug Labels

**Action:** Re-download FDA drug labels to capture recent approvals:
- Obinutuzumab (Gazyva) — lupus nephritis indication added 2025

---

#### 4.4 Add Novel Modality Handling

**File:** `service/drug_scorer.py` — `_score_clinical_regulatory()`

**Change:** Add scoring support for novel modality types that may not have traditional phase classifications:

```python
# Add modality-aware scoring
NOVEL_MODALITIES = {"Cell therapy", "Gene therapy", "Oligonucleotide"}

if candidate.drug_type in NOVEL_MODALITIES:
    # Novel modalities often have accelerated pathways
    # Don't penalize for lower phase numbers
    if candidate.max_phase >= 2:
        clinical_score = max(clinical_score, 12.0)  # Floor at Phase 2 equivalent
```

---

## Testing Strategy

### Unit Tests

1. **Live API clients:** Mock HTTP responses, verify parsing logic
2. **Pathway-hop discovery:** Verify neighbor selection, druggability ranking
3. **Causal tier mapping:** Verify Tier 1/2 → `known_driver` stratum
4. **Noise filter relaxation:** Verify Phase 3+ drugs survive with lower threshold

### Integration Tests

1. **SLE benchmark test:** Run full pipeline on `sle_dag_causal_linkage.csv` and verify:
   - ≥28 unique drug candidates returned (target: 30)
   - All 16 current drugs still present (no regression)
   - At least 12 of the 16 missing drugs now appear
   - SOC drugs (anifrolumab, belimumab, voclosporin) remain top-ranked

2. **Live API fallback test:** Disconnect Qdrant collections for CTSS and verify:
   - ChEMBL live API returns RWJ-445380, RO5459072
   - Results merge correctly with Qdrant results

3. **Pathway-hop test:** Query with C4B gene and verify:
   - Complement C5 identified as pathway neighbor
   - Eculizumab/ravulizumab found via hop
   - Scored at 85% credit (ppi_connected stratum)

### Acceptance Criteria

| Criterion | Target |
|-----------|--------|
| Total drug candidates for SLE query | ≥ 28 |
| Tier 1 gene coverage (drugs found for ≥ 80% of Tier 1 genes) | ≥ 80% |
| No regression on existing 16 drugs | 100% retained |
| Top 5 drugs by score remain clinically appropriate SOC | Yes |
| Pipeline latency (with live API fallback) | < 120 seconds |
| Live API error handling (graceful degradation) | No crashes on API timeout |

---

## File Change Summary

| File | Change Type | Phase |
|------|-------------|-------|
| `config/drug_agent_config.yaml` | Edit: max_recommendations 15→30 | Phase 1 |
| `service/schemas.py` | Edit: Add `causal_tier` to GeneContext | Phase 1 |
| `service/drug_agent_service.py` | Edit: Tier-to-stratum mapping, pathway hop discovery, live CT integration | Phase 1, 2, 3 |
| `service/drug_scorer.py` | Edit: Noise filter relaxation, pathway-hop credit, novel modality scoring | Phase 1, 3, 4 |
| `service/collection_router.py` | Edit: Live API fallback orchestration | Phase 2 |
| `service/live_api/__init__.py` | New file | Phase 2 |
| `service/live_api/chembl_live.py` | New file: ChEMBL REST API client | Phase 2 |
| `service/live_api/clinicaltrials_live.py` | New file: ClinicalTrials.gov API v2 client | Phase 2 |
| `service/live_api/opentargets_live.py` | New file: OpenTargets GraphQL client | Phase 2 |
| `chembl/config/chembl_config.yaml` | Edit: Expand molecule_types | Phase 4 |
| Streamlit app (input parser) | Edit: Pass causal_tier from CSV to GeneContext | Phase 1 |

---

## Dependency Changes

Add to `requirements.txt` or `pyproject.toml`:

```
httpx>=0.27.0  # For async-capable HTTP client (live API calls)
```

Note: `httpx` is used instead of `requests` because it supports both sync and async patterns, enabling future migration to async live API calls for better concurrency.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Live API rate limiting (ChEMBL, CT.gov) | Medium | Low | Exponential backoff + caching; graceful fallback to Qdrant-only |
| Increased latency from live APIs | Medium | Medium | Parallel execution; 15s timeout per API; cache results in Qdrant for future queries |
| Pathway-hop producing false positives | Low | Medium | Only apply to Tier 1/2 genes; cap at 3 neighbors; require established drug-target link |
| Regression in existing drug ranking | Low | High | Benchmark test: verify all 16 current drugs retained with similar relative ordering |
| OpenTargets API schema changes | Low | Low | Version-pin GraphQL queries; monitor for breaking changes |

---

## Glossary

| Term | Definition |
|------|------------|
| **Causal Linkage Tier** | Classification from MR+eQTL+GWAS analysis indicating strength of gene→disease causal chain |
| **Evidence Stratum** | Internal scoring multiplier (1.0x for known_driver, 0.5x for novel_candidate) |
| **Pathway Hop** | Drug discovery via functionally adjacent gene in same pathway (1-hop PPI/pathway edge) |
| **SOC** | Standard of Care — FDA-approved drug indicated for the target disease |
| **Live API Fallback** | Real-time REST/GraphQL query to external databases when Qdrant coverage is insufficient |
| **Noise Filter** | Minimum composite_score threshold (default 10) to exclude low-evidence candidates |
| **MR** | Mendelian Randomization — causal inference method using genetic variants as instruments |
| **eQTL** | Expression Quantitative Trait Locus — genetic variant associated with gene expression |
