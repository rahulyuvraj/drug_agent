# Drug Agent v2.1 Specification — Code Snippets Audit


---

## Design Philosophy: Traceable Reasoning

### Spec Requirement:
Every drug recommendation must include a per-drug reasoning trace that explains *why* the drug was selected — linking drug action, patient gene expression, clinical evidence, and pathway concordance.

### Code Location:
File: `drug_agent/service/drug_scorer.py`
Function: `_tier_reasoning()`
Lines: 111–162

### Actual Code:
```python
def _tier_reasoning(self, s: ScoreBreakdown, candidate: DrugCandidate) -> str:
    """Build Q1→Q2→Q3 decision-tree explanation of why a drug lands in its tier."""
    # SOC override path
    if getattr(candidate, 'is_soc_candidate', False):
        return ("Guideline-recommended backbone therapy — "
                "retained as Standard-of-Care regardless of molecular scoring.")

    from agentic_ai_wf.reporting_pipeline_agent.core_types import (
        DRUG_HIGH_PRIORITY_THRESHOLD, DRUG_MODERATE_PRIORITY_THRESHOLD,
    )
    parts = []
    # Q1: molecular concordance
    if s.target_direction_match > 0:
        parts.append(f"Q1 Concordant: drug action aligns with patient gene expression "
                     f"(Direction +{s.target_direction_match:.0f}).")
    else:
        parts.append("Q1 No direct concordance: drug target not differentially expressed "
                     "in patient transcriptome.")

    # Q2: clinical / regulatory backing
    if s.disease_indication_bonus > 0:
        parts.append(f"Q2 FDA-approved for this indication (+{s.disease_indication_bonus:.0f}).")
    elif s.clinical_regulatory_score > 0:
        parts.append(f"Q2 Clinical evidence supports use "
                     f"(Clinical +{s.clinical_regulatory_score:.0f}).")
    else:
        parts.append("Q2 No clinical/regulatory evidence for this disease context.")

    # Q3: supporting evidence breadth
    supporting = []
    if s.ot_association_score > 0:
        supporting.append(f"OpenTargets +{s.ot_association_score:.0f}")
    if s.pathway_concordance > 0:
        supporting.append(f"Pathway +{s.pathway_concordance:.0f}")
    if s.target_magnitude_match > 0:
        supporting.append(f"Magnitude +{s.target_magnitude_match:.0f}")
    if supporting:
        parts.append(f"Q3 Supporting evidence: {', '.join(supporting)}.")
    else:
        parts.append("Q3 No additional supporting evidence.")

    # Tier conclusion
    score = s.composite_score
    if score >= DRUG_HIGH_PRIORITY_THRESHOLD:
        parts.append(f"→ High Priority (score {score:.0f} ≥ {DRUG_HIGH_PRIORITY_THRESHOLD}).")
    elif score >= DRUG_MODERATE_PRIORITY_THRESHOLD:
        parts.append(f"→ Moderate Priority (score {score:.0f}; "
                     f"{DRUG_MODERATE_PRIORITY_THRESHOLD}–{DRUG_HIGH_PRIORITY_THRESHOLD - 1}).")
    else:
        parts.append(f"→ Below threshold (score {score:.0f} < {DRUG_MODERATE_PRIORITY_THRESHOLD}).")

    return ' '.join(parts)
```

Additional location — `ScoreBreakdown.tier_reasoning` field:

File: `drug_agent/service/schemas.py`
Lines: 168–179
```python
@dataclass
class ScoreBreakdown:
    target_direction_match: float = 0.0
    target_magnitude_match: float = 0.0
    clinical_regulatory_score: float = 0.0
    ot_association_score: float = 0.0
    pathway_concordance: float = 0.0
    safety_penalty: float = 0.0
    disease_indication_bonus: float = 0.0
    composite_score: float = 0.0
    pipeline_evidence_used: bool = False
    disease_relevant: bool = True
    tier_reasoning: str = ""   # Q1→Q2→Q3 decision-tree explanation of tier placement
```

---

## Section 1: Input Data Contract

### Spec Requirement:
`DrugQueryRequest` must accept: `disease_name`, `patient_deg_table` (Gene, Patient_LFC_mean/log2FC, adj.p-value), `pathway_enrichment_table` (Pathway name, FDR, direction, gene count), `cibersort_table` (Cell type, enrichment score, presence %).

### Code Location:
File: `drug_agent/service/schemas.py`
Function: `DrugQueryRequest` dataclass
Lines: 77–106

### Actual Code:
```python
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
    max_results: int = 15
    include_safety: bool = True
    include_trials: bool = True
    scoring_config: Optional[ScoringConfig] = None
    disease_context: Optional[str] = None
    disease_aliases: List[str] = field(default_factory=list)
    all_patient_genes: List[GeneContext] = field(default_factory=list)
    signature_scores: Optional[Dict] = None

@dataclass
class GeneContext:
    gene_symbol: str
    log2fc: float
    adj_p_value: float
    direction: str                                    # "up" | "down"
    role: Optional[str] = None
    composite_score: Optional[float] = None
    evidence_stratum: Optional[str] = None

@dataclass
class PathwayContext:
    pathway_name: str
    direction: str
    fdr: float
    gene_count: int
    category: Optional[str] = None
    key_genes: Optional[List[str]] = None
    disease_relevance: Optional[str] = None
```

The `build_drug_query_request()` call from `narrative_generation.py` (lines 569–577) passes gene and pathway data to the service.

File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 569–577
```python
drug_request = build_drug_query_request(
    disease, gene_mappings, pathway_mappings,
    biomarker_concordance, signature_scores, xcell_findings,
    gene_validation_results=self.gene_validation_results,
    multi_evidence_gene_results=self.multi_evidence_gene_results,
    multi_evidence_pathway_results=self.multi_evidence_pathway_results,
    strata_summary=getattr(self, 'strata_summary', None),
    dynamic_disease_context=self.dynamic_disease_context,
)
```

---

## Section 1.1: DEG Classification Thresholds

### Spec Requirement:
log2FC ≥ +0.58 = upregulated, ≤ -0.58 = downregulated, adj.p < 0.05 = significant.

### Code Location:
File: `reporting_pipeline_agent/core_types.py`
Lines: 47–50

### Actual Code:
```python
DEG_ADJ_PVALUE_THRESHOLD = 0.05  # Statistical significance threshold (FDR-corrected)
DEG_LOG2FC_THRESHOLD = 0.58     # Minimum |log2FC| for "Significant DEG" (1.5-fold)
DEG_HIGH_CONFIDENCE_LOG2FC = 1.0  # High confidence effect size threshold (2-fold)
DEG_TREND_THRESHOLD = 0.3       # Minimum |log2FC| for "Trend DEG" (below this = noise)
```

File: `drug_agent/service/drug_scorer.py`
Function: `_target_direction()`
Lines: 260–286 — threshold used in scoring:
```python
def _target_direction(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
    """Direction concordance: drug action_type vs patient gene expression."""
    w = self.config.target_direction_weight
    if not c.targets or not r.genes:
        return 0.0

    gene_map = self._build_gene_map(r)
    best = 0.0

    for t in c.targets:
        gene = gene_map.get(t.gene_symbol.upper())
        if gene:
            action = (t.action_type or "").upper()
            t.patient_gene_log2fc = gene.log2fc
            t.patient_gene_direction = gene.direction

            # Skip sub-threshold fold changes — noise should not drive scoring
            if abs(gene.log2fc) < DEG_LOG2FC_THRESHOLD:
                continue

            if action in self._INHIBITORY and gene.direction == "up":
                best = max(best, w)
            elif action in self._ACTIVATING and gene.direction == "down":
                best = max(best, w)
            elif action == "UNKNOWN" and gene.direction in ("up", "down"):
                best = max(best, w * 0.5)
            continue
```

File: `drug_agent/service/schemas.py`
Function: `get_downregulated_genes_significant()`
Lines: 102–106:
```python
def get_downregulated_genes_significant(self) -> List[GeneContext]:
    """Downregulated genes meeting clinical significance threshold (|log2FC| >= 0.58)."""
    from agentic_ai_wf.reporting_pipeline_agent.core_types import DEG_LOG2FC_THRESHOLD
    return [g for g in self.genes
            if g.direction == "down" and abs(g.log2fc) >= DEG_LOG2FC_THRESHOLD]
```

---

## Section 2: Standard-of-Care Anchor Layer

### Spec Requirement:
SOC drugs must be anchored as backbone therapies, never contraindicated based purely on gene expression. Four rules: SOC-1 (never contraindicate because target gene is downregulated), SOC-2 (never exclude if absent from DEGs), SOC-3 (never appear in contraindication table), SOC-4 (HCQ never flagged for SLE).

### Code Location — SOC Identification:
File: `drug_agent/service/drug_agent_service.py`
Lines: 257–266
```python
# ── STAGE 2.75: SOC Identification (dynamic) ─────────────────
for candidate in candidates:
    if (candidate.identity.is_fda_approved
            and candidate.identity.max_phase is not None
            and candidate.identity.max_phase >= 4
            and self.scorer._has_disease_indication(candidate, request)):
        candidate.is_soc_candidate = True
soc_count = sum(1 for c in candidates if c.is_soc_candidate)
if soc_count:
    print(f"        [Stage 2.75] {soc_count} SOC backbone candidates identified")
```

### Code Location — SOC Shield Logic (SOC-1, SOC-2, SOC-3):
File: `drug_agent/service/drug_agent_service.py`
Lines: 271–276
```python
for candidate in candidates:
    # SOC drugs are shielded from contraindication — expression data informs but never overrides
    if candidate.is_soc_candidate:
        self._collect_soc_advisories(candidate, request)
        safe_candidates.append(candidate)
        continue
    hard_contra = False
```

### Code Location — `_collect_soc_advisories()` (SOC-1 enforcement):
File: `drug_agent/service/drug_agent_service.py`
Lines: 574–584
```python
def _collect_soc_advisories(self, candidate: DrugCandidate, request: DrugQueryRequest):
    """Run contraindication checks for SOC drugs but record as advisories, not blocks."""
    for gene in request.get_downregulated_genes_significant():
        target_genes = {t.gene_symbol.upper() for t in candidate.targets}
        if gene.gene_symbol.upper() in target_genes:
            check = self.router.check_contraindication(
                candidate.identity.drug_name, gene.gene_symbol, gene.direction, gene.log2fc)
            if check.get("is_contraindicated"):
                candidate.soc_advisory_notes.append(
                    f"Target {gene.gene_symbol} is downregulated (log2FC: {gene.log2fc:.2f}) — "
                    f"noted as advisory; drug retained as backbone therapy for {request.disease}")
```

### Code Location — SOC rendering (Section 6.1):
File: `reporting_pipeline_agent/docx_generation.py`
Function: `_render_drug_agent_section()`
Lines: 4048–4070
```python
# ── SOC Backbone Therapies (appear regardless of score) ──────────
soc_drugs = [c for c in drug_response.recommendations if getattr(c, 'is_soc_candidate', False)]
non_soc_high = [c for c in high if not getattr(c, 'is_soc_candidate', False)]
non_soc_moderate = [c for c in moderate if not getattr(c, 'is_soc_candidate', False)]

if soc_drugs:
    self._add_styled_heading(doc, "Foundational / Standard-of-Care Therapies", level=3)
    soc_intro = doc.add_paragraph()
    soc_intro.add_run(
        f"The following FDA-approved therapies are guideline-recommended for {disease}. "
        f"These backbone drugs are presented first and are not subject to contraindication "
        f"based on transcriptomic expression data alone."
    )
    soc_intro.runs[0].font.size = Pt(9)
    soc_intro.runs[0].font.italic = True
    for candidate in soc_drugs:
        self._render_drug_candidate_detail(doc, candidate, disease)
        # SOC advisory notes
        for note in getattr(candidate, 'soc_advisory_notes', []):
            ap = doc.add_paragraph()
            ar = ap.add_run(f"  ℹ Advisory: {note}")
            ar.font.size = Pt(9)
            ar.font.italic = True
```

### Code Location — SOC `_tier_reasoning()` override:
File: `drug_agent/service/drug_scorer.py`
Lines: 113–116
```python
# SOC override path
if getattr(candidate, 'is_soc_candidate', False):
    return ("Guideline-recommended backbone therapy — "
            "retained as Standard-of-Care regardless of molecular scoring.")
```

---

## Section 3.1: Gene Classification — Four Categories

### Spec Requirement:
Genes must be classified into: Category A (Confirmed Disease Drivers), B (PPI-Connected Hub), C (Statistically Significant DEGs), D (Notable Expression Unknown Significance).

### Code Location:
File: `reporting_pipeline_agent/narrative_generation.py`
Function: `_generate_with_llm()` / `_build_data_package()`
Lines: 296–300 (classification), 1133–1143 (data package strata labels)

```python
# Classify genes into three categories (LLM prompt-level)
cat1_genes = [g for g in gene_mappings if g.category == GeneCategory.PATIENT_AND_DISEASE_SPECIFIC]
cat2_genes = [g for g in gene_mappings if g.category == GeneCategory.PATIENT_SPECIFIC_NOVEL]
cat3_genes = [g for g in gene_mappings if g.category == GeneCategory.KNOWN_IN_OTHER_CONDITIONS]
```

Data package includes four strata labels (lines 1133–1143):
```python
data_package += f"""📊 EVIDENCE STRATA — USE THIS TO STRUCTURE GENE DISCUSSION
--------------------------------------------------
🏆 Category A — Confirmed Disease Drivers (KG-confirmed): {sc.get('known_driver', 0)} genes → ...
🔗 Category B — PPI-Connected Hubs (no KG, high PPI): {sc.get('ppi_connected', 0)} genes → ...
📈 Category C — Expression-Significant (strong LFC): {sc.get('expression_significant', 0)} genes → ...
🆕 Category D — Novel Candidates: {sc.get('novel_candidate', 0)} genes → ...
```

LLM Section 2 prompt (lines 395–424):
```python
sections['gene_findings'] = self._llm_generate_section(
    "DISEASE-RELEVANT GENE FINDINGS",
    data_package,
    f"""Generate the DISEASE-RELEVANT GENE FINDINGS section.

USE THE "GENES BY EVIDENCE STRATUM" data to structure this section into subsections:

**A. Confirmed Disease Drivers (Category A)**
Genes tagged [known_driver]. For each: gene name, log2FC, established mechanistic role in {disease}...

**B. PPI-Connected Hub Genes (Category B)**
Genes tagged [ppi_connected]. ...

**C. Expression-Significant Genes (Category C)** (2-3 genes)
Genes tagged [expression_significant] with strongest fold-change. ...

**D. Novel Candidates (Category D)** (1-2 genes, if available)
Genes tagged [novel_candidate] that passed upstream significance filters...
```

---

## Section 3.2: Gene-to-Drug Matching Logic (5-Step)

### Spec Requirement:
Step 1: direction classification, Step 2: discovery search, Step 3: concordance, Step 4: clinical/regulatory, Step 5: safety penalty.

### Step 1 — `_target_direction()`:
File: `drug_agent/service/drug_scorer.py`
Lines: 260–321
```python
def _target_direction(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
    """Direction concordance: drug action_type vs patient gene expression."""
    w = self.config.target_direction_weight
    # ... (gene_map build)
    for t in c.targets:
        gene = gene_map.get(t.gene_symbol.upper())
        if gene:
            action = (t.action_type or "").upper()
            # Skip sub-threshold fold changes
            if abs(gene.log2fc) < DEG_LOG2FC_THRESHOLD:
                continue
            if action in self._INHIBITORY and gene.direction == "up":
                best = max(best, w)
            elif action in self._ACTIVATING and gene.direction == "down":
                best = max(best, w)
            elif action == "UNKNOWN" and gene.direction in ("up", "down"):
                best = max(best, w * 0.5)
```

### Step 2 — Discovery in `collection_router.py`:
File: `drug_agent/service/collection_router.py`
Function: `find_drugs_for_target()`
Lines: 134–212
```python
def find_drugs_for_target(self, gene_symbol: str, top_k: int = 10) -> List[Dict]:
    """Find drugs targeting a gene — queries 5 collections in parallel."""
    gene_up = gene_symbol.upper()
    semantic_q = f"{gene_symbol} inhibitor drug target"
    queries = [
        ("Drug_agent", f"{gene_symbol} drug target therapy", top_k, ...),
        ("ChEMBL_drugs", semantic_q, chembl_limit, None),
        ("OpenTargets_drugs_enriched", f"{gene_symbol} drug mechanism", top_k, None),
        ("FDA_Drug_Labels", gene_symbol, top_k, ...),
        ("Raw_csv_KG", f"{gene_symbol} drug target interaction", top_k, None),
    ]
    raw = self._parallel_search(queries)
```

### Step 3 — Concordance in `drug_scorer.py`:
Already shown in Step 1 — `_INHIBITORY` set and `gene.direction == "up"` pairing is the concordance check.

```python
_INHIBITORY = frozenset({"INHIBITOR", "ANTAGONIST", "NEGATIVE MODULATOR", "BLOCKER", "NEGATIVE ALLOSTERIC MODULATOR"})
_ACTIVATING = frozenset({"AGONIST", "POSITIVE MODULATOR", "ACTIVATOR", "POSITIVE ALLOSTERIC MODULATOR"})
```

### Step 4 — `_clinical_regulatory()`:
File: `drug_agent/service/drug_scorer.py`
Lines: 378–422
```python
def _clinical_regulatory(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
    """Merged clinical trial evidence + FDA regulatory status."""
    w = self.config.clinical_regulatory_weight
    fda = c.identity.is_fda_approved
    disease_match = self._has_disease_indication(c, r)
    # ...
    if fda and disease_match:
        base = w if (phase >= 4 or (completed and has_pval)) else w * 0.85
    elif fda:
        base = w * 0.22 if (has_trials and phase >= 2) else w * 0.18
    elif phase >= 3 and completed and has_pval:
        base = w * 0.70
    # ...
    cap = w if disease_match else w * 0.60
    return round(min(cap, base + bonus), 2)
```

### Step 5 — `_safety_penalty()`:
File: `drug_agent/service/drug_scorer.py`
Lines: 541–573
```python
def _safety_penalty(self, c: DrugCandidate) -> float:
    """Safety penalty — drugs with serious warnings should not be offset
    by high efficacy scores.  Range: 0 to safety_max_penalty (default -30).
    """
    max_penalty = self.config.safety_max_penalty
    penalty = 0.0

    if c.safety:
        penalty -= min(15, len(c.safety.boxed_warnings) * 7)

        if c.safety.serious_ratio and c.safety.serious_ratio > 0.5:
            penalty -= 5
        if c.safety.fatal_ratio and c.safety.fatal_ratio > 0.05:
            penalty -= 7

        pgx_tox = [w for w in c.safety.pgx_warnings if w.get("category", "").lower() == "toxicity"]
        penalty -= min(6, len(pgx_tox) * 3)

        active_recalls = [
            r for r in c.safety.recall_history
            if r.get("status", "").lower() not in ("terminated", "completed", "closed")
        ]
        for recall in active_recalls:
            cls = recall.get("classification", "")
            if "I" in cls and "II" not in cls and "III" not in cls:
                penalty -= 5
            elif "II" in cls:
                penalty -= 2

    if c.trial_evidence and c.trial_evidence.stopped_for_safety:
        penalty -= 5

    return round(max(max_penalty, penalty), 2)
```

---

## Section 3.3: Target Gene Absent from DEG Data

### Spec Requirement:
When a drug's target gene is not in the patient's DEG data, the system should use downstream effector checking or annotate as "not detected."

### Code Location:
File: `drug_agent/service/drug_scorer.py`
Function: `_target_direction()`
Lines: 288–320 (pathway co-member fallback) and `_render_drug_candidate_detail()` (docx)

```python
# Pathway co-membership fallback: partial credit for related patient genes
match = self._pathway_co_member(t.gene_symbol, gene_map, r)
if match:
    related_gene, pathway_name, pw_direction, effectors = match
    if abs(related_gene.log2fc) < DEG_LOG2FC_THRESHOLD:
        continue
    t.related_patient_gene = related_gene.gene_symbol
    t.related_gene_log2fc = related_gene.log2fc
    t.related_gene_direction = related_gene.direction
    t.related_gene_source = "pathway"
    t.downstream_pathway = pathway_name
    if len(effectors) >= 2:
        t.downstream_effector_genes = effectors
    # Pathway direction concordance: higher credit when drug action aligns
    action = (t.action_type or "").upper()
    pw_concordant = (
        (action in self._INHIBITORY and pw_direction == "up") or
        (action in self._ACTIVATING and pw_direction == "down")
    ) if pw_direction else False
    credit = 0.75 if pw_concordant else 0.5
    best = max(best, w * credit)
    continue

# Gene-family fallback: receptor↔ligand linking (annotation only, no score credit)
fam_match = self._gene_family_match(t.gene_symbol, gene_map)
if fam_match:
    t.related_patient_gene = fam_match.gene_symbol
    t.related_gene_log2fc = fam_match.log2fc
    t.related_gene_direction = fam_match.direction
    t.related_gene_source = "gene-family"
```

Absent target annotation in DOCX rendering:

File: `reporting_pipeline_agent/docx_generation.py`
Function: `_render_drug_candidate_detail()`
Lines: 4294–4304
```python
elif t.patient_gene_log2fc is None and not getattr(t, 'related_patient_gene', None):
    np_ = doc.add_paragraph()
    if is_indicated:
        msg = "    Established therapeutic target — drug indicated for patient's condition"
    elif has_any_match:
        msg = "    Additional pharmacological target"
    else:
        msg = "    Not detected in patient's transcriptome"
    nr = np_.add_run(msg)
    nr.font.size = Pt(9)
    nr.font.italic = True
```

---

## Section 4.1: Pathway Reporting Rules

### Spec Requirement:
Pathways must be reported with FDR threshold filtering, direction, and disease context. Top pathways by significance.

### Code Location:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 626–705 (pathway rendering in Transcriptome Overview)

```python
# Count unique pathways by direction
up_pathways = [p for p in pathway_mappings if p['regulation'] == 'Up']
down_pathways = [p for p in pathway_mappings if p['regulation'] == 'Down']

# CLARIFIED TERMINOLOGY: Explain pathway hierarchy
total_pathways_tested = summary_stats.get('total_pathways', 0)
enriched_pathways = summary_stats.get('enriched_pathways', len(pathway_mappings))
disease_relevant_count = len(pathway_mappings)

pathway_summary_para.add_run(
    f"From {total_pathways_tested:,} pathways tested, {enriched_pathways:,} were significantly enriched (FDR < 0.05). "
    f"Disease-Relevant Enriched Pathways: {disease_relevant_count} ({len(up_pathways)} upregulated, {len(down_pathways)} downregulated)"
)

# Key Upregulated Pathways table
pw_headers = ["Pathway", "FDR", "Disease Context"]
pw_rows = []
for p in deduped_up:
    pw_name = p['pathway_name']
    pw_rows.append((
        smart_truncate(pw_name, 45),
        f"{p['fdr']:.2e}" if p['fdr'] else "N/A",
        pathway_contexts.get(pw_name, '')
    ))
```

File: `drug_agent/service/drug_agent_service.py`
Lines: 113–119 (pathway-based drug discovery):
```python
# Pathway-based discovery for top pathways by significance (any direction)
sorted_pathways = sorted(request.pathways, key=lambda p: p.fdr)
for pw in sorted_pathways[:5]:
    key = f"pathway_{pw.pathway_name[:30]}"
    discovery_results[key] = self.router.get_pathway_drugs(pw.pathway_name, pw.key_genes)
```

---

## Section 4.2: CIBERSORT / Cell-Type Deconvolution

### Spec Requirement:
Cell-type deconvolution results must be reported with enrichment score and presence %. Cross-reference cell type markers against DEG data.

### Code Location — Narrative:
File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 460–487 (Section 4 prompt generation)
```python
if xcell_findings:
    print("        [4/7] Generating Deconvolution Interpretation...")
    sections['deconvolution_findings'] = self._llm_generate_section(
        "DECONVOLUTION INTERPRETATION",
        data_package,
        f"""Generate a CONCISE Deconvolution Interpretation section (maximum 1 page).
FORMAT STRICTLY:
**Overview** (2-3 sentences): Summarize dominant cell types detected.
**Key Cell Type Findings** (bullet list, max 5 items):
• Cell Type: enrichment level, clinical implication
**Integration with Molecular Findings** (1 short paragraph):
Link cell type patterns to DEG/pathway results. ...
**Clinical Relevance** (1-2 sentences): Therapeutic responsiveness implication.
```

### Code Location — DOCX rendering:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 793–855 (cell type tables)
```python
high_enriched = xcell_findings.get('high_enriched', [])
if high_enriched:
    xcell_headers = ["Cell Type", score_column_name, "Presence", "Disease Context"]
    xcell_rows = []
    for e in high_enriched[:10]:
        xcell_rows.append((
            e.cell_type,
            f"{e.median_enrichment:.3f}",
            presence_to_category(e.presence_fraction),
            cell_contexts.get(e.cell_type, '')
        ))
    self._add_table(doc, xcell_headers, xcell_rows)
```

---

## Section 5: Disease Activity Assessment

### Spec Requirement:
Molecular activity parameters must be calculated and reported with a scoring framework.

### Code Location:
File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 311–315 (pre-computation), 498–512 (structured assessment)
```python
# =====================================================================
# DISEASE ACTIVITY — computed FIRST so all LLM sections see the result
# =====================================================================
pre_disease_activity = self._compute_disease_activity_early(
    disease, gene_mappings, pathway_mappings, xcell_findings
)
# ...
# Reuse pre-computed disease activity (computed before data_package)
sections['disease_activity_assessment'] = pre_disease_activity.get('structured_assessment', {})
sections['disease_activity_format'] = 'structured'
sections['disease_knowledge'] = pre_disease_activity.get('disease_knowledge', {})
sections['dynamic_scored_parameters'] = pre_disease_activity.get('scored_parameters', [])
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 870–880 (disease activity section rendering)
```python
if disease_activity:
    section_num += 1
    da_section = section_num
    da_subsection = 0
    heading_para = self._add_styled_heading(doc, f"{section_num}. Disease Activity Assessment", level=1)
    heading_para.paragraph_format.page_break_before = True

    if disease_activity_format == 'structured' and isinstance(disease_activity, dict):
        activity_score = disease_activity.get('molecular_activity_score', {})
```

---

## Section 5.2: IFN Signature Mixed Pattern Rule

### Spec Requirement:
IFN signature gating for anifrolumab — if IFN not HIGH, reduce recommendation or add caveat. Mixed IFN pattern requires special handling.

### Code Location:
File: `drug_agent/service/drug_scorer.py`
Function: `_ifn_signature_gate()`
Lines: 58–109

```python
def _ifn_signature_gate(self, s: ScoreBreakdown, candidate: DrugCandidate,
                        request: DrugQueryRequest) -> ScoreBreakdown:
    """Gate anifrolumab (IFNAR1-targeted) on IFN-HIGH signature status.

    If the patient's IFN signature score is not available or is LOW,
    annotate the score with a caveat but preserve the composite.
    """
    drug_name = candidate.identity.drug_name.upper()
    if drug_name != 'ANIFROLUMAB' and 'ANIFROLUMAB' not in drug_name:
        return s
    # Check if any target is IFNAR1
    ifnar_target = any(t.gene_symbol.upper() == 'IFNAR1' for t in candidate.targets)
    if not ifnar_target:
        return s

    sig_scores = getattr(request, 'signature_scores', None) or {}
    ifn_data = None
    for key in sig_scores:
        if 'ifn' in key.lower() or 'interferon' in key.lower():
            ifn_data = sig_scores[key]
            break

    if ifn_data:
        level = ifn_data.get('level', '').upper()
        pct = ifn_data.get('activation_score', 0)
        if level == 'HIGH' or pct >= 75:
            # IFN-HIGH: full recommendation
            s.tier_reasoning = (
                f"IFN Signature HIGH ({pct:.0f}%) — "
                f"patient is a strong candidate for IFNAR1-targeted therapy. "
                + s.tier_reasoning
            )
        else:
            # IFN not HIGH: add caveat, reduce direction credit
            s.tier_reasoning = (
                f"IFN Signature {level or 'UNKNOWN'} ({pct:.0f}%) — "
                f"anifrolumab benefit requires IFN-HIGH status per TULIP trials. "
                + s.tier_reasoning
            )
            # Halve direction credit to reflect uncertain benefit
            if s.target_direction_match > 0:
                s.target_direction_match = s.target_direction_match * 0.5
                s.calculate()  # recalculate composite
    else:
        # No IFN data available
        s.tier_reasoning = (
            "IFN Signature NOT ASSESSED — "
            "IFN-HIGH status needed for anifrolumab candidacy (TULIP trials). "
            + s.tier_reasoning
        )
    return s
```

---

## Section 6.1: SOC Therapies Rendering

### Spec Requirement:
SOC drugs must appear first in the therapeutic section, sorted before all other drugs.

### Code Location:
File: `drug_agent/service/drug_agent_service.py`
Lines: 354–359 (SOC-first sort in Stage 5)
```python
# SOC drugs sort to front, then by composite score
safe_candidates.sort(key=lambda c: (
    not c.is_soc_candidate,
    -(c.score.composite_score if c.score else 0),
))
```

File: `reporting_pipeline_agent/docx_generation.py`
Function: `_render_drug_agent_section()`
Lines: 4048–4073 (SOC section rendered before high-priority section)
```python
soc_drugs = [c for c in drug_response.recommendations if getattr(c, 'is_soc_candidate', False)]
if soc_drugs:
    self._add_styled_heading(doc, "Foundational / Standard-of-Care Therapies", level=3)
    soc_intro.add_run(
        f"The following FDA-approved therapies are guideline-recommended for {disease}. "
        f"These backbone drugs are presented first and are not subject to contraindication "
        f"based on transcriptomic expression data alone."
    )
    for candidate in soc_drugs:
        self._render_drug_candidate_detail(doc, candidate, disease)
```

---

## Section 6.2: Biomarker-Therapy Concordance Table

### Spec Requirement:
A biomarker concordance table must classify biomarkers as Type A (RNA-assessable) or Type B (requires orthogonal testing).

### Code Location:
File: `reporting_pipeline_agent/llm_knowledge.py`
Lines: 479–526 (CONCORDANCE_RULES_PROMPT)
```python
CONCORDANCE_RULES_PROMPT = '''...
BIOMARKER TYPE CLASSIFICATION (REQUIRED for every rule):
- Type A (RNA-assessable): Biomarker whose activity can be meaningfully inferred from RNA-seq gene expression.
- Type B (requires orthogonal testing): Biomarker that CANNOT be reliably assessed from transcriptomic data alone...
Set "biomarker_type" to "A" or "B" for each rule.
...
Return 8-12 concordance rules for the main biomarkers in {disease}.
Return ONLY the JSON array.'''
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 1383–1413 (concordance table rendering)
```python
concordance_headers = ["Biomarker", "Patient Status", "RNA Interpretation", "Potential Therapies", "Required Test"]

concordance_rows = []
for entry in biomarker_concordance:
    biomarker = entry.get('Biomarker', entry.get('biomarker', 'N/A'))
    patient_status = entry.get('Patient Status', entry.get('patient_status', 'N/A'))
    rna_interp = entry.get('RNA Interpretation', entry.get('Concordance', entry.get('concordance', 'N/A')))
    therapies = entry.get('Potential Therapies', ...)
    required_test = entry.get('Required Test', ...)
    ...
    concordance_rows.append((
        str(biomarker),
        str(patient_status),
        rna_interp_display,
        smart_truncate(therapies_display, 60),
        test_display
    ))
```

---

## Section 6.3: Evidence-Based Drug Prioritization (Tier Decision Tree)

### Spec Requirement:
Spec says "Do not compute a numerical composite score." However, the codebase uses a 0-100 composite score.

### Code Location:
File: `drug_agent/service/schemas.py`
Lines: 67–73 (ScoringConfig with numeric weights):
```python
@dataclass
class ScoringConfig:
    target_direction_weight: float = 18.0
    target_magnitude_weight: float = 12.0
    clinical_regulatory_weight: float = 25.0
    ot_weight: float = 15.0
    pathway_weight: float = 15.0
    safety_max_penalty: float = -30.0
```

File: `drug_agent/service/schemas.py`
Lines: 181–190 (composite calculation):
```python
def calculate(self):
    self.composite_score = max(0.0, min(100.0,
        self.target_direction_match
        + self.target_magnitude_match
        + self.clinical_regulatory_score
        + self.ot_association_score
        + self.pathway_concordance
        + self.safety_penalty
        + self.disease_indication_bonus
    ))
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 4036–4045 (DOCX displays the score):
```python
meth.add_run(
    "Composite scores (0-100) are calculated from six evidence components: "
    "Target Direction Match ..., "
    "Target Magnitude ..., "
    "Clinical & Regulatory ..., "
    "OpenTargets Association ..., "
    "Pathway Concordance ..., "
    "and Safety Penalty ... "
    f"High-priority: ≥{DRUG_HIGH_PRIORITY_THRESHOLD}, "
    f"Moderate: {DRUG_MODERATE_PRIORITY_THRESHOLD}-{DRUG_HIGH_PRIORITY_THRESHOLD}."
)
```

---

## Section 6.4: Contraindications — Tiered Logic

### Spec Requirement:
Three-tier contraindication system (HARD/MODERATE/SOFT), SOC protection, contraindication table in DOCX.

### Code Location — Tier schema:
File: `drug_agent/service/schemas.py`
Lines: 192–202
```python
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
```

### Code Location — Gene-based contraindication:
File: `drug_agent/service/drug_agent_service.py`
Lines: 279–293
```python
# Path 1: Gene-based — drug targets a downregulated gene in a potentially harmful way
for gene in request.get_downregulated_genes_significant():
    target_genes = {t.gene_symbol.upper() for t in candidate.targets}
    if gene.gene_symbol.upper() in target_genes:
        check = self.router.check_contraindication(
            candidate.identity.drug_name, gene.gene_symbol, gene.direction, gene.log2fc)
        if check.get("is_contraindicated"):
            entry = ContraindicationEntry(
                tier=check.get("tier", 2), reason=check["reason"],
                source="gene_based", gene_symbol=gene.gene_symbol, log2fc=gene.log2fc)
            if entry.tier <= 2:
                candidate.contraindication_flags.append(entry.reason)
                hard_contra = True
            else:
                candidate.caution_notes.append(entry)
```

### Code Location — 4 contraindication paths:
File: `drug_agent/service/drug_agent_service.py`
Lines: 279–328 (Paths 1–4 shown sequentially)
- Path 1: Gene-based (lines 279–293)
- Path 2: Biomarker-aware (lines 295–300)
- Path 3: Disease-AE (lines 303–313)
- Path 4: Trial why_stopped (lines 315–328)

### Code Location — DOCX contraindication table:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 4150–4183
```python
if contra:
    self._add_styled_heading(doc, "Contraindicated / Caution", level=3)
    contra_headers = ["Drug", "Score", "Target", "Tier", "Contraindication Reason"]
    contra_rows = []
    for c in contra[:6]:
        primary = c.targets[0].gene_symbol if c.targets else '—'
        reason = '; '.join(c.contraindication_flags[:2]) if c.contraindication_flags else '—'
        tier_label = "Avoid" if any('adverse events' in f.lower() for f in c.contraindication_flags) else "Contraindicated"
        contra_rows.append((
            c.identity.drug_name,
            f"{c.score.composite_score:.0f}" if c.score else "—",
            primary,
            tier_label,
            smart_truncate(reason, 80),
        ))
    self._add_table(doc, contra_headers, contra_rows)
```

---

## Section 6.5: Gene-Targeted Unvalidated Candidates

### Spec Requirement:
Drugs that target the patient's genes but lack disease-treatment evidence must be separated into a distinct "unvalidated" or "gene-targeted only" section.

### Code Location:
File: `drug_agent/service/drug_agent_service.py`
Lines: 374–377 (split logic)
```python
# Split: drugs with disease-treatment evidence vs gene-targeted only
validated = [c for c in safe_candidates if c.score and c.score.disease_relevant]
gene_only = [c for c in safe_candidates if c.score and not c.score.disease_relevant]
gene_only.extend(reclassified)  # Claude-reclassified drugs go here with caveat
final = validated[:request.max_results]
```

File: `drug_agent/service/schemas.py`
Lines: 178 — `disease_relevant` field:
```python
disease_relevant: bool = True  # False when drug targets genes but lacks disease-treatment evidence
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 4118–4132 (DOCX rendering)
```python
# ── Gene-Targeted Candidates (lack disease-treatment evidence) ────────
gene_only = getattr(drug_response, 'gene_targeted_only', [])
if gene_only:
    self._add_styled_heading(
        doc, f"Gene-Targeted Candidates (Unvalidated for {disease})", level=3
    )
    gt_note = doc.add_paragraph()
    gt_note.add_run(
        f"The following {len(gene_only)} drug(s) target the patient's dysregulated genes "
        f"but lack direct evidence of treating {disease}. "
        f"Clinical judgment is required before considering these as therapeutic options."
    )
    gt_note.runs[0].font.italic = True
    gt_note.runs[0].font.size = Pt(9)
    self._render_drug_summary_table(doc, gene_only[:8])
```

---

## Section 6.6: Required Confirmatory Tests

### Spec Requirement:
Confirmatory tests must be generated and listed in the report for key biomarkers and drug targets.

### Code Location — Narrative generation:
File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 836–844 (LLM prompt includes confirmatory tests instruction)
```python
**E. REQUIRED CONSIDERATION SUMMARY**
❗ IMPORTANT: You MUST use this EXACT introduction sentence verbatim (do not paraphrase or change it):
"To strengthen the findings of this report, there are following considerations:"

Then list the relevant confirmatory tests based on the patient's actual biomarkers and drug targets identified in this analysis.
```

### Code Location — Concordance table includes Required Test column:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 1384, 1393–1401
```python
concordance_headers = ["Biomarker", "Patient Status", "RNA Interpretation", "Potential Therapies", "Required Test"]
# ...
required_test = entry.get('Required Test', entry.get('required_test', entry.get('Test Method', 'N/A')))
# ...
test_display = get_specific_test(str(biomarker), str(required_test))
```

### Code Location — `llm_knowledge.py` concordance rule includes test method:
Lines: 487–498
```python
{
    "biomarker": "Clinical Biomarker Name",
    "gene": "PRIMARY_GENE_SYMBOL",
    # ...
    "test_method": "FDA-approved test method",
    # ...
}
```

---

## Section 7: Report Structure

### Spec Requirement:
9 sections in the report.

### Code Location:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 340–2070 (actual section flow)

The actual DOCX section flow:
```
1. Executive Summary (line 341)
   1.1 Key Findings
   1.2 Analysis Overview
2. Clinical Context (line 414 — conditional)
3. Transcriptome Overview (line 425)
   3.1 Gene Prioritization Methodology
   3.2 Gene Findings (Disease-Relevant Genes table)
   3.3 Dysregulated Genes of Unknown Clinical Significance
   3.4 Pathway Analysis (up/down pathway tables)
4. Cell-Type Deconvolution Analysis (line 717 — conditional)
   4.1 Highly Enriched Cell Types
   4.2 Moderately Enriched Cell Types
5. Disease Activity Assessment (line 874)
   5.1 Molecular Disease Activity Score
   5.2 Molecular Phenotype Classification
   5.3 Prognostic Implications
   5.4 Monitoring Considerations
6. Therapeutic Implications (line 1357)
   6.1 Biomarker-Therapy Concordance Table
   6.2+ Evidence-Based Drug Prioritization (SOC, High, Moderate, Gene-only, Contraindicated)
[Appendix] (line 1765)
[Conclusion] (line 2070)
[References] (line 2082)
[Disclaimer] (line 2130)
```

NarrativeGenerator describes 7 sections (line 109):
```python
"""
Report Structure (7 Sections):
1. Executive Summary
2. Disease-Relevant Gene Findings
3. Pathway Analysis
4. Deconvolution Interpretation
5. Actionable Therapeutic Targets & Biomarkers
6. Integrated Biological Interpretation
7. Appendix
"""
```

---

## Section 8: Pre-Output Quality Checklist

### Spec Requirement:
A quality checklist must be run before final output to validate completeness and clinical logic.

### Code Location:
No explicit pre-output quality checklist was found in any of the audited files. The closest implementations are:

File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 654–682 — Clinical Concordance Validation (internal, not user-visible):
```python
print("        [4.6/7] Running Clinical Concordance Validation...")
clinical_validator = get_clinical_validator(disease)
clinical_validator.set_patient_state(gene_mappings, signature_scores, xcell_findings)
pathway_dicts = [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in pathway_mappings]
detected_contradictions = clinical_validator.run_full_validation(mechanistic_drug_recs, pathway_dicts)
```

File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 688–702 — Module/narrative consistency validation:
```python
if module_scores and sections.get('disease_activity_assessment'):
    disease_activity_text = str(sections.get('disease_activity_assessment', ''))
    module_inconsistencies = self.module_scorer.validate_narrative_consistency(disease_activity_text, disease)
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 1636–1646 — Section explicitly removed:
```python
# =====================================================================
# CLINICAL CONCORDANCE VALIDATION SECTION - REMOVED
# =====================================================================
# Section 6.3 Clinical Concordance Validation has been removed from the
# report per design requirements. The validation system still runs internally
# to flag drug recommendations, but the detailed contradiction section
# is no longer displayed in the final report.
```

---

## Section 9: Required & Prohibited Language Patterns

### Spec Requirement:
Required language patterns (e.g., "suggestive of," "supports the hypothesis") and prohibited phrases (e.g., "eligible for," "contraindicated") must be enforced.

### Code Location:
File: `reporting_pipeline_agent/core_types.py`
Lines: 109–135
```python
CLINICAL_LANGUAGE_TEMPLATES = {
    'supportive': [
        "suggestive of",
        "supports the hypothesis of",
        "consistent with",
        "may indicate",
        "expression pattern aligns with",
        "transcriptomic evidence supports",
    ],
    'not_supportive': [
        "does not support",
        "expression pattern inconsistent with",
        "transcriptomic data does not suggest",
        "limited RNA-level evidence for",
    ],
    'confirmation_required': [
        "requires confirmatory testing", "clinical confirmation required",
        "confirmatory testing recommended", "protein-level assessment needed",
    ],
    'forbidden_phrases': [
        "eligible for", "not eligible for", "qualifies for", "disqualifies from",
        "indicated for", "contraindicated", "approved for", "should receive",
        "should not receive", "will respond to", "will not respond to",
    ]
}
```

File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 120–203 — System prompt enforces research language:
```python
REPORT_AGENT_SYSTEM_PROMPT = """You are the "Report Generation Agent" - an expert clinical bioinformatics system.
...
❌ AVOID REPETITIVE DISCLAIMER PHRASES:
- Do NOT repeat "clinical confirmation is needed" or similar phrases
...
3. No hallucination
• Do NOT invent drug names
• Do NOT claim approvals that don't exist
• If uncertain, use: "reported in literature as..." or "studies suggest..."
"""
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 267–275 (sanitization applied to all narrative text):
```python
if 'narratives' in report_data:
    for key, value in report_data['narratives'].items():
        if isinstance(value, str):
            report_data['narratives'][key] = sanitize_clinical_text(value)
        elif isinstance(value, list):
            report_data['narratives'][key] = [
                sanitize_clinical_text(item) if isinstance(item, str) else item
                for item in value
            ]
```

---

## Section 10: SLE Disease-Specific Module

### Spec Requirement:
SLE-specific logic including HCQ, belimumab, anifrolumab, IFN signature handling.

### Code Location — IFN/anifrolumab handling:
File: `drug_agent/service/drug_scorer.py`
Lines: 58–109 (`_ifn_signature_gate()`) — shown in Section 5.2 above.

### Code Location — Disease category detection in llm_knowledge.py:
File: `reporting_pipeline_agent/llm_knowledge.py`
Lines: 1157–1160
```python
autoimmune_keywords = ["arthritis", "lupus", "sjogren", "vasculitis", "scleroderma",
                      "psoriasis", "crohn", "colitis", "multiple sclerosis", "autoimmune"]
```

### Code Location — Claude semantic validation includes SLE example:
File: `drug_agent/service/drug_agent_service.py`
Lines: 697–698
```python
f"pharmacological class that is known to be contraindicated, harmful, or "
f"ineffective in {request.disease} (e.g., TNF inhibitors in SLE, "
f"immunosuppressants in active infections), flag it as not relevant and "
```

### Code Location — Type A/B biomarker classification includes IFN example:
File: `reporting_pipeline_agent/llm_knowledge.py`
Lines: 500–502
```python
BIOMARKER TYPE CLASSIFICATION (REQUIRED for every rule):
- Type A (RNA-assessable): Biomarker whose activity can be meaningfully inferred from RNA-seq gene expression.
  Examples: IFN signature genes, cytokine pathway transcripts, cell-type marker genes.
- Type B (requires orthogonal testing): Biomarker that CANNOT be reliably assessed from transcriptomic data alone...
  Examples: anti-dsDNA antibody titers, complement C3/C4 protein levels, HER2 IHC/FISH, PD-L1 IHC.
```

---

## Section 10.1: Template for Other Diseases

### Spec Requirement:
The system must handle non-SLE diseases via a disease-specific initialization pattern.

### Code Location:
File: `reporting_pipeline_agent/llm_knowledge.py`
Function: `initialize_for_disease()`
Lines: 561–578
```python
def initialize_for_disease(self, disease: str, patient_genes: List[str] = None):
    """Pre-fetch all knowledge for a disease to minimize API calls."""
    if not self.llm_client:
        print("      Dynamic knowledge: LLM not available, using static data")
        return

    print(f"      Initializing dynamic knowledge for {disease}...")

    self._disease_cache[disease] = {
        'biomarkers': self.get_disease_biomarkers_dynamic(disease),
        'signatures': self.get_disease_signatures_dynamic(disease)
    }

    if patient_genes:
        gene_drugs = self.get_drugs_for_genes_batch(patient_genes[:50], disease)
        self._disease_cache[disease]['gene_drugs'] = gene_drugs

    self._initialized = True
    print(f"      Dynamic knowledge initialization complete.")
```

File: `drug_agent/service/drug_agent_service.py`
Lines: 81–82 (disease alias expansion):
```python
if not request.disease_aliases:
    request.disease_aliases = self.router.get_disease_aliases(request.disease)
```

---

## Section 11: Data Integrity Rules

### Spec Requirement:
No fabrication of drug names, citations, or database sources. Adj.p-value data integrity must be maintained.

### Code Location — No hallucination instructions:
File: `reporting_pipeline_agent/narrative_generation.py`
Lines: 193–196
```python
3. No hallucination
• Do NOT invent drug names
• Do NOT claim approvals that don't exist
• If uncertain, use: "reported in literature as..." or "studies suggest..."
• Be accurate about disease-specific treatments
```

### Code Location — Drug Agent validation (DRUG_VALIDATION_PENDING flag):
File: `reporting_pipeline_agent/core_types.py`
Lines: 91–96
```python
DRUG_VALIDATION_PENDING = False
DRUG_VALIDATION_DISCLAIMER = (
    "Note: Drug recommendations are pending validation by pharmacological database integration. "
    "Some listed compounds may be investigational, incorrectly identified, or placeholder entities. "
    "Do not use for prescribing decisions without independent verification."
)
```

### Code Location — Adj.p value shown in tables:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 503–504
```python
f"{adj_pval:.2e}" if adj_pval else "N/A",
```

File: `reporting_pipeline_agent/docx_generation.py`
Lines: 2058–2062 (Analysis Parameters table in appendix):
```python
param_table = [
    ("log2FC Threshold (Upregulated)", params.get('log2fc_up', '≥ 0.58 (1.5-fold)')),
    ("log2FC Threshold (Downregulated)", params.get('log2fc_down', '≤ -0.58 (1.5-fold)')),
    ("Significance Threshold", params.get('adj_p_threshold', 'adj.p < 0.05')),
```

---

## Section 12: Mandatory Clinical Disclaimer

### Spec Requirement:
A mandatory clinical disclaimer must appear in every report.

### Code Location:
File: `reporting_pipeline_agent/docx_generation.py`
Lines: 2130–2136
```python
# =====================================================================
# DISCLAIMER
# =====================================================================
doc.add_paragraph()
p = doc.add_paragraph()
p.add_run("Disclaimer: ").bold = True
p.add_run("This report is for research purposes only. All the artifacts require clinical considerations along appropriate companion diagnostics for a therapeutic decision. Clinical decisions should be made by qualified healthcare professionals in conjunction with established clinical guidelines.")
```

---
