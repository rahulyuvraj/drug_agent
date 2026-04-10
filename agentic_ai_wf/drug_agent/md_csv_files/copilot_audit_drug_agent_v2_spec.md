# Copilot Task: Drug Agent v2.1 Specification Audit & Gap Analysis

## Context

Our expert teammate (Naila) authored a comprehensive operational specification for the Drug Agent — "Drug Agent Complete System Prompt & Operational Specification v2.1." She has asked to see the exact code snippets where each instruction from her specification is implemented in our codebase. She wants to identify gaps between what the spec requires and what the code actually does.

Your job: Go through EVERY section of her specification document (summarized below), find the corresponding code in our codebase, and produce a detailed audit showing:
- WHERE each spec requirement is implemented (file, function, line numbers)
- HOW it's implemented (show the actual code snippet)
- If NOT implemented: flag it clearly as a GAP

## Important Instructions

1. Search the ENTIRE codebase — not just `drug_agent/service/`. The reporting pipeline (`reporting_pipeline_agent/`), narrative generation, DOCX generation, LLM knowledge, and core_types files all contain relevant logic.

2. For each section of the spec, provide the ACTUAL code lines. Don't summarize — show the real code.

3. Be brutally honest about gaps. If something is partially implemented, say what's there and what's missing.

4. Output format: Create a single comprehensive markdown file with sections matching the spec document's numbering. Under each section, show the code snippets (with file paths and line numbers) or mark as [GAP].

---

## THE SPECIFICATION TO AUDIT (Section by Section)

### SPEC: Design Philosophy — Traceable Reasoning

The spec requires: "For every drug recommendation or contraindication, the agent must be able to complete this sentence: '[Drug] is recommended/not recommended because [specific gene] is [up/down]regulated (log2FC: X) in this patient, and [drug] works by [mechanism] on that target.' If that sentence cannot be completed with real patient data, the recommendation should not be made, or should be labelled as SOC-based (not expression-based)."

**Find**: Where does our code construct or enforce traceable reasoning sentences? Check:
- `drug_scorer.py` — does the scoring produce traceable reasoning per drug?
- `result_aggregator.py` — does the aggregation produce reasoning text?
- `narrative_generation.py` — does the narrative generator produce the required sentence pattern?
- `docx_generation.py` — does the report output include per-drug reasoning?
- `schemas.py` — is there a `reasoning` or `rationale` field in drug candidate schemas?
- `evidence_compiler.py` — does the evidence compiler produce traceable evidence chains?

Show me the exact code for how drug-level reasoning/rationale is generated and rendered.

---

### SPEC Section 1: Input Data Contract

The spec requires these inputs:
- `disease_name` (String)
- `patient_deg_table` (CSV/TSV with Gene, Patient_LFC_mean/log2FC, adj.p-value)
- `pathway_enrichment_table` (CSV/TSV with Pathway name, FDR, direction, gene count)
- `cibersort_table` (CSV/TSV with Cell type, enrichment score, presence %)

**Find**: Show me the `DrugQueryRequest` schema in `schemas.py` and how each of these inputs is received. Also check the reporting pipeline's input handling — how does it pass data to the Drug Agent service?

Show the exact schema definition and the calling code that constructs the request.

---

### SPEC Section 1.1: DEG Classification Thresholds

The spec requires:
- Upregulated: log2FC ≥ +0.58 (≥ 1.5-fold)
- Downregulated: log2FC ≤ −0.58
- Statistically Significant: adj.p < 0.05 — mark with ✓
- Report all meeting log2FC threshold; flag adj.p < 0.05 explicitly

**Find**: Where is the `DEG_LOG2FC_THRESHOLD` defined? Show me the code that applies this threshold in:
- `drug_scorer.py` `_target_direction()` method
- `core_types.py` threshold definitions
- Any gene filtering logic in the reporting pipeline

Also check: Is adj.p < 0.05 being flagged with ✓ in tables? Search docx_generation.py for any adj.p marking logic.

---

### SPEC Section 2: Standard-of-Care Anchor Layer

The spec requires: "Backbone therapies are defined by clinical guidelines, not by gene expression. A drug's target gene being absent or downregulated in the DEG data is NOT a reason to exclude or contraindicate a backbone therapy."

**SOC Rules**:
- SOC-1: Never contraindicate a backbone drug because its target gene is downregulated
- SOC-2: Never exclude a backbone drug because its target gene is absent from DEG table
- SOC-3: A backbone drug must never appear in the contraindication table
- SOC-4 (SLE): HCQ must never be flagged as contraindicated for SLE patients

**Find**: Show me ALL code related to SOC identification and protection:
- `_identify_soc_candidates()` in `drug_agent_service.py` — how are SOC drugs identified?
- The SOC shield logic in contraindication checking — how does it prevent SOC drugs from being contraindicated?
- `llm_knowledge.py` — any SOC drug lists or knowledge?
- The contraindication check code — where does it check "is this drug SOC?" before contraindicting?
- Report rendering — where does Section 6.1 (SOC table) get rendered?

For each SOC rule (SOC-1 through SOC-4), show the specific code that enforces it or mark as [GAP].

---

### SPEC Section 3.1: Gene Classification — Four Categories

The spec requires genes classified into:
- **Category A** — Confirmed Disease Drivers (OMIM, DisGeNET, published disease association)
- **Category B** — PPI-Connected Hub Genes (PPI Score ≥ 5.0 or connects to ≥2 Category A genes)
- **Category C** — Statistically Significant DEGs (adj.p < 0.05 and |log2FC| ≥ 0.58)
- **Category D** — Notable Expression, Unknown Significance (|log2FC| ≥ 1.0, no disease connection)

**Find**: Where in the code are genes classified into these four categories? Check:
- `narrative_generation.py` — gene classification logic
- `drug_agent_service.py` — gene categorization
- Any gene ranking/classification module
- `docx_generation.py` — Section 3 rendering with categories

If genes are NOT classified into A/B/C/D categories, mark as [GAP] and note what IS done instead.

---

### SPEC Section 3.2: Gene-to-Drug Matching Logic (5-Step)

The spec requires a 5-step matching procedure:
1. Direction check — is gene up or down?
2. Drug lookup — query ChEMBL, OpenTargets, FDA, DGIdb for drugs acting on this gene
3. Concordance check — does drug mechanism align with gene direction?
4. Evidence check — FDA approval or clinical trial evidence for target disease?
5. Safety check — disease-specific risks?

**Find**: Show me the code for each of these 5 steps:
- Step 1: `_target_direction()` in `drug_scorer.py`
- Step 2: Discovery stage in `collection_router.py` — gene-based search
- Step 3: Concordance logic in `drug_scorer.py`
- Step 4: `_clinical_regulatory()` in `drug_scorer.py`
- Step 5: `_safety_penalty()` in `drug_scorer.py`

Show the actual code for each step. Note any differences between the spec's described logic and what the code actually does.

---

### SPEC Section 3.3: Target Gene Absent from DEG Data

The spec requires: When a drug's target gene is NOT in the patient DEG table:
1. Do NOT automatically exclude the drug
2. Check if drug is backbone SOC → recommend unconditionally
3. Check if downstream effector genes are in DEGs in expected direction
4. If no evidence: state "Target gene [GENE] absent from DEG data — expression-based assessment not possible"

**Find**: Where does the code handle the case where a drug's target gene is missing from patient DEGs? Check:
- `drug_scorer.py` — what happens when the gene match fails?
- `_target_direction()` — what score does a drug get when its target isn't in the DEG list?
- Downstream effector checking — is this implemented anywhere?
- Report narrative — does it produce the required language for absent targets?

This is likely a significant [GAP] — show what exists and what's missing.

---

### SPEC Section 4.1: Pathway Reporting Rules

The spec requires:
- Report all pathways with FDR < 0.05
- For each: FDR, direction, key genes, disease-specific interpretation
- Never infer pathway enrichment from gene expression alone
- Cell-cycle pathway caveat: always present as open interpretive question

**Find**: Show pathway handling code in:
- `narrative_generation.py` — pathway narrative generation
- `docx_generation.py` — pathway table rendering
- Any FDR threshold logic

---

### SPEC Section 4.2: CIBERSORT / Cell-Type Deconvolution

The spec requires:
- High enrichment > 0.10, Moderate 0.02-0.10
- Presence % is NOT cell proportion — always explain this
- Cross-reference each cell type's marker genes against DEG list
- Required language pattern for cross-validation

**Find**: Show cell-type deconvolution handling in:
- `narrative_generation.py` — cell type interpretation
- `docx_generation.py` — deconvolution table rendering
- Any code that cross-references cell type markers against DEG data

---

### SPEC Section 5: Disease Activity Assessment

The spec requires molecular activity parameters (SLE example):
- TNF Superfamily Signaling
- Adaptive Immune Regulation
- Innate Immune Defense Activity
- Cytokine Production Regulation
- Glucocorticoid Response Signature
- Interferon-Stimulated Gene Activity

Each with specific evidence genes and level assignment logic (HIGH/LOW/MIXED etc.)

**Find**: Where are disease activity parameters calculated? Check:
- `narrative_generation.py` — disease activity assessment section
- Any molecular signature or activity scoring code
- IFN signature logic specifically

---

### SPEC Section 5.2: IFN Signature Mixed Pattern Rule

The spec requires: When ISGs show a mixed pattern, count exactly how many are up vs down. State counts explicitly. For anifrolumab: IFN-LOW with residual activity = "not currently supported — monitor."

**Find**: Show the IFN signature handling code:
- `_ifn_signature_gate()` in `drug_scorer.py` or `drug_agent_service.py`
- Any IFN classification logic
- Anifrolumab-specific handling
- Required language patterns for IFN-LOW/MIXED

---

### SPEC Section 6.1: SOC Therapies (Always First)

The spec requires: SOC backbone table rendered FIRST, unconditionally. For each SOC drug, note whether target gene was in DEGs and what expression direction implies — but never as a contraindication.

**Find**: Show the SOC section rendering in:
- `docx_generation.py` — Section 6.1 rendering
- `narrative_generation.py` — SOC narrative
- How SOC drugs are ordered before other sections

---

### SPEC Section 6.2: Biomarker-Therapy Concordance Table

The spec requires Type A (transcriptomically assessable) vs Type B (requires orthogonal testing) biomarker classification. For Type B: "Requires direct [test name] — not assessable from transcriptomic data."

**CRITICAL**: The spec says "NEVER use TNF, IL1B, or any DEG as a proxy for a serological or clinical biomarker."

**Find**: Show biomarker concordance table code:
- `docx_generation.py` — concordance table rendering
- `narrative_generation.py` — biomarker assessment
- Any Type A vs Type B classification logic
- Any code that uses cytokines as surrogates for serological biomarkers (this would be a violation)

---

### SPEC Section 6.3: Evidence-Based Drug Prioritization (Tier Decision Tree)

The spec requires a decision tree:
- Q1: Is target gene concordant with drug mechanism? → If no, contraindication
- Q2: FDA-approved for target disease? → HIGH. Phase 2/3? → MODERATE. No evidence → Section 6.5
- Q3: Safety signals? → Add safety note, potentially downgrade

The spec explicitly says: "Do not compute a numerical composite score."

**Find**: Show the prioritization/scoring code:
- `drug_scorer.py` — the entire composite scoring system
- The tier assignment logic
- Note the GAP: our system DOES compute a numerical composite score (0-95), which contradicts the spec's instruction

Show where the tier assignment happens and how it maps to the spec's decision tree.

---

### SPEC Section 6.4: Contraindications — Tiered Logic

The spec requires three tiers:
- **Tier 1 — Avoid**: Drug causes/worsens target disease (class effect). Independent of expression.
- **Tier 2 — Not Supported**: Target gene downregulated + drug would further inhibit
- **Tier 3 — Use With Caution**: Target mildly downregulated but drug has SOC/prophylactic value
- **SOC — Never Contraindicate**: Remove from this table, move to 6.1

**Find**: Show all contraindication code:
- Contraindication checking in `drug_agent_service.py`
- The 3-tier (HARD/MODERATE/SOFT) system — does it map to Tier 1/2/3?
- SOC protection in contraindication
- `collection_router.py` — any contraindication logic
- `docx_generation.py` — contraindication table rendering

Show how each tier is implemented and where the spec's logic is followed or diverged.

---

### SPEC Section 6.5: Gene-Targeted Unvalidated Candidates

The spec requires: Drugs targeting patient DEGs but lacking disease-specific evidence go in a SEPARATE section with a specific disclaimer label. They must NOT appear in the priority table.

**Find**: Show the code that separates validated from unvalidated drug candidates:
- `drug_agent_service.py` — candidate splitting logic
- `docx_generation.py` — Section 6.5 rendering
- Is the required disclaimer text present?

---

### SPEC Section 6.6: Required Confirmatory Tests

The spec requires: For every High and Moderate drug, specify confirmatory tests with test type and clinical rationale.

**Find**: Show where confirmatory tests are generated:
- `narrative_generation.py` — test recommendation logic
- `docx_generation.py` — confirmatory tests table
- `llm_knowledge.py` — any confirmatory test knowledge

---

### SPEC Section 7: Report Structure

The spec requires 9 sections in exact order:
1. Executive Summary (1.1 Key Findings, 1.2 Analysis Overview)
2. Clinical Context
3. Transcriptome Overview (gene categories, pathway analysis)
4. Cell-Type Deconvolution
5. Disease Activity Assessment
6. Therapeutic Implications (6.1-6.6 in order)
7. Appendix
8. Conclusion
9. References

**Find**: Show the report structure code in `docx_generation.py` and verify section ordering matches the spec.

---

### SPEC Section 8: Pre-Output Quality Checklist

The spec requires 12 quality checks before output:
1. Foundational therapy section exists with backbone drug?
2. No backbone SOC drug in contraindication table?
3. HCQ not contraindicated for SLE?
4. Every contraindicated drug has tier stated?
5. Downstream effector evidence checked for absent target genes?
6. Corticosteroid downstream effectors checked?
7. Unvalidated candidates isolated in Section 6.5?
8. No serological biomarker assessed with DEG surrogate?
9. Type B biomarkers carry required language?
10. IFN classification matches anifrolumab discussion?
11. Executive Summary agrees with Section 6?
12. Infection susceptibility noted when innate defense is SUPPRESSED?

**Find**: Is there ANY quality checklist logic in the code? Check:
- `narrative_generation.py` — any validation/checking
- `drug_agent_service.py` — any pre-output validation
- `docx_generation.py` — any consistency checks

This is likely a significant [GAP].

---

### SPEC Section 9: Required & Prohibited Language Patterns

**Required language** for various situations (see Table 22 in spec):
- Target absent: "Primary target [GENE] not detected in DEG data..."
- Downstream effectors found: "Downstream effectors [GENE_A ↓, GENE_B ↓] are consistent with..."
- IFN-LOW: "IFN-LOW with residual ISG activity... not currently supported — recommend serial..."
- Type B biomarker: "Requires direct [test name] — not assessable from RNA expression data"
- SOC drug target absent: "[Drug] is guideline-recommended backbone therapy... Transcriptomic data does not modify..."
- Novel gene: "clinical relevance in [disease] is not established — warrants further investigation"
- Multiple contraindication tiers: "Tier 1 — [reason]. Tier 2 — [reason]. Both apply independently."

**Prohibited**:
- "not a candidate" → must use "not currently supported by this molecular profile"
- Using inflammatory cytokines as surrogates for serological biomarkers
- Gene-targeted unvalidated candidates in same table as approved therapies
- Contraindicting backbone SOC drugs based on expression direction
- Reporting CIBERSORT fraction as percentage of total cells
- Definitive interpretation of cell-cycle pathway upregulation
- Fabricating data

**Find**: Show where required/prohibited language patterns are enforced in:
- `narrative_generation.py` — prompt templates, language patterns
- `llm_knowledge.py` — any language pattern enforcement
- `docx_generation.py` — text templates

---

### SPEC Section 10: SLE Disease-Specific Module

The spec has specific SLE rules:
- HCQ never contraindicated
- Belimumab eligibility based on TNFRSF13B/BLyS expression
- Anifrolumab requires IFN-HIGH
- TNF inhibitors: always Tier 1 + Tier 2
- Canakinumab: Tier 2 if IL1B down + limited SLE evidence note
- Cell cycle upregulation: interpretive tension, not escalation signal
- Neutrophil cross-reference with CAMP, PI3, GPR84, CLEC4D
- MMF absence warning for young/male patients

**Find**: Show all SLE-specific logic:
- `llm_knowledge.py` — SLE drug knowledge
- `drug_agent_service.py` — any SLE-specific handling
- `narrative_generation.py` — SLE-specific narrative
- Belimumab, anifrolumab, HCQ specific code

---

### SPEC Section 10.1: Template for Other Diseases

The spec requires: For non-SLE diseases, retrieve backbone SOC drugs, key molecular activity genes, class-effect contraindications, and validated molecular subtypes BEFORE analysis.

**Find**: How does the system handle non-SLE diseases? Is there a generic disease setup step? Check:
- `drug_agent_service.py` — disease-specific initialization
- `llm_knowledge.py` — disease knowledge retrieval
- `narrative_generation.py` — disease-agnostic template

---

### SPEC Section 11: Data Integrity Rules

The spec requires:
- Never fabricate numbers
- Distinguish statistically significant (adj.p < 0.05) from magnitude-only DEGs
- Clearly separate provided pathway enrichment from inferred
- Report total gene/pathway counts
- Cite database source for every drug recommendation
- Novel gene findings: state known biology, unknown disease role, needed investigation

**Find**: Show data integrity enforcement in all relevant files.

---

### SPEC Section 12: Mandatory Clinical Disclaimer

The spec requires specific disclaimer text on every report's final page.

**Find**: Show the disclaimer rendering in `docx_generation.py`.

---

## OUTPUT FORMAT

Create ONE markdown file organized exactly like this:

```
# Drug Agent v2.1 Specification Audit

## Section X: [Spec Section Name]

### Spec Requirement:
[What the spec says]

### Implementation Status: ✅ IMPLEMENTED / ⚠️ PARTIAL / ❌ GAP

### Code Location:
File: [path]
Function: [name]
Lines: [start-end]

### Actual Code:
```python
[paste the actual code]
```

### Gap Analysis:
[What's missing or different from the spec]

### Recommended Fix:
[What code changes are needed to close the gap]
```

Repeat for EVERY section of the spec. Do not skip any section. If the code is spread across multiple files, show ALL relevant locations.

After the full audit, add a summary table:

```
| Spec Section | Status | Primary File | Gap Description |
|---|---|---|---|
| Design Philosophy | ⚠️ PARTIAL | drug_scorer.py | Reasoning text generated but not traceable sentence format |
| 1. Input Data | ✅ | schemas.py | All inputs present |
| ... | ... | ... | ... |
```

## Key Files to Search

These are the primary files. Search ALL of them:

**Drug Agent Service:**
- `drug_agent/service/drug_agent_service.py` — main orchestration (836 lines)
- `drug_agent/service/drug_scorer.py` — scoring logic (574 lines)
- `drug_agent/service/collection_router.py` — discovery + contraindication (1069 lines)
- `drug_agent/service/result_aggregator.py` — result merging
- `drug_agent/service/schemas.py` — data schemas
- `drug_agent/service/test_service.py` — tests

**Drug Agent Core:**
- `drug_agent/recommendation/drug_ranker.py` — ranking
- `drug_agent/recommendation/evidence_compiler.py` — evidence compilation
- `drug_agent/retrieval/hybrid_search.py` — search logic
- `drug_agent/models/data_models.py` — data models
- `drug_agent/config/settings.py` — configuration
- `drug_agent/opentargets/ot_base.py` — OpenTargets integration

**Reporting Pipeline:**
- `reporting_pipeline_agent/narrative_generation.py` — narrative/prompt generation (2884 lines)
- `reporting_pipeline_agent/docx_generation.py` — DOCX rendering (4500+ lines)
- `reporting_pipeline_agent/core_types.py` — thresholds and constants
- `reporting_pipeline_agent/llm_knowledge.py` — LLM knowledge functions (has drug-related functions)

**Search patterns to use:**
```
# Find all drug-related functions
grep -rn "def.*drug\|def.*soc\|def.*contraindic\|def.*backbone\|def.*tier\|def.*concordance\|def.*biomarker" drug_agent/ reporting_pipeline_agent/

# Find all threshold definitions  
grep -rn "THRESHOLD\|HIGH_PRIORITY\|MODERATE\|LOG2FC\|adj_p\|p_value\|FDR" drug_agent/ reporting_pipeline_agent/

# Find all language patterns
grep -rn "not a candidate\|not currently supported\|expression-based\|backbone\|guideline\|SOC\|standard.of.care" reporting_pipeline_agent/

# Find all SLE-specific logic
grep -rn "SLE\|lupus\|hydroxychloroquine\|HCQ\|belimumab\|anifrolumab\|IFN\|interferon" drug_agent/ reporting_pipeline_agent/

# Find all gene classification logic
grep -rn "category.*A\|category.*B\|category.*C\|category.*D\|disease.driver\|hub.gene\|confirmed.*driver" reporting_pipeline_agent/

# Find all disclaimer/checklist logic
grep -rn "disclaimer\|checklist\|quality.*check\|validation.*check\|pre.output" reporting_pipeline_agent/
```

## Final Deliverable

The output should be a single markdown file that the expert teammate can read to see EXACTLY where each of her instructions is (or isn't) implemented. She should be able to open the codebase and go directly to the line numbers you reference.

Be thorough. Be honest about gaps. Don't guess — show real code or mark as [GAP].
