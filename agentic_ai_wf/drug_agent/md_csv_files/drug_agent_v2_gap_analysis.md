# Drug Agent Prompt v2.1 — Gap Analysis vs Current Implementation

## Overview

This document compares each section of the **Drug Agent Prompt v2.1 specification** (the "spec") against the **current Drug Agent service** (drug_scorer.py, collection_router.py, drug_agent_service.py, result_aggregator.py, schemas.py, and the reporting pipeline's docx_generation.py / narrative_generation.py). Items are rated as ✅ Already Implemented, ⚠️ Partially Implemented, or ❌ Not Implemented.

---

## Design Philosophy & Core Principle

| Spec Requirement | Status | Notes |
|---|---|---|
| Drug recommendations traceable to specific gene + direction + mechanism | ⚠️ Partial | The scorer tracks `patient_gene_log2fc`, `direction`, and `action_type` on each target. DOCX renders this. But disease-discovered drugs often show "Not detected in patient's transcriptome" because their pharmacological target (e.g., IL4R) differs from the patient DEG (e.g., IL4). **Fix in progress**: gene-family fallback and context-aware messaging (Copilot's Fix A/B/C). |
| If the traceability sentence can't be completed, label as SOC-based | ❌ Missing | No SOC labeling system exists. All drugs go through the same scoring pipeline regardless of SOC status. |

---

## Section 1: Input Data Contract

| Spec Requirement | Status | Notes |
|---|---|---|
| Accepts disease_name, patient_deg_table, pathway_enrichment_table, cibersort_table | ✅ Done | `DrugQueryRequest` accepts `disease`, `genes` (list of GeneInput with log2fc/direction), `enriched_pathways`, and cibersort data flows through the reporting pipeline. |
| DEG thresholds: ≥ +0.58 / ≤ −0.58 for up/down classification | ⚠️ Partial | The current system uses `log2FC > 0` = upregulated, `log2FC < 0` = downregulated (no 0.58 threshold). The spec's 1.5-fold threshold is stricter. Currently, a gene with log2FC of 0.1 would be treated as "upregulated." |
| Flag adj.p < 0.05 explicitly with ✓ | ⚠️ Partial | The DOCX report shows adj.p values in tables. Statistical significance is used in the composite score (+10 for adj.p < 0.05). But there is no ✓ marker system in the gene tables. |

---

## Section 2: Standard-of-Care Anchor Layer ❌ MAJOR GAP

| Spec Requirement | Status | Notes |
|---|---|---|
| SOC backbone therapies listed BEFORE any gene-to-drug matching | ❌ Missing | No Section 6.1 "Foundational/SOC Therapies" exists. All drugs go through the same scoring pipeline. |
| SOC-1: Never contraindicate a backbone drug because its target is downregulated | ❌ Missing | The scorer gives -20 safety penalty and contraindication flags based purely on expression direction. A backbone drug like HCQ for SLE could be contraindicated if its target gene is downregulated. |
| SOC-2: Never exclude because target gene is absent from DEG table | ❌ Missing | If target gene isn't in patient DEGs, `patient_gene_log2fc = None`, direction score = 0. Drug gets no molecular credit and may score too low to appear. |
| SOC-3: Backbone drug must never appear in contraindication table | ❌ Missing | The current system has no SOC whitelist. Belimumab (FDA-approved for SLE) appeared in contraindications due to FAERS adverse event false positives. This was partially fixed with the drug-causes-disease check, but there's no fundamental SOC protection. |
| SOC-4: HCQ must never be contraindicated for SLE | ❌ Missing | No disease-specific SOC rules exist. |

**Impact**: This is the single biggest gap. The spec's design philosophy puts SOC drugs in a protected layer that expression data can inform but never override. The current system treats every drug equally through the scoring pipeline, so standard-of-care drugs can be penalized or contraindicated by molecular data.

---

## Section 3: Gene Analysis Pipeline

### 3.1 Gene Classification (A/B/C/D Categories)

| Spec Requirement | Status | Notes |
|---|---|---|
| Category A: Confirmed Disease Drivers (OMIM, DisGeNET) | ⚠️ Partial | The report has a "Disease-Relevant Gene Findings" section using Cancer Gene Census and OpenTargets data. But it doesn't use the formal A/B/C/D classification system. |
| Category B: PPI-Connected Hub Genes | ❌ Missing | No protein-protein interaction analysis. No PPI score calculation or hub gene identification. |
| Category C: Statistically Significant DEGs | ⚠️ Partial | DEGs are reported with adj.p values, but not formally classified into Category C with the ✓ marker system. |
| Category D: Notable Expression, Unknown Significance | ✅ Done | Section 3.3 "Dysregulated Genes of Unknown Clinical Significance" covers this with appropriate "clinical relevance uncertain" language. |

### 3.2 Gene-to-Drug Matching Logic

| Spec Requirement | Status | Notes |
|---|---|---|
| Step 1 — Direction check | ✅ Done | `_target_direction()` in drug_scorer.py checks patient gene log2FC direction. |
| Step 2 — Drug lookup across databases | ✅ Done | collection_router.py queries ChEMBL, OpenTargets, FDA labels, knowledge graph. |
| Step 3 — Concordance check (direction alignment) | ✅ Done | The scorer awards +9/+18 for concordant direction, applies -20 safety penalty for discordant direction. The spec's 4-case concordance matrix is implemented. |
| Step 4 — Evidence check (FDA approval, trials) | ✅ Done | `indication_bonus` (+10), OT score (+15), clinical trial data integrated. |
| Step 5 — Safety check (disease-specific risks) | ⚠️ Partial | FAERS adverse events checked, but the drug-causes-disease vs drug-treats-disease distinction was only recently added and is still being refined. |

### 3.3 Target Gene Absent from DEG Data

| Spec Requirement | Status | Notes |
|---|---|---|
| Don't automatically exclude the drug | ⚠️ Partial | Drug isn't excluded, but gets direction score = 0 and no magnitude credit, making it score low naturally. |
| Check for downstream effector genes | ❌ Missing | No downstream effector checking. The spec wants: "if ANXA3, ADM, DUSP1 downregulated → corticosteroid pathway active." This doesn't exist. |
| SOC drugs recommended unconditionally when target absent | ❌ Missing | No SOC layer (see Section 2 gap). |
| Required language: "Primary target [GENE] not detected in DEG data..." | ⚠️ Partial | Current message is "Not detected in patient's transcriptome" which is close but doesn't include the downstream effector fallback language. **Fix in progress**: Context-aware messaging (Copilot's Fix A). |

---

## Section 4: Pathway & Cell-Type Analysis

| Spec Requirement | Status | Notes |
|---|---|---|
| Report all FDR < 0.05 pathways from provided table | ✅ Done | Pathway enrichment table rendered in Section 3.4 and Appendix. |
| Disease-specific mechanistic interpretation per pathway | ✅ Done | LLM generates disease-contextualized interpretation for each pathway. |
| Distinguish table-derived vs inferred pathways | ⚠️ Partial | The system uses provided pathway data, but doesn't explicitly label whether interpretation is from table or inferred. |
| CIBERSORT: explain Presence % is NOT cell proportion | ✅ Done | Report includes: "Presence % reflects consistency of detection across analyses and does not sum to 100%." |
| Cross-reference cell types against DEG markers | ✅ Done | xCell results include "Supporting Markers" column (e.g., "IFNG↑", "KRT17↓"). |
| Cell-cycle pathway interpretive caution | ⚠️ Partial | Report notes "potential mechanism for uncontrolled cell growth" but doesn't present the spec's required "two most plausible interpretations" or recommend serial profiling. |

---

## Section 5: Disease Activity Assessment

| Spec Requirement | Status | Notes |
|---|---|---|
| Disease-specific molecular activity parameters | ✅ Done | The report includes HER2 Signaling, Hormone Receptor Status, Cell Cycle Regulation, PI3K Pathway Activity, DNA Repair Mechanisms with HIGH/MODERATE/LOW ratings. |
| Evidence gene anchors per parameter | ⚠️ Partial | Parameters reference patient genes but don't use the formal "≥2 upregulated → HIGH" counting logic from the spec. |
| IFN Signature Mixed Pattern Rule | ❌ Missing | No IFN signature-specific logic. The spec requires exact ISG counting and specific anifrolumab eligibility language. Only relevant for SLE/autoimmune, not breast cancer. |
| Molecular phenotype classification | ✅ Done | Report classifies as "HER2-DRIVEN" with TNBC subtype similarity and supporting module scores. |

---

## Section 6: Therapeutic Implications ⚠️ STRUCTURAL GAPS

### 6.1 SOC Backbone Table

| Spec Requirement | Status | Notes |
|---|---|---|
| Unconditional SOC section appearing first | ❌ Missing | No dedicated SOC section. The report goes straight to "Drug Prioritization with Mechanistic Reasoning." |
| SOC drugs listed regardless of expression data | ❌ Missing | Every drug must earn its place through the scoring pipeline. |
| Note whether target gene was detected, but never contraindicate | ❌ Missing | SOC drugs can be contraindicated by the current system. |

### 6.2 Biomarker-Therapy Concordance Table

| Spec Requirement | Status | Notes |
|---|---|---|
| Type A (transcriptomically assessable) vs Type B (requires orthogonal testing) classification | ❌ Missing | The current Biomarker-Therapy Concordance Table (Section 6.1 in current report) doesn't distinguish Type A vs Type B. It assesses all biomarkers against DEG data, including serological biomarkers that can't be inferred from RNA. |
| Type B language: "Requires direct [test] — not assessable from transcriptomic data" | ❌ Missing | Current table shows "⚠ NOT ASSESSED" when genes are absent, but doesn't explain WHY some biomarkers fundamentally can't be assessed from RNA. |
| Never use cytokines as proxy for serological biomarkers | ⚠️ Partial | The current system doesn't explicitly use cytokines as proxies, but it also doesn't prevent it with Type A/B classification. |

### 6.3 Evidence-Based Drug Prioritization

| Spec Requirement | Status | Notes |
|---|---|---|
| Tier decision tree (Q1→Q2→Q3) | ⚠️ Partial | The scorer uses a weighted composite score (0-95) instead of the spec's decision tree. The composite approach captures similar signals (direction concordance, FDA approval, safety) but as numerical weights rather than binary tier gates. |
| HIGH PRIORITY = Target concordant + FDA-approved for disease | ⚠️ Partial | Threshold is ≥55 composite score. An FDA-approved concordant drug typically scores ~65+ (direction +18, clinical +25, indication +10, OT +15), which maps roughly to HIGH. But the numerical approach allows non-concordant drugs to reach moderate priority through other score components. |
| No numerical composite score — use tier reasoning | ❌ Different | The spec explicitly says "Do not compute a numerical composite score." The current system IS a numerical composite score. This is a fundamental philosophical difference. The current approach is defensible for an automated pipeline (reproducible, deterministic) but differs from the spec's qualitative reasoning model. |
| Gene-targeted unvalidated in Section 6.5, not in priority table | ✅ Done | `gene_targeted_only` list exists as separate field on `DrugQueryResponse`. Rendered separately with disclaimer language. |

### 6.4 Contraindications — Tiered Logic

| Spec Requirement | Status | Notes |
|---|---|---|
| Tier 1 (drug causes/worsens disease) | ⚠️ Partial | Drug-causes-disease check was recently added using FAERS/AE cross-reference, but the AE-based approach has false positive issues (drug-treats-disease flares misidentified as drug-causes-disease). |
| Tier 2 (target downregulated + drug would further inhibit) | ✅ Done | The scorer applies -20 safety penalty when drug inhibits an already downregulated target. |
| Tier 3 (mildly downregulated, use with caution) | ❌ Missing | No "caution" tier. It's binary: contraindicated or not. |
| SOC drugs never in contraindication table | ❌ Missing | No SOC protection layer. |
| Multi-tier reporting (drug can be Tier 1 + Tier 2 simultaneously) | ❌ Missing | Current system outputs a single contraindication flag, not multiple tiers. |

### 6.5 Gene-Targeted Unvalidated Candidates

| Spec Requirement | Status | Notes |
|---|---|---|
| Separate section with hypothesis-generating disclaimer | ✅ Done | Exists with appropriate language in the DOCX. |
| Required label about no established evidence | ✅ Done | Disclaimer rendered in italic. |

### 6.6 Required Confirmatory Tests

| Spec Requirement | Status | Notes |
|---|---|---|
| Per-drug confirmatory test table | ✅ Done | The Biomarker-Therapy Concordance Table includes "Required Test" column (e.g., "IHC for HER2", "Genetic testing for BRCA mutations"). |
| Separate by purpose (serological, imaging, genetic) | ⚠️ Partial | Tests listed but not categorized by type. |

---

## Section 7: Report Structure

| Spec Section | Current Status | Notes |
|---|---|---|
| §1 Executive Summary | ✅ Done | 3-5 paragraph narrative with molecular classification and therapeutic conclusions. |
| §1.1 Key Findings | ✅ Done | Bullet points with gene names, log2FC, and clinical implications. |
| §1.2 Analysis Overview | ✅ Done | Table with DEG counts, pathways, cell types. |
| §2 Clinical Context | ✅ Done | Disease overview, molecular heterogeneity, clinical significance. |
| §3 Transcriptome Overview | ✅ Done | Gene prioritization, findings, pathway analysis. But uses expression-magnitude ranking, not A/B/C/D categories. |
| §4 Cell-Type Deconvolution | ✅ Done | xCell results with marker cross-validation. |
| §5 Disease Activity Assessment | ✅ Done | Molecular activity parameters, phenotype classification, prognostic features. |
| §6 Therapeutic Implications | ⚠️ Partial | Missing SOC section (6.1), missing Type A/B biomarker classification (6.2), missing tiered contraindications (6.4). |
| §7 Appendix | ✅ Done | Complete gene, pathway, and xCell tables. |
| §8 Conclusion | ✅ Done | Summary with clinical management implications. |
| §9 References | ✅ Done | Database citations included. |

---

## Section 8: Pre-Output Quality Checklist

| Check | Status | Notes |
|---|---|---|
| SOC backbone section present | ❌ Missing | No SOC section exists to check. |
| No SOC drug in contraindication table | ❌ Missing | No SOC protection. |
| HCQ never contraindicated for SLE | ❌ Missing | No disease-specific SOC rules. |
| Tiered contraindication with all applicable tiers | ❌ Missing | Single flag, not multi-tier. |
| Downstream effector evidence checked before flagging absent targets | ❌ Missing | No downstream effector checking. |
| Gene-targeted candidates isolated from priority table | ✅ Done | Separate list exists. |
| No serological biomarkers assessed via DEG surrogates | ❌ Missing | No Type A/B classification. |
| IFN classification matches anifrolumab discussion | N/A | Only relevant for SLE/autoimmune diseases. |
| Executive Summary agrees with Section 6 | ✅ Done | LLM generates consistent narrative. |

---

## Section 9: Required & Prohibited Language

| Pattern | Status | Notes |
|---|---|---|
| "Primary target [GENE] not detected in DEG data..." | ⚠️ Partial | Shows "Not detected in patient's transcriptome" — close but missing downstream effector fallback. **Fix in progress.** |
| "Primary target not in DEG data. Downstream effectors..." | ❌ Missing | No downstream effector language. |
| "[Drug] is guideline-recommended backbone therapy..." | ❌ Missing | No SOC language. |
| "[Gene] clinical relevance in [disease] is not established..." | ✅ Done | Used in Section 3.3 for novel genes. |
| Never write "not a candidate" — write "not currently supported" | ⚠️ Partial | The LLM narrative varies. Not enforced programmatically. |
| Never use cytokines as serological surrogates | ❌ Not enforced | No Type A/B gating prevents this. |
| Never contraindicate SOC based on expression | ❌ Missing | No SOC protection. |
| CIBERSORT fractions described as enrichment scores, not % of cells | ✅ Done | Correctly labeled as enrichment scores. |

---

## Section 10: Disease-Specific Modules

| Spec Requirement | Status | Notes |
|---|---|---|
| SLE-specific backbone SOC list | ❌ Missing | No disease-specific module system. |
| HCQ universal backbone rule | ❌ Missing | No SOC protection. |
| Belimumab eligibility nuance | ❌ Missing | Belimumab was previously contraindicated by FAERS false positive. |
| Anifrolumab IFN-HIGH requirement | ❌ Missing | No IFN signature gating for anifrolumab. |
| TNF inhibitor dual-tier contraindication | ❌ Missing | Single contraindication flag. |
| Template for other diseases (retrieve SOC, key genes, class contraindications) | ❌ Missing | No disease-specific retrieval module. |

---

## Section 11: Data Integrity Rules

| Rule | Status | Notes |
|---|---|---|
| Never fabricate numbers | ✅ Done | All values traced to patient data or database lookups. |
| Distinguish adj.p < 0.05 from magnitude-only findings | ⚠️ Partial | Adj.p displayed in tables but no ✓ marker system. |
| Pathway source attribution (table vs inferred) | ⚠️ Partial | Uses provided data, but doesn't explicitly label provenance. |
| Report totals (genes analyzed, thresholds, pathways tested) | ✅ Done | Analysis Overview table provides these counts. |
| Cite database sources per drug | ✅ Done | "Sources: FDA_Drug_Labels" shown per drug in the report. |

---

## Section 12: Clinical Disclaimer

| Requirement | Status | Notes |
|---|---|---|
| Verbatim disclaimer on final page | ✅ Done | Report includes "This report is for research purposes only..." disclaimer matching the spec's intent. Wording is slightly different but functionally equivalent. |

---

## Priority Summary: What Needs to Change

### 🔴 Critical Gaps (Fundamentally missing functionality)

1. **SOC Anchor Layer (Section 2)** — No backbone drug protection. This is the spec's most important architectural concept and doesn't exist at all. Requires: disease-specific SOC drug lists, whitelist protection against contraindication, unconditional rendering in Section 6.1.

2. **Biomarker Type A/B Classification (Section 6.2)** — No distinction between transcriptomically assessable biomarkers and those requiring orthogonal testing. Risk: system could use gene expression as invalid proxy for serological tests.

3. **Tiered Contraindication Logic (Section 6.4)** — Current binary contraindication (yes/no) vs spec's 3-tier system (Avoid / Not Supported / Use With Caution). Multi-tier reporting (drug can be Tier 1 + Tier 2) doesn't exist.

4. **Downstream Effector Checking (Section 3.3)** — When target gene is absent, no check for downstream pathway genes that could indicate the pathway is active.

### 🟡 Moderate Gaps (Partially implemented, needs enhancement)

5. **Gene A/B/C/D Classification (Section 3.1)** — Current system ranks by composite score and expression magnitude. Spec wants formal 4-category classification with PPI hub gene analysis (Category B).

6. **Numerical Score vs Tier Reasoning (Section 6.3)** — Philosophical difference. Current composite score is defensible for automation but conflicts with spec's "do not compute a numerical composite score" instruction. Could be reconciled by mapping score ranges to tiers with reasoning explanations.

7. **DEG Threshold 0.58 (Section 1.1)** — Current uses 0 as threshold (any positive = up, any negative = down). Spec requires ±0.58 (1.5-fold change). Would filter out low-magnitude changes.

8. **Disease-Specific Modules (Section 10)** — No per-disease retrieval of backbone SOCs, key molecular activity genes, or class-effect contraindications.

### 🟢 Already Working Well

9. Gene-to-drug matching core logic (concordance checking, direction scoring)
10. Multi-collection database architecture (ChEMBL, OpenTargets, FDA, FAERS, clinical trials)
11. Pathway and cell-type analysis with cross-validation
12. Disease activity assessment with molecular phenotyping
13. Report structure (8 of 9 sections match spec)
14. Gene-targeted unvalidated candidates separation
15. Clinical disclaimer and data integrity
16. Confirmatory test recommendations

### 🔧 Fixes Already In Progress (From Current Copilot Implementation)

17. Word-boundary regex fix (Phase 1) — addresses substring matching false positives
18. Discovery path tracking + genetic eligibility flagging (Phase 2)
19. Claude Opus semantic validation (Phase 3) — addresses mechanistic paradoxes
20. Context-aware "Not detected" messaging — addresses target-absent display issue
21. Gene-family receptor-suffix fallback — addresses IL4R↔IL4 type mismatches
22. Target sorting (matched first) — addresses display ordering
