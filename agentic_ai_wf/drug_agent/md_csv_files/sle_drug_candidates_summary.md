# SLE Drug Candidates Summary — Causal Gene-Targeted Analysis

**Date:** 2026-03-12
**Source Data:** `sle_dag_causal_linkage.csv` (248+ genes with causal tiers)
**Output:** `sle_drug_candidates.csv` (32 drug candidates mapped to causal genes)

---

## Executive Summary

This analysis maps FDA-approved drugs, clinical trial candidates, and repurposing opportunities to genetically validated causal drivers of Systemic Lupus Erythematosus (SLE). The causal linkage data identifies **24 Tier 1** (Full Causal Chain: MR + eQTL + Disease + Pathway) and **224+ Tier 2** (Strong Causal: MR + eQTL + Disease) gene targets, of which we identified **32 actionable drug candidates** spanning 15 distinct gene targets.

---

## Key Findings

### 1. FDA-Approved Drugs Already Targeting SLE Causal Genes

| Drug | Target Gene | Tier | Status |
|------|------------|------|--------|
| Belimumab (Benlysta) | BLK pathway (BAFF) | Tier 1 | FDA Approved for SLE (2011) |
| Anifrolumab (Saphnelo) | IRF4 pathway (IFNAR) | Tier 1 | FDA Approved for SLE (2021) |
| Voclosporin (Lupkynis) | HLA-DRB1/IL12RB2 (calcineurin) | Tier 1 | FDA Approved for LN (2021) |
| Obinutuzumab (Gazyva) | BLK pathway (CD20) | Tier 1 | FDA Approved for LN (2025) |

These four drugs validate the causal linkage model: they target pathways linked to Tier 1 causal genes and have demonstrated clinical efficacy in SLE.

### 2. Highest-Priority Pipeline Candidates (Phase 2/3 for SLE)

| Drug | Target Gene | Tier | Phase | Alignment |
|------|------------|------|-------|-----------|
| **Deucravacitinib** | TYK2 | Tier 2 | Phase 3 (POETYK SLE-1/2) | Strong -- TYK2 is a risk gene; selective inhibitor blocks IL-12/23 and IFN pathways |
| **Iberdomide (CC-220)** | IKZF3 | Tier 1 | Phase 2 | Complex -- degrades protective IKZF3 but reduces B cell autoimmunity |
| **Dapirolizumab pegol** | CD40 | Tier 2 | Phase 3 (positive results) | Strong -- blocks CD40L co-stimulation; significant SRI-4 improvement |
| **Iscalimab** | CD40 | Tier 2 | Phase 2 (LN) | Strong -- 42% proteinuria reduction in lupus nephritis |
| **Telitacicept** | BLK pathway (BAFF/APRIL) | Tier 1 | Phase 3 (positive) | Strong -- 67% vs 33% SRI-4 response rate |
| **Ravulizumab** | C4B pathway (C5) | Tier 1 | Phase 2 (SANCTUARY) | Strong -- complement inhibition for lupus nephritis |
| **FT819 (CAR-T)** | BLK/IKZF3 (CD19) | Tier 1 | Pivotal (planned 2026) | Strong -- iPSC-derived CAR-T eliminates autoreactive B cells |

### 3. Top Repurposing Candidates (Approved for Other Indications)

| Drug | Approved For | Target Gene | Tier | Rationale |
|------|-------------|------------|------|-----------|
| **Baricitinib** | RA | GRB2 (JAK-STAT) | Tier 1 | GRB2 is a Tier 1 risk gene in JAK-STAT; JAK1/2 inhibition directly addresses this pathway |
| **Tipifarnib** | Investigational (HNSCC) | HRAS | Tier 1 | HRAS is a Tier 1 risk gene; farnesyltransferase inhibition blocks Ras membrane localization |
| **Eculizumab** | PNH/aHUS | C4B (complement) | Tier 1 | Complement is a validated risk pathway; case reports show efficacy in severe LN |
| **Sirolimus** | Transplant | HRAS (mTOR) | Tier 1 | Included in 2025 SLE treatment guidelines; blocks mTOR downstream of MAPK |
| **Alemtuzumab** | MS | CD52 | Tier 2 | CD52 is a Tier 2 gene; lymphocyte depletion may benefit refractory SLE |

### 4. Novel Target Opportunities Without Current Drugs

Several Tier 1 causal genes lack approved drugs or active clinical programs in SLE:

| Gene | Direction | Pathway | Opportunity |
|------|-----------|---------|-------------|
| **CLDN23** | Risk (inhibit) | Cell adhesion | No known inhibitors; tight junction target |
| **FBXO6** | Risk (inhibit, log2fc=1.005) | Ubiquitin/Immune system | E3 ligase component; no clinical-stage inhibitors |
| **FCER1G** | Protective (enhance) | Immune response/T cell | Fc receptor gamma; limited direct agonists available |
| **CD226** | Protective (enhance) | IFN-gamma production | DNAM-1; potential for agonist antibodies |
| **C4B** | Risk | Complement activation | Direct C4B modulation complex; upstream complement inhibitors exist |
| **HLA-C / HLA-DMA / HLA-DRB1** | Risk | MHC/Antigen processing | Not directly druggable; CTSS inhibitors block MHC class II processing indirectly |

---

## Therapeutic Strategy Matrix

### RISK Genes (Tier 1) -- Therapeutic Goal: INHIBITION

| Gene | Drug Strategy | Best Candidates |
|------|--------------|-----------------|
| GRB2 (JAK-STAT) | JAK inhibitors | Baricitinib, Deucravacitinib (TYK2) |
| CTSS (Neutrophil degranulation) | Cathepsin S inhibitors | RWJ-445380, RO5459072, ASP1617 |
| HRAS (MAPK cascade) | Farnesyltransferase/mTOR inhibitors | Tipifarnib, Sirolimus |
| DRD4 (MAPK cascade) | Dopamine D4 antagonists | Clozapine (repurpose), L-745870 |
| HLA-DRB1/HLA-DMA (MHC) | Indirect via CTSS inhibitors | RWJ-445380, RO5459072 |
| IL12RB2 (T cell activation) | IL-12/23 pathway blockers | Deucravacitinib (TYK2 inhibitor) |
| C4B (Complement) | Complement inhibitors | Eculizumab, Ravulizumab |
| ICAM1 (Interferon signaling) | LFA-1/ICAM1 blockers | Lifitegrast (repurpose) |

### PROTECTIVE Genes (Tier 1) -- Therapeutic Goal: PATHWAY SUPPORT

| Gene | Drug Strategy | Best Candidates |
|------|--------------|-----------------|
| BLK (B cell regulation) | B cell modulators (BAFF/APRIL) | Belimumab, Telitacicept, Atacicept |
| IKZF3 (Leukocyte proliferation) | Cereblon modulators | Iberdomide (complex alignment) |
| IRF4 (Interferon signaling) | Type I IFN receptor blockade | Anifrolumab |
| CD226 (IFN-gamma production) | No approved agonists | Unmet need |
| FCER1G (Immune response) | No direct modulators | Unmet need |

---

## Notable Tier 2 Targets with Drug Potential

| Gene | Direction | Drug/Strategy | Status |
|------|-----------|--------------|--------|
| TYK2 | Risk | Deucravacitinib | Phase 3 for SLE |
| CD40 | Protective | Dapirolizumab pegol, Iscalimab | Phase 2-3 for SLE |
| C3AR1 (log2fc=1.012) | Risk | Avacopan (C5aR antagonist) | FDA approved for ANCA vasculitis |
| UBE2L3 | Risk | No direct inhibitors | Unmet need |
| TNFSF12 (TWEAK) | Risk | BIIB023 (anti-TWEAK) | Phase 2 for LN |
| CD52 | Protective | Alemtuzumab | FDA approved for MS |
| CD274 (PD-L1) | Protective | Immune checkpoint context | Complex; checkpoint inhibitors worsen SLE |

---

## Conclusions

1. **The causal linkage model is validated** by the fact that 4 FDA-approved SLE drugs target pathways linked to Tier 1 causal genes (BLK, IRF4, HLA/IL12RB2, complement).

2. **Deucravacitinib (TYK2 inhibitor)** is the most promising pipeline candidate, directly targeting a Tier 2 risk gene with Phase 3 trials underway for SLE.

3. **Telitacicept** showed remarkable Phase 3 efficacy (67% vs 33% response) and addresses the BLK-related B cell pathway.

4. **CTSS inhibitors** represent a high-value opportunity: CTSS is a Tier 1 risk gene with multiple clinical-stage inhibitors (RWJ-445380, RO5459072, ASP1617) that could be repurposed for SLE.

5. **Baricitinib** (JAK1/2 inhibitor) is a strong repurposing candidate given GRB2's role as a Tier 1 risk gene in the JAK-STAT pathway.

6. **Complement inhibitors** (eculizumab, ravulizumab) address the C4B Tier 1 risk pathway, with ravulizumab in Phase 2 for lupus nephritis.

7. **Unmet needs** remain for direct modulators of CLDN23, FBXO6, CD226, FCER1G, and UBE2L3 -- all validated causal genes without targeted therapeutics.

---

## Data Sources

- FDA drug approval database
- ClinicalTrials.gov (active SLE/lupus trials)
- ChEMBL (drug-target relationships)
- DrugBank (mechanism of action data)
- OpenTargets (genetic association evidence)
- PubMed/PMC (clinical trial publications)

## References

- [Advances in Targeted Therapy for SLE (MDPI 2025)](https://www.mdpi.com/1422-0067/26/3/929)
- [Phase 2 Trial of Iberdomide in SLE (NEJM)](https://www.nejm.org/doi/full/10.1056/NEJMoa2106535)
- [Phase 3 Trial of Telitacicept for SLE (NEJM)](https://www.nejm.org/doi/full/10.1056/NEJMoa2414719)
- [Deucravacitinib Phase 2 in SLE (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10100399/)
- [Cathepsin S Inhibitor Development (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11417842/)
- [Complement Targeting in SLE (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S017129852500049X)
- [Novel Anti-CD40 in Lupus Nephritis (Medscape)](https://www.medscape.com/viewarticle/novel-anti-cd40-antibody-shows-promise-lupus-nephritis-2025a1000mco)
- [Dapirolizumab Pegol Phase 3 (UCB)](https://www.ucb.com/newsroom/press-releases/article/dapirolizumab-pegol-phase-3-data-in-sle-presented-at-the-annual-european-congress-of-rheumatology-eular-show-improvement-in-fatigue-and-reduction-in-disease-activit)
- [Signaling Pathways in SLE (Frontiers)](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1735301/full)
- [Anti-TWEAK in Lupus Nephritis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3428508/)
