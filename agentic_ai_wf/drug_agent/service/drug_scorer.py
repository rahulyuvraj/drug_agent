"""Composite evidence-based drug scoring (0-100) with configurable weights."""

import logging
import re
import unicodedata
import numpy as np
from typing import Optional

from agentic_ai_wf.reporting_pipeline_agent.core_types import DEG_LOG2FC_THRESHOLD

# Biomarker status → activation fraction (0.0–1.0)
_STATUS_TO_SCORE = {
    "high": 1.0, "elevated": 0.8, "positive": 0.8, "overexpressed": 0.9,
    "moderate": 0.5, "intermediate": 0.5,
    "low": 0.2, "negative": 0.2, "absent": 0.0, "not_assessed": -1.0,
}


def _disease_matches(alias: str, text: str) -> bool:
    """Word-boundary match to prevent 'myopathy' matching 'cardiomyopathy'."""
    if not alias or not text:
        return False
    return bool(re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE))

from .schemas import (
    DrugCandidate, DrugQueryRequest, ScoreBreakdown, ScoringConfig,
    _GENE_TO_PROTEINS,
)

logger = logging.getLogger(__name__)


class DrugScorer:

    def __init__(self, config: Optional[ScoringConfig] = None, embedder=None):
        self.config = config or ScoringConfig()
        self.embedder = embedder
        self._moa_embed_cache: dict = {}

    def score(self, candidate: DrugCandidate, request: DrugQueryRequest) -> ScoreBreakdown:
        s = ScoreBreakdown()
        s.target_direction_match = self._target_direction(candidate, request)
        s.target_magnitude_match = self._target_magnitude(candidate, request, s.target_direction_match)
        s.clinical_regulatory_score = self._clinical_regulatory(candidate, request)
        ot_val, ot_fallback = self._ot_score(candidate, request)
        s.ot_association_score = ot_val
        s.pipeline_evidence_used = ot_fallback
        s.pathway_concordance = self._pathway_match(candidate, request)
        s.safety_penalty = self._safety_penalty(candidate)

        # Target-only mode: cap safety penalty — the user is querying by
        # target, not treating a specific patient, so extensive FAERS data
        # on a widely-used drug should not dominate the score.
        if getattr(request, 'is_target_only_query', False):
            s.safety_penalty = max(s.safety_penalty, -10.0)

        # Pathway class bonus: if the candidate was discovered via marker gene inference
        if candidate.pathway_class_match:
            s.pathway_concordance = max(s.pathway_concordance, self.config.pathway_weight * 0.6)

        s.disease_relevant = self._check_disease_relevance(candidate, request)

        # Gene evidence stratum quality for tier reasoning
        gene_map = self._build_gene_map(request)
        best_sm = 0.0
        for t in candidate.targets:
            g = gene_map.get(t.gene_symbol.upper())
            if g and abs(g.log2fc) >= DEG_LOG2FC_THRESHOLD:
                stratum = getattr(g, 'evidence_stratum', None)
                sm = self.config.stratum_multipliers.get(stratum, 1.0) if stratum else 1.0
                best_sm = max(best_sm, sm)
        s.gene_evidence_quality = best_sm if best_sm > 0 else 1.0

        # FDA-approved for this exact disease deserves a floor boost
        if candidate.identity.is_fda_approved and self._has_disease_indication(candidate, request):
            s.disease_indication_bonus = 10.0

        # Exact target match bonus: reward drugs whose targets match queried genes (+ protein aliases)
        if request.genes:
            queried_names: set = set()
            for g in request.genes:
                sym = g.gene_symbol.upper()
                queried_names.add(sym)
                queried_names.update(a.upper() for a in _GENE_TO_PROTEINS.get(sym, []))
            drug_targets: set = set()
            for t in candidate.targets:
                sym = t.gene_symbol.upper()
                drug_targets.add(sym)
                drug_targets.update(a.upper() for a in _GENE_TO_PROTEINS.get(sym, []))
            if queried_names & drug_targets:
                s.target_exact_match_bonus = self.config.target_exact_match_weight

        s.calculate()

        # --- Tier Reasoning (Q1→Q2→Q3 decision tree) ---
        s.tier_reasoning = self._tier_reasoning(s, candidate)

        # --- Signature Gating (generic biomarker→drug mechanism matching) ---
        s = self._signature_gate(s, candidate, request)

        return s

    def _signature_gate(self, s: ScoreBreakdown, candidate: DrugCandidate,
                        request: DrugQueryRequest) -> ScoreBreakdown:
        """Gate drugs on matching biomarker/signature status when relevant.

        Bridges request.biomarkers into signature_scores format, then uses
        semantic similarity between the signature name and the drug’s
        mechanism context to decide whether gating applies.
        """
        sig_scores = dict(getattr(request, 'signature_scores', None) or {})

        # Bridge BiomarkerContext → signature_scores and track relevant genes
        sig_genes: dict[str, set[str]] = {}  # sig_key → set of gene symbols
        for bm in (request.biomarkers or []):
            if getattr(bm, 'biomarker_type', None) == 'B':
                continue  # Type B biomarkers cannot be assessed from RNA
            key = bm.biomarker_name.lower().replace(" ", "_")
            if key not in sig_scores:
                status_lower = (bm.status or "").lower()
                frac = _STATUS_TO_SCORE.get(status_lower, 0.5)
                sig_scores[key] = {
                    "level": (bm.status or "UNKNOWN").upper(),
                    "activation_score": max(frac, 0) * 100,
                }
            # Always track supporting genes (even for pre-existing sig_scores)
            if bm.supporting_genes:
                sig_genes.setdefault(key, set()).update(
                    g.upper() for g in bm.supporting_genes
                )

        if not sig_scores or not self.embedder:
            return s

        # Build drug mechanism context for semantic matching
        drug_ctx = " ".join(filter(None, [
            candidate.identity.drug_name,
            candidate.identity.pharm_class_moa or "",
            " ".join(t.gene_symbol for t in candidate.targets),
        ]))
        drug_vec = self.embedder.encode(drug_ctx)
        drug_norm = np.linalg.norm(drug_vec) + 1e-10
        drug_target_genes = {t.gene_symbol.upper() for t in candidate.targets}

        for sig_key, sig_data in sig_scores.items():
            sig_vec = self.embedder.encode(sig_key.replace("_", " "))
            sim = float(np.dot(drug_vec / drug_norm, sig_vec / (np.linalg.norm(sig_vec) + 1e-10)))
            # Gene overlap: biomarker genes ∩ drug targets → strong relevance
            gene_overlap = bool(sig_genes.get(sig_key, set()) & drug_target_genes)
            if sim < 0.50 and not gene_overlap:
                continue

            level = sig_data.get("level", "").upper()
            pct = sig_data.get("activation_score", 0)
            label = sig_key.replace("_", " ").upper()

            if level == "HIGH" or pct >= 75:
                bonus = min(pct / 100 * self.config.signature_bonus_max, self.config.signature_bonus_max)
                s.signature_bonus = round(bonus, 2)
                s.tier_reasoning = (
                    f"{label} signature {level} ({pct:.0f}%) — "
                    f"patient aligns with drug's mechanism (+{s.signature_bonus:.1f}). " + s.tier_reasoning
                )
            elif level in ("LOW", "NEGATIVE", "ABSENT") or (0 <= pct < 25):
                s.tier_reasoning = (
                    f"{label} signature {level} ({pct:.0f}%) — "
                    f"drug benefit may be limited without pathway activation. " + s.tier_reasoning
                )
                if s.target_direction_match > 0:
                    s.target_direction_match *= 0.5
                    s.calculate()
            else:
                bonus = min(pct / 100 * self.config.signature_bonus_max, self.config.signature_bonus_max) * 0.5
                s.signature_bonus = round(bonus, 2)
                s.tier_reasoning = (
                    f"{label} signature {level or 'UNKNOWN'} ({pct:.0f}%) — "
                    f"intermediate activation; response uncertain (+{s.signature_bonus:.1f}). " + s.tier_reasoning
                )
            break  # Apply at most one signature gate per drug

        return s

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
            stratum_note = ""
            if s.gene_evidence_quality < 1.0:
                stratum_labels = {v: k for k, v in self.config.stratum_multipliers.items()}
                label = stratum_labels.get(s.gene_evidence_quality, "lower-evidence")
                stratum_note = f" [gene evidence: {label}, ×{s.gene_evidence_quality:.2f}]"
            parts.append(f"Q1 Concordant: drug action aligns with patient gene expression "
                         f"(Direction +{s.target_direction_match:.0f}){stratum_note}.")
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

    def _check_disease_relevance(self, c: DrugCandidate, r: DrugQueryRequest) -> bool:
        """A drug must have disease-treatment evidence, not just gene-targeting.
        Returns True if ANY of: indication match, OT disease link, successful trials,
        or MoA/EPC semantically fits the disease context.
        """
        if getattr(r, 'is_target_only_query', False):
            return True  # Target-only queries: all drugs are relevant by definition
        if self._has_disease_indication(c, r):
            return True
        if any(t.ot_association_score and t.ot_association_score > 0 for t in c.targets):
            return True
        te = c.trial_evidence
        if te and te.completed_trials >= 1:
            logger.debug(f"Trial gate passed via completed_trials={te.completed_trials} (best_p={te.best_p_value})")
            return True
        if self.embedder and getattr(r, 'disease_context', None):
            try:
                moa_parts = [t.mechanism_of_action or '' for t in c.targets]
                moa_parts += [t.fda_moa_narrative or '' for t in c.targets]
                if c.identity.pharm_class_epc:
                    moa_parts.append(c.identity.pharm_class_epc)
                if c.identity.pharm_class_moa:
                    moa_parts.append(c.identity.pharm_class_moa)
                moa_text = ' '.join(p for p in moa_parts if p)[:400]
                if moa_text:
                    moa_vec = self.embedder.encode(moa_text)
                    ctx_key = f"_ctx_{r.disease_context[:80]}"
                    if ctx_key not in self._moa_embed_cache:
                        self._moa_embed_cache[ctx_key] = self.embedder.encode(r.disease_context)
                    ctx_vec = self._moa_embed_cache[ctx_key]
                    sim = float(np.dot(
                        moa_vec / (np.linalg.norm(moa_vec) + 1e-10),
                        ctx_vec / (np.linalg.norm(ctx_vec) + 1e-10),
                    ))
                    if sim > 0.5:
                        return True
            except Exception:
                pass
        return False

    # ── Target: Direction + Magnitude ────────────────────────────────────

    _INHIBITORY = frozenset({"INHIBITOR", "ANTAGONIST", "NEGATIVE MODULATOR", "BLOCKER", "NEGATIVE ALLOSTERIC MODULATOR"})
    _ACTIVATING = frozenset({"AGONIST", "POSITIVE MODULATOR", "ACTIVATOR", "POSITIVE ALLOSTERIC MODULATOR"})

    def _build_gene_map(self, r: DrugQueryRequest) -> dict:
        """Build gene lookup from full patient profile, falling back to discovery genes."""
        source = r.all_patient_genes if r.all_patient_genes else r.genes
        return {g.gene_symbol.upper(): g for g in source}

    def _effective_evidence_stratum(self, gene) -> Optional[str]:
        """Prefer stronger causal-tier evidence without overriding stronger explicit strata."""
        explicit = getattr(gene, 'evidence_stratum', None)
        causal_tier = (getattr(gene, 'causal_tier', None) or '').lower()
        causal_stratum = None
        if 'tier 1' in causal_tier or 'tier 2' in causal_tier:
            causal_stratum = 'known_driver'
        elif 'tier 3' in causal_tier:
            causal_stratum = 'expression_significant'

        if not explicit:
            return causal_stratum
        if not causal_stratum:
            return explicit

        rank = {
            'novel_candidate': 0,
            'expression_significant': 1,
            'ppi_connected': 2,
            'known_driver': 3,
        }
        return causal_stratum if rank.get(causal_stratum, -1) > rank.get(explicit, -1) else explicit

    _RECEPTOR_SUFFIX_RE = re.compile(r'^(.{2,}?)(R[A-B]?\d*|R)$')

    def _gene_family_match(self, target_gene: str, gene_map: dict) -> Optional[tuple]:
        """Ligand↔receptor linking via gene-symbol suffix convention (annotation only)."""
        tg = target_gene.upper()
        # Target is receptor (e.g., IL4R) → look for ligand (IL4) in patient DEGs
        m = self._RECEPTOR_SUFFIX_RE.match(tg)
        if m:
            base = m.group(1)
            if base in gene_map:
                return gene_map[base]
        # Target is ligand → look for its receptor in patient DEGs
        for sym, gene in gene_map.items():
            m2 = self._RECEPTOR_SUFFIX_RE.match(sym)
            if m2 and m2.group(1) == tg:
                return gene
        return None

    def _pathway_co_member(self, target_gene: str, gene_map: dict,
                           r: DrugQueryRequest) -> Optional[tuple]:
        """Find patient genes sharing a pathway with the drug target.
        Returns (best_gene, pathway_name, pathway_direction, all_effectors) ranked by |log2fc|.
        """
        _MAX_PATHWAY_SIZE = 50
        tg = target_gene.upper()
        best_result = None
        best_fc = 0.0
        for p in r.pathways:
            if not p.key_genes or len(p.key_genes) > _MAX_PATHWAY_SIZE:
                continue
            pg_upper = {g.upper() for g in p.key_genes}
            if tg not in pg_upper:
                continue
            # Collect all significant dysregulated members in this pathway
            effectors = []
            top_gene, top_fc = None, 0.0
            for pg in pg_upper:
                if pg != tg and pg in gene_map:
                    g = gene_map[pg]
                    if abs(g.log2fc) >= DEG_LOG2FC_THRESHOLD:
                        effectors.append(f"{g.gene_symbol}{'↑' if g.direction == 'up' else '↓'}")
                        if abs(g.log2fc) > top_fc:
                            top_gene, top_fc = g, abs(g.log2fc)
            if top_gene and top_fc > best_fc:
                best_fc = top_fc
                best_result = (top_gene, p.pathway_name, p.direction, effectors)
        return best_result

    def _downstream_effector_match(self, target, gene_map: dict) -> Optional[tuple]:
        """Check if drug target's KG-resolved related genes are dysregulated in patient."""
        effectors = getattr(target, 'known_effectors', None)
        if not effectors:
            return None
        concordant = []
        best_gene, best_fc = None, 0.0
        for eff in effectors:
            gene = gene_map.get(eff.upper())
            if gene and abs(gene.log2fc) >= DEG_LOG2FC_THRESHOLD:
                concordant.append(f"{gene.gene_symbol}{'↑' if gene.direction == 'up' else '↓'}")
                if abs(gene.log2fc) > best_fc:
                    best_gene, best_fc = gene, abs(gene.log2fc)
        if len(concordant) >= self.config.min_effectors_concordant and best_gene:
            return (best_gene, concordant)
        return None

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

                # Gene evidence stratum: known drivers get full credit, novel candidates get partial
                sm = self.config.stratum_multipliers.get(
                    self._effective_evidence_stratum(gene), 1.0)

                if action in self._INHIBITORY and gene.direction == "up":
                    best = max(best, w * sm)
                elif action in self._ACTIVATING and gene.direction == "down":
                    best = max(best, w * sm)
                elif action == "INDIRECT_EFFECT" and gene.direction in ("up", "down"):
                    best = max(best, w * 0.65 * sm)
                elif action == "UNKNOWN" and gene.direction in ("up", "down"):
                    best = max(best, w * 0.5 * sm)
                continue

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
                sm = self.config.stratum_multipliers.get(
                    self._effective_evidence_stratum(related_gene), 1.0)
                logger.info(f"Pathway co-member: {t.gene_symbol} ↔ {related_gene.gene_symbol} "
                            f"via '{pathway_name}' ({len(effectors)} effectors, credit={credit:.2f}×, stratum={sm:.2f})")
                best = max(best, w * credit * sm)
                continue

            # Downstream effector: drug target's signaling cascade members in patient DEGs
            eff_match = self._downstream_effector_match(t, gene_map)
            if eff_match:
                eff_gene, eff_names = eff_match
                t.related_patient_gene = eff_gene.gene_symbol
                t.related_gene_log2fc = eff_gene.log2fc
                t.related_gene_direction = eff_gene.direction
                t.related_gene_source = "downstream_effector"
                t.downstream_effector_genes = eff_names
                sm_eff = self.config.stratum_multipliers.get(
                    self._effective_evidence_stratum(eff_gene), 1.0)
                best = max(best, w * self.config.effector_credit_fraction * sm_eff)
                continue

            # Gene-family fallback: receptor↔ligand linking (annotation only, no score credit)
            fam_match = self._gene_family_match(t.gene_symbol, gene_map)
            if fam_match:
                t.related_patient_gene = fam_match.gene_symbol
                t.related_gene_log2fc = fam_match.log2fc
                t.related_gene_direction = fam_match.direction
                t.related_gene_source = "gene-family"

        return round(best, 2)

    def _target_magnitude(self, c: DrugCandidate, r: DrugQueryRequest,
                          direction_score: float) -> float:
        """Expression magnitude bonus — gated on positive direction concordance."""
        w = self.config.target_magnitude_weight
        if direction_score <= 0 or not c.targets or not r.genes:
            return 0.0

        gene_map = self._build_gene_map(r)
        best = 0.0

        for t in c.targets:
            gene = gene_map.get(t.gene_symbol.upper())
            if not gene:
                continue

            action = (t.action_type or "").upper()
            sm = self.config.stratum_multipliers.get(
                self._effective_evidence_stratum(gene), 1.0)
            concordant = (
                (action in self._INHIBITORY and gene.direction == "up") or
                (action in self._ACTIVATING and gene.direction == "down")
            )
            if concordant:
                best = max(best, min(1.0, abs(gene.log2fc) / 8.0) * w * sm)
            elif action == "INDIRECT_EFFECT" and gene.direction in ("up", "down"):
                best = max(best, min(1.0, abs(gene.log2fc) / 8.0) * w * 0.65 * sm)
            elif action == "UNKNOWN" and gene.direction in ("up", "down"):
                best = max(best, min(1.0, abs(gene.log2fc) / 8.0) * w * 0.5 * sm)

        return round(best, 2)

    # ── Clinical + Regulatory (merged) ───────────────────────────────────

    @staticmethod
    def _ascii_fold(text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').replace("'", "").replace("\u2019", "")

    def _indication_similarity(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
        """Return 0.0–1.0 similarity between drug indication and query disease."""
        alias_set = {self._ascii_fold(a).lower() for a in ([r.disease] + getattr(r, 'disease_aliases', []))}

        # Exact word-boundary match in MoA narrative → 1.0
        if any(any(_disease_matches(alias, self._ascii_fold(t.fda_moa_narrative or "").lower()) for alias in alias_set) for t in c.targets):
            return 1.0

        if c.identity.indication_text:
            ind_lower = self._ascii_fold(c.identity.indication_text).lower()
            # Exact word-boundary match in indication text → 1.0
            if any(_disease_matches(alias, ind_lower) for alias in alias_set):
                return 1.0
            # Semantic fallback
            if self.embedder:
                try:
                    ind_vec = self.embedder.encode(c.identity.indication_text[:300])
                    dis_vec = self.embedder.encode(r.disease)
                    ind_n = ind_vec / (np.linalg.norm(ind_vec) + 1e-10)
                    dis_n = dis_vec / (np.linalg.norm(dis_vec) + 1e-10)
                    return max(0.0, float(np.dot(ind_n, dis_n)))
                except Exception:
                    pass
        return 0.0

    def _has_disease_indication(self, c: DrugCandidate, r: DrugQueryRequest,
                                   strict: bool = False) -> bool:
        """Check whether a drug's indication matches the query disease (alias-aware).

        When strict=True, only exact word-boundary matches count (no semantic
        fallback).  Used by SOC gate to prevent false-positive SOC labeling.
        """
        sim = self._indication_similarity(c, r)
        if sim >= 1.0:
            return True
        if strict:
            return False  # no semantic fallback for SOC gating
        threshold = 0.70 if len(r.disease.split()) <= 1 else 0.55
        return sim >= threshold

    def _clinical_regulatory(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
        """Merged clinical trial evidence + FDA regulatory status."""
        w = self.config.clinical_regulatory_weight
        fda = c.identity.is_fda_approved
        disease_match = self._has_disease_indication(c, r)

        te = c.trial_evidence
        has_trials = te is not None and te.total_trials > 0
        trial_phase = (te.highest_phase or 0) if has_trials else 0
        id_phase = c.identity.max_phase or 0
        phase = max(trial_phase, id_phase) if disease_match else trial_phase
        has_pval = has_trials and te.best_p_value is not None and te.best_p_value < 0.05
        completed = has_trials and te.completed_trials > 0

        if fda and disease_match:
            base = w if (phase >= 4 or (completed and has_pval)) else w * 0.85
        elif fda:
            # FDA-approved for a DIFFERENT disease — minimal credit
            base = w * 0.22 if (has_trials and phase >= 2) else w * 0.18
        elif phase >= 3 and completed and has_pval:
            base = w * 0.70
        elif phase >= 3 and completed:
            base = w * 0.60
        elif phase >= 3:
            base = w * 0.50
        elif phase >= 2 and has_pval:
            base = w * 0.40
        elif phase >= 2:
            base = w * 0.28
        elif phase >= 1:
            base = w * 0.15
        elif id_phase >= 3:
            base = w * 0.20
        else:
            return 0.0

        bonus = 0.0
        if has_trials:
            if te.completed_trials >= 3 and phase >= 3:
                bonus += w * 0.06
            if te.total_enrollment > 500:
                bonus += w * 0.04

        cap = w if disease_match else w * 0.60
        return round(min(cap, base + bonus), 2)

    def _ot_score(self, c: DrugCandidate, r: DrugQueryRequest) -> tuple:
        """Best OpenTargets target-disease association score, with pipeline-evidence fallback.
        Returns (score, pipeline_evidence_used).
        """
        w = self.config.ot_weight
        best = 0.0
        for t in c.targets:
            if t.ot_association_score is not None:
                best = max(best, t.ot_association_score)
        ot_score = round(best * w, 2)

        if ot_score > 0 or not r.disease_context or not self.embedder:
            return ot_score, False

        # Pipeline-evidence fallback: semantic similarity between disease context and drug targets
        try:
            target_desc = ' '.join(
                f"{t.gene_symbol} {t.action_type or ''}" for t in c.targets if t.gene_symbol
            )[:300]
            if not target_desc:
                return 0.0, False
            ctx_key = f"_ctx_{r.disease_context[:80]}"
            if ctx_key not in self._moa_embed_cache:
                self._moa_embed_cache[ctx_key] = self.embedder.encode(r.disease_context)
            ctx_vec = self._moa_embed_cache[ctx_key]
            tgt_vec = self.embedder.encode(target_desc)
            sim = float(np.dot(
                ctx_vec / (np.linalg.norm(ctx_vec) + 1e-10),
                tgt_vec / (np.linalg.norm(tgt_vec) + 1e-10),
            ))
            # Capped at 50% of OT weight — pipeline evidence is not independently validated
            fallback = round(min(max(0, sim) * w, w * 0.5), 2)
            return fallback, (fallback > 0)
        except Exception:
            return 0.0, False

    def _pathway_match(self, c: DrugCandidate, r: DrugQueryRequest) -> float:
        """Semantic similarity between drug MoA text and patient pathways."""
        w = self.config.pathway_weight
        if not self.embedder or not r.pathways or not c.targets:
            return 0.0

        # Use all dysregulated pathways (both up and down) for disease-agnostic scoring
        dysregulated_pathways = r.pathways
        if not dysregulated_pathways:
            return 0.0

        # Build MoA text from all available mechanism descriptions
        moa_parts = []
        for t in c.targets:
            if t.mechanism_of_action:
                moa_parts.append(t.mechanism_of_action)
            if t.fda_moa_narrative:
                moa_parts.append(t.fda_moa_narrative[:300])
        if c.identity.pharm_class_moa:
            moa_parts.append(c.identity.pharm_class_moa)

        moa_text = " ".join(moa_parts)[:500]
        if not moa_text:
            # Fallback: synthesize from drug identity + target gene symbols
            fallback = [c.identity.drug_name]
            if c.identity.pharm_class_epc:
                fallback.append(c.identity.pharm_class_epc)
            if c.identity.indication_text:
                fallback.append(c.identity.indication_text[:200])
            for t in c.targets:
                action = (t.action_type or "").strip()
                label = f"{t.gene_symbol} {action}".strip() if action and action != "UNKNOWN" else t.gene_symbol
                fallback.append(label)
            moa_text = " ".join(fallback)[:500]
            if not moa_text:
                return 0.0

        try:
            moa_key = moa_text[:100]
            if moa_key not in self._moa_embed_cache:
                self._moa_embed_cache[moa_key] = self.embedder.encode(moa_text)
            moa_vec = self._moa_embed_cache[moa_key]

            # Enrich pathway texts with disease-relevance context when available
            pathway_texts = [
                f"{p.pathway_name}: {p.disease_relevance}"[:200]
                if getattr(p, 'disease_relevance', None) else p.pathway_name
                for p in dysregulated_pathways
            ]
            pathway_vecs = self.embedder.encode(pathway_texts)

            # Cosine similarities
            moa_norm = moa_vec / (np.linalg.norm(moa_vec) + 1e-10)
            sims = []
            for pv in pathway_vecs:
                pv_norm = pv / (np.linalg.norm(pv) + 1e-10)
                sims.append(float(np.dot(moa_norm, pv_norm)))

            best_sim = max(sims) if sims else 0.0
            # Scale: similarity > 0.3 starts scoring, linear to max at 0.8
            if best_sim < 0.3:
                semantic_score = 0.0
            else:
                scaled = min(1.0, (best_sim - 0.3) / 0.5)
                semantic_score = round(scaled * w, 2)

            # Gene-overlap: direct concordance when drug targets patient pathway key genes
            drug_genes = {t.gene_symbol.upper() for t in c.targets if t.gene_symbol}
            max_overlap = 0
            for p in dysregulated_pathways:
                if p.key_genes:
                    overlap = len(drug_genes & {g.upper() for g in p.key_genes})
                    max_overlap = max(max_overlap, overlap)
            overlap_score = round(min(1.0, max_overlap / 3) * w, 2) if max_overlap else 0.0

            return max(semantic_score, overlap_score)

        except Exception as e:
            logger.warning(f"Pathway match scoring failed: {e}")
            return 0.0

    def _safety_penalty(self, c: DrugCandidate) -> float:
        """Safety penalty — drugs with serious warnings should not be offset
        by high efficacy scores.  Range: 0 to safety_max_penalty (default -30).
        """
        max_penalty = self.config.safety_max_penalty
        penalty = 0.0

        if c.safety:
            n_boxed = len(c.safety.boxed_warnings)
            if n_boxed > 0:
                penalty -= min(12, 7 + (n_boxed - 1) * 3)

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
