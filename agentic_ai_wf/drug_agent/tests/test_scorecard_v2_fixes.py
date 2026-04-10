"""Regression tests for Scorecard v2 — 9 fixes."""

import re
import pytest
from dataclasses import asdict

from agentic_ai_wf.drug_agent.service.schemas import (
    DrugCandidate, DrugIdentity, DrugQueryRequest, GeneContext,
    ScoreBreakdown, ScoringConfig, TargetEvidence, SafetyProfile,
    TrialEvidence, ContraindicationEntry,
)
from agentic_ai_wf.drug_agent.service.drug_scorer import DrugScorer
from agentic_ai_wf.drug_agent.chembl.models.chembl_models import ActionType
from agentic_ai_wf.drug_agent.service.result_aggregator import ResultAggregator


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_candidate(
    name="TESTDRUG", gene="BLK", action="INHIBITOR",
    is_fda=False, max_phase=None, indication=None,
    pharm_class_moa=None, pharm_class_epc=None,
    trial_evidence=None, safety=None, log2fc=None,
    is_soc=False, soc_confidence=0.0,
    contra_entries=None, caution_notes=None,
):
    identity = DrugIdentity(
        drug_name=name,
        is_fda_approved=is_fda,
        max_phase=max_phase,
        indication_text=indication,
        pharm_class_moa=pharm_class_moa,
        pharm_class_epc=pharm_class_epc,
    )
    targets = [TargetEvidence(gene_symbol=gene, action_type=action)]
    if log2fc is not None:
        targets[0].patient_gene_log2fc = log2fc
    return DrugCandidate(
        identity=identity,
        targets=targets,
        trial_evidence=trial_evidence,
        safety=safety,
        is_soc_candidate=is_soc,
        soc_confidence=soc_confidence,
        contraindication_entries=contra_entries or [],
        caution_notes=caution_notes or [],
    )


def _make_request(disease="melanoma", genes=None):
    if genes is None:
        genes = [GeneContext(gene_symbol="BLK", log2fc=3.5, adj_p_value=0.001, direction="up")]
    return DrugQueryRequest(disease=disease, genes=genes)


# ── Fix 1: Dedup merge propagation ──────────────────────────────────────────
# (Integration-level — tests that DrugIdentity fields survive merge are
#  validated structurally by checking the field exists and defaults.)

class TestFix1_DedupFields:
    def test_drug_identity_has_propagation_fields(self):
        """DrugIdentity must carry fields the dedup merge propagates."""
        di = DrugIdentity(drug_name="X")
        assert hasattr(di, "is_fda_approved")
        assert hasattr(di, "max_phase")
        assert hasattr(di, "indication_text")
        assert hasattr(di, "pharm_class_moa")
        assert hasattr(di, "pharm_class_epc")

    def test_drug_candidate_has_trial_and_safety(self):
        dc = DrugCandidate(identity=DrugIdentity(drug_name="X"))
        assert hasattr(dc, "trial_evidence")
        assert hasattr(dc, "safety")


# ── Fix 2: OT get_target_disease_score uses scroll ─────────────────────────
# (Full integration test requires Qdrant; unit-test the helper _disease_matches)

class TestFix2_DiseaseMatches:
    def test_exact_word_boundary(self):
        from agentic_ai_wf.drug_agent.service.drug_scorer import _disease_matches
        assert _disease_matches("melanoma", "Skin Melanoma Stage IV") is True

    def test_no_partial_match(self):
        from agentic_ai_wf.drug_agent.service.drug_scorer import _disease_matches
        assert _disease_matches("myopathy", "cardiomyopathy") is False

    def test_case_insensitive(self):
        from agentic_ai_wf.drug_agent.service.drug_scorer import _disease_matches
        assert _disease_matches("MELANOMA", "cutaneous melanoma") is True

    def test_empty_inputs(self):
        from agentic_ai_wf.drug_agent.service.drug_scorer import _disease_matches
        assert _disease_matches("", "melanoma") is False
        assert _disease_matches("melanoma", "") is False


# ── Fix 3: SOC composite + strict indication ────────────────────────────────

class TestFix3_SOCComposite:
    def test_use_soc_composite_default_true(self):
        cfg = ScoringConfig()
        assert cfg.use_soc_composite is True

    def test_strict_indication_rejects_semantic_only(self):
        """strict=True should reject when indication_similarity < 1.0."""
        scorer = DrugScorer()
        # Drug with no indication text → similarity = 0.0 → strict should return False
        c = _make_candidate(name="NODISEASE", indication=None)
        r = _make_request(disease="melanoma")
        assert scorer._has_disease_indication(c, r, strict=True) is False

    def test_strict_indication_accepts_exact_match(self):
        scorer = DrugScorer()
        c = _make_candidate(name="MELADRUG", indication="melanoma treatment")
        r = _make_request(disease="melanoma")
        assert scorer._has_disease_indication(c, r, strict=True) is True

    def test_non_strict_uses_threshold(self):
        scorer = DrugScorer()
        c = _make_candidate(name="NODISEASE", indication=None)
        r = _make_request(disease="melanoma")
        # Without indication text, similarity = 0 → non-strict also False
        assert scorer._has_disease_indication(c, r, strict=False) is False


# ── Fix 4: Signature bonus in ScoreBreakdown ────────────────────────────────

class TestFix4_SignatureBonus:
    def test_signature_bonus_field_exists(self):
        s = ScoreBreakdown()
        assert hasattr(s, "signature_bonus")
        assert s.signature_bonus == 0.0

    def test_signature_bonus_max_config(self):
        cfg = ScoringConfig()
        assert hasattr(cfg, "signature_bonus_max")
        assert cfg.signature_bonus_max == 8.0

    def test_signature_bonus_included_in_calculate(self):
        s = ScoreBreakdown()
        s.target_direction_match = 10.0
        s.signature_bonus = 5.0
        s.calculate()
        assert s.composite_score == 15.0

    def test_signature_bonus_capped_at_100(self):
        s = ScoreBreakdown()
        s.target_direction_match = 18.0
        s.clinical_regulatory_score = 25.0
        s.ot_association_score = 15.0
        s.pathway_concordance = 15.0
        s.target_magnitude_match = 12.0
        s.disease_indication_bonus = 10.0
        s.signature_bonus = 8.0
        s.calculate()
        assert s.composite_score == 100.0  # capped


# ── Fix 5: Prodrug suffixes in _SALT_RE ─────────────────────────────────────

class TestFix5_ProdrugSuffixes:
    def test_mofetil_stripped(self):
        agg = ResultAggregator()
        assert agg.normalize_drug_name("MYCOPHENOLATE MOFETIL") == "MYCOPHENOLATE"

    def test_axetil_stripped(self):
        agg = ResultAggregator()
        assert agg.normalize_drug_name("CEFUROXIME AXETIL") == "CEFUROXIME"

    def test_alafenamide_stripped(self):
        agg = ResultAggregator()
        assert agg.normalize_drug_name("TENOFOVIR ALAFENAMIDE") == "TENOFOVIR"

    def test_pivoxil_stripped(self):
        agg = ResultAggregator()
        assert agg.normalize_drug_name("CEFDITOREN PIVOXIL") == "CEFDITOREN"

    def test_medoxomil_stripped(self):
        agg = ResultAggregator()
        assert agg.normalize_drug_name("OLMESARTAN MEDOXOMIL") == "OLMESARTAN"

    def test_traditional_salts_still_work(self):
        agg = ResultAggregator()
        assert agg.normalize_drug_name("FLUOXETINE HYDROCHLORIDE") == "FLUOXETINE"


# ── Fix 6: Contra output surfacing (structural) ─────────────────────────────

class TestFix6_ContraOutputFields:
    def test_candidate_has_contra_fields(self):
        c = DrugCandidate(identity=DrugIdentity(drug_name="X"))
        assert hasattr(c, "contraindication_entries")
        assert hasattr(c, "caution_notes")
        assert hasattr(c, "is_soc_candidate")
        assert hasattr(c, "soc_confidence")

    def test_contra_entry_serialization(self):
        entry = ContraindicationEntry(
            tier=1, reason="Gene overexpressed", source="gene_based",
            gene_symbol="TP53", log2fc=4.2,
        )
        assert entry.label == "Avoid"
        d = asdict(entry)
        assert d["tier"] == 1
        assert d["gene_symbol"] == "TP53"

    def test_caution_entry_label(self):
        entry = ContraindicationEntry(tier=3, reason="Monitor", source="disease_ae")
        assert entry.label == "Use With Caution"


# ── Fix 7: INDIRECT_EFFECT action type ──────────────────────────────────────

class TestFix7_IndirectEffect:
    def test_enum_exists(self):
        assert ActionType.INDIRECT_EFFECT.value == "INDIRECT_EFFECT"

    def test_indirect_effect_direction_credit(self):
        """INDIRECT_EFFECT should get 65% of direction weight."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INDIRECT_EFFECT")
        r = _make_request(genes=[
            GeneContext(gene_symbol="BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
        ])
        score = scorer._target_direction(c, r)
        expected = round(18.0 * 0.65, 2)
        assert score == expected, f"Expected {expected}, got {score}"

    def test_indirect_effect_magnitude_credit(self):
        """INDIRECT_EFFECT should get 65% of magnitude weight when direction > 0."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INDIRECT_EFFECT")
        r = _make_request(genes=[
            GeneContext(gene_symbol="BLK", log2fc=4.0, adj_p_value=0.001, direction="up"),
        ])
        direction_score = scorer._target_direction(c, r)
        assert direction_score > 0
        mag = scorer._target_magnitude(c, r, direction_score)
        # magnitude = min(1.0, 4.0/8.0) * 12.0 * 0.65 = 0.5 * 12.0 * 0.65 = 3.9
        assert mag == round(0.5 * 12.0 * 0.65, 2)

    def test_unknown_gets_50_percent(self):
        """UNKNOWN should get 50% credit (lower than INDIRECT_EFFECT)."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="UNKNOWN")
        r = _make_request(genes=[
            GeneContext(gene_symbol="BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
        ])
        score = scorer._target_direction(c, r)
        expected = round(18.0 * 0.5, 2)
        assert score == expected


# ── Fix 8: Boxed warning diminishing returns ────────────────────────────────

class TestFix8_BoxedWarnings:
    def _score_penalty(self, n_boxed):
        scorer = DrugScorer()
        safety = SafetyProfile(boxed_warnings=["warning"] * n_boxed)
        c = _make_candidate(safety=safety)
        return scorer._safety_penalty(c)

    def test_one_boxed_warning(self):
        assert self._score_penalty(1) == -7

    def test_two_boxed_warnings(self):
        # min(12, 7 + (2-1)*3) = min(12, 10) = -10
        assert self._score_penalty(2) == -10

    def test_three_boxed_warnings_capped(self):
        # min(12, 7 + (3-1)*3) = min(12, 13) = -12
        assert self._score_penalty(3) == -12

    def test_five_boxed_warnings_still_capped(self):
        # min(12, 7 + (5-1)*3) = min(12, 19) = -12
        assert self._score_penalty(5) == -12

    def test_diminishing_returns(self):
        """Each additional warning adds less penalty (until cap)."""
        p1 = abs(self._score_penalty(1))
        p2 = abs(self._score_penalty(2))
        p3 = abs(self._score_penalty(3))
        assert p2 - p1 == 3  # +3 for 2nd
        assert p3 - p2 == 2  # +2 for 3rd (because of cap at 12)


# ── Fix 9: Contra multipliers enabled ───────────────────────────────────────

class TestFix9_ContraMultipliers:
    def test_apply_contra_multipliers_default_true(self):
        cfg = ScoringConfig()
        assert cfg.apply_contra_multipliers is True

    def test_contra_tier_multipliers_exist(self):
        cfg = ScoringConfig()
        assert cfg.contra_tier_multipliers == {1: 0.0, 2: 0.25, 3: 0.75}
