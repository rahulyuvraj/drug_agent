"""Regression tests for 5 competitive-moat logic gaps."""

import pytest
from dataclasses import asdict

from agentic_ai_wf.drug_agent.service.schemas import (
    DrugCandidate, DrugIdentity, DrugQueryRequest, GeneContext,
    BiomarkerContext, PathwayContext, ScoreBreakdown, ScoringConfig,
    TargetEvidence, SafetyProfile, TrialEvidence, ContraindicationEntry,
)
from agentic_ai_wf.drug_agent.service.drug_agent_service import DrugAgentService
from agentic_ai_wf.drug_agent.service.drug_scorer import DrugScorer


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_candidate(
    name="TESTDRUG", gene="BLK", action="INHIBITOR",
    is_fda=False, max_phase=None, indication=None,
    pharm_class_moa=None, pharm_class_epc=None,
    trial_evidence=None, safety=None,
    is_soc=False, soc_confidence=0.0,
    contra_entries=None, withdrawn=False,
    known_effectors=None,
):
    identity = DrugIdentity(
        drug_name=name,
        is_fda_approved=is_fda,
        max_phase=max_phase,
        indication_text=indication,
        pharm_class_moa=pharm_class_moa,
        pharm_class_epc=pharm_class_epc,
        withdrawn=withdrawn,
    )
    targets = [TargetEvidence(gene_symbol=gene, action_type=action,
                              known_effectors=known_effectors)]
    return DrugCandidate(
        identity=identity,
        targets=targets,
        trial_evidence=trial_evidence,
        safety=safety,
        is_soc_candidate=is_soc,
        soc_confidence=soc_confidence,
        contraindication_entries=contra_entries or [],
    )


def _make_request(disease="melanoma", genes=None, biomarkers=None, pathways=None):
    if genes is None:
        genes = [GeneContext(gene_symbol="BLK", log2fc=3.5, adj_p_value=0.001, direction="up")]
    return DrugQueryRequest(
        disease=disease, genes=genes,
        biomarkers=biomarkers or [],
        pathways=pathways or [],
    )


# ══════════════════════════════════════════════════════════════════════════════
# GAP 1: SOC Protection Layer
# ══════════════════════════════════════════════════════════════════════════════

class TestGap1_SOCProtection:
    """SOC drugs must be shielded from all contraindication paths."""

    def test_soc_candidate_skips_scoring_contras(self):
        """SOC candidate's tier_reasoning reflects SOC override."""
        scorer = DrugScorer()
        c = _make_candidate(
            name="HCQ", gene="TLR9", action="INHIBITOR",
            is_fda=True, max_phase=4, is_soc=True, soc_confidence=0.71,
            indication="systemic lupus erythematosus",
        )
        r = _make_request(disease="systemic lupus erythematosus", genes=[
            GeneContext("TLR9", log2fc=-2.5, adj_p_value=0.01, direction="down"),
        ])
        s = scorer.score(c, r)
        assert "Standard-of-Care" in s.tier_reasoning

    def test_soc_advisory_notes_structure(self):
        """SOC candidate should have advisory notes, not contra flags."""
        c = _make_candidate(is_soc=True)
        c.soc_advisory_notes.append("Test advisory")
        assert len(c.soc_advisory_notes) == 1
        assert not c.contraindication_flags


# ══════════════════════════════════════════════════════════════════════════════
# GAP 2: Tiered Contraindications
# ══════════════════════════════════════════════════════════════════════════════

class TestGap2_TieredContraindications:
    """Every contra path must create ContraindicationEntry; placement uses worst_tier."""

    def test_withdrawn_drug_gets_tier1_entry(self):
        """Withdrawn drugs should produce a Tier 1 ContraindicationEntry."""
        c = _make_candidate(withdrawn=True)
        reason = "Drug has been withdrawn from market"
        c.contraindication_flags.append(reason)
        c.contraindication_entries.append(ContraindicationEntry(
            tier=1, reason=reason, source="withdrawn"))
        assert c.contraindication_entries[0].tier == 1
        assert c.contraindication_entries[0].label == "Avoid"

    def test_tier3_not_contraindicated(self):
        """Tier 3 ('Use With Caution') should NOT place drug in contraindicated."""
        entries = [ContraindicationEntry(tier=3, reason="caution", source="gene_based")]
        worst_tier = min(e.tier for e in entries)
        is_contraindicated = worst_tier <= 2
        assert not is_contraindicated

    def test_tier1_is_contraindicated(self):
        """Tier 1 ('Avoid') should place drug in contraindicated."""
        entries = [ContraindicationEntry(tier=1, reason="AE match", source="disease_ae")]
        worst_tier = min(e.tier for e in entries)
        assert worst_tier <= 2

    def test_mixed_tiers_uses_worst(self):
        """When multiple entries exist, worst (lowest) tier drives placement."""
        entries = [
            ContraindicationEntry(tier=3, reason="caution", source="gene_based"),
            ContraindicationEntry(tier=1, reason="disease AE", source="disease_ae"),
        ]
        worst_tier = min(e.tier for e in entries)
        assert worst_tier == 1

    def test_no_entries_not_contraindicated(self):
        """No entries → drug goes to recommendations."""
        entries = []
        worst_tier = min((e.tier for e in entries), default=None)
        assert worst_tier is None

    def test_contra_entry_label_mapping(self):
        assert ContraindicationEntry(tier=1, reason="", source="").label == "Avoid"
        assert ContraindicationEntry(tier=2, reason="", source="").label == "Contraindicated"
        assert ContraindicationEntry(tier=3, reason="", source="").label == "Use With Caution"

    def test_core_types_drug_recommendation_has_entries(self):
        """DrugRecommendation in core_types must have contraindication_entries."""
        from agentic_ai_wf.reporting_pipeline_agent.core_types import DrugRecommendation
        dr = DrugRecommendation(
            drug_name="X", target_gene="Y", priority="High",
            priority_score=80.0, mechanistic_reasoning="test",
            biomarker_concordance="test", expression_support="Upregulated",
        )
        assert hasattr(dr, "contraindication_entries")
        assert isinstance(dr.contraindication_entries, list)


# ══════════════════════════════════════════════════════════════════════════════
# GAP 3: Gene Classification A/B/C/D
# ══════════════════════════════════════════════════════════════════════════════

class TestGap3_GeneClassification:
    """Gene evidence stratum modulates direction/magnitude scoring."""

    def test_stratum_multipliers_in_config(self):
        cfg = ScoringConfig()
        assert "known_driver" in cfg.stratum_multipliers
        assert cfg.stratum_multipliers["known_driver"] == 1.0
        assert cfg.stratum_multipliers["novel_candidate"] == 0.5

    def test_known_driver_full_credit(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="known_driver"),
        ])
        score = scorer._target_direction(c, r)
        assert score == 18.0  # Full credit (1.0 × 18)

    def test_novel_candidate_half_credit(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="novel_candidate"),
        ])
        score = scorer._target_direction(c, r)
        assert score == round(18.0 * 0.5, 2)  # Half credit

    def test_ppi_connected_85_percent(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="ppi_connected"),
        ])
        score = scorer._target_direction(c, r)
        assert score == round(18.0 * 0.85, 2)

    def test_expression_significant_65_percent(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="expression_significant"),
        ])
        score = scorer._target_direction(c, r)
        assert score == round(18.0 * 0.65, 2)

    def test_none_stratum_defaults_to_full(self):
        """Missing evidence_stratum should default to 1.0 (backward compat)."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum=None),
        ])
        score = scorer._target_direction(c, r)
        assert score == 18.0

    def test_magnitude_also_uses_stratum(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=4.0, adj_p_value=0.001, direction="up",
                        evidence_stratum="novel_candidate"),
        ])
        direction = scorer._target_direction(c, r)
        assert direction > 0
        mag = scorer._target_magnitude(c, r, direction)
        # min(1.0, 4.0/8.0) * 12.0 * 0.5 = 0.5 * 12.0 * 0.5 = 3.0
        assert mag == round(0.5 * 12.0 * 0.5, 2)

    def test_gene_evidence_quality_in_breakdown(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="ppi_connected"),
        ])
        s = scorer.score(c, r)
        assert s.gene_evidence_quality == 0.85

    def test_tier_reasoning_mentions_stratum(self):
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="novel_candidate"),
        ])
        s = scorer.score(c, r)
        assert "gene evidence" in s.tier_reasoning
        assert "novel_candidate" in s.tier_reasoning

    def test_driver_stratum_no_stratum_note(self):
        """known_driver (quality=1.0) should NOT add stratum note."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="known_driver"),
        ])
        s = scorer.score(c, r)
        assert "gene evidence" not in s.tier_reasoning

    def test_causal_tier_upgrades_missing_stratum(self):
        """Tier 1 causal genes should score like known drivers even without explicit stratum."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up", causal_tier="Tier 1 - Full Causal Chain"),
        ])
        score = scorer._target_direction(c, r)
        assert score == 18.0

    def test_causal_tier_tier3_upgrades_to_expression_significant(self):
        """Tier 3 causal genes should upgrade to expression_significant when weaker than explicit evidence."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up", causal_tier="Tier 3 - Supportive causal evidence"),
        ])
        score = scorer._target_direction(c, r)
        assert score == round(18.0 * 0.65, 2)

    def test_explicit_stronger_stratum_preserved_over_tier3(self):
        """Tier-derived upgrades must not downgrade stronger explicit evidence."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        r = _make_request(genes=[
            GeneContext(
                "BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                evidence_stratum="ppi_connected", causal_tier="Tier 3 - Supportive causal evidence",
            ),
        ])
        score = scorer._target_direction(c, r)
        assert score == round(18.0 * 0.85, 2)


# ══════════════════════════════════════════════════════════════════════════════
# GAP 4: Biomarker Type A/B
# ══════════════════════════════════════════════════════════════════════════════

class TestGap4_BiomarkerTypeAB:
    """Type B biomarkers forced to not_assessed; skipped in scoring and contras."""

    def test_biomarker_context_has_type_field(self):
        bm = BiomarkerContext(biomarker_name="ER", status="positive")
        assert hasattr(bm, "biomarker_type")
        assert bm.biomarker_type is None  # default

    def test_type_a_biomarker_scored(self):
        """Type A biomarkers should be bridged into signature_scores."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        bm_a = BiomarkerContext(
            biomarker_name="ER", status="positive",
            supporting_genes=["ESR1"], biomarker_type="A")
        r = _make_request(
            genes=[GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up")],
            biomarkers=[bm_a])
        # Should complete without error — Type A is processed
        s = scorer.score(c, r)
        assert s.composite_score >= 0

    def test_type_b_biomarker_skipped_in_signature_gate(self):
        """Type B biomarkers should NOT influence scoring via signature_gate."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INHIBITOR")
        s = ScoreBreakdown()
        s.target_direction_match = 18.0
        s.calculate()

        bm_b = BiomarkerContext(
            biomarker_name="anti-dsDNA", status="negative",
            supporting_genes=[], biomarker_type="B")
        r = _make_request(
            genes=[GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up")],
            biomarkers=[bm_b])

        # Type B should be skipped, leaving direction unchanged
        result = scorer._signature_gate(s, c, r)
        assert result.target_direction_match == 18.0

    def test_type_b_skipped_in_biomarker_contras(self):
        """Type B biomarkers should NOT trigger receptor-status contraindications."""
        from agentic_ai_wf.drug_agent.service.drug_agent_service import DrugAgentService
        # Direct method test
        svc = object.__new__(DrugAgentService)  # bypass __init__
        c = _make_candidate(
            name="TAMOXIFEN", gene="ESR1", action="ANTAGONIST",
            indication="estrogen receptor positive breast cancer")
        bm_b = BiomarkerContext(
            biomarker_name="ER", status="negative",
            supporting_genes=["ESR1"], biomarker_type="B")
        flags = svc._check_biomarker_contraindications(c, [bm_b])
        assert len(flags) == 0  # Type B skipped

    def test_type_a_triggers_biomarker_contra(self):
        """Type A negative biomarker SHOULD trigger contraindication."""
        from agentic_ai_wf.drug_agent.service.drug_agent_service import DrugAgentService
        svc = object.__new__(DrugAgentService)
        c = _make_candidate(
            name="TAMOXIFEN", gene="ESR1", action="ANTAGONIST",
            indication="estrogen receptor positive breast cancer")
        bm_a = BiomarkerContext(
            biomarker_name="ER", status="negative",
            supporting_genes=["ESR1"], biomarker_type="A")
        flags = svc._check_biomarker_contraindications(c, [bm_a])
        assert len(flags) >= 1

    def test_none_type_not_skipped(self):
        """Biomarkers with no type (legacy) should still be processed."""
        from agentic_ai_wf.drug_agent.service.drug_agent_service import DrugAgentService
        svc = object.__new__(DrugAgentService)
        c = _make_candidate(
            name="TAMOXIFEN", gene="ESR1", action="ANTAGONIST",
            indication="estrogen receptor positive breast cancer")
        bm_none = BiomarkerContext(
            biomarker_name="ER", status="negative",
            supporting_genes=["ESR1"], biomarker_type=None)
        flags = svc._check_biomarker_contraindications(c, [bm_none])
        assert len(flags) >= 1  # Legacy behavior preserved


# ══════════════════════════════════════════════════════════════════════════════
# GAP 5: Downstream Effector Analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestGap5_DownstreamEffectors:
    """Drug targets with KG-resolved effectors get partial credit when dysregulated."""

    def test_target_evidence_has_known_effectors(self):
        te = TargetEvidence(gene_symbol="TNF", action_type="INHIBITOR",
                            known_effectors=["STAT3", "JAK1"])
        assert te.known_effectors == ["STAT3", "JAK1"]

    def test_known_effectors_default_none(self):
        te = TargetEvidence(gene_symbol="TNF", action_type="INHIBITOR")
        assert te.known_effectors is None

    def test_config_has_effector_params(self):
        cfg = ScoringConfig()
        assert cfg.min_effectors_concordant == 2
        assert cfg.effector_credit_fraction == 0.6

    def test_effector_match_with_sufficient_genes(self):
        """≥2 concordant effectors → match returned."""
        scorer = DrugScorer()
        t = TargetEvidence(gene_symbol="TNFSF13B", action_type="INHIBITOR",
                           known_effectors=["BLK", "BANK1", "IRF5"])
        gene_map = {
            "BLK": GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
            "BANK1": GeneContext("BANK1", log2fc=2.0, adj_p_value=0.01, direction="up"),
            "IRF5": GeneContext("IRF5", log2fc=1.5, adj_p_value=0.02, direction="up"),
        }
        result = scorer._downstream_effector_match(t, gene_map)
        assert result is not None
        eff_gene, eff_names = result
        assert len(eff_names) >= 2

    def test_effector_no_match_insufficient_genes(self):
        """<2 concordant effectors → no match."""
        scorer = DrugScorer()
        t = TargetEvidence(gene_symbol="TNFSF13B", action_type="INHIBITOR",
                           known_effectors=["BLK", "BANK1", "IRF5"])
        gene_map = {
            "BLK": GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
        }
        result = scorer._downstream_effector_match(t, gene_map)
        assert result is None

    def test_effector_no_match_no_known_effectors(self):
        """Target with no known_effectors → no match (silent skip)."""
        scorer = DrugScorer()
        t = TargetEvidence(gene_symbol="RANDOMGENE", action_type="INHIBITOR")
        gene_map = {
            "BLK": GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
        }
        result = scorer._downstream_effector_match(t, gene_map)
        assert result is None

    def test_effector_sub_threshold_ignored(self):
        """Effector genes below DEG_LOG2FC_THRESHOLD should not count."""
        scorer = DrugScorer()
        t = TargetEvidence(gene_symbol="TNFSF13B", action_type="INHIBITOR",
                           known_effectors=["BLK", "BANK1", "IRF5"])
        gene_map = {
            "BLK": GeneContext("BLK", log2fc=0.3, adj_p_value=0.5, direction="up"),
            "BANK1": GeneContext("BANK1", log2fc=0.2, adj_p_value=0.5, direction="up"),
            "IRF5": GeneContext("IRF5", log2fc=0.1, adj_p_value=0.5, direction="up"),
        }
        result = scorer._downstream_effector_match(t, gene_map)
        assert result is None

    def test_effector_credit_in_direction_scoring(self):
        """Downstream effector match should grant effector_credit_fraction of direction weight."""
        scorer = DrugScorer()
        c = _make_candidate(gene="TNFSF13B", action="INHIBITOR",
                            known_effectors=["BLK", "BANK1", "IRF5"])
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
            GeneContext("BANK1", log2fc=2.0, adj_p_value=0.01, direction="up"),
            GeneContext("IRF5", log2fc=1.5, adj_p_value=0.02, direction="up"),
        ])
        score = scorer._target_direction(c, r)
        expected = round(18.0 * 0.6, 2)
        assert score == expected, f"Expected {expected}, got {score}"

    def test_effector_credit_configurable(self):
        """effector_credit_fraction should be configurable."""
        cfg = ScoringConfig(effector_credit_fraction=0.4)
        scorer = DrugScorer(config=cfg)
        c = _make_candidate(gene="TNFSF13B", action="INHIBITOR",
                            known_effectors=["BLK", "BANK1", "IRF5"])
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
            GeneContext("BANK1", log2fc=2.0, adj_p_value=0.01, direction="up"),
        ])
        score = scorer._target_direction(c, r)
        expected = round(18.0 * 0.4, 2)
        assert score == expected

    def test_effector_populates_target_evidence(self):
        """Effector match should populate TargetEvidence fields."""
        scorer = DrugScorer()
        c = _make_candidate(gene="TNFSF13B", action="INHIBITOR",
                            known_effectors=["BLK", "BANK1", "IRF5"])
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
            GeneContext("BANK1", log2fc=2.0, adj_p_value=0.01, direction="up"),
        ])
        scorer._target_direction(c, r)
        t = c.targets[0]
        assert t.related_gene_source == "downstream_effector"
        assert t.downstream_effector_genes is not None
        assert len(t.downstream_effector_genes) >= 2

    def test_empty_known_effectors_silent_skip(self):
        """Empty list (e.g., KG returned nothing for rare gene) falls through cleanly."""
        scorer = DrugScorer()
        c = _make_candidate(gene="OBSCUREGENE", action="INHIBITOR",
                            known_effectors=[])
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up"),
        ])
        score = scorer._target_direction(c, r)
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Gap Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossGapIntegration:
    """Tests verifying multiple gaps interact correctly."""

    def test_stratum_and_effector_combined(self):
        """Effector credit should also apply stratum multiplier from related gene."""
        cfg = ScoringConfig()
        scorer = DrugScorer(config=cfg)
        c = _make_candidate(gene="TNFSF13B", action="INHIBITOR",
                            known_effectors=["BLK", "BANK1", "IRF5"])
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="expression_significant"),
            GeneContext("BANK1", log2fc=2.0, adj_p_value=0.01, direction="up",
                        evidence_stratum="expression_significant"),
        ])
        score = scorer._target_direction(c, r)
        # effector_credit (0.6) × stratum (0.65) × weight (18.0)
        expected = round(18.0 * 0.6 * 0.65, 2)
        assert score == expected, f"Expected {expected}, got {score}"

    def test_indirect_effect_with_stratum(self):
        """INDIRECT_EFFECT (65%) combined with stratum multiplier."""
        scorer = DrugScorer()
        c = _make_candidate(gene="BLK", action="INDIRECT_EFFECT")
        r = _make_request(genes=[
            GeneContext("BLK", log2fc=3.5, adj_p_value=0.001, direction="up",
                        evidence_stratum="ppi_connected"),
        ])
        score = scorer._target_direction(c, r)
        expected = round(18.0 * 0.65 * 0.85, 2)
        assert score == expected

    def test_score_breakdown_fields_complete(self):
        """All new fields should be present in ScoreBreakdown."""
        s = ScoreBreakdown()
        assert hasattr(s, "gene_evidence_quality")
        assert s.gene_evidence_quality == 1.0
        assert hasattr(s, "signature_bonus")


class TestPhase1SharedServiceFiltering:
    """Phase 1 Stage 5 filter should be config-driven and clinically aware."""

    def test_noise_threshold_defaults_present(self):
        cfg = ScoringConfig()
        assert cfg.base_noise_threshold == 10.0
        assert cfg.high_clinical_noise_threshold == 5.0
        assert cfg.high_clinical_score_cutoff == 15.0

    def test_high_clinical_candidate_uses_lower_threshold(self):
        svc = object.__new__(DrugAgentService)
        cfg = ScoringConfig()
        c = _make_candidate(name="TESTDRUG")
        c.score = ScoreBreakdown(composite_score=6.0, clinical_regulatory_score=16.0)
        assert svc._noise_threshold(c, cfg) == 5.0
        assert svc._has_signal(c, cfg)

    def test_low_clinical_candidate_keeps_default_threshold(self):
        svc = object.__new__(DrugAgentService)
        cfg = ScoringConfig()
        c = _make_candidate(name="TESTDRUG")
        c.score = ScoreBreakdown(composite_score=6.0, clinical_regulatory_score=15.0)
        assert svc._noise_threshold(c, cfg) == 10.0
        assert not svc._has_signal(c, cfg)
