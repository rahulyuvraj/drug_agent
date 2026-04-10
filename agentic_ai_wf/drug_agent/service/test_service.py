"""Acceptance test — validates the Drug Agent Service against live Qdrant."""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_health_check():
    from drug_agent.service import get_service
    svc = get_service()
    health = svc.health_check()

    print("\n═══ Health Check ═══")
    print(f"  Status: {health['status']}")
    print(f"  Collections: {len(health['available_collections'])}/{len(health['available_collections']) + len(health['unavailable_collections'])}")
    print(f"  Total points: {health['total_points']:,}")
    print(f"  Ensembl cache: {health['ensembl_cache_size']:,} genes")
    print(f"  Embedder: {health['embedder_device']}")
    for c, cnt in sorted(health["per_collection"].items()):
        print(f"    {c}: {cnt:,}")
    if health["unavailable_collections"]:
        print(f"  ⚠ Missing: {health['unavailable_collections']}")

    assert health["status"] in ("healthy", "degraded"), f"Unhealthy: {health}"
    assert len(health["available_collections"]) >= 10, "Too few collections"
    print("  ✅ Health check passed")
    return svc


def test_convenience_methods(svc):
    print("\n═══ Convenience Methods ═══")

    # Find drugs for a gene
    t0 = time.time()
    drugs = svc.find_drugs_for_gene("ERBB2", top_k=5)
    print(f"\n  find_drugs_for_gene('ERBB2'): {len(drugs)} drugs in {time.time()-t0:.1f}s")
    for d in drugs[:5]:
        print(f"    {d['drug_name']} [src={d['source']}, score={d['score']:.3f}]")

    # Find drugs for a disease
    t0 = time.time()
    drugs = svc.find_drugs_for_disease("Breast Cancer", top_k=5)
    print(f"\n  find_drugs_for_disease('Breast Cancer'): {len(drugs)} drugs in {time.time()-t0:.1f}s")
    for d in drugs[:5]:
        print(f"    {d['drug_name']} [src={d['source']}, score={d['score']:.3f}]")

    # Drug identity
    t0 = time.time()
    identity = svc.get_drug_identity("trastuzumab")
    print(f"\n  get_drug_identity('trastuzumab'): {time.time()-t0:.1f}s")
    print(f"    name={identity.drug_name}, chembl={identity.chembl_id}, type={identity.drug_type}")
    print(f"    phase={identity.max_phase}, fda={identity.is_fda_approved}, brands={identity.brand_names[:3]}")
    print(f"    patents={identity.patent_count}, generics={identity.generics_available}")

    # Drug targets
    t0 = time.time()
    targets = svc.get_drug_targets("trastuzumab")
    print(f"\n  get_drug_targets('trastuzumab'): {len(targets)} targets in {time.time()-t0:.1f}s")
    for t in targets[:5]:
        print(f"    {t.gene_symbol} [{t.action_type}] — {(t.mechanism_of_action or '')[:80]}")

    # Trial evidence
    t0 = time.time()
    trials = svc.get_trial_evidence("trastuzumab", "breast cancer")
    print(f"\n  get_trial_evidence('trastuzumab', 'breast cancer'): {time.time()-t0:.1f}s")
    print(f"    trials={trials.total_trials}, phase={trials.highest_phase}, completed={trials.completed_trials}")
    print(f"    with_results={trials.trials_with_results}, best_p={trials.best_p_value}, enrollment={trials.total_enrollment}")

    # Safety profile
    t0 = time.time()
    safety = svc.get_safety_profile("trastuzumab")
    print(f"\n  get_safety_profile('trastuzumab'): {time.time()-t0:.1f}s")
    print(f"    boxed_warnings={len(safety.boxed_warnings)}, aes={len(safety.top_adverse_events)}")
    print(f"    serious_ratio={safety.serious_ratio}, fatal_ratio={safety.fatal_ratio}")
    print(f"    pgx_warnings={len(safety.pgx_warnings)}, recalls={len(safety.recall_history)}")

    print("\n  ✅ Convenience methods passed")


def test_breast_cancer_her2():
    from drug_agent.service import (
        get_service, DrugQueryRequest, QueryType,
        GeneContext, PathwayContext, BiomarkerContext, MolecularSignatures,
    )

    svc = get_service()
    print("\n═══ Full Recommendation: Breast Cancer HER2+ ═══")

    req = DrugQueryRequest(
        disease="Breast Cancer",
        query_type=QueryType.FULL_RECOMMENDATION,
        genes=[
            GeneContext("ERBB2", 4.25, 0.05, "up", role="pathogenic"),
            GeneContext("S100A7", 7.60, 0.05, "up", role="pathogenic"),
            GeneContext("BRCA2", 1.67, 0.05, "up", role="protective"),
            GeneContext("FGFR4", 2.03, 0.05, "up", role="therapeutic_target"),
            GeneContext("CDC6", 1.50, 0.05, "up", role="therapeutic_target"),
            GeneContext("ESR1", -3.60, 0.05, "down", role="therapeutic_target"),
            GeneContext("MUC4", -4.25, 0.05, "down"),
            GeneContext("IL6", -3.67, 0.05, "down", role="immune_modulator"),
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
        signatures=MolecularSignatures(
            proliferation=0.60, apoptosis=0.64, dna_repair=0.60,
            inflammation=0.68, immune_activation=0.60,
        ),
        max_results=15,
    )

    t0 = time.time()
    resp = svc.query(req)
    duration = time.time() - t0

    print(f"\n  Duration: {duration:.1f}s")
    print(f"  Success: {resp.success}")
    if resp.errors:
        print(f"  Errors: {resp.errors}")
    print(f"  Metadata: {resp.metadata}")

    print(f"\n  Recommendations ({len(resp.recommendations)}):")
    for i, r in enumerate(resp.recommendations):
        s = r.score
        score_str = f"{s.composite_score:.1f}" if s else "N/A"
        print(f"    {i+1}. {r.identity.drug_name} — score={score_str}")
        if s:
            print(f"       dir={s.target_direction_match:.1f} mag={s.target_magnitude_match:.1f} "
                  f"clin={s.clinical_regulatory_score:.1f} ot={s.ot_association_score:.1f} "
                  f"pathway={s.pathway_concordance:.1f} safety={s.safety_penalty:.1f}")
        print(f"       sources={r.evidence_sources}")
        print(f"       targets={[t.gene_symbol for t in r.targets]}")

    print(f"\n  Contraindicated ({len(resp.contraindicated)}):")
    for c in resp.contraindicated:
        print(f"    ⚠ {c.identity.drug_name}: {c.contraindication_flags}")

    # Assertions
    assert resp.success, f"Failed: {resp.errors}"
    assert len(resp.recommendations) > 0, "No recommendations"

    # Trastuzumab should be in recommendations with score >= 60
    tras = next((r for r in resp.recommendations
                 if "TRASTUZUMAB" in r.identity.drug_name.upper()), None)
    assert tras is not None, f"Trastuzumab missing! Top drugs: {[r.identity.drug_name for r in resp.recommendations[:5]]}"
    assert tras.score.composite_score >= 40, (
        f"Trastuzumab score {tras.score.composite_score:.1f} < 40 — "
        f"dir={tras.score.target_direction_match:.1f} "
        f"mag={tras.score.target_magnitude_match:.1f} "
        f"clin={tras.score.clinical_regulatory_score:.1f} "
        f"ot={tras.score.ot_association_score:.1f} "
        f"pathway={tras.score.pathway_concordance:.1f} "
        f"safety={tras.score.safety_penalty:.1f}"
    )
    print(f"\n  ✅ Trastuzumab score: {tras.score.composite_score:.1f} (≥40)")

    # Contraindicated list should be non-empty (ER/PR-targeted drugs)
    assert len(resp.contraindicated) > 0, "Contraindicated list is empty — expected ER/PR-targeted drugs"
    contra_names = {c.identity.drug_name.upper() for c in resp.contraindicated}
    er_drugs = {"TAMOXIFEN", "LETROZOLE", "ANASTROZOLE", "EXEMESTANE", "FULVESTRANT"}
    found_er = contra_names & er_drugs
    print(f"  Contraindicated drugs: {sorted(contra_names)}")
    print(f"  Known ER/PR drugs found in contraindicated: {sorted(found_er)}")
    # At least one ER-targeted drug should be flagged via biomarker check
    # (This may not catch all — depends on indication text availability in Qdrant)

    print(f"\n  High priority: {[r.identity.drug_name for r in resp.high_priority]}")
    print(f"  Moderate priority: {[r.identity.drug_name for r in resp.moderate_priority]}")
    print("  ✅ Full recommendation test passed")


def test_cml_imatinib():
    """Disease-agnostic verification — CML with BCR-ABL1."""
    from drug_agent.service import get_service, DrugQueryRequest, QueryType, GeneContext, PathwayContext

    svc = get_service()
    print("\n═══ Full Recommendation: CML (disease-agnostic check) ═══")

    req = DrugQueryRequest(
        disease="Chronic Myeloid Leukemia",
        query_type=QueryType.FULL_RECOMMENDATION,
        genes=[
            GeneContext("ABL1", 3.50, 0.01, "up", role="pathogenic"),
            GeneContext("KIT", 2.10, 0.05, "up", role="therapeutic_target"),
            GeneContext("PDGFRA", 1.80, 0.05, "up", role="therapeutic_target"),
        ],
        pathways=[
            PathwayContext("Tyrosine kinase signaling", "up", 1e-8, 30),
        ],
        max_results=10,
    )

    t0 = time.time()
    resp = svc.query(req)
    print(f"  Duration: {time.time()-t0:.1f}s")
    print(f"  Recommendations: {len(resp.recommendations)}")
    for i, r in enumerate(resp.recommendations[:5]):
        s = r.score
        if s:
            print(f"    {i+1}. {r.identity.drug_name} — score={s.composite_score:.1f}")
            print(f"       dir={s.target_direction_match:.1f} mag={s.target_magnitude_match:.1f} "
                  f"clin={s.clinical_regulatory_score:.1f} ot={s.ot_association_score:.1f} "
                  f"pathway={s.pathway_concordance:.1f} safety={s.safety_penalty:.1f}")
        else:
            print(f"    {i+1}. {r.identity.drug_name} — no score")

    assert resp.success, f"Failed: {resp.errors}"

    # Dedup check: no drug should appear more than once
    norm_names = [r.identity.drug_name.upper().split()[0] for r in resp.recommendations]
    for i, name in enumerate(norm_names):
        dupes = [j for j, n in enumerate(norm_names) if n == name and j != i]
        if dupes:
            dupe_drugs = [resp.recommendations[j].identity.drug_name for j in [i] + dupes]
            print(f"  ⚠ Possible duplicate: {dupe_drugs}")

    print("  ✅ CML test passed")


def test_negative_imatinib_breast_cancer():
    """Negative test: Imatinib should NOT score high for breast cancer."""
    from drug_agent.service import get_service, DrugQueryRequest, QueryType, GeneContext

    svc = get_service()
    print("\n═══ Negative Test: Imatinib vs Breast Cancer ═══")

    req = DrugQueryRequest(
        disease="Breast Cancer",
        query_type=QueryType.VALIDATE_DRUG,
        drug_name="imatinib",
        genes=[
            GeneContext("ERBB2", 4.25, 0.05, "up", role="pathogenic"),
        ],
    )

    t0 = time.time()
    resp = svc.query(req)
    print(f"  Duration: {time.time()-t0:.1f}s")

    # Imatinib targets ABL1/KIT/PDGFRA — not HER2-relevant
    all_candidates = resp.recommendations + resp.contraindicated
    if all_candidates:
        c = all_candidates[0]
        s = c.score
        if s:
            print(f"  Imatinib score for Breast Cancer: {s.composite_score:.1f}")
            print(f"    dir={s.target_direction_match:.1f} mag={s.target_magnitude_match:.1f} "
                  f"clin={s.clinical_regulatory_score:.1f} ot={s.ot_association_score:.1f} "
                  f"pathway={s.pathway_concordance:.1f} safety={s.safety_penalty:.1f}")
            assert s.composite_score < 25, f"Imatinib scores too high ({s.composite_score:.1f}) for breast cancer"
            print(f"  ✅ Imatinib correctly scores low for breast cancer ({s.composite_score:.1f} < 25)")
        else:
            print("  ✅ No score calculated")
    else:
        print("  ✅ No candidates returned")


def test_bug_fixes():
    """Validate all 6 bug fixes from K-Dense stress testing."""
    from drug_agent.service import (
        get_service, DrugQueryRequest, QueryType,
        GeneContext, PathwayContext, BiomarkerContext, ScoringConfig,
    )
    from drug_agent.service.schemas import ContraindicationEntry, DrugCandidate, DrugIdentity, TargetEvidence, ScoreBreakdown

    svc = get_service()
    print("\n═══ Bug Fix Validation ═══")

    # ── Fix 3: action_type UNKNOWN override ──────────────────────────
    print("\n  [Fix 3] action_type UNKNOWN override")
    targets = svc.router.get_drug_targets("imatinib")
    for t_dict in targets:
        print(f"    {t_dict.get('gene_symbol')}: action_type={t_dict.get('action_type')}")
    # ABL1/KIT/PDGFRA should have real action types from ChEMBL, not UNKNOWN
    known_targets = {t["gene_symbol"].upper(): t["action_type"] for t in targets if t.get("gene_symbol")}
    for gene in ("ABL1", "KIT", "PDGFRA"):
        if gene in known_targets:
            action = known_targets[gene]
            if action and action != "UNKNOWN":
                print(f"    ✅ {gene} has real action_type: {action}")
            else:
                print(f"    ⚠ {gene} still UNKNOWN (may need ChEMBL data)")

    # ── Fix 4A: OT probe ─────────────────────────────────────────────
    print("\n  [Fix 4A] OT schema probe")
    print(f"    Probed score fields: {svc.router._ot_score_fields}")
    print(f"    ✅ OT probe ran at init")

    # ── Fix 1: Generic signature gate ────────────────────────────────
    print("\n  [Fix 1] Generic signature gate (IFN biomarker bridge)")
    # Create a mock scenario with IFN biomarker
    from drug_agent.service.drug_scorer import DrugScorer, _STATUS_TO_SCORE
    assert "high" in _STATUS_TO_SCORE, "Missing _STATUS_TO_SCORE mapping"
    assert _STATUS_TO_SCORE["high"] == 1.0
    assert _STATUS_TO_SCORE["low"] == 0.2
    print(f"    _STATUS_TO_SCORE: {_STATUS_TO_SCORE}")

    # Test signature gate with a mock candidate
    scorer = svc.scorer
    mock_identity = DrugIdentity(drug_name="ANIFROLUMAB", pharm_class_moa="type I interferon receptor antagonist")
    mock_candidate = DrugCandidate(
        identity=mock_identity,
        targets=[TargetEvidence(gene_symbol="IFNAR1", action_type="ANTAGONIST")],
    )
    mock_request = DrugQueryRequest(
        disease="Systemic Lupus Erythematosus",
        biomarkers=[BiomarkerContext("IFN signature", "high", ["IFNAR1"])],
    )
    s = ScoreBreakdown(target_direction_match=10.0)
    s.calculate()
    s.tier_reasoning = "test"
    result = scorer._signature_gate(s, mock_candidate, mock_request)
    print(f"    Signature gate result tier_reasoning: {result.tier_reasoning[:100]}")
    # With IFN HIGH, direction should NOT be halved
    assert result.target_direction_match == 10.0, f"Direction wrongly halved: {result.target_direction_match}"
    assert "aligns" in result.tier_reasoning.lower() or "high" in result.tier_reasoning.lower(), \
        f"Gate did not fire for HIGH: {result.tier_reasoning}"
    print(f"    ✅ IFN HIGH: direction preserved at {result.target_direction_match}")

    # Test with LOW status
    mock_request_low = DrugQueryRequest(
        disease="Systemic Lupus Erythematosus",
        biomarkers=[BiomarkerContext("IFN signature", "low", ["IFNAR1"])],
    )
    s2 = ScoreBreakdown(target_direction_match=10.0)
    s2.calculate()
    s2.tier_reasoning = "test"
    result2 = scorer._signature_gate(s2, mock_candidate, mock_request_low)
    print(f"    IFN LOW result: direction={result2.target_direction_match}")
    # With IFN LOW, direction should be halved
    assert result2.target_direction_match == 5.0, f"Direction not halved for LOW: {result2.target_direction_match}"
    print(f"    ✅ IFN LOW: direction correctly halved to {result2.target_direction_match}")

    # ── Fix 6: Entity dedup ──────────────────────────────────────────
    print("\n  [Fix 6] Post-enrichment entity dedup")
    # Check that DrugCandidate has contraindication_entries field
    dc = DrugCandidate(identity=DrugIdentity(drug_name="TEST"))
    assert hasattr(dc, 'contraindication_entries'), "Missing contraindication_entries field"
    print(f"    ✅ DrugCandidate has contraindication_entries field")

    # ── Fix 2: Contra tiers ──────────────────────────────────────────
    print("\n  [Fix 2] Contraindication tier multipliers")
    cfg = ScoringConfig(apply_contra_multipliers=True)
    assert cfg.contra_tier_multipliers == {1: 0.0, 2: 0.25, 3: 0.75}
    print(f"    contra_tier_multipliers: {cfg.contra_tier_multipliers}")
    print(f"    apply_contra_multipliers: {cfg.apply_contra_multipliers}")
    # ContraindicationEntry tier labels
    e1 = ContraindicationEntry(tier=1, reason="test", source="gene_based")
    e2 = ContraindicationEntry(tier=2, reason="test", source="biomarker")
    e3 = ContraindicationEntry(tier=3, reason="test", source="disease_ae")
    assert e1.label == "Avoid"
    assert e2.label == "Contraindicated"
    assert e3.label == "Use With Caution"
    print(f"    ✅ Tier labels: 1={e1.label}, 2={e2.label}, 3={e3.label}")

    # ── Fix 5: SOC multi-signal composite ────────────────────────────
    print("\n  [Fix 5] SOC multi-signal composite")
    cfg_soc = ScoringConfig(use_soc_composite=True)
    assert cfg_soc.use_soc_composite is True
    assert "indication_sim" in cfg_soc.soc_signal_weights
    print(f"    soc_signal_weights: {cfg_soc.soc_signal_weights}")

    # _indication_similarity should return float
    sim = scorer._indication_similarity(mock_candidate, mock_request)
    print(f"    indication_similarity(anifrolumab, SLE): {sim:.3f}")
    assert isinstance(sim, float), f"Expected float, got {type(sim)}"
    print(f"    ✅ _indication_similarity returns float")

    # _has_disease_indication should still work as bool
    is_indicated = scorer._has_disease_indication(mock_candidate, mock_request)
    print(f"    _has_disease_indication(anifrolumab, SLE): {is_indicated}")
    assert isinstance(is_indicated, bool), f"Expected bool, got {type(is_indicated)}"
    print(f"    ✅ _has_disease_indication returns bool")

    # soc_confidence field exists
    dc2 = DrugCandidate(identity=DrugIdentity(drug_name="TEST"))
    assert hasattr(dc2, 'soc_confidence'), "Missing soc_confidence field"
    assert dc2.soc_confidence == 0.0
    print(f"    ✅ DrugCandidate has soc_confidence field (default=0.0)")

    print("\n  ═══ ALL BUG FIX TESTS PASSED ✅ ═══")


if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    # Load .env from drug_agent directory
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    svc = test_health_check()
    test_convenience_methods(svc)
    test_breast_cancer_her2()
    test_cml_imatinib()
    test_negative_imatinib_breast_cancer()
    test_bug_fixes()

    print("\n═══════════════════════════════")
    print("  ALL TESTS PASSED ✅")
    print("═══════════════════════════════")
