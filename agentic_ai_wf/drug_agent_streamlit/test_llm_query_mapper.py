"""Tests for llm_query_mapper.py — mocked Bedrock, structured extraction + JSON robustness."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_ai_wf.drug_agent.service.schemas import (
    DrugQueryRequest,
    GeneContext,
    PathwayContext,
    QueryType,
    TMEContext,
)
from agentic_ai_wf.drug_agent_streamlit.file_parser import FileSummary
from agentic_ai_wf.drug_agent_streamlit.llm_query_mapper import (
    _assign_gene_role,
    _assign_evidence_stratum,
    _build_disease_context,
    _build_genes_from_dataframe,
    _build_genes_from_parsed,
    _build_pathways_from_dataframe,
    _build_request,
    _build_tme_from_dataframe,
    _categorize_pathway,
    _detect_columns,
    _extract_top_n,
    _fallback_extract,
    _try_parse_json,
    classify_query,
    map_query_and_file,
)


# ─── _try_parse_json ─────────────────────────────────────────────────────────

class TestTryParseJson:
    def test_clean_json(self):
        assert _try_parse_json('{"disease": "ALS"}') == {"disease": "ALS"}

    def test_json_in_markdown_fences(self):
        text = '```json\n{"disease": "lupus"}\n```'
        assert _try_parse_json(text) == {"disease": "lupus"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"disease": "asthma", "genes": []}\nDone.'
        result = _try_parse_json(text)
        assert result["disease"] == "asthma"

    def test_empty_string(self):
        assert _try_parse_json("") == {}

    def test_totally_invalid(self):
        assert _try_parse_json("not json at all") == {}

    def test_fences_without_json_label(self):
        text = '```\n{"disease": "COPD"}\n```'
        assert _try_parse_json(text)["disease"] == "COPD"

    def test_nested_json(self):
        obj = {"disease": "RA", "genes": [{"gene_symbol": "TNF", "log2fc": 2.0}]}
        text = json.dumps(obj)
        assert _try_parse_json(text) == obj


# ─── classify_query ──────────────────────────────────────────────────────────

class TestClassifyQuery:
    def _mock_client(self, response_text):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = response_text
        client.chat.completions.create.return_value = resp
        client.model_id = "test-model"
        return client

    def test_drug_query(self):
        client = self._mock_client("DRUG")
        assert classify_query(client, "Find drugs for breast cancer") == "DRUG"

    def test_other_query(self):
        client = self._mock_client("OTHER")
        assert classify_query(client, "What are the top 10 upregulated genes?") == "OTHER"

    def test_fails_open_as_drug(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("timeout")
        client.model_id = "test"
        assert classify_query(client, "anything") == "DRUG"


# ─── _build_genes_from_parsed ────────────────────────────────────────────────

class TestBuildGenesFromParsed:
    def test_basic_gene_extraction(self):
        parsed = {
            "genes": [
                {"gene_symbol": "BRCA1", "log2fc": 2.5, "adj_p_value": 0.001, "direction": "up"},
                {"gene_symbol": "TP53", "log2fc": -1.2, "adj_p_value": 0.03, "direction": "down"},
            ]
        }
        genes = _build_genes_from_parsed(parsed)
        assert len(genes) == 2
        assert genes[0].gene_symbol == "BRCA1"
        assert genes[0].direction == "up"
        assert genes[1].gene_symbol == "TP53"
        assert genes[1].log2fc == -1.2

    def test_defaults_when_missing_values(self):
        parsed = {"genes": [{"gene_symbol": "MYC"}]}
        genes = _build_genes_from_parsed(parsed)
        assert len(genes) == 1
        assert genes[0].log2fc == 1.0
        assert genes[0].adj_p_value == 0.05

    def test_skips_empty_symbols(self):
        parsed = {"genes": [{"gene_symbol": ""}, {"gene_symbol": "JAK2", "log2fc": 1.0}]}
        genes = _build_genes_from_parsed(parsed)
        assert len(genes) == 1

    def test_direction_inferred_from_log2fc(self):
        parsed = {"genes": [{"gene_symbol": "EGFR", "log2fc": -0.8}]}
        genes = _build_genes_from_parsed(parsed)
        assert genes[0].direction == "down"


# ─── _build_genes_from_dataframe ─────────────────────────────────────────────

class TestBuildGenesFromDataframe:
    def test_standard_deg_table(self):
        df = pd.DataFrame({
            "gene_symbol": ["TNF", "IL6", "JAK2"],
            "log2FoldChange": [2.1, -1.5, 0.8],
            "padj": [0.001, 0.005, 0.02],
        })
        genes = _build_genes_from_dataframe(df)
        assert len(genes) == 3
        # Sorted by padj ascending
        assert genes[0].gene_symbol == "TNF"

    def test_alternative_column_names(self):
        df = pd.DataFrame({
            "Symbol": ["KRAS", "BRAF"],
            "logFC": [1.5, -2.0],
            "FDR": [0.01, 0.03],
        })
        genes = _build_genes_from_dataframe(df)
        assert len(genes) == 2

    def test_no_gene_column(self):
        df = pd.DataFrame({"value": [1, 2], "pval": [0.01, 0.05]})
        genes = _build_genes_from_dataframe(df)
        assert genes == []

    def test_gene_only_no_expression(self):
        df = pd.DataFrame({"gene": ["TP53", "MYC", "EGFR"]})
        genes = _build_genes_from_dataframe(df)
        assert len(genes) == 3
        assert all(g.log2fc == 1.0 for g in genes)
        assert all(g.adj_p_value == 0.05 for g in genes)

    def test_respects_max_genes(self):
        df = pd.DataFrame({
            "gene_symbol": [f"G{i}" for i in range(700)],
            "padj": [0.001 + i * 0.001 for i in range(700)],
        })
        genes = _build_genes_from_dataframe(df, max_genes=50)
        assert len(genes) == 50

    def test_id_column_with_gene_content(self):
        """Handles CSV files where gene symbols are in an 'id' column."""
        df = pd.DataFrame({
            "id": ["HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK"],
            "expression_log2fc": [0.0, 0.0, 0.0, 0.0, 0.0],
            "expression_trend": ["Up", "Up", "Up", "Down", "Down"],
            "mr_pval": [6.57e-19, 1e-10, 1e-8, 1.55e-6, 1.6e-16],
        })
        genes = _build_genes_from_dataframe(df)
        assert len(genes) == 5
        assert {g.gene_symbol for g in genes} == {"HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK"}

    def test_expression_trend_direction(self):
        """expression_trend='Up'/'Down' overrides log2fc-based direction when log2fc is 0."""
        df = pd.DataFrame({
            "gene": ["BRCA1", "TP53", "MYC"],
            "expression_log2fc": [0.0, 0.0, 0.0],
            "expression_trend": ["Up", "Down", "Up"],
        })
        genes = _build_genes_from_dataframe(df)
        dirs = {g.gene_symbol: g.direction for g in genes}
        assert dirs["BRCA1"] == "up"
        assert dirs["TP53"] == "down"
        assert dirs["MYC"] == "up"

    def test_sorts_by_significance(self):
        """Genes are sorted by p-value ascending, then |log2fc| descending."""
        df = pd.DataFrame({
            "gene_symbol": ["WEAK", "STRONG", "MID"],
            "log2FoldChange": [0.1, 3.0, 1.5],
            "padj": [0.5, 0.001, 0.01],
        })
        genes = _build_genes_from_dataframe(df)
        assert genes[0].gene_symbol == "STRONG"
        assert genes[1].gene_symbol == "MID"
        assert genes[2].gene_symbol == "WEAK"

    def test_expanded_pval_aliases(self):
        """mr_pval and min_gwas_pval are recognised as p-value columns."""
        df = pd.DataFrame({
            "gene": ["BRCA1", "TP53", "MYC"],
            "mr_pval": [1e-10, 1e-5, 0.01],
        })
        genes = _build_genes_from_dataframe(df)
        assert genes[0].adj_p_value == 1e-10

    def test_synthesized_log2fc_when_zero_with_direction(self):
        """When log2fc is explicitly 0 but direction column exists, synthesize \u00b11.0."""
        df = pd.DataFrame({
            "gene": ["BRCA1", "TP53"],
            "expression_log2fc": [0.0, 0.0],
            "expression_trend": ["Up", "Down"],
        })
        genes = _build_genes_from_dataframe(df)
        gene_map = {g.gene_symbol: g for g in genes}
        assert gene_map["BRCA1"].log2fc == 1.0
        assert gene_map["TP53"].log2fc == -1.0

    def test_paracrine_targets_skipped_in_dataframe(self):
        """Regression: _build_genes_from_dataframe uses 'id' column, not empty 'paracrine_targets'."""
        df = pd.DataFrame({
            "id": ["HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK"],
            "expression_log2fc": [0.0, 0.0, 0.0, 0.0, 0.0],
            "expression_trend": ["Up", "Up", "Up", "Down", "Down"],
            "paracrine_targets": ["", "RIPK2", "", "", ""],
            "mr_pval": [6.57e-19, 1e-10, 1e-8, 1.55e-6, 1.6e-16],
        })
        genes = _build_genes_from_dataframe(df)
        assert len(genes) == 5
        assert {g.gene_symbol for g in genes} == {"HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK"}


# ─── _extract_top_n ──────────────────────────────────────────────────────────

class TestExtractTopN:
    def test_top_n_extraction(self):
        assert _extract_top_n("find drugs for top 10 genes") == 10
        assert _extract_top_n("most significant 20 genes for lupus") == 20
        assert _extract_top_n("first 5 upregulated genes") == 5

    def test_no_top_n(self):
        assert _extract_top_n("find drugs for all genes in this file") is None
        assert _extract_top_n("recommend drugs for lupus") is None


# ─── _fallback_extract ───────────────────────────────────────────────────────

class TestFallbackExtract:
    def test_disease_and_gene_from_text(self):
        result = _fallback_extract("Recommend drugs for breast cancer targeting BRCA1")
        assert result["disease"].lower() == "breast cancer"
        assert any(g["gene_symbol"] == "BRCA1" for g in result["genes"])

    def test_multiple_genes(self):
        result = _fallback_extract("Find drugs for lupus targeting TNF IL6 JAK2")
        symbols = {g["gene_symbol"] for g in result["genes"]}
        assert {"TNF", "IL6", "JAK2"} <= symbols

    def test_no_genes_returns_disease(self):
        result = _fallback_extract("Recommend drugs for Crohn's disease")
        assert "crohn" in result["disease"].lower()

    def test_empty_query(self):
        result = _fallback_extract("")
        assert result["disease"] == "unknown"


# ─── _build_request ──────────────────────────────────────────────────────────

class TestBuildRequest:
    def test_full_recommendation(self):
        parsed = {
            "disease": "Crohn's Disease",
            "query_type": "full_recommendation",
            "genes": [{"gene_symbol": "TNF", "log2fc": 2.0, "adj_p_value": 0.001, "direction": "up"}],
            "pathways": [{"pathway_name": "NFkB", "direction": "up", "fdr": 0.01, "gene_count": 15}],
            "disease_aliases": ["CD", "Crohn disease"],
        }
        req = _build_request(parsed, 15)
        assert req.disease == "Crohn's Disease"
        assert req.query_type == QueryType.FULL_RECOMMENDATION
        assert len(req.genes) == 1
        assert len(req.pathways) == 1
        # disease_aliases forced empty so Stage 0 Qdrant expansion always fires
        assert req.disease_aliases == []

    def test_validate_drug(self):
        parsed = {
            "disease": "Rheumatoid Arthritis",
            "query_type": "validate_drug",
            "drug_name": "Methotrexate",
            "genes": [],
        }
        req = _build_request(parsed, 10)
        assert req.query_type == QueryType.VALIDATE_DRUG
        assert req.drug_name == "Methotrexate"

    def test_unknown_query_type_defaults(self):
        parsed = {"disease": "ALS", "query_type": "nonsense"}
        req = _build_request(parsed, 15)
        assert req.query_type == QueryType.FULL_RECOMMENDATION

    def test_all_patient_genes_populated(self):
        parsed = {
            "disease": "Asthma",
            "genes": [{"gene_symbol": "IL4", "log2fc": 1.2, "adj_p_value": 0.01, "direction": "up"}],
        }
        req = _build_request(parsed, 15)
        assert len(req.all_patient_genes) == len(req.genes)

    def test_dataframe_first_over_llm(self):
        """When file has a DataFrame, its genes take priority over LLM-extracted genes."""
        parsed = {
            "disease": "Lupus",
            "genes": [{"gene_symbol": "FAKE_LLM", "log2fc": 1.0}],
        }
        df = pd.DataFrame({
            "gene_symbol": ["TNF", "IL6", "JAK2"],
            "log2FoldChange": [2.1, -1.5, 0.8],
            "padj": [0.001, 0.005, 0.02],
        })
        fs = FileSummary(
            raw_preview="...", data_type="tabular",
            columns=list(df.columns), row_count=3,
            sample_genes=["TNF", "IL6", "JAK2"], dataframe=df,
        )
        req = _build_request(parsed, 15, fs)
        symbols = {g.gene_symbol for g in req.genes}
        assert "TNF" in symbols
        assert "IL6" in symbols
        # LLM-only gene merged in
        assert "FAKE_LLM" in symbols
        assert len(req.genes) == 4

    def test_top_n_limits_genes(self):
        """top_n caps genes from DataFrame."""
        df = pd.DataFrame({
            "gene_symbol": [f"G{i}" for i in range(100)],
            "padj": [i * 0.001 for i in range(100)],
        })
        fs = FileSummary(
            raw_preview="...", data_type="tabular",
            columns=list(df.columns), row_count=100,
            sample_genes=[], dataframe=df,
        )
        req = _build_request({"disease": "Test"}, 15, fs, top_n=10)
        assert len(req.genes) == 10


# ─── map_query_and_file (integration with mocked LLM) ───────────────────────

class TestMapQueryAndFile:
    def _mock_client(self, response_json: dict):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(response_json)
        client.chat.completions.create.return_value = resp
        client.model_id = "test-model"
        return client

    def test_basic_mapping(self):
        llm_response = {
            "disease": "Sarcoidosis",
            "query_type": "full_recommendation",
            "genes": [{"gene_symbol": "ACE", "log2fc": 1.8, "adj_p_value": 0.005, "direction": "up"}],
            "pathways": [],
            "biomarkers": [],
            "max_results": 15,
        }
        client = self._mock_client(llm_response)
        req, raw, parsed = map_query_and_file(client, "Find drugs for sarcoidosis")
        assert req.disease == "Sarcoidosis"
        assert len(req.genes) == 1
        assert req.genes[0].gene_symbol == "ACE"

    def test_with_file_summary(self):
        llm_response = {
            "disease": "Breast Cancer",
            "genes": [
                {"gene_symbol": "BRCA1", "log2fc": 2.5, "adj_p_value": 0.001, "direction": "up"},
                {"gene_symbol": "HER2", "log2fc": 3.0, "adj_p_value": 0.0001, "direction": "up"},
            ],
        }
        client = self._mock_client(llm_response)
        fs = FileSummary(
            raw_preview="gene_symbol,log2fc,padj\nBRCA1,2.5,0.001\nHER2,3.0,0.0001",
            data_type="tabular",
            columns=["gene_symbol", "log2fc", "padj"],
            row_count=2,
            sample_genes=["BRCA1", "HER2"],
        )
        req, _, _ = map_query_and_file(client, "Find drugs for breast cancer", fs)
        assert req.disease == "Breast Cancer"
        assert len(req.genes) == 2

    def test_markdown_fenced_response(self):
        raw_text = '```json\n{"disease": "COPD", "genes": []}\n```'
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = raw_text
        client.chat.completions.create.return_value = resp
        client.model_id = "test"
        req, _, _ = map_query_and_file(client, "drugs for COPD")
        assert req.disease == "COPD"

    def test_fallback_to_dataframe_genes(self):
        """When LLM returns no genes but file has a DataFrame, DataFrame extraction kicks in."""
        llm_response = {"disease": "Lung Cancer", "genes": []}
        client = self._mock_client(llm_response)
        df = pd.DataFrame({
            "gene_symbol": ["EGFR", "KRAS", "ALK"],
            "log2FoldChange": [2.0, -1.5, 1.2],
            "padj": [0.001, 0.01, 0.05],
        })
        fs = FileSummary(
            raw_preview="...",
            data_type="tabular",
            columns=list(df.columns),
            row_count=3,
            sample_genes=["EGFR", "KRAS", "ALK"],
            dataframe=df,
        )
        req, _, _ = map_query_and_file(client, "Find drugs for lung cancer", fs)
        assert len(req.genes) == 3

    def test_dataframe_takes_priority_over_llm_genes(self):
        """DataFrame genes are primary even when LLM also returns genes."""
        llm_response = {
            "disease": "Lupus",
            "genes": [{"gene_symbol": "LLM_ONLY", "log2fc": 1.0}],
        }
        client = self._mock_client(llm_response)
        df = pd.DataFrame({
            "gene_symbol": ["TNF", "IL6"],
            "log2FoldChange": [2.0, -1.0],
            "padj": [0.001, 0.01],
        })
        fs = FileSummary(
            raw_preview="...", data_type="tabular",
            columns=list(df.columns), row_count=2,
            sample_genes=["TNF", "IL6"], dataframe=df,
        )
        req, _, _ = map_query_and_file(client, "drugs for lupus", fs)
        symbols = {g.gene_symbol for g in req.genes}
        assert "TNF" in symbols
        assert "IL6" in symbols
        assert "LLM_ONLY" in symbols  # merged

    def test_top_n_from_query(self):
        """'top 10' in query limits genes from DataFrame."""
        llm_response = {"disease": "Test", "genes": []}
        client = self._mock_client(llm_response)
        df = pd.DataFrame({
            "gene_symbol": [f"G{i}" for i in range(100)],
            "padj": [i * 0.001 for i in range(100)],
        })
        fs = FileSummary(
            raw_preview="...", data_type="tabular",
            columns=list(df.columns), row_count=100,
            sample_genes=[], dataframe=df,
        )
        req, _, _ = map_query_and_file(client, "find drugs for top 10 genes", fs)
        assert len(req.genes) == 10

    def test_llm_failure_uses_fallback(self):
        """When LLM fails entirely, regex fallback extracts disease + genes from query text."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Bedrock throttled")
        client.model_id = "test"
        req, raw, parsed = map_query_and_file(
            client, "Recommend drugs for breast cancer targeting BRCA1"
        )
        assert "breast cancer" in req.disease.lower()
        assert any(g.gene_symbol == "BRCA1" for g in req.genes)


# ─── _detect_columns ────────────────────────────────────────────────────────

class TestDetectColumns:
    def test_detects_role_columns(self):
        cols = ["id", "therapeutic_recommendation", "eqtl_causal_direction",
                "strategy_type", "causal_linkage_tier"]
        detected = _detect_columns(cols)
        assert detected["therapeutic_recommendation"] == "therapeutic_recommendation"
        assert detected["eqtl_causal_direction"] == "eqtl_causal_direction"
        assert detected["strategy_type"] == "strategy_type"
        assert detected["causal_linkage_tier"] == "causal_linkage_tier"

    def test_detects_stratum_columns(self):
        cols = ["id", "has_disease_link", "has_pathway_link", "mr_pval"]
        detected = _detect_columns(cols)
        assert detected["has_disease_link"] == "has_disease_link"
        assert detected["has_pathway_link"] == "has_pathway_link"
        assert detected["mr_pval"] == "mr_pval"

    def test_empty_columns(self):
        detected = _detect_columns(["id", "value"])
        # All keys map to None (no matching role/stratum columns found)
        assert all(v is None for v in detected.values())


# ─── _assign_gene_role ───────────────────────────────────────────────────────

class TestAssignGeneRole:
    """Verify the 5-priority cascade for gene role assignment."""

    def _make_row(self, **kwargs):
        return pd.Series(kwargs)

    def test_therapeutic_recommendation_target(self):
        """Priority 1: 'therapeutic_recommendation' containing 'target'."""
        row = self._make_row(therapeutic_recommendation="primary target")
        detected = {"therapeutic_recommendation": "therapeutic_recommendation"}
        assert _assign_gene_role(row, "up", detected) == "therapeutic_target"

    def test_eqtl_protective(self):
        """Priority 2: eqtl_causal_direction='Protective'."""
        row = self._make_row(eqtl_causal_direction="Protective")
        detected = {"eqtl_causal_direction": "eqtl_causal_direction"}
        assert _assign_gene_role(row, "up", detected) == "protective"

    def test_eqtl_risk_up(self):
        """eqtl_causal_direction='Risk' + up → pathogenic."""
        row = self._make_row(eqtl_causal_direction="Risk")
        detected = {"eqtl_causal_direction": "eqtl_causal_direction"}
        assert _assign_gene_role(row, "up", detected) == "pathogenic"

    def test_eqtl_risk_down(self):
        """eqtl_causal_direction='Risk' + down → protective."""
        row = self._make_row(eqtl_causal_direction="Risk")
        detected = {"eqtl_causal_direction": "eqtl_causal_direction"}
        assert _assign_gene_role(row, "down", detected) == "protective"

    def test_strategy_immune_keyword(self):
        """Priority 3: strategy_type with immune keyword."""
        row = self._make_row(strategy_type="Modulate Immune Response")
        detected = {"strategy_type": "strategy_type"}
        assert _assign_gene_role(row, "up", detected) == "immune_modulator"

    def test_causal_tier_validated_up(self):
        """Priority 4: causal_tier with 'Validated' + up → pathogenic."""
        row = self._make_row(causal_linkage_tier="Tier 1 — Validated Driver (causal)")
        detected = {"causal_linkage_tier": "causal_linkage_tier"}
        assert _assign_gene_role(row, "up", detected) == "pathogenic"

    def test_causal_tier_validated_down(self):
        """Priority 4: causal_tier with 'Validated' + down → protective."""
        row = self._make_row(causal_linkage_tier="Tier 1 — Validated Driver (causal)")
        detected = {"causal_linkage_tier": "causal_linkage_tier"}
        assert _assign_gene_role(row, "down", detected) == "protective"

    def test_directional_fallback_up(self):
        """Priority 5: no columns match → up → pathogenic."""
        row = self._make_row()
        assert _assign_gene_role(row, "up", {}) == "pathogenic"

    def test_directional_fallback_down(self):
        """Priority 5: no columns match → down → protective (SOC fix)."""
        row = self._make_row()
        assert _assign_gene_role(row, "down", {}) == "protective"


# ─── _assign_evidence_stratum ────────────────────────────────────────────────

class TestAssignEvidenceStratum:
    def _make_row(self, **kwargs):
        return pd.Series(kwargs)

    def test_full_causal_chain(self):
        row = self._make_row(
            causal_linkage_tier="Tier 1 — Full Causal Chain (genetic variant → expression → disease)"
        )
        detected = {"causal_linkage_tier": "causal_linkage_tier"}
        assert _assign_evidence_stratum(row, detected) == "known_driver"

    def test_disease_and_pathway_linked(self):
        row = self._make_row(has_disease_link="Yes", has_pathway_link="Yes")
        detected = {
            "has_disease_link": "has_disease_link",
            "has_pathway_link": "has_pathway_link",
        }
        assert _assign_evidence_stratum(row, detected) == "ppi_connected"

    def test_mr_significant(self):
        row = self._make_row(mr_pval=0.001)
        detected = {"mr_pval": "mr_pval"}
        assert _assign_evidence_stratum(row, detected) == "expression_significant"

    def test_novel_candidate_fallback(self):
        """When stratum columns exist but no criteria met → novel_candidate."""
        row = self._make_row(has_disease_link="No", mr_pval=0.5)
        detected = {
            "has_disease_link": "has_disease_link",
            "mr_pval": "mr_pval",
        }
        assert _assign_evidence_stratum(row, detected) == "novel_candidate"

    def test_no_stratum_columns_returns_none(self):
        """When no stratum columns exist at all → None (backward compat multiplier 1.0)."""
        row = self._make_row(gene="TNF")
        assert _assign_evidence_stratum(row, {}) is None


# ─── _categorize_pathway ─────────────────────────────────────────────────────

class TestCategorizePathway:
    def test_complement_pathway(self):
        assert _categorize_pathway("Complement activation") == "Immune/Complement"

    def test_interferon_pathway(self):
        assert _categorize_pathway("Type I interferon signaling") == "Immune/Interferon"

    def test_t_cell_pathway(self):
        assert _categorize_pathway("T cell receptor signaling") == "Immune/T Cell"

    def test_unknown_pathway(self):
        # "pathway" keyword matches the Signaling category pattern
        assert _categorize_pathway("Some random pathway") == "Signaling"

    def test_truly_unknown_pathway(self):
        assert _categorize_pathway("Unrelated biological process") is None


# ─── _build_pathways_from_dataframe ──────────────────────────────────────────

class TestBuildPathwaysFromDataframe:
    def test_pipe_delimited_extraction(self):
        """Standard SLE-like CSV with pipe-delimited pathways column."""
        df = pd.DataFrame({
            "id": ["C4B", "HLA-C", "IRF5"],
            "pathways": [
                "Complement activation | Initial triggering of complement",
                "Antigen processing and presentation",
                "Interferon signaling | Type I interferon production",
            ],
            "expression_trend": ["Down", "Up", "Up"],
            "expression_log2fc": [0.0, 0.0, 0.0],
            "mr_pval": [1e-10, 1e-15, 1e-8],
        })
        pathways = _build_pathways_from_dataframe(df, disease="SLE")
        assert len(pathways) >= 3
        names = {p.pathway_name for p in pathways}
        assert "Complement activation" in names
        assert "Interferon signaling" in names

    def test_no_pathway_column(self):
        df = pd.DataFrame({"gene": ["TNF"], "padj": [0.01]})
        assert _build_pathways_from_dataframe(df, disease="Test") == []

    def test_max_pathways_cap(self):
        """Respects max_pathways limit."""
        rows = []
        for i in range(100):
            rows.append({"id": f"G{i}", "pathways": f"Pathway_{i}", "expression_trend": "Up"})
        df = pd.DataFrame(rows)
        pathways = _build_pathways_from_dataframe(df, disease="Test", max_pathways=20)
        assert len(pathways) <= 20

    def test_key_genes_populated(self):
        """Each pathway has key_genes listing member genes."""
        df = pd.DataFrame({
            "id": ["TNF", "IL6", "NFKB1"],
            "pathways": [
                "NFkB signaling",
                "NFkB signaling | Cytokine signaling",
                "NFkB signaling",
            ],
            "expression_log2fc": [2.0, -1.5, 1.0],
        })
        pathways = _build_pathways_from_dataframe(df, disease="RA")
        nfkb = [p for p in pathways if p.pathway_name == "NFkB signaling"][0]
        assert "TNF" in nfkb.key_genes
        assert nfkb.gene_count == 3

    def test_disease_relevance_set(self):
        """disease_relevance set when disease name appears in pathway name."""
        df = pd.DataFrame({
            "id": ["C4B", "IRF5"],
            "pathways": ["Lupus nephritis immune pathway", "Interferon signaling"],
            "expression_trend": ["Down", "Up"],
        })
        pathways = _build_pathways_from_dataframe(df, disease="Lupus")
        lupus_pw = [p for p in pathways if "Lupus" in p.pathway_name]
        assert len(lupus_pw) == 1
        assert lupus_pw[0].disease_relevance is not None


# ─── _build_tme_from_dataframe ───────────────────────────────────────────────

class TestBuildTmeFromDataframe:
    def test_extracts_cell_types(self):
        """TME splits by pipe delimiter (matching SLE CSV format)."""
        df = pd.DataFrame({
            "id": ["TNF", "IL6", "BRCA1", "TP53", "KRAS"],
            "cell_types_active_in": [
                "T cells|B cells",
                "T cells|Macrophages",
                "",
                "NK cells",
                "T cells",
            ],
        })
        tme = _build_tme_from_dataframe(df)
        assert tme is not None
        # T cells appears 3 times (top quartile)
        assert "T cells" in tme.highly_enriched_cells

    def test_no_cell_type_column(self):
        df = pd.DataFrame({"gene": ["TNF"], "padj": [0.01]})
        assert _build_tme_from_dataframe(df) is None


# ─── _build_disease_context ─────────────────────────────────────────────────

class TestBuildDiseaseContext:
    def test_basic_context(self):
        genes = [
            GeneContext(gene_symbol="TNF", log2fc=2.0, adj_p_value=0.001, direction="up"),
            GeneContext(gene_symbol="IL6", log2fc=-1.5, adj_p_value=0.005, direction="down"),
        ]
        pathways = [
            PathwayContext(pathway_name="NFkB signaling", direction="up", fdr=0.01, gene_count=10),
        ]
        ctx = _build_disease_context("Lupus", genes, pathways)
        assert ctx is not None
        assert "Lupus" in ctx
        assert "TNF" in ctx
        assert len(ctx) <= 500

    def test_no_genes(self):
        """Empty genes list → context still generated (has 0 DEGs)."""
        ctx = _build_disease_context("ALS", [], [])
        assert ctx is not None
        assert "0 DEGs" in ctx

    def test_unknown_disease(self):
        genes = [GeneContext(gene_symbol="A", log2fc=1.0, adj_p_value=0.01, direction="up")]
        ctx = _build_disease_context("unknown", genes, [])
        assert ctx is None


# ─── disease_aliases always forced empty ─────────────────────────────────────

class TestDiseaseAliasesForced:
    def test_aliases_always_empty(self):
        """SOC shield fix: disease_aliases must always be [] to force Qdrant expansion."""
        parsed = {
            "disease": "SLE",
            "disease_aliases": ["Lupus", "Systemic Lupus Erythematosus"],
            "genes": [{"gene_symbol": "TNF", "log2fc": 1.0}],
        }
        req = _build_request(parsed, 15)
        assert req.disease_aliases == []


# ─── downregulated genes get role ────────────────────────────────────────────

class TestDownregulatedGenesRole:
    def test_down_genes_get_protective_role(self):
        """Down-regulated genes must have role='protective' so Stage 1 includes them."""
        df = pd.DataFrame({
            "id": ["C4B", "HLA-C", "TNF"],
            "expression_log2fc": [0.0, 0.0, 0.0],
            "expression_trend": ["Down", "Up", "Up"],
        })
        genes = _build_genes_from_dataframe(df)
        gene_map = {g.gene_symbol: g for g in genes}
        assert gene_map["C4B"].direction == "down"
        assert gene_map["C4B"].role == "protective"
        assert gene_map["HLA-C"].direction == "up"
        assert gene_map["HLA-C"].role == "pathogenic"

    def test_causal_tier_overrides_direction(self):
        """When causal_linkage_tier is present, it takes priority over plain direction."""
        df = pd.DataFrame({
            "id": ["C4B", "IRF5"],
            "expression_trend": ["Down", "Up"],
            "expression_log2fc": [-1.0, 2.0],
            "causal_linkage_tier": [
                "Tier 1 — Validated Driver",
                "Tier 1 — Validated Driver",
            ],
        })
        genes = _build_genes_from_dataframe(df)
        gene_map = {g.gene_symbol: g for g in genes}
        # Down + Validated → protective
        assert gene_map["C4B"].role == "protective"
        # Up + Validated → pathogenic
        assert gene_map["IRF5"].role == "pathogenic"


# ─── composite_score from DataFrame ──────────────────────────────────────────

class TestCompositeScore:
    def test_composite_score_from_csv(self):
        df = pd.DataFrame({
            "id": ["TNF", "IL6"],
            "Gene_Genetic_Confidence_Score": [0.92, 0.78],
        })
        genes = _build_genes_from_dataframe(df)
        gene_map = {g.gene_symbol: g for g in genes}
        assert gene_map["TNF"].composite_score == 0.92
        assert gene_map["IL6"].composite_score == 0.78

    def test_missing_composite_score_defaults_none(self):
        """Composite score stays None when CSV has no confidence column."""
        df = pd.DataFrame({"id": ["TNF", "IL6"], "padj": [0.01, 0.05]})
        genes = _build_genes_from_dataframe(df)
        assert len(genes) == 2
        assert genes[0].composite_score is None


# ─── evidence_stratum integration ────────────────────────────────────────────

class TestEvidenceStratumIntegration:
    def test_stratum_set_on_dataframe_genes(self):
        """Genes extracted from DataFrame get evidence_stratum from CSV columns."""
        df = pd.DataFrame({
            "id": ["TNF", "IL6"],
            "causal_linkage_tier": [
                "Tier 1 — Full Causal Chain (genetic variant → expression → disease)",
                "Tier 2 — some other",
            ],
            "has_disease_link": ["Yes", "Yes"],
            "has_pathway_link": ["Yes", "No"],
            "mr_pval": [1e-15, 0.5],
        })
        genes = _build_genes_from_dataframe(df)
        gene_map = {g.gene_symbol: g for g in genes}
        assert gene_map["TNF"].evidence_stratum == "known_driver"
        assert gene_map["TNF"].causal_tier.startswith("Tier 1")

    def test_stratum_none_when_no_columns(self):
        """When no stratum columns exist, evidence_stratum remains None."""
        df = pd.DataFrame({"gene_symbol": ["TNF"], "padj": [0.01]})
        genes = _build_genes_from_dataframe(df)
        assert genes[0].evidence_stratum is None
        assert genes[0].causal_tier is None


# ---------- Gene Rescue ----------
from agentic_ai_wf.drug_agent_streamlit.llm_query_mapper import (
    _rescue_genes_from_query,
    _clean_disease_name,
)


class TestRescueGenesFromQuery:
    """Tests for case-insensitive gene rescue from query text."""

    def test_lowercase_gene_extracted(self):
        """'erbb2' in query is rescued as 'ERBB2'."""
        genes = _rescue_genes_from_query(
            "can you recommend drugs for breast cancer for erbb2",
            "breast cancer",
        )
        symbols = [g["gene_symbol"] for g in genes]
        assert "ERBB2" in symbols

    def test_mixed_case_gene(self):
        """'Brca1' is rescued as 'BRCA1'."""
        genes = _rescue_genes_from_query(
            "recommend drugs for breast cancer targeting Brca1",
            "breast cancer",
        )
        symbols = [g["gene_symbol"] for g in genes]
        assert "BRCA1" in symbols

    def test_multiple_genes_rescued(self):
        """Multiple gene symbols are all rescued."""
        genes = _rescue_genes_from_query(
            "drugs for lung cancer with egfr and alk",
            "lung cancer",
        )
        symbols = {g["gene_symbol"] for g in genes}
        assert "EGFR" in symbols
        assert "ALK" in symbols

    def test_disease_words_not_rescued(self):
        """Common English words and disease terms are excluded."""
        genes = _rescue_genes_from_query(
            "can you recommend drugs for breast cancer",
            "breast cancer",
        )
        symbols = {g["gene_symbol"] for g in genes}
        assert "CAN" not in symbols
        assert "DRUGS" not in symbols

    def test_no_genes_returns_empty(self):
        """Query with no gene symbols returns empty list."""
        genes = _rescue_genes_from_query(
            "recommend drugs for diabetes",
            "diabetes",
        )
        assert genes == []

    def test_uppercase_gene_still_works(self):
        """Uppercase gene symbol is still rescued correctly."""
        genes = _rescue_genes_from_query(
            "drugs for SLE targeting TNFSF13B",
            "SLE",
        )
        symbols = [g["gene_symbol"] for g in genes]
        assert "TNFSF13B" in symbols


class TestCleanDiseaseName:
    """Tests for stripping gene symbols from disease names."""

    def test_strip_trailing_gene(self):
        """'breast cancer for erbb2' → 'breast cancer'."""
        result = _clean_disease_name(
            "breast cancer for erbb2",
            [{"gene_symbol": "ERBB2"}],
        )
        assert result == "breast cancer"

    def test_strip_targeting_gene(self):
        """'lung cancer targeting EGFR' → 'lung cancer'."""
        result = _clean_disease_name(
            "lung cancer targeting EGFR",
            [{"gene_symbol": "EGFR"}],
        )
        assert result == "lung cancer"

    def test_no_genes_unchanged(self):
        """Disease name unchanged when no genes provided."""
        result = _clean_disease_name("breast cancer", [])
        assert result == "breast cancer"

    def test_unrelated_gene_unchanged(self):
        """Disease name unchanged when gene not in the name."""
        result = _clean_disease_name(
            "breast cancer",
            [{"gene_symbol": "TP53"}],
        )
        assert result == "breast cancer"

    def test_with_suffix_pattern(self):
        """'crohn disease with NOD2' → 'crohn disease'."""
        result = _clean_disease_name(
            "crohn disease with NOD2",
            [{"gene_symbol": "NOD2"}],
        )
        assert result == "crohn disease"

    def test_strip_query_boilerplate_and_file_noise(self):
        """Request scaffolding should not be treated as the disease name."""
        result = _clean_disease_name(
            "drug candidates for lupus for attached file and",
            [],
            "drug candidates for lupus for attached file and",
        )
        assert result == "lupus"

    def test_strip_uploaded_file_suffix(self):
        """Trailing file references should be removed from disease strings."""
        result = _clean_disease_name(
            "crohn disease using uploaded file",
            [],
            "recommend drugs for crohn disease using uploaded file",
        )
        assert result == "crohn disease"
