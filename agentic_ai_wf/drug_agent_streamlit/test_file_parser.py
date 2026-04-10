"""Tests for file_parser.py — all Phase 1 formats + edge cases."""

import io
import json
import textwrap

import pandas as pd
import pytest

from agentic_ai_wf.drug_agent_streamlit.file_parser import (
    SUPPORTED_EXTENSIONS,
    FileSummary,
    parse_uploaded_file,
    _find_gene_column,
    _find_direction_column,
    _looks_like_gene_column,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

class FakeUpload:
    """Mimics streamlit.runtime.uploaded_file_manager.UploadedFile."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self.size = len(content)
        self._buf = io.BytesIO(content)

    def read(self):
        self._buf.seek(0)
        return self._buf.read()

    def seek(self, pos):
        self._buf.seek(pos)


# ─── CSV ─────────────────────────────────────────────────────────────────────

class TestCSV:
    def test_basic_csv(self):
        data = "gene_symbol,log2FoldChange,padj\nBRCA1,2.5,0.001\nTP53,-1.2,0.03\n"
        fs = parse_uploaded_file(FakeUpload("deg.csv", data.encode()))
        assert fs.data_type == "tabular"
        assert fs.row_count == 2
        assert "gene_symbol" in fs.columns
        assert fs.has_data
        assert "BRCA1" in fs.sample_genes

    def test_semicolon_delimited(self):
        data = "gene;logfc;pval\nMYC;1.8;0.01\nEGFR;-0.9;0.04\n"
        fs = parse_uploaded_file(FakeUpload("data.csv", data.encode()))
        assert fs.data_type == "tabular"
        assert fs.row_count == 2
        assert "gene" in fs.columns

    def test_empty_csv(self):
        fs = parse_uploaded_file(FakeUpload("empty.csv", b"col1,col2\n"))
        assert fs.data_type == "tabular"
        assert fs.row_count == 0


# ─── TSV ─────────────────────────────────────────────────────────────────────

class TestTSV:
    def test_basic_tsv(self):
        data = "Symbol\tlog2FC\tFDR\nJAK2\t3.1\t0.001\nSTAT3\t-2.0\t0.005\n"
        fs = parse_uploaded_file(FakeUpload("genes.tsv", data.encode()))
        assert fs.data_type == "tabular"
        assert fs.row_count == 2
        assert "JAK2" in fs.sample_genes


# ─── XLSX ────────────────────────────────────────────────────────────────────

class TestXLSX:
    def test_basic_xlsx(self):
        buf = io.BytesIO()
        df = pd.DataFrame({"gene_symbol": ["TNF", "IL6"], "log2fc": [1.5, -0.8]})
        df.to_excel(buf, index=False, engine="openpyxl")
        fs = parse_uploaded_file(FakeUpload("data.xlsx", buf.getvalue()))
        assert fs.data_type == "tabular"
        assert fs.row_count == 2
        assert "TNF" in fs.sample_genes


# ─── TXT ─────────────────────────────────────────────────────────────────────

class TestTXT:
    def test_tabular_txt(self):
        data = "gene\tvalue\nBRCA2\t1.1\nKRAS\t2.3\n"
        fs = parse_uploaded_file(FakeUpload("results.txt", data.encode()))
        assert fs.data_type == "tabular"
        assert fs.row_count == 2

    def test_plain_text(self):
        data = "This is just a plain text file with no structure at all.\nLine 2\n"
        fs = parse_uploaded_file(FakeUpload("notes.txt", data.encode()))
        assert fs.data_type == "plain_text"


# ─── JSON ────────────────────────────────────────────────────────────────────

class TestJSON:
    def test_json_array(self):
        data = json.dumps([
            {"gene_symbol": "TP53", "log2fc": -1.5},
            {"gene_symbol": "MYC", "log2fc": 2.0},
        ])
        fs = parse_uploaded_file(FakeUpload("data.json", data.encode()))
        assert fs.data_type == "json_array"
        assert fs.row_count == 2
        assert fs.dataframe is not None

    def test_json_object(self):
        data = json.dumps({"disease": "lung cancer", "genes": ["EGFR", "KRAS"]})
        fs = parse_uploaded_file(FakeUpload("meta.json", data.encode()))
        assert fs.data_type == "json_object"
        assert "disease" in fs.columns

    def test_invalid_json(self):
        fs = parse_uploaded_file(FakeUpload("bad.json", b"{broken json"))
        assert fs.error is not None


# ─── Unsupported / Edge cases ────────────────────────────────────────────────

class TestEdgeCases:
    def test_unsupported_extension(self):
        fs = parse_uploaded_file(FakeUpload("image.png", b"\x89PNG"))
        assert fs.data_type == "unsupported"
        assert not fs.has_data
        assert ".png" in fs.error

    def test_supported_extensions_registered(self):
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".tsv" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".xls" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".json" in SUPPORTED_EXTENSIONS

    def test_gene_detection_case_insensitive_columns(self):
        data = "Gene_Name,logFC\nBRCA1,1.5\nTP53,-2.0\n"
        fs = parse_uploaded_file(FakeUpload("caps.csv", data.encode()))
        assert len(fs.sample_genes) > 0

    def test_large_gene_list_truncated(self):
        genes = [f"GENE{i}" for i in range(700)]
        rows = "\n".join(f"{g},0.5,0.01" for g in genes)
        data = f"gene_symbol,log2fc,padj\n{rows}\n"
        fs = parse_uploaded_file(FakeUpload("big.csv", data.encode()))
        assert len(fs.sample_genes) <= 1000


# ─── Smart gene column detection ────────────────────────────────────────────

class TestSmartGeneDetection:
    def test_id_column_with_gene_symbols(self):
        """CSV like sle_dag_causal_linkage.csv where gene symbols are in an 'id' column."""
        data = "id,expression_trend,expression_log2fc,mr_pval\n"
        data += "\n".join(
            f"{g},Up,0.0,0.001" for g in ["HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK", "ITGAX"]
        )
        fs = parse_uploaded_file(FakeUpload("genes.csv", data.encode()))
        assert len(fs.sample_genes) == 6
        assert "HLA-C" in fs.sample_genes

    def test_id_column_with_numeric_ids_rejected(self):
        """Numeric 'id' columns should NOT be treated as gene columns."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "value": [0.1, 0.2, 0.3, 0.4, 0.5]})
        assert _find_gene_column(df) is None

    def test_first_column_fallback(self):
        """When no column name matches, first column with gene-like content wins."""
        df = pd.DataFrame({
            "probe": ["BRCA1", "TP53", "MYC", "EGFR", "JAK2"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        assert _find_gene_column(df) == "probe"

    def test_first_column_rejected_non_genes(self):
        df = pd.DataFrame({
            "sample": ["sample_001", "sample_002", "sample_003", "sample_004"],
            "value": [1, 2, 3, 4],
        })
        assert _find_gene_column(df) is None

    def test_looks_like_gene_column(self):
        assert _looks_like_gene_column(pd.Series(["BRCA1", "TP53", "MYC", "EGFR"]))
        assert not _looks_like_gene_column(pd.Series([1, 2, 3, 4]))
        assert not _looks_like_gene_column(pd.Series(["hello world", "foo bar"]))

    def test_direction_column_detection(self):
        df = pd.DataFrame({
            "id": ["BRCA1", "TP53"],
            "expression_trend": ["Up", "Down"],
        })
        assert _find_direction_column(df) == "expression_trend"

    def test_no_direction_column(self):
        df = pd.DataFrame({"gene": ["BRCA1", "TP53"], "log2fc": [1.0, -1.0]})
        assert _find_direction_column(df) is None

    def test_explicit_gene_col_takes_priority(self):
        """gene_symbol column with valid gene content should win over 'id' column."""
        df = pd.DataFrame({
            "id": ["BRCA1", "TP53", "MYC"],
            "gene_symbol": ["KRAS", "BRAF", "NRAS"],
        })
        assert _find_gene_column(df) == "gene_symbol"

    def test_preview_wide_file_stays_readable(self):
        """Wide files (>15 cols) should prioritise informative columns in preview."""
        cols = {f"col_{i}": range(5) for i in range(20)}
        cols["gene_symbol"] = ["BRCA1", "TP53", "MYC", "JAK2", "EGFR"]
        cols["log2fc"] = [1.0, -1.5, 2.0, 0.8, -0.3]
        df = pd.DataFrame(cols)
        from agentic_ai_wf.drug_agent_streamlit.file_parser import _build_preview
        preview = _build_preview(df)
        assert "gene_symbol" in preview
        assert "log2fc" in preview

    def test_paracrine_targets_skipped(self):
        """Regression: 'paracrine_targets' column must NOT match as gene column.

        The SLE CSV has genes in 'id' and a mostly-empty 'paracrine_targets' column.
        The old regex matched 'target' inside 'paracrine_targets' without content validation.
        """
        df = pd.DataFrame({
            "id": ["HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK", "ITGAX"],
            "expression_trend": ["Up", "Up", "Up", "Down", "Down", "Up"],
            "paracrine_targets": ["", "RIPK2", "", "", "", ""],
            "has_disease_link": ["yes", "yes", "yes", "no", "yes", "no"],
        })
        assert _find_gene_column(df) == "id"

    def test_paracrine_targets_integration(self):
        """Full integration: parse_uploaded_file extracts genes from 'id', not 'paracrine_targets'."""
        data = "id,expression_trend,paracrine_targets,mr_pval\n"
        data += "\n".join(
            f"{g},Up,,0.001" for g in ["HLA-C", "RIPK2", "TYK2", "IL12RB2", "BLK"]
        )
        fs = parse_uploaded_file(FakeUpload("sle.csv", data.encode()))
        assert len(fs.sample_genes) == 5
        assert "HLA-C" in fs.sample_genes
