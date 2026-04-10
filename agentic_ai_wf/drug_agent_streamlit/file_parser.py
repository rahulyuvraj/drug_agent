"""
Universal scientific file parser with plugin-pattern extensibility.

Phase 1 formats: CSV, TSV, XLSX, XLS, TXT (tabular), JSON.
To add a new format (e.g. h5ad, FASTA, VCF):
    1. Write a function: def _parse_h5ad(uploaded, raw: bytes) -> FileSummary
    2. Register it:      _PARSERS[".h5ad"] = _parse_h5ad
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable

import pandas as pd


# ─── Standardized output ────────────────────────────────────────────────────

@dataclass
class FileSummary:
    raw_preview: str = ""
    data_type: str = "unknown"
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    sample_genes: List[str] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None
    error: Optional[str] = None

    @property
    def has_data(self) -> bool:
        return self.data_type not in ("unknown", "unsupported") and self.error is None


# ─── Gene-symbol heuristic ──────────────────────────────────────────────────

# Column names strongly associated with gene symbols (word-bounded to avoid substring matches)
_GENE_COL_PATTERN = re.compile(
    r"\b(?:gene|symbol|gene_symbol|gene_name|geneid|hgnc|target)\b", re.IGNORECASE
)

# Gene symbols are uppercase alphanumeric, 1-20 chars, may include hyphens/dots (HLA-DRB1, TP53, IL12RB2)
_GENE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9]+([-./][A-Z0-9]+)*$")

# Direction-like columns with "Up"/"Down" categorical values
_DIRECTION_COL_PATTERN = re.compile(
    r"expression_trend|direction|regulation|trend|fold_direction", re.IGNORECASE
)


def _looks_like_gene_column(series: pd.Series, threshold: float = 0.4) -> bool:
    """Content-based validation: ≥threshold fraction of values match gene-symbol patterns."""
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals.str.len().between(1, 20)]
    if len(vals) < 2:
        return False
    matches = vals.apply(lambda v: bool(_GENE_SYMBOL_RE.match(v)))
    return (matches.sum() / len(vals)) >= threshold


def _find_gene_column(df: pd.DataFrame) -> Optional[str]:
    """Smart gene column detection: name-match + content-validation, then fallback."""
    # Pass 1: explicit gene-related column names — require content validation to avoid
    # false positives (e.g. "paracrine_targets" matching on bare "target" substring)
    for col in df.columns:
        if _GENE_COL_PATTERN.search(str(col)):
            if _looks_like_gene_column(df[col]):
                return col

    # Pass 2: columns named "id", "name", "identifier" — accept only if content looks like genes
    for col in df.columns:
        if str(col).lower().strip() in ("id", "name", "identifier"):
            if _looks_like_gene_column(df[col]):
                return col

    # Pass 3: first column content check (most gene-centric files put symbols in col 0)
    if len(df.columns) > 0 and _looks_like_gene_column(df.iloc[:, 0]):
        return df.columns[0]

    return None


def _find_direction_column(df: pd.DataFrame) -> Optional[str]:
    """Detect columns carrying Up/Down expression direction."""
    for col in df.columns:
        if _DIRECTION_COL_PATTERN.search(str(col)):
            vals = df[col].dropna().astype(str).str.lower().unique()
            if {"up", "down"} & set(vals):
                return col
    return None


def _extract_sample_genes(df: pd.DataFrame, max_genes: int = 1000) -> List[str]:
    """Auto-detect gene symbol column and return unique values."""
    gene_col = _find_gene_column(df)
    if not gene_col:
        return []
    genes = (
        df[gene_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.len().between(1, 20)]
        .unique()
        .tolist()
    )
    return genes[:max_genes]


def _build_preview(df: pd.DataFrame, max_chars: int = 5000) -> str:
    """Compact textual summary optimised for LLM consumption of wide scientific files."""
    header = f"Columns ({len(df.columns)}): {', '.join(df.columns[:60])}"
    # For wide files, show only the most informative columns in the sample
    display_df = df
    if len(df.columns) > 15:
        priority = re.compile(
            r"gene|symbol|id|log2|lfc|padj|pval|fdr|direction|trend|disease|pathway|tier",
            re.IGNORECASE,
        )
        keep = [c for c in df.columns if priority.search(str(c))]
        if len(keep) < 3:
            keep = list(df.columns[:8])
        display_df = df[keep]
    sample = display_df.head(25).to_string(index=False, max_colwidth=40)
    preview = f"{header}\n\n{sample}"
    return preview[:max_chars]


# ─── Individual parsers ─────────────────────────────────────────────────────

def _parse_csv(uploaded, raw: bytes) -> FileSummary:
    text_head = raw[:8192].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text_head, delimiters=",\t;|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","

    df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
    return FileSummary(
        raw_preview=_build_preview(df),
        data_type="tabular",
        columns=list(df.columns),
        row_count=len(df),
        sample_genes=_extract_sample_genes(df),
        dataframe=df,
    )


def _parse_tsv(uploaded, raw: bytes) -> FileSummary:
    df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
    return FileSummary(
        raw_preview=_build_preview(df),
        data_type="tabular",
        columns=list(df.columns),
        row_count=len(df),
        sample_genes=_extract_sample_genes(df),
        dataframe=df,
    )


def _parse_excel(uploaded, raw: bytes) -> FileSummary:
    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    return FileSummary(
        raw_preview=_build_preview(df),
        data_type="tabular",
        columns=list(df.columns),
        row_count=len(df),
        sample_genes=_extract_sample_genes(df),
        dataframe=df,
    )


def _parse_xls(uploaded, raw: bytes) -> FileSummary:
    df = pd.read_excel(io.BytesIO(raw), engine="xlrd")
    return FileSummary(
        raw_preview=_build_preview(df),
        data_type="tabular",
        columns=list(df.columns),
        row_count=len(df),
        sample_genes=_extract_sample_genes(df),
        dataframe=df,
    )


def _parse_txt(uploaded, raw: bytes) -> FileSummary:
    """TXT files: attempt tabular parse with delimiter sniffing, fall back to plain text."""
    text_head = raw[:8192].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text_head, delimiters=",\t;|")
        sep = dialect.delimiter
        df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
        if len(df.columns) > 1:
            return FileSummary(
                raw_preview=_build_preview(df),
                data_type="tabular",
                columns=list(df.columns),
                row_count=len(df),
                sample_genes=_extract_sample_genes(df),
                dataframe=df,
            )
    except Exception:
        pass

    text = raw.decode("utf-8", errors="replace")
    return FileSummary(
        raw_preview=text[:3000],
        data_type="plain_text",
        columns=[],
        row_count=text.count("\n"),
    )


def _parse_json(uploaded, raw: bytes) -> FileSummary:
    data = json.loads(raw.decode("utf-8", errors="replace"))

    if isinstance(data, list) and data and isinstance(data[0], dict):
        df = pd.json_normalize(data)
        return FileSummary(
            raw_preview=_build_preview(df),
            data_type="json_array",
            columns=list(df.columns),
            row_count=len(df),
            sample_genes=_extract_sample_genes(df),
            dataframe=df,
        )

    if isinstance(data, dict):
        preview = json.dumps(data, indent=2, default=str)[:3000]
        flat_keys = list(data.keys())
        return FileSummary(
            raw_preview=preview,
            data_type="json_object",
            columns=flat_keys,
            row_count=1,
        )

    return FileSummary(raw_preview=str(data)[:3000], data_type="json_other")


# ─── Parser registry ────────────────────────────────────────────────────────

_PARSERS: Dict[str, Callable] = {
    ".csv": _parse_csv,
    ".tsv": _parse_tsv,
    ".xlsx": _parse_excel,
    ".xls": _parse_xls,
    ".txt": _parse_txt,
    ".json": _parse_json,
}

SUPPORTED_EXTENSIONS = list(_PARSERS.keys())


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> FileSummary:
    """Parse a Streamlit UploadedFile into a FileSummary.

    Args:
        uploaded_file: A streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns:
        FileSummary with parsed data or error details.
    """
    name = getattr(uploaded_file, "name", "unknown")
    ext = Path(name).suffix.lower()

    parser = _PARSERS.get(ext)
    if parser is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return FileSummary(
            data_type="unsupported",
            error=f"Unsupported format '{ext}'. Supported: {supported}",
        )

    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        return parser(uploaded_file, raw)
    except Exception as e:
        return FileSummary(
            data_type="error",
            error=f"Failed to parse {name}: {e}",
        )
