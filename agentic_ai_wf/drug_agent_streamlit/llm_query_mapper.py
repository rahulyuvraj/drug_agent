"""
Claude-powered query classification + structured file→DrugQueryRequest mapping.

Two-step LLM pipeline:
  1. classify_query  → Is this a drug discovery question?
  2. map_query_and_file → Extract disease, genes, pathways, biomarkers → DrugQueryRequest
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from agentic_ai_wf.drug_agent.service.schemas import (
    BiomarkerContext,
    DrugQueryRequest,
    GeneContext,
    PathwayContext,
    QueryType,
    ScoringConfig,
    TMEContext,
    _GENE_TO_PROTEINS,
    _PROTEIN_TO_GENE,
)
from agentic_ai_wf.reporting_pipeline_agent.llm_factory import BedrockLLMClient
from agentic_ai_wf.drug_agent_streamlit.file_parser import (
    FileSummary, _find_gene_column, _find_direction_column, _looks_like_gene_column,
)

logger = logging.getLogger(__name__)

# ─── .env credential loading ────────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / "drug_agent" / ".env"
load_dotenv(_ENV_PATH)


def _get_llm_client() -> BedrockLLMClient:
    """Build BedrockLLMClient from env vars (cached at Streamlit level via @st.cache_resource)."""
    return BedrockLLMClient(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
    )


# ─── Robust JSON parser (adapted from core_report.py llm_json pattern) ─────

def _try_parse_json(text: str) -> dict:
    if not text:
        return {}
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    return {}


def _llm_call(client: BedrockLLMClient, system: str, user: str,
              max_tokens: int = 4096) -> str:
    """Single Claude call → raw text."""
    resp = client.chat.completions.create(
        model=client.model_id,
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _llm_json(client: BedrockLLMClient, system: str, user: str,
              max_tokens: int = 4096, repair_retries: int = 2) -> dict:
    """Claude call → parsed JSON with retry + repair. Never raises."""
    raw = ""
    try:
        raw = _llm_call(client, system, user + "\n\nReturn ONLY valid JSON.", max_tokens)
        parsed = _try_parse_json(raw)
        if parsed:
            return parsed
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")

    for attempt in range(repair_retries):
        try:
            repair_prompt = (
                "The following text was intended to be a JSON object but is invalid.\n"
                "Return ONLY a corrected JSON object — no explanation.\n\n"
                f"INVALID TEXT:\n{raw[:8000]}"
            )
            raw = _llm_call(client, "Fix the JSON. Return ONLY valid JSON.", repair_prompt, max_tokens)
            parsed = _try_parse_json(raw)
            if parsed:
                return parsed
        except Exception:
            pass

    logger.error("LLM JSON extraction failed after retries")
    return {}


# ─── Step 1: Query classification ───────────────────────────────────────────

_CLASSIFY_SYSTEM = (
    "You are a query classifier for a drug discovery platform.\n"
    "Determine if the user's question is about drug discovery — meaning drug recommendations, "
    "drug validation, drug safety, or finding drugs for a disease/genes.\n"
    "Return ONLY the single word DRUG or OTHER. Nothing else."
)


def classify_query(client: BedrockLLMClient, user_query: str) -> str:
    """Returns 'DRUG' or 'OTHER'."""
    try:
        raw = _llm_call(client, _CLASSIFY_SYSTEM, user_query, max_tokens=10)
        token = raw.strip().upper().split()[0] if raw.strip() else "OTHER"
        return "DRUG" if "DRUG" in token else "OTHER"
    except Exception:
        return "DRUG"  # fail-open: let the mapper handle it


# ─── Step 2: Structured extraction ──────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are a drug discovery query mapper. Given a user's question and optionally a summary of their uploaded data file, extract structured parameters for a drug recommendation engine.

RESPOND WITH VALID JSON ONLY — no markdown, no code fences, no explanation, no commentary.

OUTPUT SCHEMA:
{
  "disease": "string — the disease/condition to find drugs for (REQUIRED)",
  "query_type": "full_recommendation | validate_drug",
  "drug_name": "string or null — required if query_type is validate_drug",
  "disease_aliases": ["list of alternative names for the disease"],
  "genes": [
    {
      "gene_symbol": "HGNC gene symbol (e.g. BRCA1)",
      "log2fc": 0.0,
      "adj_p_value": 0.05,
      "direction": "up or down",
      "role": "pathogenic | protective | therapeutic_target | immune_modulator | null"
    }
  ],
  "pathways": [
    {
      "pathway_name": "pathway or gene set name",
      "direction": "up or down",
      "fdr": 0.05,
      "gene_count": 10
    }
  ],
  "biomarkers": [
    {
      "biomarker_name": "name",
      "status": "positive | negative | not_assessed"
    }
  ],
  "max_results": 15
}

IMPORTANT — GENE EXTRACTION RULES:
- When the user uploads a data FILE: do NOT try to list all genes. Gene extraction from files is handled programmatically. Only extract genes the user EXPLICITLY mentions by name in their query text.
- When the user types gene names WITHOUT a file: extract those genes into the genes array.
- Always extract the disease, query_type, pathways, biomarkers, and disease_aliases from the query.
- When the user mentions pathways by name in their query text (e.g. "JAK-STAT", "NF-kB", "PI3K-AKT-mTOR", "MAPK"), extract them into the pathways array with fdr: 0.01 and direction: "up".
- If the user mentions specific pathways from the file, include them as well.

QUERY TYPE DETECTION:
- If the user asks to "validate", "check", or "assess" a specific drug → query_type = "validate_drug" and extract drug_name
- Otherwise → query_type = "full_recommendation"
"""


def map_query_and_file(
    client: BedrockLLMClient,
    user_query: str,
    file_summary: Optional[FileSummary] = None,
    max_results: int = 15,
    discovery_genes: int = 500,
) -> tuple[DrugQueryRequest, str, dict]:
    """Map natural language + file into a DrugQueryRequest.

    Returns:
        (request, raw_llm_text, parsed_json) — the last two for debugging display.
    """
    user_content = f"USER QUERY: {user_query}\n"
    if file_summary and file_summary.has_data:
        user_content += f"\nUPLOADED FILE SUMMARY:\n{file_summary.raw_preview}\n"
        if file_summary.columns:
            user_content += f"\nCOLUMN NAMES: {', '.join(file_summary.columns)}\n"
        if file_summary.sample_genes:
            total = len(file_summary.sample_genes)
            shown = file_summary.sample_genes[:100]
            user_content += (
                f"\nGENES DETECTED ({total} total, extracted programmatically): "
                f"{', '.join(shown)}\n"
                f"NOTE: All {total} genes will be used automatically. "
                f"Do NOT re-extract genes from the file.\n"
            )
        user_content += f"\nROW COUNT: {file_summary.row_count}\n"

    raw = ""
    try:
        raw = _llm_call(client, _EXTRACTION_SYSTEM, user_content, max_tokens=4096)
    except Exception as e:
        logger.error(f"Extraction LLM call failed: {e}")

    parsed = _try_parse_json(raw)
    if not parsed:
        parsed = _llm_json(client, _EXTRACTION_SYSTEM, user_content)

    # Last resort: regex fallback when LLM completely fails (e.g. API down, malformed response)
    if not parsed.get("disease") or parsed.get("disease", "").lower() == "unknown":
        fallback = _fallback_extract(user_query)
        if not parsed:
            parsed = fallback
        else:
            # Merge: prefer LLM values when present, fill gaps from fallback
            if parsed.get("disease", "unknown").lower() == "unknown" and fallback["disease"] != "unknown":
                parsed["disease"] = fallback["disease"]
            if not parsed.get("genes") and fallback.get("genes"):
                parsed["genes"] = fallback["genes"]

    # Gene rescue: if LLM returned a disease but missed genes, scan query for gene symbols
    # Skip for validate_drug — the LLM correctly returns zero genes; rescue would inject drug name as gene
    if (parsed and parsed.get("disease") and not parsed.get("genes")
            and parsed.get("query_type") != "validate_drug"):
        rescued = _rescue_genes_from_query(user_query, parsed["disease"])
        if rescued:
            parsed["genes"] = rescued
            logger.info(f"Gene rescue extracted {len(rescued)} gene(s) from query text")

    # Pathway rescue: scan query text for known pathway class names
    if parsed and not parsed.get("pathways") and parsed.get("query_type") != "validate_drug":
        rescued_pw = _rescue_pathways_from_query(user_query)
        if rescued_pw:
            parsed["pathways"] = rescued_pw
            logger.info(f"Pathway rescue extracted {len(rescued_pw)} pathway(s) from query text")

    # Disease cleanup: strip gene symbols that LLM folded into disease name
    if parsed and parsed.get("disease"):
        parsed["disease"] = _clean_disease_name(parsed["disease"], parsed.get("genes", []), user_query)

    # Parse user intent for gene count limiting (e.g. "top 10 genes", "most significant 50")
    top_n = _extract_top_n(user_query)

    request = _build_request(parsed, max_results, file_summary, top_n, discovery_genes, user_query)
    return request, raw, parsed


def _extract_top_n(query: str) -> Optional[int]:
    """Regex-based extraction of explicit gene count from user query."""
    m = re.search(r"(?:top|first|most\s+significant|highest)\s+(\d+)", query, re.IGNORECASE)
    return int(m.group(1)) if m else None


# Gene symbol pattern (imported regex from file_parser for consistency)
_GENE_RE = re.compile(r"^[A-Z][A-Z0-9]+([-./][A-Z0-9]+)*$")

# Common English words and query verbs to exclude from gene rescue
_GENE_STOP = {
    "THE", "AND", "FOR", "WITH", "FROM", "USING", "ALL", "THIS", "THAT",
    "FIND", "GIVE", "DRUGS", "DRUG", "RECOMMEND", "RUN", "FILE", "SLE",
    "TARGETING", "TARGETED", "MOST", "SIGNIFICANT", "GENES", "GENE",
    "CAN", "YOU", "PLEASE", "CANCER", "DISEASE", "TREATMENT", "THERAPY",
    "TOP", "BEST", "NEW", "USE", "SHOW", "LIST", "GET", "HELP",
    "VALIDATE", "CHECK", "ASSESS", "EVALUATE", "ANALYZE", "COMPARE",
    "REVIEW", "INVESTIGATE", "REPURPOSE", "CONFIRM", "EXPLORE",
}

# Drug name suffixes — tokens ending with these are drug names, not gene symbols
_DRUG_SUFFIXES = (
    "mab", "nib", "lib", "zumab", "tinib", "ciclib", "lisib",
    "parin", "statin", "sartan", "prazole", "olol", "vir",
    "taxel", "platin", "rubicin", "mide", "sertib", "rafenib",
)


def _normalize_pathway_name(name: str) -> str:
    """Map user/LLM pathway names to ScoringConfig class names when possible."""
    cfg = ScoringConfig()
    # Exact match
    if name in cfg.pathway_drug_class_map:
        return name
    name_lower = name.lower()
    name_words = set(re.findall(r"[a-z0-9]{2,}", name_lower))
    # Case-insensitive exact match
    for cls_name in cfg.pathway_drug_class_map:
        if cls_name.lower() == name_lower:
            return cls_name
    # Bidirectional keyword overlap
    for cls_name in cfg.pathway_drug_class_map:
        cls_keywords = set(re.findall(r"[a-z0-9]{3,}", cls_name.lower()))
        non_trivial = {w for w in name_words if len(w) >= 3 and w not in {"for", "the", "and", "drugs", "pathway"}}
        if non_trivial and non_trivial <= cls_keywords:
            return cls_name
    # Alias lookup (shared with _rescue_pathways_from_query)
    for alias, cls_name in _PATHWAY_CLASS_ALIASES.items():
        if alias in name_lower and cls_name in cfg.pathway_drug_class_map:
            return cls_name
    return name


# Shared alias table for pathway normalization and rescue
_PATHWAY_CLASS_ALIASES = {
    "cell cycle": "CDK-cell-cycle", "cdk": "CDK-cell-cycle",
    "mtor": "PI3K-AKT-mTOR", "pi3k": "PI3K-AKT-mTOR", "akt": "PI3K-AKT-mTOR",
    "jak stat": "JAK-STAT", "jak-stat": "JAK-STAT",
    "nfkb": "NF-kB", "nf kappa b": "NF-kB", "nf-kb": "NF-kB",
    "wnt": "Wnt-beta-catenin", "beta catenin": "Wnt-beta-catenin",
    "pd-1": "PD1-PDL1", "pdl1": "PD1-PDL1", "pd-l1": "PD1-PDL1", "pd1": "PD1-PDL1",
    "her2": "HER2-ERBB", "erbb": "HER2-ERBB",
    "mapk": "MAPK-ERK", "erk": "MAPK-ERK", "ras": "MAPK-ERK",
    "bcl2": "BCL2-apoptosis", "bcl-2": "BCL2-apoptosis", "apoptosis": "BCL2-apoptosis",
    "tnf": "TNF", "tnf alpha": "TNF",
    "complement": "Complement",
    "hedgehog": "Hedgehog", "shh": "Hedgehog",
    "notch": "Notch",
    "tgf beta": "TGF-beta", "tgfb": "TGF-beta",
    "vegf": "VEGF",
    "egfr": "EGFR",
    "bcr-abl": "BCR-ABL", "bcr abl": "BCR-ABL",
    "btk": "BTK",
    "parp": "PARP",
    "il6": "IL6", "il-6": "IL6",
    "il17": "IL17", "il-17": "IL17",
    "il23": "IL23", "il-23": "IL23",
    "il4": "IL4-IL13", "il13": "IL4-IL13", "il-4": "IL4-IL13", "il-13": "IL4-IL13",
    "ctla4": "CTLA4", "ctla-4": "CTLA4",
}


def _rescue_pathways_from_query(query: str) -> list:
    """Scan query text for known pathway class names from ScoringConfig.

    Uses three strategies: (1) direct substring match of class name variants,
    (2) bidirectional keyword overlap, (3) common alias lookup.
    """
    cfg = ScoringConfig()
    query_lower = query.lower()
    query_words = set(re.findall(r"[a-z0-9]{2,}", query_lower))
    found = []
    seen: set = set()

    def _add(cls_name: str):
        if cls_name not in seen:
            seen.add(cls_name)
            found.append({"pathway_name": cls_name, "direction": "up", "fdr": 0.01, "gene_count": 0})

    # Strategy 1: direct variant matching (existing logic)
    for cls_name in cfg.pathway_drug_class_map:
        variants = [cls_name.lower(), cls_name.lower().replace("-", " "), cls_name.lower().replace("-", "")]
        for v in variants:
            if v in query_lower:
                _add(cls_name)

    # Strategy 2: bidirectional keyword overlap — user's query words vs class name keywords
    for cls_name in cfg.pathway_drug_class_map:
        if cls_name in seen:
            continue
        cls_keywords = set(re.findall(r"[a-z0-9]{3,}", cls_name.lower()))
        # All meaningful query words (≥3 chars) that also appear in class keywords
        overlap = query_words & cls_keywords
        non_trivial_query = {w for w in query_words if len(w) >= 3 and w not in {"for", "the", "and", "drugs", "pathway"}}
        if non_trivial_query and non_trivial_query <= cls_keywords:
            _add(cls_name)

    # Strategy 3: common alias lookup using shared _PATHWAY_CLASS_ALIASES
    for alias, cls_name in _PATHWAY_CLASS_ALIASES.items():
        if alias in query_lower and cls_name in cfg.pathway_drug_class_map:
            _add(cls_name)

    return found


def _rescue_genes_from_query(query: str, disease: str) -> list:
    """Case-insensitive gene rescue from query text.

    When the LLM identifies a disease but returns zero genes, scan the
    original query for tokens that look like gene symbols (case-insensitive).
    Strips the disease name from the query first to avoid false positives.
    """
    # Remove the disease portion to avoid matching disease words as genes
    clean = re.sub(re.escape(disease), "", query, flags=re.IGNORECASE).strip()
    # Uppercase all tokens and match gene-symbol pattern
    tokens = re.findall(r"\b([A-Za-z][A-Za-z0-9](?:[A-Za-z0-9\-./]*[A-Za-z0-9])?)\b", clean)
    genes = []
    seen = set()
    for t in tokens:
        upper = t.upper()
        if upper in seen or upper in _GENE_STOP or len(upper) < 2:
            continue
        if t.lower().endswith(_DRUG_SUFFIXES):
            continue
        if _GENE_RE.match(upper):
            seen.add(upper)
            genes.append(
                {"gene_symbol": upper, "log2fc": 1.0, "adj_p_value": 0.05, "direction": "up"}
            )
    return genes


def _clean_disease_name(disease: str, genes: list, user_query: str = "") -> str:
    """Strip gene symbols that the LLM folded into the disease name.

    E.g. 'breast cancer for erbb2' → 'breast cancer'
    """
    cleaned = re.sub(r"\s+", " ", disease).strip(" ,.;:-")

    # Remove leading request boilerplate while keeping the disease phrase.
    cleaned = re.sub(
        r"^(?:(?:can\s+you|please)\s+)?(?:(?:recommend|find|show|list|give|suggest|identify|query|look\s+up)\s+)?"
        r"(?:drugs?|drug\s+candidates?|treatments?|therap(?:y|ies)|medications?)\s+for\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    gene_symbols = {g["gene_symbol"].upper() if isinstance(g, dict) else g.upper() for g in genes}
    if gene_symbols:
        cleaned = re.sub(
            r"\s+(?:for|targeting|with)\s+(" + "|".join(re.escape(s) for s in gene_symbols) + r")\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    # Strip trailing file/data scaffolding the LLM sometimes folds into disease.
    cleaned = re.sub(
        r"\s+(?:for|with|from|using|based\s+on)\s+(?:the\s+)?"
        r"(?:(?:attached|uploaded|provided|input)\s+)?"
        r"(?:file|data|dataset|csv|tsv|xlsx|sheet)(?:\s+\w+){0,3}\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    cleaned = re.sub(r"\s+(?:and|or|with|from|using|based\s+on)\s*$", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:-")

    if not cleaned and user_query:
        fallback = _fallback_extract(user_query)
        if fallback.get("disease") and fallback["disease"] != "unknown":
            cleaned = fallback["disease"]

    return cleaned or disease


def _fallback_extract(query: str) -> dict:
    """Regex-only fallback when the LLM call fails entirely.

    Extracts disease from 'for <disease>' patterns, and gene symbols from uppercase tokens.
    Not as accurate as LLM extraction, but ensures text-only queries never produce empty results.
    """
    # Disease: "for <disease> targeting/with/using..." or "for <disease>" at end
    disease = "unknown"
    m = re.search(
        r"\bfor\s+(.+?)(?:\s+(?:targeting|with|using|genes?|from)\b|$)",
        query, re.IGNORECASE,
    )
    if m:
        disease = m.group(1).strip().rstrip(".,;:")

    # Genes: uppercase tokens matching gene-symbol pattern, excluding common English words
    _STOP = {"THE", "AND", "FOR", "WITH", "FROM", "USING", "ALL", "THIS", "THAT",
             "FIND", "GIVE", "DRUGS", "DRUG", "RECOMMEND", "RUN", "FILE", "SLE",
             "TARGETING", "TARGETED", "MOST", "SIGNIFICANT", "GENES", "GENE"}
    tokens = re.findall(r"\b([A-Z][A-Z0-9](?:[A-Z0-9-./]*[A-Z0-9])?)\b", query)
    genes = [
        {"gene_symbol": t, "log2fc": 1.0, "adj_p_value": 0.05, "direction": "up"}
        for t in dict.fromkeys(tokens)  # dedup, preserve order
        if t not in _STOP and _GENE_RE.match(t)
    ]
    return {"disease": disease, "genes": genes, "query_type": "full_recommendation"}


def _build_request(
    parsed: dict,
    max_results: int,
    file_summary: Optional[FileSummary] = None,
    top_n: Optional[int] = None,
    discovery_genes: int = 500,
    user_query: str = "",
) -> DrugQueryRequest:
    """Convert parsed JSON dict into a DrugQueryRequest dataclass.

    Gene extraction priority: DataFrame (deterministic, complete) > LLM (limited by token window).
    The LLM handles disease/intent/pathways; genes come from the file when available.
    """
    disease = parsed.get("disease", "unknown")

    # Query type
    qt_str = parsed.get("query_type", "full_recommendation")
    try:
        query_type = QueryType(qt_str)
    except ValueError:
        query_type = QueryType.FULL_RECOMMENDATION

    # Genes — DataFrame-first: deterministic extraction beats LLM for tabular data
    gene_limit = top_n or min(discovery_genes, 1000)
    df_genes: List[GeneContext] = []
    has_df = file_summary and file_summary.dataframe is not None
    if has_df:
        df_genes = _build_genes_from_dataframe(file_summary.dataframe, max_genes=gene_limit)

    llm_genes = _build_genes_from_parsed(parsed)

    if df_genes:
        # Merge any LLM-only genes (e.g. user typed specific genes in query text)
        df_symbols = {g.gene_symbol for g in df_genes}
        for g in llm_genes:
            if g.gene_symbol not in df_symbols:
                df_genes.append(g)
        all_genes = df_genes
    else:
        all_genes = llm_genes

    # Pathways — DataFrame-extracted (with key_genes) merged with LLM-extracted
    df_pathways: List[PathwayContext] = []
    if has_df:
        df_pathways = _build_pathways_from_dataframe(file_summary.dataframe, disease=disease)

    llm_pathways = [
        PathwayContext(
            pathway_name=_normalize_pathway_name(p.get("pathway_name", "")),
            direction=p.get("direction") or "up",
            fdr=float(p.get("fdr") or 0.05),
            gene_count=int(p.get("gene_count") or 0),
        )
        for p in parsed.get("pathways", [])
        if p.get("pathway_name")
    ]

    # Merge: DataFrame pathways win (they carry key_genes); add LLM-only ones
    pw_names_seen = {p.pathway_name.lower() for p in df_pathways}
    for lp in llm_pathways:
        if lp.pathway_name.lower() not in pw_names_seen:
            df_pathways.append(lp)
            pw_names_seen.add(lp.pathway_name.lower())
    pathways = df_pathways

    # TME
    tme = None
    if has_df:
        tme = _build_tme_from_dataframe(file_summary.dataframe)

    # Disease context — built from LLM-extracted disease name, independent of aliases
    disease_context = _build_disease_context(disease, all_genes, pathways)

    # Biomarkers
    biomarkers = [
        BiomarkerContext(
            biomarker_name=b.get("biomarker_name", ""),
            status=b.get("status", "not_assessed"),
        )
        for b in parsed.get("biomarkers", [])
        if b.get("biomarker_name")
    ]

    # ── Protein-alias rescue: resolve common protein names to HGNC gene symbols ──
    # E.g. "CD20" → MS4A1, "HER2" → ERBB2, "PD-1" → PDCD1
    force_gene_inferred = False
    if disease and not has_df and not all_genes:
        disease_upper = disease.upper().replace("-", "-")  # preserve hyphens for PD-1 etc.
        # Check protein→gene mapping first (CD20→MS4A1)
        resolved_gene = _PROTEIN_TO_GENE.get(disease_upper)
        # Also check if disease itself is a gene symbol with protein aliases
        if not resolved_gene and disease_upper in _GENE_TO_PROTEINS:
            resolved_gene = disease_upper
        # Last resort: looks like a gene symbol pattern?
        if not resolved_gene and _GENE_RE.match(disease_upper) and disease_upper not in _GENE_STOP:
            resolved_gene = disease_upper
        if resolved_gene:
            all_genes = [GeneContext(gene_symbol=resolved_gene, log2fc=1.0, adj_p_value=0.05, direction="up")]
            force_gene_inferred = True
            logger.info(f"Protein-alias rescue: '{disease}' → gene '{resolved_gene}'")

    # Detect gene-inferred disease: disease name matches a gene symbol from the extracted genes
    gene_symbols_upper = {g.gene_symbol.upper() if isinstance(g, GeneContext) else g.get("gene_symbol", "").upper()
                          for g in all_genes}
    is_gene_inferred = force_gene_inferred or bool(
        disease and disease.upper() in gene_symbols_upper
        and not has_df  # file-based queries always have real diseases
    )

    # Target-only detection: query mentions a gene/protein explicitly but disease is LLM-inferred
    # E.g. "find drugs for CD20 gene" → LLM returns disease="B-cell lymphoma", genes=[MS4A1]
    # The disease was NOT in the user's query, so this is target-centric.
    target_only = is_gene_inferred and not has_df
    if not target_only and not has_df and all_genes and disease:
        query_upper = user_query.upper()
        # Check if disease words are absent from query (= LLM-inferred disease)
        disease_words = [w for w in disease.lower().split() if len(w) > 3]
        disease_in_query = any(w in user_query.lower() for w in disease_words)
        if not disease_in_query:
            # Confirm query contains a gene symbol or protein alias
            for g in all_genes:
                sym = g.gene_symbol.upper()
                if sym in query_upper:
                    target_only = True
                    is_gene_inferred = True
                    break
                for alias in _GENE_TO_PROTEINS.get(sym, []):
                    if alias.upper() in query_upper:
                        target_only = True
                        is_gene_inferred = True
                        break
                if target_only:
                    break
            if target_only:
                logger.info(f"Target-only query detected: disease '{disease}' not in query, gene/alias found")

    # Force empty disease_aliases so Stage 0's Qdrant expansion always fires
    # (SOC shield fix: ensures Belimumab/HCQ get recognized as SOC for SLE)
    return DrugQueryRequest(
        disease=disease,
        query_type=query_type,
        genes=all_genes,
        pathways=pathways,
        biomarkers=biomarkers,
        tme=tme,
        drug_name=parsed.get("drug_name"),
        max_results=parsed.get("max_results", max_results),
        disease_aliases=[],
        all_patient_genes=all_genes,
        disease_context=disease_context,
        is_gene_inferred_disease=is_gene_inferred,
        is_target_only_query=target_only,
    )


def _build_genes_from_parsed(parsed: dict) -> List[GeneContext]:
    genes = []
    for g in parsed.get("genes", []):
        symbol = g.get("gene_symbol", "")
        if not symbol:
            continue
        try:
            log2fc = float(g.get("log2fc", 1.0))
        except (ValueError, TypeError):
            log2fc = 1.0
        try:
            pval = float(g.get("adj_p_value", 0.05))
        except (ValueError, TypeError):
            pval = 0.05
        direction = g.get("direction", "up" if log2fc > 0 else "down")
        genes.append(GeneContext(
            gene_symbol=symbol.strip().upper(),
            log2fc=log2fc,
            adj_p_value=pval,
            direction=direction,
            role=g.get("role"),
        ))
    return genes


# Column-name aliases for direct DataFrame mapping
_LOG2FC_ALIASES = {
    "log2foldchange", "logfc", "log2fc", "lfc", "log_fc",
    "expression_log2fc", "log2_fold_change",
}
_PVAL_ALIASES = {
    "padj", "adj_p_value", "fdr", "q_value", "adj.p.val", "p_adj", "qvalue",
    "mr_pval", "min_gwas_pval", "pvalue", "p_value",
}
_GENE_ALIASES = {
    "gene", "symbol", "gene_symbol", "gene.symbol", "gene_name", "geneid",
    "hgnc_symbol", "gene_id", "target_gene", "id",
}
_DIRECTION_ALIASES = {
    "expression_trend", "direction", "regulation", "trend", "fold_direction",
}
_PATHWAY_ALIASES = {
    "pathways", "pathway", "pathway_name", "kegg_pathway", "go_term", "reactome",
}
_CELL_TYPE_ALIASES = {
    "cell_types_active_in", "cell_type", "cell_types", "immune_cells",
}

# Role-relevant column sets (detected dynamically per-file)
_ROLE_COLS = {
    "therapeutic_recommendation", "eqtl_causal_direction", "strategy_type",
    "causal_tier",
}
_STRATUM_COLS = {
    "causal_linkage_tier", "has_disease_link", "has_pathway_link",
    "expression_log2fc", "mr_pval",
}
_COMPOSITE_SCORE_ALIASES = {
    "gene_genetic_confidence_score", "composite_score", "evidence_strength_score",
}

# Pathway category detection — keyword → category label
_PATHWAY_CATEGORIES = [
    (re.compile(r"complement|classical pathway|lectin pathway", re.I), "Immune/Complement"),
    (re.compile(r"interferon|ifn[- ]?[abg]", re.I), "Immune/Interferon"),
    (re.compile(r"MHC|antigen.+presentation|HLA", re.I), "Immune/Antigen Presentation"),
    (re.compile(r"T.?cell|Th1|Th2|Th17|TCR", re.I), "Immune/T Cell"),
    (re.compile(r"B.?cell|immunoglobulin|IgA|antibod", re.I), "Immune/B Cell"),
    (re.compile(r"NF.?kB|TNF|inflamm", re.I), "Inflammatory"),
    (re.compile(r"apoptosi|caspase|programmed cell death", re.I), "Apoptosis"),
    (re.compile(r"signaling|pathway|cascade", re.I), "Signaling"),
]


def _find_column(df_columns: List[str], aliases: set) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in df_columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _detect_columns(df_columns: List[str]) -> Dict[str, Optional[str]]:
    """One-pass detection of all role/stratum/metadata columns present in the DataFrame."""
    lower_map = {c.lower().strip(): c for c in df_columns}
    detected: Dict[str, Optional[str]] = {}
    for alias_set_name, alias_set in [
        ("therapeutic_recommendation", {"therapeutic_recommendation"}),
        ("eqtl_causal_direction", {"eqtl_causal_direction"}),
        ("strategy_type", {"strategy_type"}),
        ("causal_tier", {"causal_tier"}),
        ("causal_linkage_tier", {"causal_linkage_tier"}),
        ("has_disease_link", {"has_disease_link"}),
        ("has_pathway_link", {"has_pathway_link"}),
        ("mr_pval", {"mr_pval"}),
    ]:
        detected[alias_set_name] = _find_column(df_columns, alias_set)
    detected["composite_score"] = _find_column(df_columns, _COMPOSITE_SCORE_ALIASES)
    return detected


def _assign_gene_role(row, direction: str, detected: Dict[str, Optional[str]]) -> Optional[str]:
    """Derive gene role from CSV columns — column-existence-gated, fully dynamic."""
    import pandas as pd

    # Priority 1: explicit therapeutic target
    col = detected.get("therapeutic_recommendation")
    if col and pd.notna(row.get(col)):
        val = str(row[col]).lower()
        if "target" in val and "context" not in val:
            return "therapeutic_target"

    # Priority 2: causal direction
    col = detected.get("eqtl_causal_direction")
    if col and pd.notna(row.get(col)):
        causal = str(row[col]).strip().lower()
        if causal == "protective":
            return "protective"
        if causal == "risk":
            return "pathogenic" if direction == "up" else "protective"

    # Priority 3: immune-modulator from strategy
    col = detected.get("strategy_type")
    if col and pd.notna(row.get(col)):
        strat = str(row[col]).strip().lower()
        if any(kw in strat for kw in ("immune", "cytokine", "interferon", "checkpoint")):
            return "immune_modulator"

    # Priority 4: validated driver tier
    col = detected.get("causal_tier")
    if col and pd.notna(row.get(col)):
        tier = str(row[col]).strip().lower()
        if "validated" in tier or "driver" in tier:
            return "pathogenic" if direction == "up" else "protective"

    # Directional fallback — ensures downregulated genes are never silently excluded
    return "pathogenic" if direction == "up" else "protective"


def _assign_evidence_stratum(row, detected: Dict[str, Optional[str]]) -> Optional[str]:
    """Derive evidence stratum from CSV columns — column-existence-gated."""
    import pandas as pd

    # Tier 1: Full causal chain
    col = detected.get("causal_linkage_tier")
    if col and pd.notna(row.get(col)):
        if "full causal chain" in str(row[col]).lower():
            return "known_driver"

    # Tier 2: Disease + pathway linked
    dl_col = detected.get("has_disease_link")
    pl_col = detected.get("has_pathway_link")
    if dl_col and pl_col:
        dl = str(row.get(dl_col, "")).strip().lower()
        pl = str(row.get(pl_col, "")).strip().lower()
        if dl == "yes" and pl == "yes":
            return "ppi_connected"

    # Tier 3: Expression or MR significance
    mr_col = detected.get("mr_pval")
    if mr_col and pd.notna(row.get(mr_col)):
        try:
            if float(row[mr_col]) < 0.05:
                return "expression_significant"
        except (ValueError, TypeError):
            pass

    # If any role-relevant columns exist at all, classify as novel_candidate
    if any(detected.get(k) for k in ("causal_linkage_tier", "has_disease_link", "causal_tier")):
        return "novel_candidate"

    # No stratum columns present → None (backward-compatible multiplier 1.0)
    return None


def _categorize_pathway(name: str) -> Optional[str]:
    for pattern, cat in _PATHWAY_CATEGORIES:
        if pattern.search(name):
            return cat
    return None


def _build_pathways_from_dataframe(
    df, disease: str = "unknown", max_pathways: int = 50,
) -> List[PathwayContext]:
    """Extract pathways from a pipe-delimited column in the DataFrame."""
    import pandas as pd

    cols = list(df.columns)
    pw_col = _find_column(cols, _PATHWAY_ALIASES)
    if not pw_col:
        return []

    gene_col = _find_column(cols, _GENE_ALIASES)
    if not gene_col:
        gene_col = _find_gene_column(df)
    lfc_col = _find_column(cols, _LOG2FC_ALIASES)
    pval_col = _find_column(cols, _PVAL_ALIASES)
    dir_col = _find_column(cols, _DIRECTION_ALIASES)

    # Build pathway → member gene index
    pw_index: Dict[str, dict] = {}  # pathway_name → {genes, directions, pvals}
    for _, row in df.iterrows():
        raw_pw = row.get(pw_col)
        if pd.isna(raw_pw) or not str(raw_pw).strip():
            continue
        gene = str(row.get(gene_col, "")).strip().upper() if gene_col else None
        lfc = 0.0
        if lfc_col and pd.notna(row.get(lfc_col)):
            try:
                lfc = float(row[lfc_col])
            except (ValueError, TypeError):
                pass
        pval = 1.0
        if pval_col and pd.notna(row.get(pval_col)):
            try:
                pval = float(row[pval_col])
            except (ValueError, TypeError):
                pass

        direction = "up"
        if dir_col and pd.notna(row.get(dir_col)):
            direction = "down" if str(row[dir_col]).strip().lower() in ("down", "downregulated", "negative") else "up"
        elif lfc < 0:
            direction = "down"

        for pw_name in str(raw_pw).split("|"):
            pw_name = pw_name.strip()
            if not pw_name:
                continue
            if pw_name not in pw_index:
                pw_index[pw_name] = {"genes": [], "directions": [], "pvals": [], "lfcs": []}
            entry = pw_index[pw_name]
            if gene:
                entry["genes"].append(gene)
            entry["directions"].append(direction)
            entry["pvals"].append(pval)
            entry["lfcs"].append(abs(lfc))

    disease_lower = disease.lower()
    pathways = []
    for pw_name, data in pw_index.items():
        up_count = data["directions"].count("up")
        down_count = data["directions"].count("down")
        direction = "up" if up_count >= down_count else "down"

        # Top key genes by |log2fc|
        gene_lfc = sorted(zip(data["genes"], data["lfcs"]), key=lambda x: -x[1])
        key_genes = list(dict.fromkeys(g for g, _ in gene_lfc[:10]))

        min_pval = min(data["pvals"]) if data["pvals"] else 0.05

        relevance = None
        if disease_lower != "unknown" and disease_lower in pw_name.lower():
            relevance = "Directly disease-associated pathway"

        pathways.append(PathwayContext(
            pathway_name=pw_name,
            direction=direction,
            fdr=min_pval,
            gene_count=len(data["genes"]),
            category=_categorize_pathway(pw_name),
            key_genes=key_genes or None,
            disease_relevance=relevance,
        ))

    pathways.sort(key=lambda p: -p.gene_count)
    return pathways[:max_pathways]


def _build_tme_from_dataframe(df) -> Optional[TMEContext]:
    """Extract TME context from a cell-type column in the DataFrame."""
    import pandas as pd

    ct_col = _find_column(list(df.columns), _CELL_TYPE_ALIASES)
    if not ct_col:
        return None

    cell_counts: Dict[str, int] = {}
    for val in df[ct_col].dropna():
        for ct in str(val).split("|"):
            ct = ct.strip()
            if ct:
                cell_counts[ct] = cell_counts.get(ct, 0) + 1

    if not cell_counts:
        return None

    sorted_cells = sorted(cell_counts.items(), key=lambda x: -x[1])
    n = len(sorted_cells)
    q1 = max(1, n // 4)
    highly = [c for c, _ in sorted_cells[:q1]]
    moderately = [c for c, _ in sorted_cells[q1:]]

    if n > 5:
        level = "high"
    elif n >= 2:
        level = "moderate"
    else:
        level = "low"

    return TMEContext(
        highly_enriched_cells=highly,
        moderately_enriched_cells=moderately,
        immune_infiltration_level=level,
    )


def _build_disease_context(
    disease: str, genes: List[GeneContext], pathways: List[PathwayContext],
) -> Optional[str]:
    """Synthetic disease context for OT fallback scoring."""
    if disease.lower() == "unknown":
        return None
    up = [g for g in genes if g.direction == "up"]
    down = [g for g in genes if g.direction == "down"]
    parts = [f"{disease} patient with {len(genes)} DEGs ({len(up)} up, {len(down)} down)."]
    if pathways:
        top_pw = ", ".join(p.pathway_name for p in pathways[:3])
        parts.append(f"Top pathways: {top_pw}.")
    if up:
        parts.append(f"Key upregulated: {', '.join(g.gene_symbol for g in up[:5])}.")
    if down:
        parts.append(f"Key downregulated: {', '.join(g.gene_symbol for g in down[:5])}.")
    return " ".join(parts)[:500]


def _build_genes_from_dataframe(df, max_genes: int = 1000) -> List[GeneContext]:
    """Deterministic gene extraction from DataFrame — the primary extraction path for uploaded files.

    Uses file_parser's smart column detection (name-match → content-validated "id"/"name" → first-col fallback).
    Sorts by significance and caps at max_genes to prevent Qdrant query explosion.
    """
    import pandas as pd

    cols = list(df.columns)

    # Gene column: try explicit aliases first, then smart detection from file_parser
    gene_col = _find_column(cols, _GENE_ALIASES)
    # Content-gate for generic column names (e.g. "id") to avoid numeric primary keys
    if gene_col and len(gene_col) <= 4 and not _looks_like_gene_column(df[gene_col]):
        gene_col = None
    if not gene_col:
        gene_col = _find_gene_column(df)
    if not gene_col:
        return []

    lfc_col = _find_column(cols, _LOG2FC_ALIASES)
    pval_col = _find_column(cols, _PVAL_ALIASES)
    dir_col = _find_column(cols, _DIRECTION_ALIASES)
    detected = _detect_columns(cols)

    # Sort by significance: p-value ascending, then |log2fc| descending
    work = df.copy()
    if pval_col:
        work[pval_col] = pd.to_numeric(work[pval_col], errors="coerce")
        if lfc_col:
            work["_abs_lfc"] = pd.to_numeric(work[lfc_col], errors="coerce").abs()
            work = work.sort_values([pval_col, "_abs_lfc"], ascending=[True, False])
        else:
            work = work.sort_values(pval_col, ascending=True)
    elif lfc_col:
        work["_abs_lfc"] = pd.to_numeric(work[lfc_col], errors="coerce").abs()
        work = work.sort_values("_abs_lfc", ascending=False)

    work = work.head(max_genes)

    genes = []
    for _, row in work.iterrows():
        symbol = str(row.get(gene_col, "")).strip()
        if not symbol or symbol.lower() in ("nan", ""):
            continue

        log2fc = 1.0
        if lfc_col and pd.notna(row.get(lfc_col)):
            try:
                log2fc = float(row[lfc_col])
            except (ValueError, TypeError):
                log2fc = 1.0

        pval = 0.05
        if pval_col and pd.notna(row.get(pval_col)):
            try:
                pval = float(row[pval_col])
            except (ValueError, TypeError):
                pval = 0.05

        # Direction: prefer explicit direction column (e.g. expression_trend="Up"/"Down")
        # over log2fc sign — critical when log2fc values are all 0.
        if dir_col and pd.notna(row.get(dir_col)):
            raw_dir = str(row[dir_col]).strip().lower()
            direction = "down" if raw_dir in ("down", "downregulated", "negative") else "up"
            # Synthesize magnitude when log2fc is explicitly 0 but direction is known,
            # so the magnitude scorer has a nonzero value to differentiate genes.
            if log2fc == 0.0:
                log2fc = -1.0 if direction == "down" else 1.0
        else:
            direction = "up" if log2fc >= 0 else "down"

        role = _assign_gene_role(row, direction, detected)
        stratum = _assign_evidence_stratum(row, detected)
        causal_tier = None
        tier_col = detected.get("causal_linkage_tier")
        if tier_col and pd.notna(row.get(tier_col)):
            causal_tier = str(row[tier_col]).strip() or None

        comp_score = None
        comp_col = detected.get("composite_score")
        if comp_col and pd.notna(row.get(comp_col)):
            try:
                comp_score = float(row[comp_col])
            except (ValueError, TypeError):
                pass

        genes.append(GeneContext(
            gene_symbol=symbol.upper(),
            log2fc=log2fc,
            adj_p_value=pval,
            direction=direction,
            role=role,
            evidence_stratum=stratum,
            composite_score=comp_score,
            causal_tier=causal_tier,
        ))
    return genes
