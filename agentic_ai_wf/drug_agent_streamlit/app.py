"""
Drug Agent — AI-Powered Drug Discovery (Streamlit UI)

Upload any tabular scientific file, type a natural language query, and get
drug recommendations powered by Claude (Bedrock) + Qdrant vector search.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from agentic_ai_wf.drug_agent.service.drug_agent_service import get_service
from agentic_ai_wf.drug_agent.service.schemas import DrugCandidate, DrugQueryResponse
from agentic_ai_wf.drug_agent.service.collection_router import ALL_COLLECTIONS
from agentic_ai_wf.drug_agent_streamlit.file_parser import SUPPORTED_EXTENSIONS, FileSummary, parse_uploaded_file
from agentic_ai_wf.drug_agent_streamlit.llm_query_mapper import (
    _get_llm_client,
    classify_query,
    map_query_and_file,
)

logger = logging.getLogger(__name__)

# ─── Cached singletons ──────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading Bedrock LLM client…")
def get_llm():
    return _get_llm_client()


@st.cache_resource(show_spinner="Initializing Drug Agent Service (Qdrant + PubMedBERT)…")
def get_drug_service():
    return get_service()


# ─── Session state ───────────────────────────────────────────────────────────


def _init_state():
    defaults = {
        "parsed_file": None,
        "last_request_json": None,
        "last_response": None,
        "llm_raw": "",
        "query_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Sidebar ─────────────────────────────────────────────────────────────────


def _render_sidebar():
    with st.sidebar:
        st.header("Upload & Configure")

        uploaded = st.file_uploader(
            "Upload your data file",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            help="Supported: CSV, TSV, XLSX, XLS, TXT, JSON",
        )

        if uploaded:
            cache_key = f"{uploaded.name}_{uploaded.size}"
            if (
                st.session_state.parsed_file is None
                or getattr(st.session_state, "_file_cache_key", None) != cache_key
            ):
                with st.spinner("Parsing file…"):
                    st.session_state.parsed_file = parse_uploaded_file(uploaded)
                    st.session_state._file_cache_key = cache_key

            fs: FileSummary = st.session_state.parsed_file
            if fs.error:
                st.error(fs.error)
            else:
                with st.expander(f"File preview — {uploaded.name}", expanded=False):
                    st.caption(f"{fs.data_type} · {fs.row_count} rows · {len(fs.columns)} columns")
                    if fs.sample_genes:
                        st.caption(f"Detected genes: {', '.join(fs.sample_genes[:20])}")
                    if fs.dataframe is not None:
                        st.dataframe(_sanitize_for_display(fs.dataframe.head(10).copy()))
                    else:
                        st.code(fs.raw_preview[:1500], language="text")
        else:
            st.session_state.parsed_file = None

        st.divider()

        max_results = st.slider("Max results", 5, 50, 15, key="max_results_slider")

        discovery_genes = st.slider(
            "Discovery genes",
            min_value=50, max_value=1000, value=500,
            help="Number of top genes sent for Qdrant drug discovery queries",
            key="discovery_genes_slider",
        )
        if discovery_genes > 500:
            st.caption("\u26a0\ufe0f Discovery with >500 genes may take 3\u20135 minutes")

        # Query history
        if st.session_state.query_history:
            st.divider()
            st.subheader("Query History")
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
                label = f"{entry['disease']} — {entry['gene_count']}g, {entry['rec_count']}r"
                st.caption(f"**{entry['timestamp']}**  \n{label}")

    return max_results, discovery_genes


# ─── Results rendering ───────────────────────────────────────────────────────


def _sanitize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure uniform column types so PyArrow serialization never fails."""
    for col in df.columns:
        if df[col].apply(type).nunique() > 1:
            df[col] = df[col].astype(str)
    return df


def _candidate_to_row(c: DrugCandidate) -> dict:
    target_genes = ", ".join(t.gene_symbol for t in c.targets[:5])
    moa = c.identity.pharm_class_moa or (c.targets[0].mechanism_of_action if c.targets else "—")
    indications = ", ".join(
        i["disease_name"] for i in (c.identity.approved_indications or [])[:5]
        if i.get("disease_name")
    ) or "—"
    pw_parts = []
    if c.pathway_class_match:
        pw_parts.append(c.pathway_class_match)
    for p in (c.enriched_pathways or []):
        if p not in pw_parts:
            pw_parts.append(p)
    pathways = ", ".join(pw_parts[:4]) or "—"
    return {
        "Drug": c.identity.drug_name,
        "Score": round(c.score.composite_score, 1) if c.score else 0.0,
        "Phase": str(c.identity.max_phase) if c.identity.max_phase is not None else "—",
        "FDA": "Yes" if c.identity.is_fda_approved else "No",
        "Indications": indications,
        "Pathways": pathways,
        "Mechanism": (moa or "—")[:60],
        "Targets": target_genes or "—",
        "Sources": ", ".join(c.evidence_sources[:4]) if c.evidence_sources else "—",
        "Discovery": ", ".join(c.discovery_paths) if c.discovery_paths else "—",
    }


def _render_candidate_detail(c: DrugCandidate):
    """Expandable detail for a single drug candidate."""
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Identity**")
        st.caption(f"Type: {c.identity.drug_type or '—'}")
        st.caption(f"ChEMBL: {c.identity.chembl_id or '—'}")
        if c.identity.approved_indications:
            top_ind = "; ".join(
                f"{i['disease_name']} (Ph{i['phase']})" if i.get('phase') else i['disease_name']
                for i in c.identity.approved_indications[:8]
            )
            st.caption(f"Indications: {top_ind}")
        else:
            st.caption(f"Indication: {(c.identity.indication_text or '—')[:120]}")
        if c.identity.brand_names:
            st.caption(f"Brands: {', '.join(c.identity.brand_names[:3])}")
        _all_pw = []
        if c.pathway_class_match:
            _all_pw.append(f"{c.pathway_class_match} [class]")
        for p in (c.enriched_pathways or []):
            if p not in _all_pw:
                _all_pw.append(p)
        if _all_pw:
            st.caption(f"Pathways: {'; '.join(_all_pw[:8])}")
    with cols[1]:
        if c.score:
            st.markdown("**Score Breakdown**")
            st.caption(f"Direction match: {c.score.target_direction_match:.1f}")
            st.caption(f"Magnitude: {c.score.target_magnitude_match:.1f}")
            st.caption(f"Clinical: {c.score.clinical_regulatory_score:.1f}")
            st.caption(f"OT association: {c.score.ot_association_score:.1f}")
            st.caption(f"Pathway: {c.score.pathway_concordance:.1f}")
            st.caption(f"Safety penalty: {c.score.safety_penalty:.1f}")
            st.caption(f"Indication bonus: {c.score.disease_indication_bonus:.1f}")
            if c.score.tier_reasoning:
                st.caption(f"Tier: {c.score.tier_reasoning[:100]}")
    with cols[2]:
        if c.trial_evidence and c.trial_evidence.total_trials:
            st.markdown("**Trials**")
            st.caption(f"Total: {c.trial_evidence.total_trials}")
            st.caption(f"Completed: {c.trial_evidence.completed_trials}")
            st.caption(f"Highest phase: {c.trial_evidence.highest_phase}")
        if c.safety:
            st.markdown("**Safety**")
            if c.safety.boxed_warnings:
                st.warning(f"Boxed warnings: {len(c.safety.boxed_warnings)}")
            if c.safety.contraindications:
                st.caption(f"Contraindications: {len(c.safety.contraindications)}")

    if c.validation_caveat:
        st.info(f"Validation note: {c.validation_caveat}")


def _render_recommendations(candidates: list[DrugCandidate], label: str):
    if not candidates:
        st.info(f"No {label.lower()} found.")
        return

    rows = [_candidate_to_row(c) for c in candidates]
    df = _sanitize_for_display(pd.DataFrame(rows))
    st.dataframe(df, hide_index=True)

    for c in candidates:
        with st.expander(f"{c.identity.drug_name} — details"):
            _render_candidate_detail(c)

    csv = df.to_csv(index=False)
    st.download_button(f"Download {label} CSV", csv, f"{label.lower().replace(' ', '_')}.csv", "text/csv")


def _render_contraindicated(candidates: list[DrugCandidate]):
    if not candidates:
        st.info("No contraindicated drugs identified.")
        return

    rows = []
    for c in candidates:
        reasons = [e.reason for e in c.contraindication_entries[:3]]
        tiers = [e.label for e in c.contraindication_entries[:3]]
        rows.append({
            "Drug": c.identity.drug_name,
            "Tier": ", ".join(tiers) if tiers else ", ".join(c.contraindication_flags[:3]),
            "Reason": "; ".join(reasons) if reasons else "—",
            "Score": round(c.score.composite_score, 1) if c.score else 0,
        })
    st.dataframe(_sanitize_for_display(pd.DataFrame(rows)), hide_index=True)


def _render_validation_summary(c: DrugCandidate, disease: str):
    """Quick at-a-glance summary for validate_drug queries."""
    name = c.identity.drug_name
    drug_type = c.identity.drug_type or "drug"
    phase = c.identity.max_phase
    fda = c.identity.is_fda_approved
    score = c.score.composite_score if c.score else 0
    clinical = c.score.clinical_regulatory_score if c.score else 0
    penalty = c.score.safety_penalty if c.score else 0
    bw_count = len(c.safety.boxed_warnings) if c.safety else 0

    # Check if queried disease appears in approved indications
    disease_lower = disease.lower()
    matched_ind = next(
        (i for i in (c.identity.approved_indications or [])
         if disease_lower in i.get("disease_name", "").lower()),
        None,
    )

    parts = [f"**{name}**"]
    if fda and phase and phase >= 4:
        parts.append(f"is FDA-approved (Phase {phase})")
    elif phase:
        parts.append(f"is in Phase {phase} ({drug_type})")
    else:
        parts.append(f"({drug_type})")

    if matched_ind:
        parts.append(f"with indication for **{disease}**")
    else:
        parts.append(f"— no specific indication found for **{disease}**")

    parts.append(f"| Clinical evidence: {clinical:.0f}/25")
    if bw_count:
        parts.append(f"| Safety: {bw_count} boxed warning{'s' if bw_count != 1 else ''} ({penalty:+.0f})")
    parts.append(f"| Overall: **{score:.1f}**/100")

    banner = " ".join(parts)
    (st.success if (fda and matched_ind) else st.info)(banner)


def _render_results(response: DrugQueryResponse):
    tab_rec, tab_contra, tab_gene, tab_debug = st.tabs([
        f"Recommended ({len(response.recommendations)})",
        f"Contraindicated ({len(response.contraindicated)})",
        f"Gene-Targeted Only ({len(response.gene_targeted_only)})",
        "Query Debug",
    ])

    with tab_rec:
        if response.query_type == "validate_drug" and response.recommendations:
            _render_validation_summary(response.recommendations[0], response.disease)
        if response.targets_without_drugs:
            st.warning(
                f"No drug candidates found directly targeting: "
                f"**{', '.join(response.targets_without_drugs)}**. "
                f"Consider searching for drugs targeting interacting genes."
            )
        target_matched = [c for c in response.recommendations if c.match_group == "target"]
        disease_matched = [c for c in response.recommendations if c.match_group == "disease"]
        if target_matched and disease_matched:
            st.markdown("### 🎯 Matching Your Targets")
            _render_recommendations(target_matched, "Target-Matched Drugs")
            st.markdown("### 💊 Other Disease-Relevant")
            _render_recommendations(disease_matched, "Disease-Relevant Drugs")
        else:
            _render_recommendations(response.recommendations, "Recommended Drugs")
    with tab_contra:
        _render_contraindicated(response.contraindicated)
    with tab_gene:
        if response.disease and response.disease.lower() != "unknown":
            st.caption(
                f"These drugs target genes from your dataset but are **not yet validated "
                f"for {response.disease}**. They may be candidates for repurposing research."
            )
        _render_recommendations(response.gene_targeted_only, "Gene-Targeted Drugs")
    with tab_debug:
        st.subheader("DrugQueryRequest sent")
        st.json(st.session_state.last_request_json or {})
        with st.expander("Raw LLM response"):
            st.code(st.session_state.llm_raw, language="json")
        if response.metadata:
            st.subheader("Metadata")
            st.json(response.metadata)
        if response.errors:
            st.error("Errors: " + "; ".join(response.errors))


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="Drug Agent", page_icon="💊", layout="wide")
    _init_state()

    st.title("💊 Drug Agent — AI-Powered Drug Discovery")
    st.caption(
        "Upload a DEG table or gene list, describe your query in plain English, "
        f"and get scored drug recommendations from {len(ALL_COLLECTIONS)} Qdrant collections."
    )

    max_results, discovery_genes = _render_sidebar()

    query = st.text_area(
        "What drugs are you looking for?",
        placeholder=(
            "Examples:\n"
            "• Find drugs for Crohn's Disease targeting TNF, IL6, JAK2\n"
            "• Validate Infliximab for Ulcerative Colitis\n"
            "• Recommend treatments for breast cancer using the uploaded DEG file"
        ),
        height=100,
    )

    run = st.button("Run Analysis", type="primary", width="stretch")

    if run and query.strip():
        llm = get_llm()

        with st.status("Processing…", expanded=True) as status:

            # Step 1: classify
            st.write("Classifying query…")
            intent = classify_query(llm, query)
            if intent == "OTHER":
                status.update(label="Not a drug discovery query", state="error")
                st.error(
                    "This tool is specifically for drug discovery queries — "
                    "drug recommendations or drug validation for a disease. "
                    "Please rephrase your question."
                )
                return

            # Step 2: map
            st.write("Extracting disease, genes, pathways from your query…")
            file_summary = st.session_state.parsed_file
            request, raw_llm, parsed_json = map_query_and_file(
                llm, query, file_summary, max_results, discovery_genes
            )
            st.session_state.llm_raw = raw_llm

            try:
                from dataclasses import asdict
                st.session_state.last_request_json = asdict(request)
            except Exception:
                st.session_state.last_request_json = {"disease": request.disease}

            # Confirmation banner
            gene_count = len(request.genes)
            pathway_count = len(request.pathways)

            # Abort early if LLM extraction returned nothing useful
            if request.disease.lower() in ("unknown", "") and gene_count == 0:
                status.update(label="Could not extract query details", state="error")
                st.error(
                    "The LLM could not identify a disease or genes from your input. "
                    "Please provide a clearer query (e.g. *'Recommend drugs for "
                    "breast cancer targeting BRCA1'*) or upload a file with gene data."
                )
                with st.expander("Debug info"):
                    st.code(raw_llm or "(LLM returned no response — possible API error)", language="text")
                    st.json(parsed_json or {})
                return

            # Dynamic time estimate based on query complexity
            if gene_count == 0:
                _time_est = "1–3 minutes (disease-only search)"
            elif gene_count <= 50:
                _time_est = "1–5 minutes"
            elif gene_count <= 200:
                _time_est = "3–10 minutes"
            else:
                _time_est = "10–30 minutes"
            st.info(
                f"Querying drug agent for **{request.disease}** with "
                f"**{gene_count} genes**, **{pathway_count} pathways**. "
                f"This takes {_time_est}…"
            )

            # Step 3: query drug agent
            st.write("Searching Qdrant collections…")
            t0 = time.time()
            svc = get_drug_service()
            response = svc.query(request)
            elapsed = time.time() - t0

            st.session_state.last_response = response

            # History
            st.session_state.query_history.append({
                "query": query[:80],
                "disease": request.disease,
                "gene_count": gene_count,
                "rec_count": len(response.recommendations),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

            status.update(
                label=f"Done — {len(response.recommendations)} recommendations in {elapsed:.1f}s",
                state="complete",
            )

    if st.session_state.last_response:
        _render_results(st.session_state.last_response)
    elif not run:
        st.markdown(
            "---\n"
            "*Upload a file in the sidebar (optional), type your query above, "
            "and click **Run Analysis**.*"
        )


if __name__ == "__main__":
    main()
