"""
Drug Discovery Agent - Multi-Collection Streamlit Application
=============================================================
Searches across ALL Qdrant collections:
- Drug_agent (GeneALaCart genes)
- ChEMBL_drugs (ChEMBL drug compounds)
- Raw_csv_KG (Knowledge Graph)
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.models import Filter, FieldCondition, Range

# Load environment
SCRIPT_DIR = Path(__file__).parent.resolve()
load_dotenv(SCRIPT_DIR / ".env")

st.set_page_config(page_title="Drug Discovery Agent", page_icon="💊", layout="wide")

# Define collections with their metadata
COLLECTIONS = {
    "Drug_agent": {
        "name": "GeneALaCart Genes",
        "icon": "🧬",
        "description": "Gene-disease-drug associations from GeneALaCart",
        "fields": ["gene_symbol", "diseases", "drugs", "summary", "pathways", "phenotypes"]
    },
    "ChEMBL_drugs": {
        "name": "ChEMBL Drugs",
        "icon": "💊",
        "description": "Drug compounds from ChEMBL database",
        "fields": ["drug_name", "mechanism", "targets", "indications", "phase"]
    },
    "Raw_csv_KG": {
        "name": "Knowledge Graph",
        "icon": "🔗",
        "description": "Biomedical knowledge graph relationships",
        "fields": ["subject", "predicate", "object", "source"]
    },
    "OpenTargets_data": {
        "name": "Open Targets",
        "icon": "🎯",
        "description": "Drug targets, diseases, and associations from Open Targets",
        "fields": ["name", "entity_type", "description", "text_content", "score"]
    },
    "OpenTargets_drugs_enriched": {
        "name": "🔬 Open Targets Drugs (Enriched)",
        "icon": "💉",
        "description": "Detailed drug data with mechanisms, targets, and indications for drug recommendations",
        "fields": ["name", "drug_type", "mechanisms", "indications", "linked_targets", "linked_diseases"]
    },
    "OpenTargets_adverse_events": {
        "name": "Adverse Events (FAERS)",
        "icon": "⚠️",
        "description": "FDA adverse event reports and drug safety warnings from Open Targets",
        "fields": ["drug_name", "event_name", "report_count", "log_lr", "toxicity_class"]
    },
    "OpenTargets_pharmacogenomics": {
        "name": "Pharmacogenomics",
        "icon": "🧬",
        "description": "Variant-drug interactions affecting efficacy, toxicity, and dosing",
        "fields": ["gene_symbol", "variant_rs_id", "drug_name", "pgx_category", "evidence_level"]
    },
    "FDA_Orange_Book": {
        "name": "Orange Book",
        "icon": "📙",
        "description": "FDA-approved products with patents, exclusivity, and therapeutic equivalence",
        "fields": ["trade_name", "ingredient", "te_code", "nda_number", "patents", "exclusivities"]
    },
    "FDA_DrugsFDA": {
        "name": "Drugs@FDA",
        "icon": "🏛️",
        "description": "FDA regulatory submission history, approvals, and pharmacological classification",
        "fields": ["brand_name", "generic_name", "sponsor_name", "application_number", "pharm_class_moa"]
    },
    "FDA_FAERS": {
        "name": "FDA FAERS",
        "icon": "🚨",
        "description": "FDA adverse event reporting system — reaction counts, seriousness, and outcomes",
        "fields": ["drug_name", "reaction_term", "reaction_count", "serious_pct", "fatal_pct"]
    },
    "FDA_Drug_Labels": {
        "name": "Drug Labels",
        "icon": "📋",
        "description": "FDA drug labeling sections — indications, mechanisms, adverse reactions, warnings",
        "fields": ["brand_name", "generic_name", "section_name", "pharm_class_moa"]
    },
    "FDA_Enforcement": {
        "name": "Enforcement",
        "icon": "⛔",
        "description": "FDA drug recalls, market withdrawals, and enforcement actions",
        "fields": ["classification", "product_description", "reason_for_recall", "recalling_firm"]
    },
    "ClinicalTrials_summaries": {
        "name": "Clinical Trials",
        "icon": "🏥",
        "description": "ClinicalTrials.gov Phase 2-4 drug/biological trial protocols — conditions, interventions, design, endpoints",
        "fields": ["nct_id", "brief_title", "phase", "overall_status", "conditions", "drug_names", "sponsor"]
    },
    "ClinicalTrials_results": {
        "name": "Trial Results",
        "icon": "📊",
        "description": "Clinical trial outcome measures, efficacy data, p-values, serious adverse events",
        "fields": ["nct_id", "brief_title", "phase", "conditions", "drug_names", "primary_outcome_titles", "p_values"]
    },
}


@st.cache_resource
def get_qdrant_client():
    """Get Qdrant client with Basic Auth."""
    try:
        from qdrant_client import QdrantClient
        import httpx
        
        url = os.getenv("QDRANT_URL", "https://vector.f420.ai")
        username = os.getenv("QDRANT_USERNAME", "admin")
        password = os.getenv("QDRANT_PASSWORD", "4-2i!CW~5ic+")
        
        client = QdrantClient(
            url=url,
            port=443,
            timeout=60,
            prefer_grpc=False,
            https=True,
        )
        
        # Patch with Basic Auth
        auth = httpx.BasicAuth(username, password)
        custom_http = httpx.Client(auth=auth, timeout=60.0)
        
        http_apis = client._client.http
        for api_name in ['collections_api', 'points_api', 'service_api', 'search_api']:
            api = getattr(http_apis, api_name, None)
            if api and hasattr(api, 'api_client'):
                api.api_client._client = custom_http
        
        return client, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_embedder():
    """Load PubMedBERT embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    except Exception as e:
        st.error(f"Failed to load embedder: {e}")
        return None


def get_collection_info(client, collection_name):
    """Get info about a collection."""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points": info.points_count,
            "status": str(info.status)
        }
    except Exception as e:
        return {"name": collection_name, "points": 0, "status": f"Error: {e}"}


# Collections that store max_phase in their payloads
PHASE_FILTERABLE = {"OpenTargets_drugs_enriched", "OpenTargets_adverse_events", "OpenTargets_pharmacogenomics", "FDA_FAERS", "ClinicalTrials_summaries", "ClinicalTrials_results"}


def search_collection(client, collection_name, query_vector, limit=5, min_phase=None):
    """Search a single collection with optional phase filter."""
    try:
        qf = None
        if min_phase and collection_name in PHASE_FILTERABLE:
            qf = Filter(must=[FieldCondition(key="max_phase", range=Range(gte=min_phase))])
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qf,
            limit=limit,
            with_payload=True
        )
        return collection_name, results.points, None
    except Exception as e:
        return collection_name, [], str(e)


def get_field(payload, *keys, default=""):
    """Get first available field value from multiple possible keys."""
    for key in keys:
        if key in payload and payload[key]:
            return payload[key]
    return default


def format_result(payload, collection_name):
    """Format a result based on collection type."""
    if collection_name == "Drug_agent":
        return {
            "title": payload.get("gene_symbol", "Unknown Gene"),
            "subtitle": "Gene",
            "diseases": payload.get("diseases", []),
            "drugs": payload.get("drugs", []),
            "summary": payload.get("summary", payload.get("text_content", "")[:500]),
            "pathways": payload.get("pathways", []),
            "phenotypes": payload.get("phenotypes", [])
        }
    elif collection_name == "ChEMBL_drugs":
        return {
            "title": payload.get("drug_name", payload.get("molecule_name", "Unknown Drug")),
            "subtitle": "Drug Compound",
            "mechanism": payload.get("mechanism_of_action", payload.get("mechanism", "")),
            "targets": payload.get("targets", payload.get("target_name", "")),
            "indications": payload.get("indications", payload.get("disease", "")),
            "phase": payload.get("max_phase", payload.get("phase", "")),
            "summary": payload.get("text_content", "")[:500]
        }
    elif collection_name == "Raw_csv_KG":
        subj = get_field(payload, "subject", "x_name")
        pred = get_field(payload, "predicate", "display_relation", "relation")
        obj = get_field(payload, "object", "y_name")
        return {
            "title": f"{subj or 'Unknown'} → {obj or 'Unknown'}",
            "subtitle": pred or "Relationship",
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "source": get_field(payload, "source", "x_source", "data_source"),
            "x_type": payload.get("x_type", ""),
            "y_type": payload.get("y_type", ""),
            "summary": payload.get("text_content", str(payload))[:500]
        }
    elif collection_name == "OpenTargets_data":
        entity_type = payload.get("entity_type", "unknown")
        name = payload.get("name", payload.get("approved_symbol", payload.get("id", "Unknown")))
        
        if entity_type == "target":
            return {
                "title": f"🎯 {name}",
                "subtitle": f"Target - {payload.get('biotype', 'protein_coding')}",
                "name": name,
                "approved_name": payload.get("approved_name", ""),
                "biotype": payload.get("biotype", ""),
                "description": payload.get("description", "")[:300],
                "summary": payload.get("text_content", "")[:500]
            }
        elif entity_type == "disease":
            return {
                "title": f"🏥 {name}",
                "subtitle": "Disease",
                "name": name,
                "description": payload.get("description", ""),
                "therapeutic_areas": payload.get("therapeutic_areas", []),
                "summary": payload.get("text_content", "")[:500]
            }
        elif entity_type == "drug":
            return {
                "title": f"💊 {name}",
                "subtitle": f"Drug - Phase {payload.get('max_phase', 'N/A')}",
                "name": name,
                "drug_type": payload.get("drug_type", ""),
                "max_phase": payload.get("max_phase", ""),
                "mechanisms": payload.get("mechanisms", []),
                "indications": payload.get("indications", []),
                "summary": payload.get("text_content", "")[:500]
            }
        elif entity_type == "association":
            return {
                "title": f"🔗 {payload.get('target_name', '')} ↔ {payload.get('disease_name', '')}",
                "subtitle": f"Association (score: {payload.get('score', 0):.3f})",
                "target": payload.get("target_name", ""),
                "disease": payload.get("disease_name", ""),
                "score": payload.get("score", 0),
                "evidence_types": payload.get("evidence_types", {}),
                "summary": payload.get("text_content", "")[:500]
            }
        else:
            return {
                "title": name,
                "subtitle": f"Open Targets - {entity_type}",
                "summary": payload.get("text_content", str(payload))[:500]
            }
    elif collection_name == "OpenTargets_drugs_enriched":
        entity_type = payload.get("entity_type", "drug_enriched")
        name = payload.get("name", payload.get("drug_name", "Unknown Drug"))
        
        if entity_type == "drug_enriched":
            mechanisms = payload.get("mechanisms", [])
            indications = payload.get("indications", [])
            targets = payload.get("mechanism_targets", payload.get("linked_targets", []))
            
            # Format indications for display
            indication_names = []
            for ind in indications[:5]:
                if isinstance(ind, dict):
                    indication_names.append(f"{ind.get('disease_name', '')} (Phase {ind.get('phase', '')})")
                else:
                    indication_names.append(str(ind))
            
            return {
                "title": f"💉 {name}",
                "subtitle": f"{payload.get('drug_type', 'Drug')} - Phase {payload.get('max_phase', 'N/A')}",
                "name": name,
                "drug_type": payload.get("drug_type", ""),
                "max_phase": payload.get("max_phase", ""),
                "mechanisms": mechanisms[:5] if mechanisms else [],
                "targets": targets[:10] if targets else [],
                "indications": indication_names,
                "linked_diseases": payload.get("linked_diseases", [])[:10],
                "description": payload.get("description", "")[:300],
                "summary": payload.get("text_content", "")[:500]
            }
        elif entity_type == "drug_indication":
            return {
                "title": f"💊 {payload.get('drug_name', '')} → {payload.get('disease_name', '')}",
                "subtitle": f"Drug Indication - Phase {payload.get('clinical_phase', 'N/A')}",
                "drug_name": payload.get("drug_name", ""),
                "disease_name": payload.get("disease_name", ""),
                "mechanism": payload.get("mechanism_of_action", ""),
                "target": payload.get("target_name", ""),
                "phase": payload.get("clinical_phase", ""),
                "status": payload.get("clinical_status", ""),
                "summary": payload.get("text_content", "")[:500]
            }
        else:
            return {
                "title": name,
                "subtitle": f"Enriched Drug Data - {entity_type}",
                "summary": payload.get("text_content", str(payload))[:500]
            }
    elif collection_name == "OpenTargets_adverse_events":
        entity_type = payload.get("entity_type", "")
        if entity_type == "drug_warning":
            return {
                "title": f"🚨 {payload.get('drug_name', '')} - {payload.get('toxicity_class', '')}",
                "subtitle": payload.get("warning_type", "Drug Warning"),
                "drug_name": payload.get("drug_name", ""),
                "toxicity_class": payload.get("toxicity_class", ""),
                "country": payload.get("country", ""),
                "summary": payload.get("text_content", "")[:500]
            }
        else:
            return {
                "title": f"⚠️ {payload.get('drug_name', '')} → {payload.get('event_name', '')}",
                "subtitle": f"logLR={payload.get('log_lr', 0):.1f} | {payload.get('report_count', 0):,} reports",
                "drug_name": payload.get("drug_name", ""),
                "event_name": payload.get("event_name", ""),
                "log_lr": payload.get("log_lr", 0),
                "report_count": payload.get("report_count", 0),
                "meddra_code": payload.get("meddra_code", ""),
                "summary": payload.get("text_content", "")[:500]
            }
    elif collection_name == "OpenTargets_pharmacogenomics":
        entity_type = payload.get("entity_type", "")
        if entity_type == "target_safety":
            return {
                "title": f"🛡️ {payload.get('gene_symbol', '')} - {payload.get('event', '')}",
                "subtitle": f"Safety | {payload.get('datasource', '')}",
                "gene_symbol": payload.get("gene_symbol", ""),
                "event": payload.get("event", ""),
                "summary": payload.get("text_content", "")[:500]
            }
        else:
            return {
                "title": f"🧬 {payload.get('gene_symbol', '')} {payload.get('variant_rs_id', '')} → {payload.get('drug_name', '')}",
                "subtitle": f"{payload.get('pgx_category', '')} | Evidence: {payload.get('evidence_level', '')}",
                "gene_symbol": payload.get("gene_symbol", ""),
                "variant_rs_id": payload.get("variant_rs_id", ""),
                "drug_name": payload.get("drug_name", ""),
                "pgx_category": payload.get("pgx_category", ""),
                "evidence_level": payload.get("evidence_level", ""),
                "phenotype": payload.get("phenotype_text", ""),
                "summary": payload.get("text_content", "")[:500]
            }
    elif collection_name == "FDA_Orange_Book":
        patents = payload.get("patents", [])
        excls = payload.get("exclusivities", [])
        patent_str = f"{len(patents)} patent(s)" if patents else "No patents"
        te = payload.get("te_code", "")
        return {
            "title": f"📙 {payload.get('trade_name', '')} ({payload.get('ingredient', '')})",
            "subtitle": f"{payload.get('nda_type', '')}{payload.get('nda_number', '')} | {payload.get('product_type', '')} | TE: {te or 'N/A'}",
            "trade_name": payload.get("trade_name", ""),
            "ingredient": payload.get("ingredient", ""),
            "approval_date": payload.get("approval_date", ""),
            "patent_str": patent_str,
            "exclusivity_count": len(excls),
            "applicant": payload.get("applicant", ""),
            "summary": payload.get("text_content", "")[:500]
        }
    elif collection_name == "FDA_DrugsFDA":
        return {
            "title": f"🏛️ {payload.get('brand_name', '')} ({payload.get('generic_name', '')})",
            "subtitle": f"{payload.get('application_number', '')} | {payload.get('sponsor_name', '')} | {payload.get('submission_count', 0)} submissions",
            "brand_name": payload.get("brand_name", ""),
            "generic_name": payload.get("generic_name", ""),
            "sponsor_name": payload.get("sponsor_name", ""),
            "pharm_class": payload.get("pharm_class_epc", ""),
            "mechanism": payload.get("pharm_class_moa", ""),
            "summary": payload.get("text_content", "")[:500]
        }
    elif collection_name == "FDA_FAERS":
        entity_type = payload.get("entity_type", "")
        if entity_type == "faers_summary":
            return {
                "title": f"🚨 {payload.get('drug_name', '')} — FAERS Summary",
                "subtitle": f"{payload.get('total_reports', 0):,} reports | {payload.get('serious_pct', 0)}% serious | {payload.get('fatal_pct', 0)}% fatal",
                "drug_name": payload.get("drug_name", ""),
                "total_reports": payload.get("total_reports", 0),
                "serious_pct": payload.get("serious_pct", 0),
                "fatal_pct": payload.get("fatal_pct", 0),
                "summary": payload.get("text_content", "")[:500]
            }
        else:
            return {
                "title": f"🚨 {payload.get('drug_name', '')} → {payload.get('reaction_term', '')}",
                "subtitle": f"{payload.get('reaction_count', 0):,} reports | {payload.get('serious_pct', 0)}% serious",
                "drug_name": payload.get("drug_name", ""),
                "reaction_term": payload.get("reaction_term", ""),
                "reaction_count": payload.get("reaction_count", 0),
                "summary": payload.get("text_content", "")[:500]
            }
    elif collection_name == "FDA_Drug_Labels":
        return {
            "title": f"📋 {payload.get('brand_name', '')} — {payload.get('section_title', payload.get('section_name', ''))}",
            "subtitle": f"{payload.get('generic_name', '')} | {payload.get('product_type', '')}",
            "brand_name": payload.get("brand_name", ""),
            "section_name": payload.get("section_name", ""),
            "section_title": payload.get("section_title", ""),
            "pharm_class": payload.get("pharm_class_moa", ""),
            "summary": payload.get("text_content", "")[:800]
        }
    elif collection_name == "FDA_Enforcement":
        return {
            "title": f"⛔ {payload.get('classification', '')} — {payload.get('recalling_firm', '')}",
            "subtitle": f"{payload.get('status', '')} | {payload.get('voluntary_mandated', '')}",
            "classification": payload.get("classification", ""),
            "product_description": payload.get("product_description", "")[:200],
            "reason": payload.get("reason_for_recall", "")[:200],
            "brand_name": payload.get("brand_name", ""),
            "summary": payload.get("text_content", "")[:500]
        }
    elif collection_name == "ClinicalTrials_summaries":
        drugs = payload.get("drug_names", [])
        conds = payload.get("conditions", [])
        return {
            "title": f"🏥 {payload.get('nct_id', '')} — {payload.get('brief_title', '')}",
            "subtitle": f"{payload.get('phase', '')} | {payload.get('overall_status', '')} | {payload.get('sponsor', '')}",
            "nct_id": payload.get("nct_id", ""),
            "phase": payload.get("phase", ""),
            "enrollment": payload.get("enrollment", 0),
            "conditions": ", ".join(conds[:5]) if conds else "",
            "drugs": ", ".join(drugs[:5]) if drugs else "",
            "has_results": payload.get("has_results", False),
            "summary": payload.get("text_content", "")[:600]
        }
    elif collection_name == "ClinicalTrials_results":
        drugs = payload.get("drug_names", [])
        pvals = payload.get("p_values", [])
        return {
            "title": f"📊 {payload.get('nct_id', '')} — {payload.get('brief_title', '')}",
            "subtitle": f"{payload.get('phase', '')} | {payload.get('num_primary_outcomes', 0)} outcomes | {payload.get('num_serious_aes', 0)} serious AEs",
            "nct_id": payload.get("nct_id", ""),
            "phase": payload.get("phase", ""),
            "drugs": ", ".join(drugs[:5]) if drugs else "",
            "p_values": ", ".join(pvals[:5]) if pvals else "",
            "primary_outcomes": ", ".join(payload.get("primary_outcome_titles", [])[:3]),
            "summary": payload.get("text_content", "")[:800]
        }
    else:
        return {
            "title": str(payload.get("name", payload.get("id", "Unknown"))),
            "subtitle": collection_name,
            "summary": str(payload)[:500]
        }


def main():
    st.title("💊 Drug Discovery Agent")
    st.markdown("**Multi-Collection Search** - Search across genes, drugs, and knowledge graph")
    
    # Session state
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    # Get client
    client, error = get_qdrant_client()
    
    # Sidebar - Collection Status
    with st.sidebar:
        st.header("📊 Collections")
        
        if error:
            st.error(f"❌ Connection Error: {error}")
        elif client:
            total_docs = 0
            for col_name, col_meta in COLLECTIONS.items():
                info = get_collection_info(client, col_name)
                total_docs += info["points"]
                
                with st.expander(f"{col_meta['icon']} {col_meta['name']}", expanded=True):
                    if info["points"] > 0:
                        st.success(f"✅ Connected")
                        st.metric("Documents", f"{info['points']:,}")
                    else:
                        st.warning(f"⚠️ {info['status']}")
            
            st.markdown("---")
            st.metric("📚 Total Documents", f"{total_docs:,}")
        
        st.markdown("---")
        st.header("🔬 Filters")
        phase_options = {"All phases": None, "Phase 4 (Approved)": 4, "Phase 3+": 3, "Phase 2+": 2}
        phase_label = st.selectbox("Min clinical phase", options=list(phase_options.keys()), index=0,
                                   help="Filter drugs by minimum clinical trial phase (applies to Enriched Drugs, AE, PGx)")
        min_phase = phase_options[phase_label]

        st.markdown("---")
        st.header("ℹ️ Search Tips")
        st.markdown("""
        **Try searching for:**
        - Disease: `Breast Cancer`, `Diabetes`
        - Gene: `BRCA1`, `TP53`, `EGFR`
        - Drug: `Metformin`, `Imatinib`
        - Pathway: `apoptosis`, `inflammation`
        """)
    
    # Main search interface
    st.markdown("---")
    
    # Collection selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_collections = st.multiselect(
            "🗂️ Search in collections:",
            options=list(COLLECTIONS.keys()),
            default=list(COLLECTIONS.keys()),
            format_func=lambda x: f"{COLLECTIONS[x]['icon']} {COLLECTIONS[x]['name']}"
        )
    with col2:
        results_per_collection = st.number_input("Results per collection", min_value=1, max_value=20, value=5)
    
    # Search box
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("🔍 Search Query", placeholder="Enter disease, gene, drug, or pathway...")
    with col2:
        st.write("")
        st.write("")
        search = st.button("Search", type="primary", use_container_width=True)
    
    # Perform search
    if search and query and client and selected_collections:
        with st.spinner("Searching across collections..."):
            try:
                # Load embedder
                embedder = get_embedder()
                if not embedder:
                    st.error("Failed to load embedder")
                    return
                
                # Generate embedding
                query_vector = embedder.encode(query).tolist()
                
                # Search all selected collections in parallel
                all_results = {}
                with ThreadPoolExecutor(max_workers=len(selected_collections)) as executor:
                    futures = {
                        executor.submit(search_collection, client, col, query_vector,
                                        results_per_collection, min_phase): col 
                        for col in selected_collections
                    }
                    
                    for future in as_completed(futures):
                        col_name, results, error = future.result()
                        if error:
                            st.warning(f"⚠️ Error searching {col_name}: {error}")
                        all_results[col_name] = results
                
                st.session_state.results = all_results
                st.session_state.query = query
                
                total_results = sum(len(r) for r in all_results.values())
                st.success(f"✅ Found {total_results} results across {len(selected_collections)} collections!")
                
            except Exception as e:
                st.error(f"Search failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        st.subheader(f"🔎 Results for: '{st.session_state.query}'")
        
        # Create tabs for each collection
        tabs = st.tabs([
            f"{COLLECTIONS[col]['icon']} {COLLECTIONS[col]['name']} ({len(st.session_state.results.get(col, []))})"
            for col in selected_collections if col in st.session_state.results
        ])
        
        for tab, col_name in zip(tabs, [c for c in selected_collections if c in st.session_state.results]):
            with tab:
                results = st.session_state.results.get(col_name, [])
                
                if not results:
                    st.info(f"No results found in {COLLECTIONS[col_name]['name']}")
                    continue
                
                for i, r in enumerate(results):
                    payload = r.payload or {}
                    formatted = format_result(payload, col_name)
                    
                    with st.expander(
                        f"**{i+1}. {formatted['title']}** ({formatted['subtitle']}) - Score: {r.score:.2%}",
                        expanded=(i < 2)
                    ):
                        # Different display based on collection
                        if col_name == "Drug_agent":
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**🧬 Gene:**")
                                st.write(formatted["title"])
                                
                                if formatted.get("diseases"):
                                    st.markdown("**🏥 Diseases:**")
                                    diseases = formatted["diseases"]
                                    if isinstance(diseases, list):
                                        for d in diseases[:10]:
                                            name = d.get("name", d) if isinstance(d, dict) else d
                                            st.write(f"• {name}")
                                    else:
                                        st.write(diseases[:500])
                            
                            with col_b:
                                if formatted.get("drugs"):
                                    st.markdown("**💊 Drugs:**")
                                    drugs = formatted["drugs"]
                                    if isinstance(drugs, list):
                                        for d in drugs[:10]:
                                            name = d.get("name", d) if isinstance(d, dict) else d
                                            st.write(f"• {name}")
                                    else:
                                        st.write(drugs[:500])
                                
                                if formatted.get("pathways"):
                                    st.markdown("**🛤️ Pathways:**")
                                    pathways = formatted["pathways"]
                                    if isinstance(pathways, list):
                                        for p in pathways[:5]:
                                            st.write(f"• {p}")
                            
                            if formatted.get("summary"):
                                st.markdown("**📝 Summary:**")
                                st.write(formatted["summary"])
                        
                        elif col_name == "ChEMBL_drugs":
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**💊 Drug:**")
                                st.write(formatted["title"])
                                
                                if formatted.get("mechanism"):
                                    st.markdown("**⚙️ Mechanism:**")
                                    st.write(formatted["mechanism"])
                            
                            with col_b:
                                if formatted.get("targets"):
                                    st.markdown("**🎯 Targets:**")
                                    st.write(formatted["targets"])
                                
                                if formatted.get("indications"):
                                    st.markdown("**🏥 Indications:**")
                                    st.write(formatted["indications"])
                                
                                if formatted.get("phase"):
                                    st.markdown("**📊 Phase:**")
                                    st.write(formatted["phase"])
                            
                            if formatted.get("summary"):
                                st.markdown("**📝 Details:**")
                                st.write(formatted["summary"])
                        
                        elif col_name == "Raw_csv_KG":
                            st.markdown("**🔗 Knowledge Graph Triple:**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.markdown("**Subject:**")
                                st.info(formatted.get("subject", "N/A"))
                            with col_b:
                                st.markdown("**Predicate:**")
                                st.warning(formatted.get("predicate", "N/A"))
                            with col_c:
                                st.markdown("**Object:**")
                                st.success(formatted.get("object", "N/A"))
                            
                            if formatted.get("source"):
                                st.markdown(f"**Source:** {formatted['source']}")
                        
                        else:
                            st.json(payload)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "Drug Discovery Agent | Multi-Collection Search | Powered by Qdrant & PubMedBERT"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
