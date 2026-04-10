"""
Drug Discovery Agent - Streamlit UI
====================================

A web interface for querying drug recommendations based on disease.

Usage:
    cd drug_agent
    streamlit run app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the drug_agent directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Collection metadata for display
COLLECTION_META = {
    "Drug_agent": {"icon": "🧬", "name": "GeneALaCart Genes"},
    "ChEMBL_drugs": {"icon": "💊", "name": "ChEMBL Drugs"},
    "Raw_csv_KG": {"icon": "🔗", "name": "Knowledge Graph"},
}

# Page configuration
st.set_page_config(
    page_title="Drug Discovery Agent",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .gene-tag {
        background: #e3f2fd;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.85rem;
    }
    .approved-badge {
        background: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .trial-badge {
        background: #FF9800;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_qdrant_client():
    """Initialize Qdrant client with support for Basic Auth (nginx proxy) or API key."""
    import importlib.util
    
    # Direct import to avoid package __init__.py which has relative imports
    module_path = SCRIPT_DIR / "storage" / "basic_auth_qdrant.py"
    spec = importlib.util.spec_from_file_location("basic_auth_qdrant", module_path)
    basic_auth_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic_auth_module)
    
    env_path = SCRIPT_DIR / ".env"
    client, _, error = basic_auth_module.get_qdrant_client_from_env(str(env_path))
    return client, error


def get_all_collections(client):
    """Fetch all collections dynamically from Qdrant."""
    try:
        collections = client.get_collections().collections
        result = {}
        for c in collections:
            info = client.get_collection(c.name)
            meta = COLLECTION_META.get(c.name, {"icon": "📁", "name": c.name})
            result[c.name] = {"count": info.points_count, **meta}
        return result
    except Exception as e:
        return {}


@st.cache_resource
def get_embedder():
    """Initialize the embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
        return model
    except Exception as e:
        st.error(f"Failed to load embedder: {e}")
        return None


def search_collection(client, collection_name, query_vector, limit=10):
    """Search a single collection and return results."""
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return collection_name, results.points, None
    except Exception as e:
        return collection_name, [], str(e)


def extract_result_fields(payload, collection_name):
    """Extract display fields from payload based on collection type."""
    # Helper to get first non-empty value
    def get(*keys, default=""):
        for k in keys:
            if k in payload and payload[k]:
                return payload[k]
        return default
    
    if collection_name == "Drug_agent":
        # GeneALaCart: has gene_symbol, diseases, drugs (nested)
        drugs_list = payload.get("drugs", [])
        drug_names = [d.get("name", d) if isinstance(d, dict) else str(d) for d in drugs_list[:5]] if isinstance(drugs_list, list) else []
        return {
            "title": get("gene_symbol", "name", default="Unknown Gene"),
            "drug_name": ", ".join(drug_names) if drug_names else "",
            "gene_symbol": get("gene_symbol"),
            "mechanism": get("summary", "text_content")[:300] if get("summary", "text_content") else "",
            "approval_status": "Gene Target",
            "drug_type": "Gene-based",
            "diseases": payload.get("diseases", []),
            "drugs": drugs_list,
            "collection": collection_name
        }
    elif collection_name == "ChEMBL_drugs":
        return {
            "title": get("drug_name", "molecule_name", "pref_name", default="Unknown Drug"),
            "drug_name": get("drug_name", "molecule_name", "pref_name"),
            "gene_symbol": get("target_name", "targets"),
            "mechanism": get("mechanism_of_action", "mechanism", "text_content")[:300] if get("mechanism_of_action", "mechanism", "text_content") else "",
            "approval_status": get("max_phase", "phase", default="ChEMBL"),
            "drug_type": get("drug_type", "molecule_type", default="Small Molecule"),
            "indication": get("indications", "disease"),
            "collection": collection_name
        }
    elif collection_name == "Raw_csv_KG":
        # KG uses: x_name, y_name, x_type, y_type, display_relation, relation
        x_name = get("x_name", "subject")
        y_name = get("y_name", "object")
        x_type = get("x_type", "").lower()
        y_type = get("y_type", "").lower()
        predicate = get("display_relation", "relation")
        
        # Smart gene/drug detection based on type
        gene_symbol = ""
        drug_name = ""
        if "gene" in x_type or "protein" in x_type:
            gene_symbol = x_name
        if "gene" in y_type or "protein" in y_type:
            gene_symbol = gene_symbol or y_name
        if "drug" in x_type or "compound" in x_type:
            drug_name = x_name
        if "drug" in y_type or "compound" in y_type:
            drug_name = drug_name or y_name
        
        return {
            "title": f"{x_name} → {y_name}" if x_name and y_name else "KG Relation",
            "x_name": x_name,
            "y_name": y_name,
            "x_type": get("x_type"),
            "y_type": get("y_type"),
            "predicate": predicate,
            "drug_name": drug_name,
            "gene_symbol": gene_symbol,
            "source": get("x_source", "source", default="KG"),
            "collection": collection_name
        }
    else:
        return {
            "title": get("name", "title", "id", default="Result"),
            "drug_name": get("name", "drug_name"),
            "gene_symbol": get("gene_symbol", "gene"),
            "mechanism": get("description", "text_content", "summary")[:300] if get("description", "text_content", "summary") else "",
            "approval_status": "Unknown",
            "drug_type": "Unknown",
            "collection": collection_name
        }


def search_drugs(client, collections, embedder, disease_name, genes=None, top_k=20):
    """Search for drugs across multiple collections, return grouped by collection."""
    
    # Build search query
    query_text = f"{disease_name} treatment therapy drug"
    if genes:
        query_text += " " + " ".join(genes[:5])
    
    query_vector = embedder.encode(query_text).tolist()
    
    # Search all collections in parallel, group results by collection
    results_by_collection = {col: [] for col in collections}
    
    with ThreadPoolExecutor(max_workers=len(collections)) as executor:
        futures = {
            executor.submit(search_collection, client, col, query_vector, top_k): col
            for col in collections
        }
        for future in as_completed(futures):
            col_name, points, error = future.result()
            if error:
                st.warning(f"⚠️ {col_name}: {error}")
            for r in points:
                payload = r.payload or {}
                fields = extract_result_fields(payload, col_name)
                fields["score"] = r.score if hasattr(r, 'score') else 0.0
                fields["_payload"] = payload  # Keep raw payload for debugging
                results_by_collection[col_name].append(fields)
    
    # Sort each collection's results by score
    for col in results_by_collection:
        results_by_collection[col].sort(key=lambda x: x["score"], reverse=True)
    
    return results_by_collection


def get_status_badge(status: str) -> str:
    """Get HTML badge for approval status."""
    status_lower = status.lower() if status else ""
    if "approved" in status_lower or "fda" in status_lower:
        return f'<span class="approved-badge">✓ {status}</span>'
    elif "phase" in status_lower:
        return f'<span class="trial-badge">⏳ {status}</span>'
    return status or "Unknown"


def main():
    # Header
    st.markdown('<p class="main-header">💊 Drug Discovery Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered drug recommendations based on disease and genomic data</p>', unsafe_allow_html=True)
    
    # Load clients
    client, error = get_qdrant_client()
    
    if error:
        st.error(f"❌ Failed to connect to Qdrant: {error}")
        st.info("Make sure your `.env` file in drug_agent folder has QDRANT_URL and QDRANT_API_KEY")
        return
    
    embedder = get_embedder()
    
    if embedder is None:
        st.error("Failed to load embedding model")
        return
    
    # Fetch all collections dynamically
    collections_info = get_all_collections(client)
    
    # Sidebar - Health Check
    with st.sidebar:
        st.header("🔧 System Status")
        
        if collections_info:
            st.success("Qdrant ✓ Connected")
            st.subheader("📚 Collections")
            total = 0
            for name, info in collections_info.items():
                st.caption(f"{info['icon']} **{info['name']}**: {info['count']:,} docs")
                total += info['count']
            st.metric("Total Documents", f"{total:,}")
            
            from importlib.metadata import version
            st.caption(f"qdrant-client: {version('qdrant-client')}")
        else:
            st.error("No collections found")
        
        st.divider()
        st.header("ℹ️ About")
        st.markdown("""
        This tool queries a knowledge base of gene-drug-disease 
        relationships to recommend potential therapeutic options.
        """)
    
    # Main content
    st.header("🔍 Search for Drug Recommendations")
    
    # Collection selection
    selected_collections = st.multiselect(
        "🗂️ Search in collections:",
        options=list(collections_info.keys()),
        default=list(collections_info.keys()),
        format_func=lambda x: f"{collections_info[x]['icon']} {collections_info[x]['name']}"
    )
    
    # Disease input (mandatory)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        disease_name = st.text_input(
            "Disease Name *",
            placeholder="e.g., Breast Cancer, Lung Cancer, Alzheimer's Disease",
            help="Enter the disease name to search for relevant drug recommendations"
        )
    
    with col2:
        max_results = st.slider("Max Results", 5, 50, 20)
    
    # Optional: Gene inputs
    with st.expander("🧬 Add Genes (Optional)", expanded=False):
        genes_input = st.text_area(
            "Gene Symbols (one per line or comma-separated)",
            placeholder="ERBB2\nESR1\nBRCA1\nPIK3CA",
            height=100
        )
    
    # Search button
    st.divider()
    
    if st.button("🔍 Search for Drugs", type="primary", use_container_width=True):
        if not disease_name.strip():
            st.error("⚠️ Please enter a disease name")
            return
        
        with st.spinner(f"Searching for drugs related to {disease_name}..."):
            try:
                # Parse genes
                genes = []
                if genes_input.strip():
                    genes = [g.strip().upper() for g in genes_input.replace(",", "\n").split("\n") if g.strip()]
                
                # Search across selected collections
                if not selected_collections:
                    st.warning("Please select at least one collection")
                    return
                
                results = search_drugs(client, selected_collections, embedder, disease_name, genes, max_results)
                
                # Display results
                display_results(results, disease_name, collections_info)
                
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def display_results(results_by_collection, disease_name, collections_info):
    """Display the drug recommendation results grouped by collection."""
    
    st.divider()
    st.header(f"🔎 Results for: '{disease_name}'")
    
    # Filter to collections that have results
    active_collections = {k: v for k, v in results_by_collection.items() if v}
    
    if not active_collections:
        st.warning("No drug recommendations found for this query.")
        return
    
    total_results = sum(len(v) for v in active_collections.values())
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Results", total_results)
    with col2:
        st.metric("Collections Searched", len(active_collections))
    
    st.divider()
    
    # Create tabs per collection (like app_multi.py)
    tab_labels = [
        f"{collections_info.get(col, {}).get('icon', '📁')} {collections_info.get(col, {}).get('name', col)} ({len(results)})"
        for col, results in active_collections.items()
    ]
    tabs = st.tabs(tab_labels)
    
    for tab, (col_name, results) in zip(tabs, active_collections.items()):
        with tab:
            if not results:
                st.info(f"No results in {col_name}")
                continue
            
            # Collection-specific display
            for i, rec in enumerate(results, 1):
                score_pct = rec['score'] * 100 if rec['score'] <= 1 else rec['score']
                
                with st.expander(f"**{i}. {rec.get('title', 'Result')}** - Score: {score_pct:.2f}%", expanded=(i <= 2)):
                    
                    if col_name == "Raw_csv_KG":
                        # KG Triple display
                        st.markdown("**🔗 Knowledge Graph Triple:**")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**Subject:**")
                            st.info(rec.get("x_name") or "N/A")
                            if rec.get("x_type"):
                                st.caption(f"Type: {rec['x_type']}")
                        with c2:
                            st.markdown("**Predicate:**")
                            st.warning(rec.get("predicate") or "N/A")
                        with c3:
                            st.markdown("**Object:**")
                            st.success(rec.get("y_name") or "N/A")
                            if rec.get("y_type"):
                                st.caption(f"Type: {rec['y_type']}")
                        
                        if rec.get("source"):
                            st.caption(f"Source: {rec['source']}")
                    
                    elif col_name == "Drug_agent":
                        # Gene-centric display
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**🧬 Gene:** `{rec.get('gene_symbol', 'N/A')}`")
                            if rec.get("diseases"):
                                st.markdown("**🏥 Diseases:**")
                                diseases = rec["diseases"]
                                if isinstance(diseases, list):
                                    for d in diseases[:5]:
                                        name = d.get("name", d) if isinstance(d, dict) else str(d)
                                        st.write(f"• {name}")
                        with c2:
                            if rec.get("drugs"):
                                st.markdown("**💊 Associated Drugs:**")
                                drugs = rec["drugs"]
                                if isinstance(drugs, list):
                                    for d in drugs[:5]:
                                        name = d.get("name", d) if isinstance(d, dict) else str(d)
                                        st.write(f"• {name}")
                        
                        if rec.get("mechanism"):
                            st.markdown(f"**📝 Summary:** {rec['mechanism'][:300]}")
                    
                    elif col_name == "ChEMBL_drugs":
                        # Drug-centric display
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**💊 Drug:** {rec.get('drug_name', 'N/A')}")
                            if rec.get("mechanism"):
                                st.markdown(f"**⚙️ Mechanism:** {rec['mechanism'][:200]}")
                        with c2:
                            if rec.get("gene_symbol"):
                                st.markdown(f"**🎯 Target:** `{rec['gene_symbol']}`")
                            if rec.get("indication"):
                                st.markdown(f"**🏥 Indication:** {rec['indication']}")
                            if rec.get("approval_status"):
                                st.markdown(f"**📊 Phase:** {rec['approval_status']}")
                    
                    else:
                        # Generic display
                        if rec.get("gene_symbol"):
                            st.markdown(f"**Gene:** `{rec['gene_symbol']}`")
                        if rec.get("drug_name"):
                            st.markdown(f"**Drug:** {rec['drug_name']}")
                        if rec.get("mechanism"):
                            st.markdown(f"**Details:** {rec['mechanism'][:300]}")
    
    # Export section below tabs
    st.divider()
    with st.expander("📥 Export Results"):
        import pandas as pd
        import json
        
        # Flatten all results for export
        all_results = []
        for col_name, results in active_collections.items():
            for r in results:
                export_rec = {k: v for k, v in r.items() if k != "_payload"}
                all_results.append(export_rec)
        
        c1, c2 = st.columns(2)
        with c1:
            df = pd.DataFrame(all_results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"drugs_{disease_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        with c2:
            json_str = json.dumps(all_results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"drugs_{disease_name.replace(' ', '_')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()