"""
Drug Discovery Agent - Streamlit Application
Simple version with direct Qdrant connection
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
SCRIPT_DIR = Path(__file__).parent.resolve()
load_dotenv(SCRIPT_DIR / ".env")

st.set_page_config(page_title="Drug Discovery Agent", page_icon="💊", layout="wide")


@st.cache_resource
def get_qdrant_client():
    """Get Qdrant client with Basic Auth."""
    try:
        from qdrant_client import QdrantClient
        import httpx
        
        url = os.getenv("QDRANT_URL", "https://vector.f420.ai")
        username = os.getenv("QDRANT_USERNAME", "admin")
        password = os.getenv("QDRANT_PASSWORD", "4-2i!CW~5ic+")
        collection = os.getenv("QDRANT_COLLECTION", "Drug_agent")
        
        # Create client with port 443 for nginx
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
        
        return client, collection, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def get_embedder():
    """Load PubMedBERT embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    except Exception as e:
        st.error(f"Failed to load embedder: {e}")
        return None


def main():
    st.title("💊 Drug Discovery Agent")
    st.markdown("Search for genes, diseases, and drug interactions")
    
    # Session state
    if "results" not in st.session_state:
        st.session_state.results = []
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    # Connect
    client, collection, error = get_qdrant_client()
    
    # Sidebar
    with st.sidebar:
        st.header("🔌 Status")
        if error:
            st.error(f"❌ {error}")
        elif client:
            try:
                info = client.get_collection(collection)
                st.success("✅ Connected")
                st.write(f"**Collection:** {collection}")
                st.write(f"**Documents:** {info.points_count:,}")
            except Exception as e:
                st.warning(f"⚠️ {e}")
        
        st.markdown("---")
        st.markdown("""
        **Try searching:**
        - Breast Cancer
        - Alzheimer Disease
        - BRCA1
        - TP53
        - diabetes
        """)
    
    # Search
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("🔍 Search", placeholder="Enter disease, gene, or drug...")
    with col2:
        st.write("")
        st.write("")
        search = st.button("Search", type="primary", use_container_width=True)
    
    if search and query and client:
        try:
            with st.spinner("Loading embedder..."):
                embedder = get_embedder()
            
            with st.spinner("Searching..."):
                query_vector = embedder.encode(query).tolist()
                results = client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=10,
                    with_payload=True
                )
            
            st.session_state.results = results.points
            st.session_state.query = query
            st.success(f"Found {len(results.points)} results!")
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        st.subheader(f"Results for: '{st.session_state.query}'")
        
        for i, r in enumerate(st.session_state.results):
            p = r.payload or {}
            gene = p.get("gene_symbol", "Unknown")
            diseases = p.get("diseases", [])
            drugs = p.get("drugs", [])
            summary = p.get("summary", "")
            text = p.get("text_content", "")
            
            with st.expander(f"**{i+1}. {gene}** (Score: {r.score:.2%})", expanded=(i < 3)):
                st.markdown(f"### 🧬 {gene}")
                
                tab1, tab2, tab3 = st.tabs(["🏥 Diseases", "💊 Drugs", "📝 Summary"])
                
                with tab1:
                    if diseases:
                        if isinstance(diseases, list):
                            for d in diseases[:15]:
                                name = d.get("name", d) if isinstance(d, dict) else d
                                st.write(f"• {name}")
                        else:
                            st.write(diseases)
                    else:
                        st.write("No disease associations.")
                
                with tab2:
                    if drugs:
                        if isinstance(drugs, list):
                            for d in drugs[:15]:
                                name = d.get("name", d) if isinstance(d, dict) else d
                                st.write(f"• {name}")
                        else:
                            st.write(drugs)
                    else:
                        st.write("No drug associations.")
                
                with tab3:
                    if summary:
                        st.write(summary)
                    elif text:
                        st.write(text[:1000])
                    else:
                        st.write("No summary available.")


if __name__ == "__main__":
    main()
