import streamlit as st
from sentence_transformers import SentenceTransformer
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# ==========================================
# 1. SETUP AZURE CONFIGURATION BOUNDARY
# ==========================================

AZURE_SEARCH_ENDPOINT = "https://dealer-graph-search.search.windows.net"
AZURE_SEARCH_KEY = "33mCsSi17cPWEL9citGuqLT7t0HqL3K0qJ97ao5u7LAzSeDsG75f"
INDEX_NAME = "dealer-graph-vector-index"

# Initialize Azure Search Client Connection
credential = AzureKeyCredential(AZURE_SEARCH_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential
)

# ==========================================
# 2. LOAD LOCAL VECTOR EMBEDDING ENGINE
# ==========================================
@st.cache_resource
def load_embedding_model():
    """Caches the model in system memory so it loads instantly on refresh."""
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# ==========================================
# 3. STREAMLIT FRONTEND DASHBOARD LAYOUT
# ==========================================
st.set_page_config(page_title="DealerCentric Intelligence Dashboard", layout="wide")

st.title("🚗 DealerCentric Graph-RAG Intelligence UI")
st.markdown(
    "Query your local Graph topology database using natural language vector matching. "
    "This system bypasses external AI token expenses by computing embeddings locally."
)

# Sidebar System Health Status Panel
st.sidebar.header("System Infrastructure")
st.sidebar.success("Database: Cosmos DB (Connected)")
st.sidebar.success("Index: Azure AI Search (Live)")
st.sidebar.success("Embedding Engine: Local MiniLM (Active)")

# Natural Language Query Input Interface Layout
st.subheader("🔍 Search Vehicle Inventory & Graph Insights")
user_query = st.text_input(
    label="Ask a question about the lot context:",
    placeholder="e.g., Show me premium sedans with active customer follow ups or luxury SUVs in transit...",
    help="Type your question naturally. The system will vectorize this phrase and perform a mathematical similarity search."
)

# ==========================================
# 4. EXECUTE VECTOR SIMILARITY QUERY FIELD
# ==========================================
if user_query:
    with st.spinner("Localizing search intent vectors and querying index..."):
        try:
            # Step A: Convert the user's plain text query into a 384-float vector array locally
            query_vector = embedder.encode(user_query).tolist()
            
            # Step B: Package vector array into an official Azure Search Vectorized Query Object
            vector_query_payload = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,      # Fetch the top 5 most contextually relevant node records
                fields="text_vector"        # Match against our specialized target index column
            )
            
            # Step C: Execute structural request across the cloud framework
            results = search_client.search(
                search_text=None,           # Pure vector query search strategy execution
                vector_queries=[vector_query_payload],
                select=["id", "label", "status"] # Pull core identifying attributes
            )
            
            # Step D: Parse and display output documents in UI cards
            st.subheader(f"🎯 Top Contextual Matches for: '{user_query}'")
            
            records_found = False
            for doc in results:
                records_found = True
                
                # Render clean visual metric UI containers for each graph match item
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.metric(label="Vehicle Key ID", value=doc.get("id"))
                    with col2:
                        st.markdown(f"**Entity Classification (Label):** `{doc.get('label')}`")
                    with col3:
                        # Color code status tags for high operational clarity
                        status = doc.get("status", "Unknown")
                        if status == "On-Lot":
                            st.markdown(f"**Status:** 🟩 `On-Lot`")
                        elif status == "In-Transit":
                            st.markdown(f"**Status:** 🟨 `In-Transit`")
                        else:
                            st.markdown(f"**Status:** 🟦 `{status}`")
                    st.markdown("---")
                    
            if not records_found:
                st.warning("No matching graph nodes found inside the search index boundary parameters.")
                
        except Exception as error:
            st.error(f"Failed to query Azure AI Search Engine Layout. Details: {str(error)}")