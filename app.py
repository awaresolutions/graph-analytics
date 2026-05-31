import streamlit as st
from sentence_transformers import SentenceTransformer
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# ==========================================
# 1. SETUP AZURE CONFIGURATION BOUNDARY
# ==========================================
# Target Indexing infrastructure routing addresses
AZURE_SEARCH_ENDPOINT = "https://dealer-graph-search.search.windows.net"
INDEX_NAME = "dealer-graph-vector-index"

# Fetch credentials securely from local secrets.toml or Streamlit Cloud Console
AZURE_SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]

# Initialize Azure Search Client Connection Gate
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

# Initialize session state tracking variable for the query text box if not present
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Action callback routine to empty out the session state string tracker when clicked
def clear_search_callback():
    st.session_state.search_query = ""

# Natural Language Query Input Interface Layout with dynamic side-by-side buttons
st.subheader("🔍 Search Vehicle Inventory & Graph Insights")

# Split the layout into two horizontal columns (85% width for input bar, 15% for button)
input_col, button_col = st.columns([17, 3])

with input_col:
    user_query = st.text_input(
        label="Ask a question about the lot context:",
        value=st.session_state.search_query,
        key="search_query",
        placeholder="e.g., Show me premium sedans with active customer follow ups or luxury SUVs in transit...",
        label_visibility="collapsed" # Hides text headers for a cleaner layout line
    )

with button_col:
    st.button(
        label="🧹 Clear Search", 
        on_click=clear_search_callback, 
        use_container_width=True
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