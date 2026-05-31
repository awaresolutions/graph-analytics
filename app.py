import sys
import asyncio
import streamlit as st

# WINDOWS ENGINES PATCH: Handle loop initialization safely if running a local Windows test
if sys.platform == 'win32':
    try:
        asyncio.get_event_loop_policy()
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from gremlin_python.driver import client, serializer

# Set up page configurations for a clean data- maturity presentation look
st.set_page_config(page_title="Automotive Graph DB Analytics", layout="wide")
st.title("📊 Real-Time Graph DB Dealer Ecosystem Analytics")
st.caption("Live Property Graph Traversals powered by Azure Cosmos DB (Gremlin API)")

# ==========================================
# RETRIEVE CREDENTIALS FROM STREAMLIT SECRETS
# ==========================================
# When deploying publicly, we never hardcode keys. We pull them safely from environment slots.
ENDPOINT = st.secrets["COSMOS_ENDPOINT"]
PRIMARY_KEY = st.secrets["COSMOS_KEY"]
DATABASE_NAME = st.secrets["COSMOS_DB"]
CONTAINER_NAME = st.secrets["COSMOS_CONTAINER"]
USERNAME = f"/dbs/{DATABASE_NAME}/colls/{CONTAINER_NAME}"

# ==========================================
# CONNECTION POOL CACHING ENGINE
# ==========================================
@st.cache_resource
def get_gremlin_client():
    """Establishes a single, persistent network driver connection pool shared across all web sessions."""
    return client.Client(
        url=ENDPOINT,
        traversal_source='g',
        username=USERNAME,
        password=PRIMARY_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0()
    )

try:
    db_client = get_gremlin_client()
except Exception as conn_err:
    st.error(f"Failed to bridge network channel to Azure Cluster: {conn_err}")
    st.stop()

def run_graph_query(query_string):
    """Submits text queries synchronously over the global cached driver stream."""
    try:
        result_set = db_client.submit(query_string)
        return result_set.all().result()
    except Exception as query_err:
        st.error(f"Traversal Exception: {query_err}")
        return None

# ==========================================
# Step 5: Frontend Visualization Matrix
# ==========================================

# KPI Scorecard Layout Board
st.subheader("🏎️ Executive Inventory Matrix Primitives")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

# Metrics Aggregation Traversal Lookups
total_vertices = run_graph_query("g.V().count()")
pipeline_distribution = run_graph_query("g.V().hasLabel('vehicle').groupCount().by('status')")

if total_vertices:
    kpi_col1.metric("Total Graph Nodes", f"{total_vertices[0]} Vertices")

if pipeline_distribution and len(pipeline_distribution) > 0:
    dist = pipeline_distribution[0]
    kpi_col2.metric("Vehicles Handled (Sold)", f"{dist.get('Sold', 0)} Units")
    kpi_col3.metric("Active Lot Inventory", f"{dist.get('On-Lot', 0)} Units")

st.markdown("---")

# Split Dashboard Pane: Advanced Traversal Data Frames
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Lead Generation Conversion Funnels")
    st.write("Outbound path traversal tracing `TrueCar` referral vertices across hardware partition boundaries:")
    
    # Run the optimized conversion lookup traversal
    conversion_query = "g.V('TrueCar').has('model','Taxonomy').bothE('GENERATED').bothV().hasLabel('vehicle').bothE('RESULTED_IN').bothV().hasLabel('sales_outcome').id().groupCount()"
    conversion_data = run_graph_query(conversion_query)
    
    if conversion_data and len(conversion_data) > 0:
        c_dict = conversion_data[0]
        # Format the unstructured graph return matrix into a clean data dashboard dataframe
        st.bar_chart(c_dict)
        st.json(c_dict)

with col_right:
    st.subheader("📈 Marketing Channel Pipeline Driving Origination")
    st.write("Inbound backward path traversal tracing back from active negotiation pipelines to discover high-yield channels:")
    
    # Run the optimized inbound origination traversal
    origination_query = "g.V('Active Follow-up').has('model','Taxonomy').bothE('RESULTED_IN').bothV().hasLabel('vehicle').bothE('GENERATED').bothV().hasLabel('lead_source').id().groupCount()"
    origination_data = run_graph_query(origination_query)
    
    if origination_data and len(origination_data) > 0:
        o_dict = origination_data[0]
        st.bar_chart(o_dict)
        st.json(o_dict)