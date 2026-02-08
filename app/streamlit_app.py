import streamlit as st
import requests
import json

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Customer Complaint RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: gray;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION
# ============================================

API_URL = "http://localhost:8000"

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Query settings
    st.subheader("Query Parameters")
    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=3,
        help="How many similar complaints to retrieve"
    )
    
    max_tokens = st.slider(
        "Max response length",
        min_value=100,
        max_value=1000,
        value=500,
        step=50
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative"
    )
    
    st.markdown("---")
    
    # System status
    st.subheader("🔧 System Status")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200 and health.json().get("rag_loaded"):
            st.success("✅ System Online")
            
            # Get stats
            stats = requests.get(f"{API_URL}/stats", timeout=5)
            if stats.status_code == 200:
                stats_data = stats.json()
                st.info(f"""
                **Model:** {stats_data.get('llm_model', 'N/A')}  
                **Embeddings:** {stats_data.get('embedding_model', 'N/A')}  
                **Vector DB:** {stats_data.get('vector_db', 'N/A')}
                """)
        else:
            st.warning("⚠️ System Loading...")
    except:
        st.error("❌ API Offline")
        st.info("Run: `python app/main.py`")
    
    st.markdown("---")
    st.info("""
    **Customer Complaint RAG**
    
    AI-powered support system using:
    - 🤖 LLaMA 3.2
    - 📚 FAISS Vector DB
    - 🔍 Semantic Search
    """)

# ============================================
# MAIN INTERFACE
# ============================================

st.markdown('<p class="main-header">🤖 Customer Support AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about app issues, complaints, and resolutions</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# EXAMPLE QUESTIONS
# ============================================

st.subheader("💡 Try These Questions:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📱 App Crashes", use_container_width=True):
        st.session_state.query = "Why does the app crash when placing orders?"

with col2:
    if st.button("🛒 Cart Issues", use_container_width=True):
        st.session_state.query = "Items disappearing from cart"

with col3:
    if st.button("💰 Refund Problems", use_container_width=True):
        st.session_state.query = "How do I get a refund?"

st.markdown("---")

# ============================================
# QUERY INPUT
# ============================================

user_query = st.text_input(
    "Your Question:",
    value=st.session_state.get('query', ''),
    placeholder="e.g., Why is the app crashing?",
    help="Ask about app issues, bugs, or support"
)

# Clear session state
if 'query' in st.session_state:
    del st.session_state.query

# Submit button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit = st.button("🔍 Get Answer", type="primary", use_container_width=True)

# ============================================
# PROCESS QUERY
# ============================================

if submit:
    if not user_query:
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("🔄 Analyzing complaints and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": user_query,
                        "top_k": top_k,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display answer
                    st.markdown("### 💬 AI Response")
                    st.markdown(
                        f'<div class="answer-box">{data["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display sources
                    st.markdown("---")
                    st.markdown(f"### 📚 Related Complaints ({data['num_sources']})")
                    
                    for i, source in enumerate(data['sources'], 1):
                        relevance = abs(source.get('score', 0))
                        with st.expander(
                            f"📄 Complaint {i} - Relevance: {relevance:.2f}"
                        ):
                            st.markdown(f"**Source:** `{source['source']}`")
                            st.markdown(f"**ID:** `{source['id']}`")
                            st.markdown("**Details:**")
                            st.text(source['text'])
                
                elif response.status_code == 503:
                    st.error("❌ RAG system not initialized. Please wait or restart the API.")
                else:
                    st.error(f"❌ Error: {response.status_code}")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try again.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Powered by LLaMA 3.2 • FAISS • Sentence Transformers</p>
        <p style='font-size: 0.8rem;'>Customer Complaint RAG System v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)