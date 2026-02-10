import streamlit as st

# Import your RAG pipeline function
from rag.pipeline import run_pipeline


# ----------------------------------------
# Streamlit Page Config
# ----------------------------------------
st.set_page_config(
    page_title="Customer Complaint Root-Cause Analyzer",
    layout="centered"
)

st.title("📊 Customer Complaint Root-Cause Analyzer")
st.write(
    "Analyze customer complaints to identify recurring issues, root causes, "
    "and trends before and after product updates."
)

# ----------------------------------------
# User Input
# ----------------------------------------
query = st.text_input(
    "Ask a business question (example: What are the top recurring complaints?)"
)

# ----------------------------------------
# Run RAG Pipeline
# ----------------------------------------
if query:
    with st.spinner("Analyzing complaints..."):
        try:
            answer = run_pipeline(query)

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error("Something went wrong while processing your request.")
            st.exception(e)

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.markdown(
    "Built with Retrieval-Augmented Generation (RAG) using open-source models."
)
