import streamlit as st
from rag_core import answer_query

st.set_page_config(
    page_title="Sanskrit RAG System",
    layout="wide"
)

st.title("📜 Sanskrit Document Question Answering (RAG)")
st.markdown(
    "Ask questions based on the provided Sanskrit documents."
)

query = st.text_input(
    "Enter your Sanskrit question:",
    placeholder="उदाहरणः – घण्टाकर्णः कः आसीत् ?"
)

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Retrieving answer..."):
            answer, docs = answer_query(query)

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📚 Retrieved Context")
        for d in docs:
            st.markdown(f"- {d}")
    else:
        st.warning("Please enter a question.")
