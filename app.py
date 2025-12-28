import streamlit as st
from rag_core import answer_query

st.set_page_config(page_title="Sanskrit RAG", layout="wide")
st.title("📜 Sanskrit RAG System")

query = st.text_input("Enter your Sanskrit question")

if st.button("Get Answer"):
    if query:
        answer, docs = answer_query(query)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Context")
        for d in docs:
            st.markdown(f"- {d}")
