# app.py
import streamlit as st
from embedding_reranker import load_vectorstore, retrieve_chunks, rerank_chunks, generate_answer

FILE_PATH = r"demo.txt"

st.set_page_config(page_title="RAG QA ", page_icon="📄")
st.title("RAG QA  📄")

vectordb = load_vectorstore(FILE_PATH)

query = st.text_input("Enter your question:")
initial_k = st.slider("Retrieve Top-K (from Chroma)", 1, 15, 5)
rerank_k = st.slider("Final Top-K (after reranking)", 1, 10, 3)

if query:
    st.subheader("📘 Before Reranking (Chroma Scores)")
    retrieved = retrieve_chunks(vectordb, query, top_k=initial_k)
    for i, (doc, score) in enumerate(retrieved):
        st.write(f"Chunk {i+1} — Chroma Score: {score:.4f}")
        st.write(doc.page_content[:250] + "...")

    st.subheader("📗 After Reranking")
    reranked = rerank_chunks(query, retrieved, rerank_top_k=rerank_k)
    for i, (doc, score) in enumerate(reranked):
        st.write(f"Chunk {i+1} — Reranker Score: {score:.4f}")
        st.write(doc.page_content[:250] + "...")

    st.subheader("🔹 Final Answer")
    answer = generate_answer(reranked, query, model_name="gpt-4o-mini")
    st.write(answer)
