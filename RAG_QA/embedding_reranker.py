# main.py
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import CrossEncoder
import os

from dotenv import load_dotenv
load_dotenv()  

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# --------------------------------------------------------
# Load vectorstore
# --------------------------------------------------------
def load_vectorstore(file_path):
    loader = TextLoader("file_path", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    # Remove duplicates
    unique_texts = []
    seen = set()
    for doc in texts:
        if doc.page_content not in seen:
            unique_texts.append(doc)
            seen.add(doc.page_content)

    texts = unique_texts

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(texts, embeddings)
    return vectordb


# --------------------------------------------------------
# Retrieve chunks (before reranking)
# --------------------------------------------------------
def retrieve_chunks(vectordb, query, top_k):
    return vectordb.similarity_search_with_score(query, k=top_k)


# --------------------------------------------------------
# Rerank using BGE CrossEncoder
# --------------------------------------------------------
def rerank_chunks(query, retrieved_docs, rerank_top_k):
    reranker = CrossEncoder(
        model_name=BGE_RERANKER_MODEL,
        device="cpu"  # change to "cuda" if available
    )

    # Prepare query-doc pairs for scoring
    pairs = [(query, doc.page_content) for doc, _ in retrieved_docs]

    # CrossEncoder gives relevance scores
    rerank_scores = reranker.predict(pairs)

    # Attach scores & sort
    reranked = [
        (retrieved_docs[i][0], rerank_scores[i])  
        for i in range(len(rerank_scores))
    ]

    reranked.sort(key=lambda x: x[1], reverse=True)

    # Apply top_k after reranking
    return reranked[:rerank_top_k]


# --------------------------------------------------------
# Generate answer
# --------------------------------------------------------
#def generate_answer(reranked_docs, query, model_name):
 #   llm = ChatOpenAI(model=model_name)

  #  context = "\n".join([doc.page_content for doc, _ in reranked_docs])

  #  prompt = f"""
   # Use the context below to answer the question.

   # Context:
   # {context}

   # Question: {query}

   # Answer:
   # """

   # return llm(prompt)


def generate_answer(reranked_docs, query, model_name="gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)

    # unpack (doc, score)
    context = "\n\n".join(doc.page_content for doc, _ in reranked_docs)

    prompt = f"""
You are a QA assistant. Answer the question using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content