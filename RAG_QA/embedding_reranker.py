#importss
import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import CrossEncoder


from dotenv import load_dotenv
load_dotenv()  

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# Loading vectors

def load_vectorstore(file_path):
    loader = TextLoader("file_path", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    
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



# Retrieve chunks (before reranking)
def retrieve_chunks(vectordb, query, top_k):
    return vectordb.similarity_search_with_score(query, k=top_k)



# Reranking  using BGE CrossEncoder

def rerank_chunks(query, retrieved_docs, rerank_top_k):
    reranker = CrossEncoder(
        model_name=BGE_RERANKER_MODEL,
        device="cpu"  
    )

    pairs = [(query, doc.page_content) for doc, _ in retrieved_docs]

   
    rerank_scores = reranker.predict(pairs)

    reranked = [
        (retrieved_docs[i][0], rerank_scores[i])  
        for i in range(len(rerank_scores))
    ]

    reranked.sort(key=lambda x: x[1], reverse=True)

    # Apply top_k after reranking
    return reranked[:rerank_top_k]


   # Question: {query}

   # Answer:
   # """

   # return llm(prompt)


def generate_answer(reranked_docs, query, model_name="gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)

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
