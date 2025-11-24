# 📄 RAG QA System

A Retrieval-Augmented Generation (RAG) based Question-Answering system that loads a text document, chunks it, retrieves relevant pieces using embeddings, **reranks them using a reranker (BAAI/bge-reranker-v2-m3)**, and finally generates an answer using OpenAI’s ChatGPT models.  
The project also includes an interactive **Streamlit UI** for testing queries based on text.

---

## Prerequisites

- Python 3.12 or higher  
- Conda (recommended)  
- OpenAI API key

---

## Installation

1. **Clone the repository**
```bash
git clone <repository_url>

```

2. **Create virtual environment**
```bash
conda create -p venv python=3.12 -y
```
(Any other method to create a Python environment can also be used.)

3. **Activate environment**
```bash
conda activate ./venv
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Create `.env` file** in project root:
```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=groq_api_key
```

---

## Usage

###  Run the backend RAG logic (no UI)
```bash
conda activate ./venv
python main.py
```

### Run the Streamlit Interface
```bash
conda activate ./venv
streamlit run app.py
```

### How to Interact

1. Type your question in the Streamlit text box  
2. See:
   - Chunks retrieved from Chroma *before reranking*
   - Chunks *after reranking* using BGE CrossEncoder
3. View the **final LLM answer** generated using ChatOpenAI  
4. Transparency: All chunks used are displayed to the user

---

## Example Query

**Query:**  
```
When did the delhi blast happen?
```

You will see:
- Top-k chunks retrieved via Chroma
- Top-k chunks after reranking
- Final answer produced by GPT-4o-mini

---

## Features

- Load and split long text documents into semantic chunks  
- Embedding generation using **OpenAI text-embedding-3-small**  
- Vector search powered by **Chroma**  
- Reranking of retrieved chunks using:  
  ```
  BAAI/bge-reranker-v2-m3
  ```
- Final answer generation with **ChatOpenAI**
- Clean and simple **Streamlit UI**

---

## Workflow

1. **Document Loading**
   - A `.txt` file is loaded and chunked using `RecursiveCharacterTextSplitter`

2. **Embedding + Vector Indexing**
   - Each chunk is embedded using OpenAI embeddings  
   - Chroma stores vectors and performs similarity search

3. **Initial Retrieval**
   - Top-k most relevant chunks retrieved using vector similarity

4. **Reranking**
   - CrossEncoder computes a *true relevance score* for each chunk  
   - Chunks are sorted again based on reranker scores  
   - Final top-k chosen

5. **LLM Answer Generation**
   - Reranked chunks are passed as context to ChatOpenAI  
   - The final natural-language answer is generated

6. **User Interface**
   - Streamlit UI displays everything:  
     - Retrieved chunks  
     - Reranked chunks  
     - Final answer  

---



## License

This project is licensed under the terms included in the LICENSE file.

---

## Author

**Anjali Bheemireddy**  
(anjalinature156@gmail.com)

