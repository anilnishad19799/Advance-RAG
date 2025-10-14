# 🧠 MultiDomainRAG

**MultiDomainRAG** is an Advanced Retrieval-Augmented Generation (RAG) pipeline that supports multiple domains — including *medical*, *sports*, and *general* text.  
It integrates document advance chunking, indexing, hybrid retrieval, reranking, and semantic routing to provide accurate, context-aware answers from diverse data sources.

---

## 📂 Folder Structure

```
|---.gitignore
|---README.md
|   
+---data
|   +---database - storing all type like medical, legal, other domain vectordb and exact match word data here
|   +---raw - original pdf or text save here
|   +---texts - convesion of pdf to text save here
+---MultiDomainRAG
|   |   .env
|   |   requirements.txt
|   |   
|   +---frontend
|   |       index.html
|   |       
|   \---src
|       +---api
|       |   |   app.py
|       |   |   main.py
|       |   |   __init__.py
|       |
|       +---chain
|       |   |   react_chain.py
|       |
|       +---chunking
|       |   |   sentence_window_chunker.py
|       |
|       +---documents
|       |       medical.txt
|       |       sample.txt
|       |       sport.txt
|       |
|       +---indexing
|       |   |   bm25_indexer.py
|       |   |   chromadb_indexer.py
|       |
|       +---reranking
|       |   |   cohererank.py
|       |
|       +---retriever
|       |   |   hybrid_adapter.py
|       |   |   hybrid_retriever.py
|       |
|       +---routing
|       |   |   semantic_router.py
|       |
|       \---utils
|           |   file_loader.py
|           |   __init__.py
|
\---notebook
    |   advance_retrieval.ipynb - sentence window retrieval and auto merging retrieval
    |   chunking.ipynb - all chunking technique
    |   example.db - database for working with milvus and weaviate
    |   indexing.ipynb - all vector database like faiss, chromadb, pinecone, milvus, weaviate
    |   query_construction.ipynb - includes vectordb, graphdb, relationaldb queries
    |   query_translation.ipynb - multi-query, RAG-fusion
    |   reranking.ipynb - Cohererank
    |   retrieval.ipynb - exact match retrieval, embedding-based retrieval, hybrid retrieval
    |   routing.ipynb - logical and semantic routing
```

---

## 🚀 Topic — **Advanced RAG**

The **MultiDomainRAG** project implements advanced RAG concepts and techniques.

### 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **PyMuPDFLoader** | PDF processing |
| **Sentence Window Chunking** | Context-preserving document chunking |
| **Indexing** | ChromaDB + BM25 Encoder |
| **Retrieval** | Hybrid Retriever (semantic + lexical) |
| **Reranking** | Cohererank for coherence-based ordering |
| **Routing** | Semantic routing across domains |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd MultiDomainRAG
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file in the `MultiDomainRAG` directory (if required):
```bash
OPENAI_API_KEY=your_openai_api_key_here
CO_API_KEY=your_cohererank_api_key_here
```

---

## ▶️ Running the Project

### Step 1 — Start the Backend Server
Go to the `MultiDomainRAG` folder and run:
```bash
uvicorn src.api.app:app --reload
```

- The FastAPI server will start at:  
  👉 http://127.0.0.1:8000  
- (Optional) API documentation:  
  👉 http://127.0.0.1:8000/docs

---

### Step 2 — Open the Frontend

1. Go to the folder:  
   `MultiDomainRAG/frontend`
2. Double-click on **index.html** to open it in your browser.  
3. It will open a **chat-based RAG application**.

---

### Step 3 — Interact with the App

In the chat interface:
1. **Upload documents and index document** (PDF or text documents will be chunked, encoded & stored)  
3. **Ask questions** related to the uploaded documents  
4. The app will:
   - There are three vector database for each domain - medical, legal, other
   - Route query to relevant domain  
   - Retrieve relevant chunks using **Hybrid Retriever**  
   - Rerank results using **Cohererank**  
   - Generate accurate, context-aware answers  

---

## 📘 Notebooks Overview

All notebooks are available in the `notebook` folder — each covers a different advanced RAG concept.

| Notebook | Description |
|-----------|-------------|
| **advance_retrieval.ipynb** | Sentence-window retrieval & auto-merging retrieval |
| **chunking.ipynb** | Different chunking techniques like fixed, recursivecharactersplitter etc. |
| **indexing.ipynb** | Indexing with FAISS, ChromaDB, Pinecone, Milvus, Weaviate |
| **retrieval.ipynb** | Exact match retrieval, embedding-based retrieval, hybrid retrieval |
| **reranking.ipynb** | Reranking using Cohererank |
| **routing.ipynb** | Logical & semantic routing |
| **query_construction.ipynb** | Query creation for vector, graph, and relational DBs |
| **query_translation.ipynb** | Multi-query, RAG-fusion |
| **example.db** | Example database for testing Milvus/Weaviate setups |

To open:
```bash
jupyter notebook
```

---

## 🔍 How It Works (High-Level)

1. **Document Loading** — PDFs or text files processed via `PyMuPDFLoader`.
2. **Chunking** — Split using **Sentence Window Chunking**.
3. **Indexing** —  
   - **Vector Indexing:** ChromaDB / FAISS  
   - **Lexical Indexing:** BM25 Encoder  
4. **Retrieval** — Combined lexical + semantic retrieval using Hybrid Retriever.
5. **Reranking** — Refined with **Cohererank**.
6. **Routing** — Queries dynamically routed with **Semantic Router**.
7. **Frontend** — Simple UI (`index.html`) to interact with the RAG system.

---

## 💡 Example Commands

| Action | Command |
|--------|----------|
| Start server | `uvicorn src.api.app:app --reload` |
| Install dependencies | `pip install -r requirements.txt` |
| Run notebooks | `jupyter notebook` |

---

## 🧰 Troubleshooting

| Issue | Solution |
|--------|-----------|
| Server not starting | Check if port 8000 is free, or use `--port 8080` |
| CORS / request error | Run frontend via local server → `python -m http.server 8001` |
| Dependencies missing | Run `pip install -r requirements.txt` again |
| Vector DB connection error | Check `.env` file configuration |
| Large PDF handling | Adjust chunk size & overlap in `sentence_window_chunker.py` |

---

## 🧑‍💻 Contributing

Contributions are welcome!  
1. Fork the repository  
2. Create a feature branch  
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes  
   ```bash
   git commit -m "Added new feature"
   ```
4. Push and create a Pull Request  

---


## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.
