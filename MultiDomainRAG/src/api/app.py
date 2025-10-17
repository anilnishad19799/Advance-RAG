# src/api/app.py
import os
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from langchain.docstore.document import Document
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.file_loader import FileLoader
from chunking.sentence_window_chunker import SentenceWindowChunker
from indexing.chromadb_indexer import ChromaDBIndexer
from indexing.bm25_indexer import BM25Indexer
from retriever.hybrid_retriever import HybridRetriever
from retriever.hybrid_adapter import HybridRetrieverAdapter
from reranking.cohererank import CohererankRetriever
from routing.semantic_router import SemanticRouterRetriever
from chain.react_chain import ReactAnswerGenerator
from langchain.embeddings import OpenAIEmbeddings


# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Advance RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["file://"] for local file
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TEXT_DIR = DATA_DIR / "texts"
# Path to templates
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

# ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def get_index():
    """Serve the index.html page."""
    return FileResponse(INDEX_HTML)


def save_uploaded_file_to_data(upload_file: UploadFile, target_path: Path) -> Path:
    """Save uploaded file to disk and return path."""
    with target_path.open("wb") as f:
        f.write(upload_file.file.read())
    return target_path


@app.post("/upload_and_index")
async def upload_and_index_file(
    dataset: Literal["medical", "legal", "other"] = Form(...),
    file: UploadFile = File(...),
    chunk_size: int = Form(3),
    chunk_overlap: int = Form(1),
):
    """
    Upload a PDF or TXT, extract text, chunk it, and build indexes automatically.
    """
    dataset = dataset.lower()
    raw_dataset_dir = RAW_DIR / dataset
    raw_dataset_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename).name
    raw_path = raw_dataset_dir / filename
    save_uploaded_file_to_data(file, raw_path)

    # Load and extract text
    loader = FileLoader(str(raw_path))
    try:
        docs = loader.load()
    except Exception as e:
        return JSONResponse({"error": f"Failed to load file: {e}"}, status_code=400)

    # Save combined text
    full_text = "\n\n".join([d.page_content for d in docs])
    text_name = f"{dataset}__{filename.rsplit('.',1)[0]}.txt"
    text_path = TEXT_DIR / text_name
    text_path.write_text(full_text, encoding="utf-8")

    # Chunking
    chunker = SentenceWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = chunker.chunk_documents(docs)

    # Prepare dataset directories
    ds_dir = DATA_DIR / "database" / dataset
    vector_db_dir = ds_dir / "vector_database"
    bm25_dir = ds_dir / "bm25"
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    # Build Chroma index
    chroma_indexer = ChromaDBIndexer(persist_directory=str(vector_db_dir))
    chroma_indexer.build_index(chunked_docs)

    # Build BM25 index
    bm25_indexer = BM25Indexer(persist_directory=str(bm25_dir), top_k=10)
    bm25_indexer.build_index(chunked_docs)

    return {
        "status": "saved_and_indexed",
        "dataset": dataset,
        "raw_path": str(raw_path),
        "text_path": str(text_path),
        "chunks": len(chunked_docs),
        "vector_db": str(vector_db_dir),
        "bm25": str(bm25_dir),
    }


@app.post("/query")
def query_rag(question: str = Form(...), top_k: int = Form(5)):
    """
    Query endpoint performing:
     - domain routing
     - hybrid retrieval (BM25 + Chroma)
     - Cohere rerank on top-k
     - LLM final answer generation
    """
    # Ensure API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)
    if not os.environ.get("CO_API_KEY"):
        return JSONResponse({"error": "CO_API_KEY not set"}, status_code=500)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Prepare paths for each dataset
    paths = {}
    for d in ["medical", "legal", "other"]:
        base = PROJECT_ROOT / "data" / "database" / d
        paths[d] = {"bm25": base / "bm25", "chroma": base / "vector_database"}

    # Create HybridRetriever instances
    hybrid_retrievers = {
        "medical": HybridRetriever(
            bm25_index_path=str(paths["medical"]["bm25"]),
            chroma_index_path=str(paths["medical"]["chroma"]),
            top_k=10,
        ),
        "legal": HybridRetriever(
            bm25_index_path=str(paths["legal"]["bm25"]),
            chroma_index_path=str(paths["legal"]["chroma"]),
            top_k=10,
        ),
        "other": HybridRetriever(
            bm25_index_path=str(paths["other"]["bm25"]),
            chroma_index_path=str(paths["other"]["chroma"]),
            top_k=10,
        ),
    }

    # Domain routing
    router = SemanticRouterRetriever(embeddings=embeddings)
    domain = router.route_domain(question)  # returns 'medical', 'legal', or 'other'

    chosen_hybrid = hybrid_retrievers.get(domain, hybrid_retrievers["other"])
    adapter = HybridRetrieverAdapter(chosen_hybrid)
    coherer = CohererankRetriever(base_retriever=adapter)
    reranked_docs = coherer.get_relevant_documents(question, top_k=top_k)

    docs_resp = []
    for d in reranked_docs:
        docs_resp.append({"text": d.page_content, "metadata": {"domain": domain}})

    # LLM final answer generation
    context = "\n\n".join([d.page_content for d in reranked_docs])
    react_chain = ReactAnswerGenerator()
    final_answer = react_chain.generate_answer(
        domain=domain, context=context, question=question
    )

    return {
        "domain": domain,
        "top_k": top_k,
        "reranked_count": len(reranked_docs),
        "reranked_docs": docs_resp,
        "final_answer": final_answer,
    }
