# src/api/app.py
import os
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain.docstore.document import Document
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware


# make src visible if running module directly
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.file_loader import FileLoader
from src.chunking.sentence_window_chunker import SentenceWindowChunker
from src.indexing.chromadb_indexer import ChromaDBIndexer
from src.indexing.bm25_indexer import BM25Indexer
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.hybrid_adapter import HybridRetrieverAdapter
from src.reranking.cohererank import CohererankRetriever
from src.routing.semantic_router import SemanticRouterRetriever
from src.chain.react_chain import ReactAnswerGenerator
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
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

# ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


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


from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


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

    # Build response documents
    # docs_resp = [
    #     {"text": d.page_content, "metadata": getattr(d, "metadata", {})}
    #     for d in reranked_docs
    # ]

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


# # src/api/app.py
# import os
# from pathlib import Path
# from typing import List
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from langchain.docstore.document import Document

# # make src visible if running module directly
# import sys

# sys.path.append(str(Path(__file__).parent.parent))

# from src.utils.file_loader import FileLoader
# from src.chunking.sentence_window_chunker import (
#     SentenceWindowChunker,
# )
# from typing import Literal
# from src.indexing.chromadb_indexer import ChromaDBIndexer
# from src.indexing.bm25_indexer import BM25Indexer
# from src.retriever.hybrid_retriever import HybridRetriever
# from src.retriever.hybrid_adapter import HybridRetrieverAdapter
# from src.reranking.cohererank import CohererankRetriever
# from src.routing.semantic_router import SemanticRouterRetriever
# from src.chain.react_chain import ReactAnswerGenerator
# from langchain.embeddings import OpenAIEmbeddings
# from dotenv import load_dotenv

# # Get the path to the .env file (one level up)
# env_path = Path(__file__).resolve().parent.parent.parent / ".env"
# # Load the .env file
# load_dotenv(dotenv_path=env_path)

# app = FastAPI(title="Advance RAG API")

# PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# print("PROJECT_ROOT", PROJECT_ROOT)
# DATA_DIR = PROJECT_ROOT / "data"
# RAW_DIR = DATA_DIR / "raw"
# TEXT_DIR = DATA_DIR / "texts"

# # ensure directories exist
# RAW_DIR.mkdir(parents=True, exist_ok=True)
# TEXT_DIR.mkdir(parents=True, exist_ok=True)


# def save_uploaded_file_to_data(upload_file: UploadFile, target_path: Path) -> Path:
#     """
#     Save uploaded file to disk and return path.
#     """
#     with target_path.open("wb") as f:
#         content = upload_file.file.read()
#         f.write(content)
#     return target_path


# @app.post("/upload")
# async def upload_file(
#     dataset: Literal["medical", "legal", "other"] = Form(...),
#     file: UploadFile = File(...),
# ):
#     """
#     Upload a PDF or TXT. Save raw file into data/raw/<dataset>/<filename>
#     Extract text using existing FileLoader and save to data/texts/<dataset>_<filename>.txt.
#     """
#     dataset = dataset.lower()
#     raw_dataset_dir = RAW_DIR / dataset
#     raw_dataset_dir.mkdir(parents=True, exist_ok=True)

#     filename = Path(file.filename).name
#     raw_path = raw_dataset_dir / filename
#     save_uploaded_file_to_data(file, raw_path)

#     # Use FileLoader to load and extract text
#     loader = FileLoader(str(raw_path))
#     try:
#         docs = loader.load()
#     except Exception as e:
#         return JSONResponse({"error": f"Failed to load file: {e}"}, status_code=400)

#     # Combine text and save
#     full_text = "\n\n".join([d.page_content for d in docs])
#     text_name = f"{dataset}__{filename.rsplit('.',1)[0]}.txt"
#     text_path = TEXT_DIR / text_name
#     text_path.write_text(full_text, encoding="utf-8")

#     return {"status": "saved", "raw_path": str(raw_path), "text_path": str(text_path)}


# @app.post("/index")
# def build_index(
#     dataset: Literal["medical", "legal", "other"] = Form(...),
#     chunk_size: int = Form(3),
#     chunk_overlap: int = Form(1),
# ):
#     """
#     Build or rebuild indexes for the given dataset.
#     - chunks text files under data/texts that match the dataset prefix
#     - persist ChromaDB at data/<dataset>/vector_database
#     - persist BM25 at data/<dataset>/bm25
#     """
#     dataset = dataset.lower()
#     # find text files for dataset
#     text_files = list(TEXT_DIR.glob(f"{dataset}__*.txt"))
#     if not text_files:
#         return JSONResponse(
#             {
#                 "error": f"No uploaded text files found for dataset '{dataset}' in {TEXT_DIR}"
#             },
#             status_code=400,
#         )

#     # Read all files and form documents (langchain Document)

#     docs = []
#     for t in text_files:
#         txt = t.read_text(encoding="utf-8")
#         docs.append(Document(page_content=txt, metadata={"source": str(t.name)}))

#     # Chunk
#     chunker = SentenceWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunked_docs = chunker.chunk_documents(docs)

#     # prepare dataset directories
#     ds_dir = DATA_DIR / "database" / dataset
#     vector_db_dir = ds_dir / "vector_database"
#     bm25_dir = ds_dir / "bm25"
#     vector_db_dir.mkdir(parents=True, exist_ok=True)
#     bm25_dir.mkdir(parents=True, exist_ok=True)

#     # Build Chroma index
#     chroma_indexer = ChromaDBIndexer(persist_directory=str(vector_db_dir))
#     chroma_indexer.build_index(chunked_docs)

#     # Build BM25 index
#     bm25_indexer = BM25Indexer(persist_directory=str(bm25_dir), top_k=10)
#     bm25_indexer.build_index(chunked_docs)

#     return {
#         "status": "indexed",
#         "dataset": dataset,
#         "chunks": len(chunked_docs),
#         "vector_db": str(vector_db_dir),
#         "bm25": str(bm25_dir),
#     }


# @app.post("/query")
# def query_rag(question: str = Form(...), top_k: int = Form(5)):
#     """
#     Query endpoint performing:
#      - semantic routing between 'medical' and 'sport'
#      - hybrid retrieval (BM25 + Chroma) via HybridRetriever
#      - Cohere rerank on top-k
#      - auto-merge results into a single context
#     Returns JSON with domain, top documents (text + metadata) and merged context.
#     """
#     # Ensure API keys set externally:
#     if not os.environ.get("OPENAI_API_KEY"):
#         return JSONResponse(
#             {"error": "OPENAI_API_KEY not set in environment"}, status_code=500
#         )
#     if not os.environ.get("CO_API_KEY"):
#         return JSONResponse(
#             {"error": "CO_API_KEY not set in environment"}, status_code=500
#         )

#     # prepare embeddings (shared)
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#     # Build HybridRetrievers for both datasets (they expect persisted indexes exist)
#     project_root = PROJECT_ROOT
#     medical_bm25_path = project_root / "data" / "database" / "medical" / "bm25"
#     medical_chroma_path = (
#         project_root / "data" / "database" / "medical" / "vector_database"
#     )
#     legal_bm25_path = project_root / "data" / "database" / "legal" / "bm25"
#     legal_chroma_path = project_root / "data" / "database" / "legal" / "vector_database"
#     other_bm25_path = project_root / "data" / "database" / "other" / "bm25"
#     other_chroma_path = project_root / "data" / "database" / "other" / "vector_database"

#     # Create hybrid retriever instances (these instantiate BM25Indexer/ChromaDBIndexer internally)
#     medical_hybrid = HybridRetriever(
#         bm25_index_path=str(medical_bm25_path),
#         chroma_index_path=str(medical_chroma_path),
#         top_k=10,
#     )
#     legal_hybrid = HybridRetriever(
#         bm25_index_path=str(legal_bm25_path),
#         chroma_index_path=str(legal_chroma_path),
#         top_k=10,
#     )

#     other_hybrid = HybridRetriever(
#         bm25_index_path=str(other_bm25_path),
#         chroma_index_path=str(other_chroma_path),
#         top_k=10,
#     )

#     # Router focused only on domain detection (returns 'medical' or 'sport')
#     router = SemanticRouterRetriever(embeddings=embeddings)
#     domain = router.route_domain(question)  # route_domain returns the domain key only

#     # pick corresponding hybrid retriever and adapter
#     if domain == "medical":
#         chosen_hybrid = medical_hybrid
#     elif domain == "legal":
#         chosen_hybrid = legal_hybrid
#     else:
#         chosen_hybrid = other_hybrid

#     adapter = HybridRetrieverAdapter(chosen_hybrid)
#     coherer = CohererankRetriever(base_retriever=adapter)
#     reranked_docs = coherer.get_relevant_documents(question, top_k=top_k)

#     # build response
#     docs_resp = []
#     for d in reranked_docs:
#         docs_resp.append(
#             {"text": d.page_content, "metadata": getattr(d, "metadata", {})}
#         )

#     # ================================
#     # 🧩 LLM FINAL ANSWER GENERATION
#     # ================================
#     context = "\n\n".join([d.page_content for d in reranked_docs])
#     react_chain = ReactAnswerGenerator()
#     final_answer = react_chain.generate_answer(
#         domain=domain, context=context, question=question
#     )

#     return {
#         "domain": domain,
#         "top_k": top_k,
#         "reranked_count": len(reranked_docs),
#         "reranked_docs": docs_resp,
#         "final_answer": final_answer,
#     }
