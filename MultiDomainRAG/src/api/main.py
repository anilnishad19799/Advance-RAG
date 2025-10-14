# src/api/main.py

import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # make src visible

from src.utils.file_loader import FileLoader
from src.chunking.sentence_window_chunker import (
    SentenceWindowChunker,
)
from src.indexing.chromadb_indexer import ChromaDBIndexer
from src.indexing.bm25_indexer import BM25Indexer
from src.query_translation.multiquery_generator import MultiQueryGenerator
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranking.cohererank import CohererankRetriever
from langchain.embeddings import OpenAIEmbeddings
from src.retriever.hybrid_adapter import HybridRetrieverAdapter
from dotenv import load_dotenv


# Get the path to the .env file (one level up)
env_path = Path(__file__).resolve().parent.parent / ".env"

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Path to any PDF or TXT
medical_file_path = "C:/Users/aniln/Desktop/github_celery_redis/Advance_RAG2/Project/src/documents/medical.txt"
sport_file_path = "C:/Users/aniln/Desktop/github_celery_redis/Advance_RAG2/Project/src/documents/sport.txt"

medical_loader = FileLoader(medical_file_path)
medical_docs = medical_loader.load()

sport_loader = FileLoader(sport_file_path)
sport_docs = sport_loader.load()

# os.environ["CO_API_KEY"] = (
#     "WeCRCffL5Mf9JzbsNdnuFI5xyxO4ahygB5Z3HoGF"  # replace with your key
# )

# -------------------------------
# 2️⃣ Chunk Documents (Sentence Window)
# # -------------------------------
medical_chunker = SentenceWindowChunker(chunk_size=3, chunk_overlap=1)
medical_chunked_docs = medical_chunker.chunk_documents(medical_docs)

sport_chunker = SentenceWindowChunker(chunk_size=3, chunk_overlap=1)
sport_chunked_docs = sport_chunker.chunk_documents(sport_docs)

# # # semantic routing
from src.routing.semantic_router import SemanticRouterRetriever
from src.indexing.chromadb_indexer import ChromaDBIndexer

# # Suppose these are your prebuilt ChromaDB indexes
project_root = Path(__file__).parent.parent.parent
medical_chormadb_path = "data/medical/vector_database"
medical_vector_db_path = project_root / medical_chormadb_path
chroma_medical_index = ChromaDBIndexer(persist_directory=medical_vector_db_path)
medical_bm25_path = project_root / "data/medical/bm25"
medical_bm25 = BM25Indexer(persist_directory=medical_bm25_path, top_k=3)

sport_chormadb_path = "data/sport/vector_database"
sport_vector_db_path = project_root / sport_chormadb_path
chroma_sport_index = ChromaDBIndexer(persist_directory=sport_vector_db_path)
sport_bm25_path = project_root / "data/sport/bm25"
sport_bm25 = BM25Indexer(persist_directory=sport_bm25_path, top_k=3)

# Pass retrievers to semantic router
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
semantic_router = SemanticRouterRetriever(
    embeddings=embeddings,
)

medical_retriever = HybridRetriever(
    bm25_index_path=str(medical_bm25_path),
    chroma_index_path=str(medical_chormadb_path),
    top_k=10,
)
sport_retriever = HybridRetriever(
    bm25_index_path=str(sport_bm25_path),
    chroma_index_path=str(sport_chormadb_path),
    top_k=10,
)

query = "Explain all disease related topic"
domain = semantic_router.route_domain(query)

if domain == "medical":
    print("domain", domain)
    medical_adapter = HybridRetrieverAdapter(medical_retriever)
    medical_cohere_retriever = CohererankRetriever(base_retriever=medical_adapter)
    medical_top_docs = medical_cohere_retriever.get_relevant_documents(query, top_k=5)
    print("medical_top_docs", medical_top_docs)
else:
    print("domain", domain)
    sport_adapter = HybridRetrieverAdapter(sport_retriever)
    sport_cohere_retriever = CohererankRetriever(base_retriever=sport_adapter)
    sport_top_docs = sport_cohere_retriever.get_relevant_documents(query, top_k=5)
    print("sport_top_docs", sport_top_docs)
