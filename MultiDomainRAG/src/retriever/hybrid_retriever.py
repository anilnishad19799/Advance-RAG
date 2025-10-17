# src/retriever/hybrid_retriever.py

from typing import List
from langchain.docstore.document import Document
from indexing.bm25_indexer import BM25Indexer
from indexing.chromadb_indexer import ChromaDBIndexer


class HybridRetriever:
    """
    Hybrid retriever combining BM25 + ChromaDB embeddings.
    Supports multiple queries and returns deduplicated documents.
    """

    def __init__(
        self,
        bm25_index_path: str = None,
        chroma_index_path: str = None,
        top_k: int = 3,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_index_path (str, optional): Directory for BM25 persistent index.
            chroma_index_path (str, optional): Directory for ChromaDB persistent index.
            top_k (int): Number of documents to retrieve from each retriever.
        """
        self.bm25_indexer = BM25Indexer(persist_directory=bm25_index_path, top_k=top_k)
        self.chromadb_indexer = ChromaDBIndexer(persist_directory=chroma_index_path)
        self.top_k = top_k

    def get_relevant_documents(self, queries: List[str]) -> List[Document]:
        """
        Retrieve documents from both BM25 and ChromaDB for multiple queries.

        Args:
            queries (List[str]): List of queries.

        Returns:
            List[Document]: Deduplicated list of relevant documents.
        """
        all_docs = []

        for query in queries:
            # BM25 retrieval
            bm25_docs = self.bm25_indexer.retrieve(query)
            all_docs.extend(bm25_docs)

            # ChromaDB retrieval
            chroma_retriever = self.chromadb_indexer.as_retriever(k=self.top_k)
            chroma_docs = chroma_retriever.get_relevant_documents(query)
            all_docs.extend(chroma_docs)

        # Deduplicate while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs
