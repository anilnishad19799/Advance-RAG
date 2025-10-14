# src/indexing/bm25_indexer.py

import pickle
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document


class BM25Indexer:
    def __init__(self, persist_directory: str = None, top_k: int = 3):
        """
        Args:
            persist_directory (str, optional): Path to store BM25 index.
            top_k (int): Number of top documents to retrieve.
        """
        # Compute project root dynamically (src folder is inside Project/)
        project_root = Path(__file__).parent.parent.parent

        # Default: save in project_root/data/bm25
        self.persist_directory = (
            Path(persist_directory)
            if persist_directory
            else project_root / "data" / "bm25"
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.top_k = top_k
        self.documents: List[Document] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: BM25Okapi = None
        self.index_file = self.persist_directory / "bm25_index.pkl"

    def build_index(self, docs: List[Document]):
        """
        Build BM25 index from chunked documents.

        Args:
            docs (List[Document]): Chunked documents to index.
        """
        self.documents = docs
        self.tokenized_docs = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self._save_index()
        print(f"✅ BM25 index saved at {self.index_file}")

    def _save_index(self):
        """Persist BM25 index to disk."""
        with open(self.index_file, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenized_docs": self.tokenized_docs,
                    "bm25": self.bm25,
                    "top_k": self.top_k,
                },
                f,
            )

    def load_index(self):
        """Load BM25 index from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"No BM25 index found at {self.index_file}")
        with open(self.index_file, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.tokenized_docs = data["tokenized_docs"]
        self.bm25 = data["bm25"]
        self.top_k = data["top_k"]
        print(f"✅ BM25 index loaded from {self.index_file}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve top-k documents for a query.

        Args:
            query (str): User query.

        Returns:
            List[Document]: Top-k relevant documents.
        """
        if self.bm25 is None:
            self.load_index()
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_docs = [
            doc
            for _, doc in sorted(
                zip(scores, self.documents), key=lambda x: x[0], reverse=True
            )
        ]
        return ranked_docs[: self.top_k]
