# src/indexing/chromadb_indexer.py

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path
from typing import List
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"

print("*" * 100)
print("env_path", env_path)
load_dotenv(dotenv_path=env_path)


class ChromaDBIndexer:
    """
    ChromaDB indexer for storing and retrieving chunked documents.
    """

    def __init__(self, persist_directory: str = "data/vector_database"):
        """
        Args:
            persist_directory (str): Path to store the ChromaDB database.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def build_index(self, docs: List[Document]):
        """
        Build a ChromaDB index from documents.

        Args:
            docs (List[Document]): Chunked documents to index.
        """
        self.vectorstore = Chroma.from_documents(
            docs,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
        )
        self.vectorstore.persist()
        print(f"✅ ChromaDB index persisted at {self.persist_directory}")

    def as_retriever(self, search_type: str = "similarity", k: int = 3):
        """
        Return a retriever for querying the vector store.

        Args:
            search_type (str): Type of search ('similarity' or 'mmr').
            k (int): Number of documents to retrieve.

        Returns:
            retriever object
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
            )
        return self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )
