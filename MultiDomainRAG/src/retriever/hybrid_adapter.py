# src/retriever/hybrid_adapter.py

from langchain.schema import BaseRetriever, Document
from typing import List
from retriever.hybrid_retriever import HybridRetriever
from pydantic import PrivateAttr


class HybridRetrieverAdapter(BaseRetriever):
    """
    Adapter to make HybridRetriever compatible with LangChain retrievers.
    Converts single query input to a list for HybridRetriever.
    """

    # Use PrivateAttr to store non-validated attributes
    _hybrid_retriever: HybridRetriever = PrivateAttr()

    def __init__(self, hybrid_retriever: HybridRetriever):
        super().__init__()
        self._hybrid_retriever = hybrid_retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Wrap single string query into list to call HybridRetriever.
        """
        return self._hybrid_retriever.get_relevant_documents([query])
