# src/reranking/cohererank.py

import os
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.docstore.document import Document
from typing import List


class CohererankRetriever:
    """
    Wraps a base retriever and reranks retrieved documents using CohereRerank.
    """

    def __init__(self, base_retriever, cohere_model: str = "rerank-english-v3.0"):
        """
        Args:
            base_retriever: Any retriever object (Chroma, FAISS, Hybrid)
            cohere_model (str): Cohere rerank model name
        """
        if not os.environ.get("CO_API_KEY"):
            raise ValueError("Please set CO_API_KEY environment variable.")

        self.base_retriever = base_retriever
        self.cohere_model = cohere_model
        self.compressor = CohereRerank(model=self.cohere_model)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever,
            top_k=10,
        )

    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve and rerank top documents.

        Args:
            query (str): Input query
            top_k (int): Number of top documents to return

        Returns:
            List[Document]: Reranked top documents
        """
        # Use the same method as your snippet: invoke
        docs = self.compression_retriever.invoke(query)

        print("docs", len(docs))
        print("top k", top_k)
        return docs[:top_k]
