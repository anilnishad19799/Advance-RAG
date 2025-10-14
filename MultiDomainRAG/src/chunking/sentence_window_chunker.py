from langchain.docstore.document import Document
from typing import List
import re


class SentenceWindowChunker:
    """
    Sentence-window chunking: true sliding window by sentences.
    """

    def __init__(self, chunk_size: int = 3, chunk_overlap: int = 1):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        chunks = []

        for doc in docs:
            # Split text into sentences
            sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
            sentences = [s for s in sentences if s]

            start = 0
            while start < len(sentences):
                end = start + self.chunk_size
                chunk_sentences = sentences[start:end]
                chunks.append(Document(page_content=" ".join(chunk_sentences)))
                start += self.chunk_size - self.chunk_overlap
                if start < 0:  # safety
                    start = 0

        return chunks
