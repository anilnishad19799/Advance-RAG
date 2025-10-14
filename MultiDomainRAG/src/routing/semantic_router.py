# src/routing/semantic_router.py

import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.utils.math import cosine_similarity
from langchain.docstore.document import Document


class SemanticRouterRetriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.domain_prompts = {
            "medical": (
                "Medical domain: diseases, symptoms, treatments, health conditions, "
                "diagnoses, doctors' notes, clinical studies, patient care, medications, "
                "medical procedures, medical terminology."
            ),
            "legal": (
                "Legal domain: case law, court judgments, statutes, legal contracts, "
                "regulations, legal procedures, court filings, legal arguments, "
                "law terminology, rights and obligations."
            ),
            "other": (
                "Other domain: general knowledge, sports, technology, finance, entertainment, "
                "history, arts, or any topic that does not fall under medical or legal domains."
            ),
        }
        self.domain_embeddings = {
            domain: self.embeddings.embed_query(text)
            for domain, text in self.domain_prompts.items()
        }

    def route_domain(self, query: str) -> str:
        """Return 'medical' or 'sport' based on query similarity."""
        query_emb = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
        scores = {}
        for domain, dom_emb in self.domain_embeddings.items():
            dom_emb_2d = np.array(dom_emb).reshape(1, -1)
            scores[domain] = cosine_similarity(query_emb, dom_emb_2d)[0][0]
        best_domain = max(scores, key=scores.get)
        return best_domain
