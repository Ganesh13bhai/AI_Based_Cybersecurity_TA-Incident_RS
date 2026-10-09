
import math

from typing import Any

from rag.document_ingestion import DocumentIngestion
from rag.embedding_service import EmbeddingService


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match.")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class RetrievalService:
    """Retrieve relevant cybersecurity documents using local embeddings."""

    def __init__(self):
        self.ingestion = DocumentIngestion()
        self.embedding = EmbeddingService()

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty retrieval query is required.")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        documents = self.ingestion.list_documents()
        if not documents:
            return []

        query_embedding = self.embedding.embed_query(query)

        document_embeddings = self.embedding.embed_documents(
            [document["text"] for document in documents]
        )

        results = []

        for document, vector in zip(documents, document_embeddings):
            score = cosine_similarity(query_embedding, vector)
            results.append({
                "document_id": document["document_id"],
                "title": document["title"],
                "source": document["source"],
                "text": document["text"],
                "metadata": document.get("metadata", {}),
                "similarity": round(score, 6),
            })

        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:top_k]
