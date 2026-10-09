
import os
import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text:latest",
)


class EmbeddingService:
    """Generate local embeddings using Ollama."""

    def __init__(self):
        self.base_url = OLLAMA_URL.rstrip("/")
        self.model = EMBEDDING_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Provide a non-empty list of text documents.")

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        embeddings = response.json().get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response.")

        if any(not isinstance(v, list) or not v for v in embeddings):
            raise RuntimeError("Ollama returned an invalid embedding vector.")

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
