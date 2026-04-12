from langchain_ollama import OllamaEmbeddings
from functools import lru_cache
from agentic_rag.config import get_settings


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    """Returns cached embedding model."""
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts.
    Returns list of vectors in same order as input.
    """
    embedder = get_embeddings()
    return embedder.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    embedder = get_embeddings()
    return embedder.embed_query(text)