from langchain_ollama import ChatOllama
from ..config import get_settings


def get_llm() -> ChatOllama:
    """
    Returns the LLM — gemma4:e2b via Ollama.
    """
    settings = get_settings()

    return ChatOllama(
        model=settings.ollama_model,        # gemma4:e2b
        base_url=settings.ollama_base_url,  # http://localhost:11434
        temperature=0.1,    # low = more factual, less creative. Good for RAG
        num_ctx=8192,       # context window to use. gemma4 supports 128K but 8K is enough for RAG
    )