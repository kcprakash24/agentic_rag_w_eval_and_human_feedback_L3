from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "AgenticRAG"
    log_level: str = "INFO"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"

    # Embeddings
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "docmind"
    postgres_user: str = "docmind"
    postgres_password: str = "docmind"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 3600
    redis_similarity_threshold: float = 0.15

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    # Memory
    memory_last_n_messages: int = 6
    memory_summarize_after: int = 10

    # Retrieval
    retrieval_top_k: int = 20        # candidates from pgvector
    rerank_top_n: int = 4            # final chunks after reranking

    # Self-RAG
    max_retrieval_attempts: int = 2  # prevent infinite loops

    # Evaluation
    eval_judge_model: str = "gemma4:e2b"   # LLM used as judge
    golden_dataset_name: str = "rag-golden-dataset"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()