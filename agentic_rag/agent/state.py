from typing import TypedDict


class AgentState(TypedDict):
    # Input
    user_id: str
    session_id: str
    question: str
    trace_id: str

    # Collection routing
    target_collection: str

    # Memory
    summary: str
    recent_messages: list[dict]

    # Retrieval
    retrieved_chunks: list[dict]    # top-20 from pgvector
    reranked_chunks: list[dict]     # top-4 after cross-encoder
    context: str                    # formatted for prompt
    sources: list[dict]             # for UI display

    # Cache
    cache_hit: bool

    # Output
    answer: str