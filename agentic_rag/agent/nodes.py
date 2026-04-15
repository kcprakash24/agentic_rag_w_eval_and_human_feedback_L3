import uuid
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agentic_rag.agent.state import AgentState
from agentic_rag.config import get_settings
from agentic_rag.embeddings.embedder import embed_query
from agentic_rag.retrieval.retriever import retrieve
from agentic_rag.retrieval.reranker import rerank
from agentic_rag.memory.pg_memory import (
    save_message,
    should_summarize,
    get_recent_messages,
    get_latest_summary,
)
from agentic_rag.memory.summarizer import summarize_and_compress
from agentic_rag.cache.redis_cache import cache_lookup, cache_store
from agentic_rag.ingestion.collection_router import route_document
from agentic_rag.llm.provider import get_llm
from agentic_rag.observability.langfuse_client import (
    get_langfuse_handler,
    trace_node,
)

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research assistant. Answer the question using ONLY the context provided.
If the context does not contain enough information, say "I don't have enough context to answer this."
Do not use prior knowledge. Always cite the chunk ID that supports your answer.

{summary_section}

{history_section}

Retrieved Context:
{context}

Question: {question}

Answer:
""")


def _format_summary_section(summary: str) -> str:
    if not summary:
        return ""
    return f"Conversation Summary (older context):\n{summary}"


def _format_history_section(recent_messages: list[dict]) -> str:
    if not recent_messages:
        return ""
    lines = []
    for m in recent_messages:
        role = "User" if m["role"] == "human" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "Recent Conversation:\n" + "\n".join(lines)


def load_memory(state: AgentState) -> AgentState:
    """Node 1: Generate trace ID + load memory from PostgreSQL."""
    trace_id = uuid.uuid4().hex
    user_id = state["user_id"]
    session_id = state["session_id"]

    summary = get_latest_summary(user_id, session_id) or ""
    recent = get_recent_messages(user_id, session_id)

    trace_node(
        trace_id=trace_id,
        node_name="load_memory",
        user_id=user_id,
        session_id=session_id,
        input_data={"user_id": user_id, "session_id": session_id},
        output_data={
            "has_summary": bool(summary),
            "recent_message_count": len(recent),
        },
    )

    return {
        **state,
        "trace_id": trace_id,
        "summary": summary,
        "recent_messages": recent,
    }


def check_cache(state: AgentState) -> AgentState:
    """Node 2: Semantic cache lookup."""
    question = state["question"]
    question_embedding = embed_query(question)
    result = cache_lookup(question, question_embedding=question_embedding)

    cache_hit = bool(result)

    trace_node(
        trace_id=state["trace_id"],
        node_name="check_cache",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"question": question},
        output_data={"cache_hit": cache_hit},
    )

    if result:
        return {
            **state,
            "cache_hit": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "context": "",
            "retrieved_chunks": [],
            "reranked_chunks": [],
        }

    return {
        **state,
        "cache_hit": False,
        "answer": "",
        "sources": [],
        "context": "",
        "retrieved_chunks": [],
        "reranked_chunks": [],
    }


def route_collection(state: AgentState) -> AgentState:
    """Node 3: Decide which collection to search."""
    question = state["question"]
    collection = route_document(question)

    trace_node(
        trace_id=state["trace_id"],
        node_name="route_collection",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"question": question},
        output_data={"target_collection": collection},
    )

    return {**state, "target_collection": collection}


def retrieve_chunks(state: AgentState) -> AgentState:
    """Node 4: Bi-encoder retrieval — top-20 from pgvector."""
    question = state["question"]
    collection = state.get("target_collection")

    # Fall back to None (search all) if collection not in DB
    chunks = retrieve(question, collection=collection, k=20)

    # If collection search returned nothing search all
    if not chunks:
        logger.warning(f"No results in '{collection}' — searching all collections")
        chunks = retrieve(question, collection=None, k=20)

    trace_node(
        trace_id=state["trace_id"],
        node_name="retrieve",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"question": question, "collection": collection},
        output_data={"chunks_retrieved": len(chunks)},
    )

    return {**state, "retrieved_chunks": chunks}


def rerank_chunks(state: AgentState) -> AgentState:
    """Node 5: Cross-encoder reranking — top-20 → top-4."""
    settings = get_settings()
    question = state["question"]
    candidates = state["retrieved_chunks"]

    reranked = rerank(question, candidates, top_n=settings.rerank_top_n)

    # Format context string
    context_parts = [
        f"[{c['chunk_id']}]\n{c['content']}"
        for c in reranked
    ]
    context = "\n\n---\n\n".join(context_parts)

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "source": c["metadata"].get("source", ""),
            "collection": c.get("collection", ""),
            "preview": c["content"][:200],
            "similarity": round(c.get("similarity", 0), 4),
            "rerank_score": round(c.get("rerank_score", 0), 4),
        }
        for c in reranked
    ]

    trace_node(
        trace_id=state["trace_id"],
        node_name="rerank",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"candidates": len(candidates)},
        output_data={
            "reranked_to": len(reranked),
            "top_rerank_score": reranked[0]["rerank_score"] if reranked else 0,
            "chunk_ids": [c["chunk_id"] for c in reranked],
        },
    )

    return {
        **state,
        "reranked_chunks": reranked,
        "context": context,
        "sources": sources,
    }


def generate(state: AgentState) -> AgentState:
    """Node 6: Generate answer with memory context."""
    llm = get_llm()

    handler = get_langfuse_handler(
        session_id=state["session_id"],
        user_id=state["user_id"],
        trace_name="rag_generate",
    )

    prompt_input = {
        "summary_section": _format_summary_section(state["summary"]),
        "history_section": _format_history_section(state["recent_messages"]),
        "context": state["context"],
        "question": state["question"],
    }

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        prompt_input,
        config={"callbacks": [handler]},
    )

    return {**state, "answer": answer}


def save_memory(state: AgentState) -> AgentState:
    """Node 7: Persist messages + trigger summarization if needed."""
    user_id = state["user_id"]
    session_id = state["session_id"]

    save_message(user_id, session_id, "human", state["question"])
    save_message(user_id, session_id, "assistant", state["answer"])

    summarized = False
    if should_summarize(user_id, session_id):
        summarize_and_compress(user_id, session_id)
        summarized = True

    trace_node(
        trace_id=state["trace_id"],
        node_name="save_memory",
        user_id=user_id,
        session_id=session_id,
        input_data={"cache_hit": state["cache_hit"]},
        output_data={
            "messages_saved": 2,
            "summarization_triggered": summarized,
        },
    )

    return state


def cache_response(state: AgentState) -> AgentState:
    """Node 8: Store answer in Redis on cache miss."""
    if not state["cache_hit"]:
        question_embedding = embed_query(state["question"])
        cache_store(
            question=state["question"],
            answer=state["answer"],
            sources=state["sources"],
            question_embedding=question_embedding,
        )

    trace_node(
        trace_id=state["trace_id"],
        node_name="cache_response",
        user_id=state["user_id"],
        session_id=state["session_id"],
        input_data={"cache_hit": state["cache_hit"]},
        output_data={"stored_in_cache": not state["cache_hit"]},
    )

    return state


def score_eval(state: AgentState) -> AgentState:
    """
    Node 9: Run RAGAS eval scores in background thread.
    Does not block the response — fires and forgets.
    """
    import threading
    from agentic_rag.evaluation.scorer import score_live

    if state["cache_hit"] or not state["context"]:
        return state

    def _run_eval():
        try:
            score_live(
                trace_id=state["trace_id"],
                question=state["question"],
                answer=state["answer"],
                contexts=[c["content"] for c in state["reranked_chunks"]],
            )
        except Exception as e:
            logger.warning(f"Background eval failed: {e}")

    thread = threading.Thread(target=_run_eval, daemon=True)
    thread.start()

    return state


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_cache(state: AgentState) -> str:
    """Route based on cache hit — skip retrieval if hit."""
    if state["cache_hit"]:
        return "save_memory"
    return "route_collection"