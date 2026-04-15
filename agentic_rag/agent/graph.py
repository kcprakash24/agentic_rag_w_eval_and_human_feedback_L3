import logging
from functools import lru_cache
from langgraph.graph import StateGraph, END

from agentic_rag.agent.state import AgentState
from agentic_rag.agent.nodes import (
    load_memory,
    check_cache,
    route_collection,
    retrieve_chunks,
    rerank_chunks,
    generate,
    save_memory,
    cache_response,
    score_eval,
    route_after_cache,
)

logging.getLogger("langfuse").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("load_memory", load_memory)
    graph.add_node("check_cache", check_cache)
    graph.add_node("route_collection", route_collection)
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("rerank", rerank_chunks)
    graph.add_node("generate", generate)
    graph.add_node("save_memory", save_memory)
    graph.add_node("cache_response", cache_response)
    graph.add_node("score_eval", score_eval)

    # Entry point
    graph.set_entry_point("load_memory")

    # Linear edges
    graph.add_edge("load_memory", "check_cache")

    # Conditional — cache hit skips retrieval
    graph.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {
            "route_collection": "route_collection",
            "save_memory": "save_memory",
        }
    )

    graph.add_edge("route_collection", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "save_memory")
    graph.add_edge("save_memory", "cache_response")
    graph.add_edge("cache_response", "score_eval")
    graph.add_edge("score_eval", END)

    return graph.compile()


@lru_cache
def get_agent():
    """Singleton agent — built once, reused."""
    return build_graph()


def ask(
    question: str,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Run the full agentic RAG pipeline.

    Args:
        question: User question
        user_id: e.g. 'dave' or 'mike'
        session_id: Unique session identifier

    Returns:
        Dict with answer, sources, cache_hit, trace_id
    """
    agent = get_agent()

    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "trace_id": "",
        "target_collection": "",
        "summary": "",
        "recent_messages": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "context": "",
        "sources": [],
        "cache_hit": False,
        "answer": "",
    }

    final_state = agent.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "cache_hit": final_state["cache_hit"],
        "trace_id": final_state["trace_id"],
        "user_id": user_id,
        "session_id": session_id,
    }