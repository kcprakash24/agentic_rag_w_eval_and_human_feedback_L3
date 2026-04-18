import os
import uuid
import logging
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)


def _set_langfuse_env() -> None:
    settings = get_settings()
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url


def get_langfuse():
    _set_langfuse_env()
    return get_client()


def get_langfuse_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str = "rag_generate",
) -> CallbackHandler:
    _set_langfuse_env()

    trace_context = {
        "trace_id": uuid.uuid4().hex,  # 32 hex chars, no hyphens
        "name": trace_name,
    }
    if session_id:
        trace_context["session_id"] = session_id
    if user_id:
        trace_context["user_id"] = user_id

    return CallbackHandler(trace_context=trace_context)


def trace_node(
    trace_id: str,
    node_name: str,
    user_id: str,
    session_id: str,
    input_data: dict,
    output_data: dict,
    metadata: dict | None = None,
) -> None:
    """Trace a LangGraph node using Langfuse v4 API."""
    try:
        _set_langfuse_env()
        langfuse = get_client()

        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
        ):
            with langfuse.start_as_current_observation(
                as_type="span",
                name=node_name,
                input=input_data,
                metadata=metadata or {},
                trace_context={"trace_id": trace_id},
            ) as span:
                span.update(output=output_data)

        langfuse.flush()

    except Exception as e:
        logger.warning(f"Langfuse trace failed for node '{node_name}': {e}")


def verify_langfuse_connection() -> bool:
    _set_langfuse_env()
    client = get_client()
    if client.auth_check():
        print("Langfuse: authenticated and ready")
        return True
    print("Langfuse: authentication failed")
    return False

def submit_feedback(
    trace_id: str,
    thumbs_up: bool,
    comment: str | None = None,
) -> None:
    """
    Logs human feedback as a Langfuse score on an existing trace.
    score name: 'human_feedback' | value: 1.0 (thumbs up) or 0.0 (thumbs down)
    """
    try:
        langfuse = get_langfuse()
        langfuse.create_score(
            trace_id=trace_id,
            name="human_feedback",
            value=1.0 if thumbs_up else 0.0,
            comment=comment or "",
        )
        langfuse.flush()
        logger.info(f"Feedback logged to Langfuse | trace={trace_id} | thumbs_up={thumbs_up}")
    except Exception as e:
        logger.warning(f"Langfuse feedback logging failed: {e}")