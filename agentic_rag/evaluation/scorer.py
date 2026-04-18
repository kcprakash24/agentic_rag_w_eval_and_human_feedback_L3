import logging

logger = logging.getLogger(__name__)


def score_live(
    trace_id: str,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Stub — full RAGAS implementation in Step 7.
    Returns empty dict for now.
    """
    logger.info(f"score_live called for trace {trace_id[:8]} — stub, skipping")
    return {}