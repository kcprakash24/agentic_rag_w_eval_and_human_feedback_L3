"""
RAGAS-based scoring using non-LLM metrics only.

score_live()  — called by score_eval node (background thread, single question)
score_batch() — called by run_evals.py (full dataset run)
"""

import logging
from ragas import SingleTurnSample
from ragas.metrics import BleuScore, RougeScore

from agentic_rag.observability.langfuse_client import get_langfuse

logger = logging.getLogger(__name__)

# Instantiate once — these are stateless, no LLM needed
_bleu = BleuScore()
# _rouge = RougeScore(rouge_type="rougeL", measure_type="f1")
_rouge = RougeScore(rouge_type="rougeL", mode="fmeasure")



def _compute_scores(question: str, answer: str, reference: str) -> dict[str, float]:
    """
    Computes BLEU and ROUGE-L F1 scores.
    Both metrics compare `answer` against `reference` (ground truth string).
    Returns scores as floats in [0, 1].
    """
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        reference=reference,
    )
    bleu = _bleu.single_turn_score(sample)
    rouge = _rouge.single_turn_score(sample)
    return {"bleu": round(bleu, 4), "rouge_l": round(rouge, 4)}


def score_live(
    trace_id: str,
    question: str,
    answer: str,
    contexts: list[str],  # kept in signature for API compatibility; not used by non-LLM metrics
    reference: str | None = None,
) -> None:
    """
    Called by score_eval node during live agent runs.
    If no reference is provided (normal live traffic), skips scoring silently.
    When called from run_evals.py, reference is injected.
    Logs scores to Langfuse as trace-level scores.
    """
    if not reference:
        # Live traffic with no ground truth — nothing to score against
        return

    try:
        scores = _compute_scores(question, answer, reference)
        langfuse = get_langfuse()

        langfuse.create_score(
            trace_id=trace_id,
            name="bleu",
            value=scores["bleu"],
            comment="RAGAS BleuScore (non-LLM)",
        )
        langfuse.create_score(
            trace_id=trace_id,
            name="rouge_l",
            value=scores["rouge_l"],
            comment="RAGAS RougeScore rougeL f1 (non-LLM)",
        )
        langfuse.flush()

        logger.info(f"Scores logged | trace={trace_id} | {scores}")

    except Exception as e:
        logger.warning(f"score_live failed for trace {trace_id}: {e}")


def score_batch(
    trace_id: str,
    question: str,
    answer: str,
    reference: str,
    contexts: list[str],
) -> dict[str, float]:
    """
    Called by run_evals.py. Scores and returns the dict for DB insertion.
    Also logs to Langfuse.
    """
    scores = _compute_scores(question, answer, reference)

    try:
        langfuse = get_langfuse()
        langfuse.create_score(trace_id=trace_id, name="bleu", value=scores["bleu"])
        langfuse.create_score(trace_id=trace_id, name="rouge_l", value=scores["rouge_l"])
        langfuse.flush()
    except Exception as e:
        logger.warning(f"Langfuse score logging failed: {e}")

    return scores