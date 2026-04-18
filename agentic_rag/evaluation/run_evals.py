"""
Offline eval runner.

Flow:
  1. Fetch all items from the Langfuse golden dataset
  2. Run each question through your existing ask() agent
  3. Score with RAGAS (BleuScore + RougeScore)
  4. Log scores to Langfuse (trace scores)
  5. Insert row into eval_results table in Postgres

Run: python -m agentic_rag.evaluation.run_evals
"""

import logging
import uuid
from datetime import datetime, timezone

import psycopg2

from agentic_rag.observability.langfuse_client import get_langfuse
from agentic_rag.evaluation.dataset import DATASET_NAME
from agentic_rag.evaluation.scorer import score_batch
from agentic_rag.agent.graph import ask  # adjust import path if different
from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)

DATASET_RUN_NAME = f"eval_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _get_db_conn():
    settings = get_settings()
    return psycopg2.connect(settings.postgres_url)


def _insert_eval_result(
    conn,
    trace_id: str,
    question: str,
    answer: str,
    reference: str,
    bleu: float,
    rouge: float,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_results
                (trace_id, question, answer, reference, bleu_score, rouge_score, dataset_run, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (trace_id, question, answer, reference, bleu, rouge, DATASET_RUN_NAME, datetime.now(timezone.utc)),
        )
    conn.commit()


def run_evals(user_id: str = "eval_runner", session_id: str | None = None) -> None:
    langfuse = get_langfuse()
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = dataset.items
    logger.info(f"Loaded {len(items)} items from dataset '{DATASET_NAME}'.")

    conn = _get_db_conn()
    results = []

    for i, item in enumerate(items):
        question = item.input["question"]
        reference = item.expected_output["reference"]
        run_session = session_id or f"eval_{uuid.uuid4().hex[:8]}"

        logger.info(f"[{i+1}/{len(items)}] Running: {question[:60]}...")

        try:
            result = ask(
                question=question,
                user_id=user_id,
                session_id=run_session,
            )

            answer = result.get("answer", "")
            trace_id = result.get("trace_id", uuid.uuid4().hex)
            contexts = [c["content"] for c in result.get("reranked_chunks", [])]

            scores = score_batch(
                trace_id=trace_id,
                question=question,
                answer=answer,
                reference=reference,
                contexts=contexts,
            )

            _insert_eval_result(
                conn=conn,
                trace_id=trace_id,
                question=question,
                answer=answer,
                reference=reference,
                bleu=scores["bleu"],
                rouge=scores["rouge_l"],
            )

            # Link this run to the Langfuse dataset item
            item.link(
                run_name=DATASET_RUN_NAME,
                trace_or_observation=langfuse.get_trace(trace_id),
            )

            results.append({"question": question, **scores})
            logger.info(f"  → bleu={scores['bleu']:.4f} | rouge_l={scores['rouge_l']:.4f}")

        except Exception as e:
            logger.error(f"Failed on question {i+1}: {e}")
            continue

    conn.close()
    langfuse.flush()

    # Summary
    if results:
        avg_bleu = sum(r["bleu"] for r in results) / len(results)
        avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
        print(f"\n{'='*50}")
        print(f"Run: {DATASET_RUN_NAME}")
        print(f"Questions evaluated: {len(results)}/{len(items)}")
        print(f"Avg BLEU:    {avg_bleu:.4f}")
        print(f"Avg ROUGE-L: {avg_rouge:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evals()