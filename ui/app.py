"""
Streamlit UI for Agentic RAG L3.

Main chat:
  - Question input → agent → answer + sources
  - 👍 / 👎 buttons + optional comment → logged to Langfuse + Postgres

Admin panel (sidebar toggle):
  - Avg BLEU + ROUGE-L from eval_results
  - Recent feedback from user_feedback
  - Eval score trend chart across runs
"""

import uuid
import psycopg2
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_rag.agent.graph import ask
from agentic_rag.config import get_settings
from agentic_rag.observability.langfuse_client import submit_feedback
from agentic_rag.vectorstore.pgvector_store import save_feedback


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
)

settings = get_settings()

# ── Session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_default"

if "messages" not in st.session_state:
    # Each entry: {role, content, trace_id, question, answer, feedback_given}
    st.session_state.messages = []

if "show_admin" not in st.session_state:
    st.session_state.show_admin = False


# ── DB helpers (admin panel) ──────────────────────────────────────────────────
def _get_db_conn():
    return psycopg2.connect(settings.postgres_url)


def get_eval_summary() -> dict:
    """Avg BLEU + ROUGE-L across all eval runs."""
    conn = _get_db_conn()
    try:
        df = pd.read_sql(
            "SELECT AVG(bleu_score) as avg_bleu, AVG(rouge_score) as avg_rouge FROM eval_results",
            conn,
        )
        return {
            "avg_bleu": round(df["avg_bleu"].iloc[0] or 0.0, 4),
            "avg_rouge": round(df["avg_rouge"].iloc[0] or 0.0, 4),
        }
    finally:
        conn.close()


def get_eval_trend() -> pd.DataFrame:
    """Per-run average scores for trend chart."""
    conn = _get_db_conn()
    try:
        return pd.read_sql(
            """
            SELECT
                dataset_run,
                AVG(bleu_score)  AS avg_bleu,
                AVG(rouge_score) AS avg_rouge,
                COUNT(*)         AS n_questions,
                MIN(created_at)  AS run_date
            FROM eval_results
            GROUP BY dataset_run
            ORDER BY run_date ASC
            """,
            conn,
        )
    finally:
        conn.close()


def get_recent_feedback(limit: int = 10) -> pd.DataFrame:
    """Most recent human feedback rows."""
    conn = _get_db_conn()
    try:
        return pd.read_sql(
            f"""
            SELECT
                user_id,
                SUBSTRING(question, 1, 60) AS question,
                thumbs_up,
                COALESCE(comment, '') AS comment,
                created_at
            FROM user_feedback
            ORDER BY created_at DESC
            LIMIT {limit}
            """,
            conn,
        )
    finally:
        conn.close()


# ── Feedback handler ──────────────────────────────────────────────────────────
def handle_feedback(msg_index: int, thumbs_up: bool, comment: str) -> None:
    msg = st.session_state.messages[msg_index]
    if msg.get("feedback_given"):
        return

    # Write to Langfuse + Postgres
    submit_feedback(
        trace_id=msg["trace_id"],
        thumbs_up=thumbs_up,
        comment=comment or None,
    )
    save_feedback(
        trace_id=msg["trace_id"],
        user_id=st.session_state.user_id,
        question=msg["question"],
        answer=msg["answer"],
        thumbs_up=thumbs_up,
        comment=comment or None,
    )

    st.session_state.messages[msg_index]["feedback_given"] = True
    st.session_state.messages[msg_index]["feedback_value"] = thumbs_up


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Agentic RAG")
    st.caption(f"Session: `{st.session_state.session_id}`")

    st.divider()
    st.session_state.user_id = st.text_input(
        "User ID", value=st.session_state.user_id
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:8]
        st.rerun()

    st.divider()
    st.session_state.show_admin = st.toggle("🛠️ Admin panel", value=st.session_state.show_admin)


# ── Admin panel ───────────────────────────────────────────────────────────────
if st.session_state.show_admin:
    st.header("🛠️ Admin Panel")

    # Row 1: eval summary metrics
    summary = get_eval_summary()
    col1, col2 = st.columns(2)
    col1.metric("Avg BLEU (all runs)", summary["avg_bleu"])
    col2.metric("Avg ROUGE-L (all runs)", summary["avg_rouge"])

    st.divider()

    # Row 2: eval trend chart
    st.subheader("Eval Score Trend")
    trend_df = get_eval_trend()
    if trend_df.empty:
        st.info("No eval runs found. Run `run_evals.py` first.")
    else:
        chart_df = trend_df.set_index("dataset_run")[["avg_bleu", "avg_rouge"]]
        st.line_chart(chart_df)
        st.caption(f"{len(trend_df)} eval run(s) found.")

    st.divider()

    # Row 3: recent human feedback
    st.subheader("Recent Human Feedback")
    feedback_df = get_recent_feedback()
    if feedback_df.empty:
        st.info("No feedback submitted yet.")
    else:
        # Render thumbs as emoji for readability
        feedback_df["thumbs_up"] = feedback_df["thumbs_up"].map(
            {True: "👍", False: "👎"}
        )
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)

    st.divider()

# ── Main chat ─────────────────────────────────────────────────────────────────
st.header("💬 Ask your documents")

# Render existing messages + feedback widgets
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Feedback widget — only on assistant messages, only if not yet given
        if msg["role"] == "assistant":
            if msg.get("feedback_given"):
                icon = "👍" if msg.get("feedback_value") else "👎"
                st.caption(f"Feedback recorded {icon}")
            else:
                # Comment box + buttons in a compact row
                comment = st.text_input(
                    "Optional comment",
                    key=f"comment_{i}",
                    placeholder="What was wrong or right about this answer?",
                    label_visibility="collapsed",
                )
                col_up, col_down, _ = st.columns([1, 1, 8])
                with col_up:
                    if st.button("👍", key=f"up_{i}"):
                        handle_feedback(i, thumbs_up=True, comment=comment)
                        st.rerun()
                with col_down:
                    if st.button("👎", key=f"down_{i}"):
                        handle_feedback(i, thumbs_up=False, comment=comment)
                        st.rerun()

            # Sources expander
            if msg.get("sources"):
                with st.expander("Sources", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"[{s['rerank_score']:.3f}] {s['chunk_id']}")

# Chat input
if question := st.chat_input("Ask something about your documents..."):
    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(
                question=question,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
            )

        answer = result["answer"]
        trace_id = result["trace_id"]
        sources = result.get("sources", [])
        cache_hit = result.get("cache_hit", False)

        st.markdown(answer)

        if cache_hit:
            st.caption("⚡ Cache hit")

        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.caption(f"[{s['rerank_score']:.3f}] {s['chunk_id']}")

    # Store in session state with all metadata needed for feedback
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace_id": trace_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "feedback_given": False,
        "feedback_value": None,
    })
    st.rerun()