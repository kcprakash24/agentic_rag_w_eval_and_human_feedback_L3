import psycopg2
import psycopg2.extras
from datetime import datetime
from agentic_rag.config import get_settings


def get_connection():
    settings = get_settings()
    return psycopg2.connect(settings.postgres_url)


def save_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
) -> None:
    """
    Save a single message to PostgreSQL.

    Args:
        user_id: e.g. 'dave' or 'mike'
        session_id: unique session identifier
        role: 'human' or 'assistant'
        content: message text
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_messages (user_id, session_id, role, content)
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, role, content))
        conn.commit()
    finally:
        conn.close()


def get_recent_messages(
    user_id: str,
    session_id: str,
    n: int | None = None,
) -> list[dict]:
    """
    Fetch the last N messages for a user+session.
    Returns in chronological order (oldest first).

    Args:
        user_id: user identifier
        session_id: session identifier
        n: number of messages to fetch (defaults to config value * 2)
    """
    settings = get_settings()
    limit = n if n else settings.memory_last_n_messages * 2

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT role, content, created_at
            FROM chat_messages
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, session_id, limit))

        # Reverse to get chronological order
        messages = list(reversed(cur.fetchall()))
        return [dict(m) for m in messages]
    finally:
        conn.close()


def get_message_count(user_id: str, session_id: str) -> int:
    """Count total messages for a user+session."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM chat_messages
            WHERE user_id = %s AND session_id = %s
        """, (user_id, session_id))
        return cur.fetchone()[0]
    finally:
        conn.close()


def get_oldest_messages(
    user_id: str,
    session_id: str,
    n: int,
) -> list[dict]:
    """Fetch the oldest N messages — used before summarization."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, role, content, created_at
            FROM chat_messages
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (user_id, session_id, n))
        return [dict(m) for m in cur.fetchall()]
    finally:
        conn.close()


def delete_messages_by_ids(message_ids: list[int]) -> int:
    """Delete specific messages by their IDs."""
    if not message_ids:
        return 0

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM chat_messages
            WHERE id = ANY(%s)
        """, (message_ids,))
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def save_summary(
    user_id: str,
    session_id: str,
    summary: str,
    message_count: int,
) -> None:
    """Save a compressed summary to chat_summaries table."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_summaries
                (user_id, session_id, summary, message_count)
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, summary, message_count))
        conn.commit()
    finally:
        conn.close()


def get_latest_summary(
    user_id: str,
    session_id: str,
) -> str | None:
    """
    Fetch the most recent summary for a user+session.
    Returns None if no summary exists yet.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT summary FROM chat_summaries
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id, session_id))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def should_summarize(user_id: str, session_id: str) -> bool:
    """
    Check if message count exceeds summarization threshold.
    Returns True if we should compress history.
    """
    settings = get_settings()
    count = get_message_count(user_id, session_id)
    threshold = settings.memory_summarize_after * 2  # pairs → messages
    return count >= threshold