import psycopg2
import psycopg2.extras
from agentic_rag.config import get_settings
from agentic_rag.ingestion.chunker import DocumentChunk
from datetime import datetime, timezone, timedelta

def get_connection():
    """Raw psycopg2 connection."""
    settings = get_settings()
    return psycopg2.connect(settings.postgres_url)


def add_chunks(
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> int:
    """
    Store chunks + embeddings in pgvector.
    Idempotent — skips existing chunk_ids.
    Collection is read from chunk metadata.

    Args:
        chunks: Output from chunk_document()
        embeddings: One vector per chunk, same order

    Returns:
        Number of new chunks inserted
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    inserted = 0
    conn = get_connection()

    try:
        cur = conn.cursor()

        for chunk, embedding in zip(chunks, embeddings):
            collection = chunk.metadata.get("collection", "general")

            cur.execute("""
                INSERT INTO documents
                    (chunk_id, content, metadata, embedding, collection)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
            """, (
                chunk.chunk_id,
                chunk.text,
                psycopg2.extras.Json(chunk.metadata),
                embedding,
                collection,
            ))

            if cur.rowcount > 0:
                inserted += 1

        conn.commit()
        print(f"  Inserted {inserted} new chunks "
              f"({len(chunks) - inserted} already existed)")

    finally:
        conn.close()

    return inserted


def similarity_search(
    query_embedding: list[float],
    k: int = 20,
    collection: str | None = None,
) -> list[dict]:
    """
    Find top-k similar chunks using cosine distance.
    Optionally filter by collection.

    Args:
        query_embedding: Embedded query vector
        k: Number of results (default 20 — reranker reduces to top-4)
        collection: Filter to specific collection (None = search all)

    Returns:
        List of dicts with content, metadata, similarity score
    """
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if collection:
            cur.execute("""
                SELECT
                    chunk_id,
                    content,
                    metadata,
                    collection,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE collection = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, collection, query_embedding, k))
        else:
            cur.execute("""
                SELECT
                    chunk_id,
                    content,
                    metadata,
                    collection,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, k))

        return [dict(row) for row in cur.fetchall()]

    finally:
        conn.close()


def get_collection_stats() -> dict:
    """Summary of stored chunks per collection."""
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents")
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT collection, COUNT(*) as chunks
            FROM documents
            GROUP BY collection
            ORDER BY chunks DESC
        """)

        return {
            "total_chunks": total,
            "sources": [
                {"source": row[0], "chunks": row[1]}
                for row in cur.fetchall()
            ]
        }
    finally:
        conn.close()


def get_available_collections() -> list[str]:
    """Return all collection names currently in the database."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT collection
            FROM documents
            ORDER BY collection
        """)
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def delete_collection(collection: str) -> int:
    """Delete all chunks from a collection."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM documents WHERE collection = %s",
            (collection,)
        )
        deleted = cur.rowcount
        conn.commit()
        print(f"  Deleted {deleted} chunks from '{collection}'")
        return deleted
    finally:
        conn.close()


def delete_source(source_name: str) -> int:
    """Delete all chunks from a specific source file."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM documents
            WHERE metadata->>'source' = %s
        """, (source_name,))
        deleted = cur.rowcount
        conn.commit()
        print(f"  Deleted {deleted} chunks for '{source_name}'")
        return deleted
    finally:
        conn.close()


def explain_query(query_embedding: list[float], collection: str | None = None) -> str:
    """
    Show PostgreSQL query plan — confirms HNSW index is being used.
    Useful for debugging retrieval performance.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        if collection:
            cur.execute("""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE collection = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 20
            """, (query_embedding, collection, query_embedding))
        else:
            cur.execute("""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT 20
            """, (query_embedding, query_embedding))

        plan = "\n".join(row[0] for row in cur.fetchall())
        return plan

    finally:
        conn.close()

def save_feedback(
    trace_id: str,
    user_id: str,
    question: str,
    answer: str,
    thumbs_up: bool,
    comment: str | None = None,
) -> None:
    """Persists human feedback to user_feedback table."""
    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_feedback
                    (trace_id, user_id, question, answer, thumbs_up, comment, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (trace_id, user_id, question, answer, thumbs_up, comment, datetime.now(timezone.utc)),
            )
        conn.commit()
    finally:
        conn.close()