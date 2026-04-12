import json
import redis
import numpy as np
from agentic_rag.config import get_settings
from agentic_rag.embeddings.embedder import embed_query


def get_redis_client() -> redis.Redis:
    """Returns a Redis client."""
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=False)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Compute cosine distance between two vectors.
    Returns 0.0 (identical) to 2.0 (opposite).
    Lower = more similar.
    """
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def cache_lookup(
    question: str,
    question_embedding: list[float] | None = None,
) -> dict | None:
    """
    Look up a question in the semantic cache.

    Flow:
        1. Embed the question (or use provided embedding)
        2. Scan all cached entries
        3. Compute cosine distance to each cached question
        4. Return cached answer if distance < threshold

    Args:
        question: The user's question
        question_embedding: Pre-computed embedding (avoids re-embedding)

    Returns:
        Cached result dict or None if cache miss
    """
    settings = get_settings()
    r = get_redis_client()

    # Embed the question if not provided
    if question_embedding is None:
        question_embedding = embed_query(question)

    # Get all cache keys
    keys = r.keys("rag_cache:*")
    if not keys:
        return None

    best_distance = float("inf")
    best_result = None

    for key in keys:
        raw = r.get(key)
        if not raw:
            continue

        try:
            entry = json.loads(raw.decode("utf-8"))
            cached_embedding = entry["question_embedding"]
            distance = _cosine_distance(question_embedding, cached_embedding)

            if distance < best_distance:
                best_distance = distance
                best_result = entry

        except (json.JSONDecodeError, KeyError):
            continue

    # Return if within threshold
    if best_distance <= settings.redis_similarity_threshold:
        print(f"  Cache HIT (distance={round(best_distance, 4)})")
        return best_result

    print(f"  Cache MISS (best distance={round(best_distance, 4)})")
    return None


def cache_store(
    question: str,
    answer: str,
    sources: list[dict],
    question_embedding: list[float] | None = None,
) -> None:
    """
    Store a question+answer in the semantic cache.

    Args:
        question: Original question text
        answer: LLM answer
        sources: Retrieved source chunks
        question_embedding: Pre-computed embedding
    """
    settings = get_settings()
    r = get_redis_client()

    if question_embedding is None:
        question_embedding = embed_query(question)

    # Use hash of question as key suffix for uniqueness
    import hashlib
    key_suffix = hashlib.md5(question.encode()).hexdigest()[:12]
    cache_key = f"rag_cache:{key_suffix}"

    entry = {
        "question": question,
        "question_embedding": question_embedding,
        "answer": answer,
        "sources": sources,
    }

    r.setex(
        name=cache_key,
        time=settings.redis_ttl_seconds,   # TTL — auto-expires
        value=json.dumps(entry),
    )

    print(f"  Cached question (key={cache_key}, ttl={settings.redis_ttl_seconds}s)")


def cache_clear() -> int:
    """Clear all RAG cache entries. Useful for testing."""
    r = get_redis_client()
    keys = r.keys("rag_cache:*")
    if keys:
        r.delete(*keys)
    print(f"  Cleared {len(keys)} cache entries")
    return len(keys)


def cache_stats() -> dict:
    """Summary of current cache state."""
    r = get_redis_client()
    keys = r.keys("rag_cache:*")

    entries = []
    for key in keys:
        raw = r.get(key)
        if raw:
            try:
                entry = json.loads(raw.decode("utf-8"))
                ttl = r.ttl(key)
                entries.append({
                    "key": key.decode("utf-8"),
                    "question": entry.get("question", "")[:80],
                    "ttl_seconds": ttl,
                })
            except Exception:
                continue

    return {
        "total_cached": len(entries),
        "entries": entries,
    }