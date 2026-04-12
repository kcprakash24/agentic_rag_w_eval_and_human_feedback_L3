from agentic_rag.llm.provider import get_llm
from agentic_rag.memory.pg_memory import (
    get_oldest_messages,
    delete_messages_by_ids,
    save_summary,
    get_latest_summary,
)
from agentic_rag.config import get_settings


SUMMARIZE_PROMPT = """You are summarizing a conversation between a user and a research assistant.
Create a concise summary that captures:
- The main topics and questions discussed
- Key facts or answers that were established
- Any important context for future questions

Keep the summary under 200 words.

Conversation to summarize:
{conversation}

Summary:"""


def summarize_and_compress(user_id: str, session_id: str) -> str | None:
    """
    Summarize the oldest messages and compress them into a single summary.
    Deletes the summarized messages from the database.

    Flow:
        1. Fetch oldest N messages
        2. Build conversation string
        3. Ask gemma4 to summarize
        4. Save summary to chat_summaries
        5. Delete the summarized messages
        6. Return the summary text

    Returns:
        Summary text if summarization happened, None otherwise
    """
    settings = get_settings()
    n_to_summarize = settings.memory_summarize_after * 2  # pairs → messages

    # Fetch oldest messages
    messages = get_oldest_messages(user_id, session_id, n=n_to_summarize)

    if not messages:
        return None

    # Build conversation string for the prompt
    conversation = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
    ])

    # Ask gemma4 to summarize
    llm = get_llm()
    prompt = SUMMARIZE_PROMPT.format(conversation=conversation)
    response = llm.invoke(prompt)
    summary = response.content.strip()

    # Save summary and delete compressed messages
    message_ids = [m["id"] for m in messages]
    save_summary(user_id, session_id, summary, len(messages))
    delete_messages_by_ids(message_ids)

    print(f"  Summarized {len(messages)} messages for {user_id}/{session_id}")

    return summary


def get_memory_context(user_id: str, session_id: str) -> dict:
    """
    Get full memory context for a user+session.
    Returns both the latest summary and recent messages.

    This is what gets injected into the agent prompt.
    """
    summary = get_latest_summary(user_id, session_id)
    from agentic_rag.memory.pg_memory import get_recent_messages
    recent = get_recent_messages(user_id, session_id)

    return {
        "summary": summary or "",
        "recent_messages": recent,
    }