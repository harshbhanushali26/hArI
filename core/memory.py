"""
memory.py
---------
Manages conversation memory for hArI's token-efficient chat pipeline.

Responsibilities:
    - Append user and assistant messages to session buffer
    - Check if buffer has exceeded MAX_CACHE_SIZE
    - Summarize older messages via Groq when buffer is full
    - Trim buffer to recent messages after summarization
    - Build compressed context string for LLM prompt injection

Design decisions:
    - Stateless functions — all state lives in st.session_state
      (memory_buffer, compressed_summary), not in this module
    - summarize() only compresses the older half of the buffer,
      keeping the most recent messages raw for better continuity
    - get_context() returns a single formatted string — ready to
      inject directly into system prompt construction
    - trim_buffer() always called after summarize() — never standalone

Flow (called from app.py):
    add_message(role, text)
    → should_summarize()?
        → True  → summarize() → trim_buffer()
        → False → continue
    → get_context() → passed to responser.py

Public API:
    add_message(role, text)   -> None
    get_context()             -> str
    should_summarize()        -> bool
    summarize()               -> str
    trim_buffer(summary)      -> None
"""

import streamlit as st
from config import SUMMARY_MODEL, MAX_CACHE_SIZE, MAX_SUMMARY_TOKENS
from core.utils import get_groq_client



# ── Message Management ────────────────────────────────────────────────────────

def add_message(role: str, text: str) -> None:
    """
    Appends a message to the session memory buffer.

    Args:
        role : "user" or "assistant"
        text : message content string

    Raises:
        ValueError if role is not "user" or "assistant"
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

    st.session_state.memory_buffer.append({"role": role, "content": text})


# ── Context Builder ───────────────────────────────────────────────────────────

def get_context() -> str:
    """
    Builds a formatted context string from compressed summary + raw buffer.
    Injected into system prompt before every LLM call.

    Format:
        [Conversation Summary]
        <compressed_summary if exists>

        [Recent Conversation]
        User: ...
        Assistant: ...

    Returns:
        Formatted string — empty string if buffer and summary are both empty.
    """
    parts = []

    # prepend compressed summary if exists
    summary = st.session_state.get("compressed_summary", "")
    if summary:
        parts.append(f"[Conversation Summary]\n{summary}")

    # append raw recent messages
    buffer = st.session_state.get("memory_buffer", [])
    if buffer:
        recent = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in buffer
        )
        parts.append(f"[Recent Conversation]\n{recent}")

    return "\n\n".join(parts)


# ── Summarization Trigger ─────────────────────────────────────────────────────

def should_summarize() -> bool:
    """
    Returns True if memory buffer has exceeded MAX_CACHE_SIZE.
    Called after every add_message() in app.py.
    """
    return len(st.session_state.get("memory_buffer", [])) > MAX_CACHE_SIZE


# ── Summarizer ────────────────────────────────────────────────────────────────

def summarize() -> str:
    """
    Compresses the older half of the memory buffer into a summary
    using a Groq API call.

    Only summarizes the older half — keeps recent messages raw
    for better conversation continuity.

    Returns:
        summary string from Groq
    Raises:
        RuntimeError if Groq API call fails
    """
    buffer = st.session_state.get("memory_buffer", [])

    # older half goes to summarization, recent half stays raw
    split_point  = len(buffer) // 2
    old_messages = buffer[:split_point]

    # format old messages for summarization prompt
    conversation = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in old_messages
    )

    # prepend existing summary if present — rolling summary
    existing_summary = st.session_state.get("compressed_summary", "")
    if existing_summary:
        conversation = f"Previous Summary:\n{existing_summary}\n\nNew Messages:\n{conversation}"

    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversation summarizer. "
                        "Summarize the following conversation concisely, "
                        "preserving all key facts, questions asked, and conclusions reached. "
                        "Keep the summary under 150 words."
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize this conversation:\n\n{conversation}"
                }
            ],
            max_tokens=MAX_SUMMARY_TOKENS
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"Groq summarization failed: {e}")


# ── Buffer Trimmer ────────────────────────────────────────────────────────────

def trim_buffer(summary: str) -> None:
    """
    Called immediately after summarize().
    Stores the new summary and keeps only the recent half of the buffer.

    Args:
        summary: compressed summary string returned by summarize()
    """
    buffer      = st.session_state.get("memory_buffer", [])
    split_point = len(buffer) // 2

    st.session_state.compressed_summary = summary
    st.session_state.memory_buffer      = buffer[split_point:]