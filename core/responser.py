"""
responser.py
------------
Final LLM response layer for hArI.

Responsibilities:
    - Load system_prompt.md and parse mode-specific sections
    - Build mode-aware prompt (PDF or CSV) with injected context
    - Call Groq API with correct model per intent
    - Return final response string to app.py

Design decisions:
    - load_prompt() reads file once — caller can cache in st.session_state
    - Mode-specific prompt built per call — LLM only sees relevant instructions
        (Identity + mode section + General Rules) — no noise from other mode
    - PDF prompt injects retrieved chunks with source + page citations
    - CSV prompt injects dataframe schema + analysis result
    - Memory context always injected regardless of mode
    - RESPONSE_MODEL for PDF, ANALYSIS_MODEL for CSV — set in config
    - get_response() is the only function app.py calls — single entry point

Flow (called from app.py):
    intent = get_intent(files, query)
    response = get_response(
        query    = user query,
        intent   = "pdf" or "csv",
        context  = retrieved chunks OR df schema + result,
        memory   = get_context() from memory.py
    )

Public API:
    get_response(query, intent, context, memory)  -> str
    load_prompt()                                 -> dict

Internal:
    _build_pdf_system_prompt(sections, memory)    -> str
    _build_csv_system_prompt(sections, memory)    -> str
    _build_pdf_user_message(query, context)       -> str
    _build_csv_user_message(query, context)       -> str
    _parse_prompt_sections(raw)                   -> dict
"""

from pathlib import Path
from config import RESPONSE_MODEL, ANALYSIS_MODEL
from core.utils import get_groq_client

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.md"


# ── Prompt Loader ─────────────────────────────────────────────────────────────

def load_prompt() -> dict:
    """
    Reads system_prompt.md and parses it into sections.
    Returns dict with keys: 'identity', 'pdf', 'csv', 'general'

    Should be called once and cached in st.session_state:
        st.session_state.prompt_sections = load_prompt()

    Raises:
        FileNotFoundError if system_prompt.md is missing
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"system_prompt.md not found at {PROMPT_PATH}")

    raw = PROMPT_PATH.read_text(encoding="utf-8")
    return _parse_prompt_sections(raw)


def _parse_prompt_sections(raw: str) -> dict:
    """
    Parses markdown sections from system_prompt.md into a dict.

    Splits on ## headers and maps to keys:
        Identity        -> "identity"
        PDF Mode        -> "pdf"
        CSV / Excel Mode-> "csv"
        General Rules   -> "general"

    Args:
        raw: full string content of system_prompt.md

    Returns:
        dict with section content strings
    """
    sections = {"identity": "", "pdf": "", "csv": "", "general": ""}
    current  = None

    for line in raw.splitlines():
        if line.startswith("## Identity"):
            current = "identity"
        elif line.startswith("## PDF Mode"):
            current = "pdf"
        elif line.startswith("## CSV"):
            current = "csv"
        elif line.startswith("## General Rules"):
            current = "general"
        elif line.startswith("---"):
            continue
        elif current:
            sections[current] += line + "\n"

    return {k: v.strip() for k, v in sections.items()}


# ── System Prompt Builders ────────────────────────────────────────────────────

def _build_pdf_system_prompt(sections: dict, memory: str) -> str:
    """
    Builds system prompt for PDF RAG mode.
    Combines: Identity + PDF Mode + General Rules + memory context.

    Args:
        sections: parsed prompt sections from load_prompt()
        memory  : formatted conversation context from memory.get_context()

    Returns:
        complete system prompt string for PDF queries
    """
    parts = [
        sections["identity"],
        sections["pdf"],
        sections["general"]
    ]

    if memory:
        parts.append(f"## Conversation History\n{memory}")

    return "\n\n".join(parts)


def _build_csv_system_prompt(sections: dict, memory: str) -> str:
    """
    Builds system prompt for CSV/Excel analysis mode.
    Combines: Identity + CSV Mode + General Rules + memory context.

    Args:
        sections: parsed prompt sections from load_prompt()
        memory  : formatted conversation context from memory.get_context()

    Returns:
        complete system prompt string for CSV queries
    """
    parts = [
        sections["identity"],
        sections["csv"],
        sections["general"]
    ]

    if memory:
        parts.append(f"## Conversation History\n{memory}")

    return "\n\n".join(parts)


# ── User Message Builders ─────────────────────────────────────────────────────

def _build_pdf_user_message(query: str, context: list) -> str:
    """
    Builds user message for PDF mode with injected RAG chunks.

    Args:
        query  : user's question string
        context: list of dicts from retriever.retrieve()
                    each dict has "content", "metadata" (source, page)

    Returns:
        formatted user message string with context chunks injected
    """
    if not context:
        return f"Query: {query}\n\nNo relevant context found in the uploaded documents."

    chunks = []
    for i, chunk in enumerate(context, start=1):
        source = chunk["metadata"].get("source", "unknown")
        page   = chunk["metadata"].get("page", "?")
        chunks.append(
            f"[Chunk {i} — {source}, Page {page}]\n{chunk['content']}"
        )

    context_str = "\n\n".join(chunks)

    return (
        f"Context from uploaded documents:\n\n"
        f"{context_str}\n\n"
        f"---\n"
        f"User Query: {query}"
    )


def _build_csv_user_message(query: str, context: dict) -> str:
    """
    Builds user message for CSV mode with injected schema and analysis result.

    Args:
        query  : user's question string
        context: dict with keys:
                    "metadata"        -> df metadata (columns, dtypes, preview, rows)
                    "analysis_result" -> string result from pandas code execution

    Returns:
        formatted user message string with schema and result injected
    """
    metadata        = context.get("metadata", {})
    analysis_result = context.get("analysis_result", "No result available.")

    filename = metadata.get("source", "unknown")
    rows     = metadata.get("rows", "?")
    columns  = metadata.get("columns", [])
    dtypes   = metadata.get("dtypes", {})
    preview  = metadata.get("preview", [])

    schema_str  = "\n".join(f"  - {col}: {dtypes.get(col, '?')}" for col in columns)
    preview_str = "\n".join(str(row) for row in preview)

    return (
        f"File: {filename} ({rows} rows)\n\n"
        f"Schema:\n{schema_str}\n\n"
        f"Preview (first 3 rows):\n{preview_str}\n\n"
        f"Analysis Result:\n{analysis_result}\n\n"
        f"---\n"
        f"User Query: {query}"
    )


# ── Master Entry Point ────────────────────────────────────────────────────────

def get_response(
    query   : str,
    intent  : str,
    context,
    memory  : str,
    sections: dict,
    placeholder=None
) -> str:
    """
    Master function called by app.py to get final LLM response.

    Args:
        query   : user's question string
        intent  : "pdf" or "csv" from query_intent.get_intent()
        context : list of chunk dicts (PDF) or dict with schema+result (CSV)
        memory  : conversation context string from memory.get_context()
        sections: parsed prompt sections from load_prompt()

    Returns:
        final response string from Groq LLM

    Raises:
        RuntimeError if Groq API call fails
    """
    if intent == "pdf":
        system_prompt = _build_pdf_system_prompt(sections, memory)
        user_message  = _build_pdf_user_message(query, context)
        model         = RESPONSE_MODEL

    else:  # csv / excel
        system_prompt = _build_csv_system_prompt(sections, memory)
        user_message  = _build_csv_user_message(query, context)
        model         = ANALYSIS_MODEL

    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            max_tokens  = 1024,
            temperature = 0.2,    # low temp — factual, precise responses
            stream=True
        )
        full_response = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            if placeholder:
                placeholder.markdown(full_response + "▌")  # ← cursor effect

        if placeholder:
            placeholder.markdown(full_response)  # ← final render, remove cursor

        return full_response
        # return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"Groq response generation failed: {e}")