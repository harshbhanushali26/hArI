"""
query_intent.py
---------------
Classifies user query intent to route between PDF RAG and CSV analysis pipelines.

Responsibilities:
    - Detect file types present in current session
    - Route directly if only one file type uploaded (no LLM call needed)
    - Use LLM classifier only when both PDF and CSV/Excel files are present
    - Fallback to "pdf" if LLM classification fails or returns unclear response

Design decisions:
    - get_groq_client() imported from utils — shared across memory, responser
    - get_file_extension() imported from utils — shared with file_processor
    - max_tokens=5 on LLM call — forces single word response, saves tokens
    - Response safety guard — strips, lowercases, checks for "csv" keyword
      defaults to "pdf" if unclear (PDF RAG is safer fallback than code exec)
    - get_intent() is the only function app.py calls — single entry point

Routing logic:
    only PDFs  → "pdf"  (no LLM call)
    only CSV/Excel → "csv"  (no LLM call)
    both types → llm_intent_classifier(query) → "pdf" or "csv"
    LLM fails  → fallback "pdf"

Public API:
    get_intent(files, query)          -> str  ("pdf" or "csv")

Internal:
    query_classifier(files)           -> str  ("pdf", "csv", "mix")
    llm_intent_classifier(query)      -> str  ("pdf" or "csv")
"""

from config import INTENT_MODEL
from core.utils import get_groq_client


# ── File Type Classifier ──────────────────────────────────────────────────────

def query_classifier(files) -> str:
    """
    Checks uploaded file types and returns routing decision.

    Args:
        files: list of file metadata dicts from st.session_state.uploaded_files
               each dict has keys: filename, file_type, detail, size

    Returns:
        "pdf"  — only PDF files present
        "csv"  — only CSV/Excel files present
        "mix"  — both types present, needs LLM classification
    """
    has_pdf = any(f["file_type"] == "pdf"            for f in files)
    has_csv = any(f["file_type"] in ["csv", "excel"] for f in files)

    if has_pdf and not has_csv:
        return "pdf"
    if has_csv and not has_pdf:
        return "csv"
    return "mix"




# ── LLM Intent Classifier ─────────────────────────────────────────────────────

def llm_intent_classifier(query: str) -> str:
    """
    Uses INTENT_MODEL to classify query as "pdf" or "csv".
    Only called when both file types are present in session.

    Args:
        query: user's question string

    Returns:
        "pdf" or "csv"

    Raises:
        RuntimeError if Groq API call fails
    """
    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. "
                        "The user has uploaded both PDF documents and CSV/Excel files. "
                        "Classify whether the user's query is about the PDF documents "
                        "or the CSV/Excel data. "
                        "Reply with ONE word only: 'pdf' or 'csv'."
                    )
                },
                {
                    "role": "user",
                    "content": f"Classify this query:\n\n{query}"
                }
            ],
            max_tokens=5   # one word response only
        )

        intent = response.choices[0].message.content.strip().lower()

        # safety guard — LLM might return "csv file" or "PDF" etc.
        if "csv" in intent or "excel" in intent or "data" in intent:
            return "csv"
        return "pdf"  # default to pdf if unclear

    except Exception as e:
        raise RuntimeError(f"Groq intent classification failed: {e}")


# ── Master Orchestrator ───────────────────────────────────────────────────────

def get_intent(files, query: str) -> str:
    """
    Master entry point for intent detection.
    Called by app.py before every LLM query.

    Flow:
        1. query_classifier checks file types
        2. If only one type → return directly (no LLM call)
        3. If mixed → llm_intent_classifier decides
        4. If LLM fails → fallback to "pdf"

    Args:
        files : list of uploaded file objects from st.session_state
        query : user's question string

    Returns:
        "pdf" or "csv"
    """
    result = query_classifier(files)

    if result != "mix":
        return result

    try:
        return llm_intent_classifier(query)
    except RuntimeError:
        return "pdf"  # safe fallback — PDF RAG safer than failed code exec