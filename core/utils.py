"""
utils.py
--------
Shared utility functions used across multiple hArI modules.

Responsibilities:
    - Provide a single Groq client factory (used by memory, query_intent, responser)
    - Provide file extension helper (used by file_processor, query_intent)

Design decisions:
    - Functions here are pure helpers — no state, no side effects
    - get_groq_client() creates a new client instance each call — lightweight,
      no need to persist a single client across modules
    - get_file_extension() is the single source of truth for extension detection —
      avoids duplicating Path logic across files

Public API:
    get_groq_client()        -> Groq
    get_file_extension(file) -> str
"""

from pathlib import Path
from groq import Groq
from config import GROQ_API_KEY


def get_groq_client() -> Groq:
    """
    Returns a Groq client instance using API key from config.
    Used by memory.py, query_intent.py, and responser.py.

    Raises:
        ValueError if GROQ_API_KEY is not set in .env
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Check your .env file.")
    return Groq(api_key=GROQ_API_KEY)


def get_file_extension(file) -> str:
    """
    Returns lowercase file extension from a Streamlit uploaded file object.
    e.g. 'report.PDF' -> '.pdf', 'sales.CSV' -> '.csv'

    Args:
        file: Streamlit UploadedFile object with a .name attribute

    Returns:
        lowercase extension string including dot e.g. '.pdf', '.csv', '.xlsx'
    """
    return Path(file.name).suffix.lower()


def strip_thinking(text: str) -> str:
    """
    Strips qwen/reasoning model thinking tags from response.
    Models like qwen-qwq-32b wrap chain-of-thought in <think>...</think>.
    We only want the final answer after the thinking block.
    """
    import re
    # remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()