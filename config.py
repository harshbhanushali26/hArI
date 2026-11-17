"""
config.py
---------
Central configuration for hArI.
All constants, model names, and environment variables live here.
Import from this file — never hardcode values in other modules.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── API ───────────────────────────────────────────────────────────────────────
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")


# ── Models ────────────────────────────────────────────────────────────────────
INTENT_MODEL   = "openai/gpt-oss-20b"                        # fast classifier — pdf or csv
ANALYSIS_MODEL = "openai/gpt-oss-120b"                       # SQL generation for DuckDB
RESPONSE_MODEL = "openai/gpt-oss-120b"                       # Grounded response generation
SUMMARY_MODEL  = "openai/gpt-oss-20b"                        # Memory compression
TITLE_MODEL    = "openai/gpt-oss-20b"                        # Chat session titling


# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
# SCORE_THRESHOLD = 0.35


# ── Memory ────────────────────────────────────────────────────────────────────
MAX_CACHE_SIZE     = 8    # msgs before summarization triggers
MAX_SUMMARY_TOKENS = 400  # max output tokens for Groq summary call


# ── File Upload ───────────────────────────────────────────────────────────────
MAX_FILES_UPLOAD     = 4
SUPPORTED_EXTENSIONS = [".pdf", ".csv", ".xlsx", ".xls"]


