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
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# ── Models ────────────────────────────────────────────────────────────────────
# Available models on Groq:
# "qwen/qwen3-32b"
# "groq/compound-beta"
# "groq/compound-beta-mini"
# "openai/gpt-oss-120b"
# "openai/gpt-oss-20b"
# "meta-llama/llama-4-scout-17b-16e-instruct"

INTENT_MODEL   = "compound-beta-mini"                        # fast classifier — pdf or csv
ANALYSIS_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # CSV pandas code gen — clean output
RESPONSE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # PDF RAG response — no thinking noise
SUMMARY_MODEL  = "compound-beta-mini"                        # memory compression (simple task)


# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 75


# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
SCORE_THRESHOLD = 0.35


# ── Memory ────────────────────────────────────────────────────────────────────
MAX_CACHE_SIZE     = 8    # msgs before summarization triggers
MAX_SUMMARY_TOKENS = 400  # max output tokens for Groq summary call


# ── File Upload ───────────────────────────────────────────────────────────────
MAX_FILES_UPLOAD     = 4
SUPPORTED_EXTENSIONS = [".pdf", ".csv", ".xlsx", ".xls"]


# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR     = "data/chroma_store"
CHROMA_COLLECTION_NAME = "hari_docs"