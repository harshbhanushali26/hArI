"""
ui/state.py
-----------
Session state initialization and chat memory management.

Responsibilities:
    - Initialize all st.session_state keys on first run
    - Provide clear_chat() to reset conversation without losing indexed files
"""

import streamlit as st

from rag.embedder import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

from core.responser import load_prompt


def init_session_state():
    """Initialize all session state keys on first run."""

    if "uploaded_files"  not in st.session_state:
        st.session_state.uploaded_files     = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files    = set()
    if "dataframes"      not in st.session_state:
        st.session_state.dataframes         = {}

    if "vector_store"      not in st.session_state:
        st.session_state.vector_store       = VectorStore()
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager  = EmbeddingManager()
    if "retriever"         not in st.session_state:
        st.session_state.retriever          = RAGRetriever(
            st.session_state.vector_store,
            st.session_state.embedding_manager
        )

    if "chat_history"       not in st.session_state:
        st.session_state.chat_history       = []
    if "memory_buffer"      not in st.session_state:
        st.session_state.memory_buffer      = []
    if "compressed_summary" not in st.session_state:
        st.session_state.compressed_summary = ""

    if "prompt_sections" not in st.session_state:
        st.session_state.prompt_sections    = load_prompt()

    if "session_ready" not in st.session_state:
        st.session_state.session_ready      = False


def clear_chat():
    """Clears chat history and memory buffer. Keeps files indexed."""
    st.session_state.chat_history       = []
    st.session_state.memory_buffer      = []
    st.session_state.compressed_summary = ""

