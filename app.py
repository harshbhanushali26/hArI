"""
app.py
------
Main Streamlit application for hArI.
Entry point for hArI. Composes UI from modular components.

Responsibilities:
    - Configure Streamlit page settings
    - Call inject_styles() and init_session_state() on startup
    - Orchestrate render sequence via ui/components.py

Session State:
    uploaded_files      : list  — metadata of uploaded files
    processed_files     : set   — filenames already indexed
    dataframes          : dict  — {filename: (df, metadata)} for CSV/Excel
    vector_store        : VectorStore instance
    embedding_manager   : EmbeddingManager instance
    retriever           : RAGRetriever instance
    chat_history        : list  — {role, content, sources?} for UI display
    memory_buffer       : list  — last N msgs for LLM context
    compressed_summary  : str   — Groq-summarized older messages
    prompt_sections     : dict  — parsed system_prompt.md sections
    session_ready       : bool  — True once at least 1 file indexed

Pipeline Flow:
    Upload:
        load_file() -> split_docs() -> generate_embeddings()
        -> vector_store.add_documents() [PDF]
        -> st.session_state.dataframes [CSV/Excel]

    Query:
        get_intent() -> retrieve() [PDF] or pandas exec [CSV]
        -> get_response() -> add_message() -> display
"""

import streamlit as st
from ui.styles import inject_styles
from ui.state import init_session_state
from ui.components import render_header, render_status_bar, render_suggestion_pills, render_upload_section, render_file_list, render_chat_input, render_chat_history, render_action_buttons


st.set_page_config(
    page_title="hArI",
    page_icon="🤖◈",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    inject_styles()
    init_session_state()

    render_header()
    render_suggestion_pills()
    render_upload_section()
    render_file_list()
    render_status_bar()
    render_chat_history()
    render_chat_input()
    render_action_buttons()


if __name__ == "__main__":
    main()