"""
ui/components.py
----------------
All Streamlit UI render functions for hArI.

Responsibilities:
    - Render header, suggestion pills, upload section, file list
    - Render status bar, chat history, chat input, action buttons
    - Wire UI events to handlers and state functions
"""

import html
import streamlit as st

from ui.handlers import ingest_file, remove_file, handle_query, clear_chat, reset_session
from config import MAX_FILES_UPLOAD, SUPPORTED_EXTENSIONS


def render_header():
    """Renders hArI logo + title."""
    st.markdown("""
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:24px; padding-top:8px">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                <rect width="36" height="36" rx="8" fill="#1a1a1a"/>
                <circle cx="18" cy="18" r="3" fill="#7F77DD"/>
                <line x1="18" y1="8"  x2="18" y2="13" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="18" y1="23" x2="18" y2="28" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="8"  y1="18" x2="13" y2="18" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="23" y1="18" x2="28" y2="18" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="11.5" y1="11.5" x2="14.9" y2="14.9" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                <line x1="21.1" y1="21.1" x2="24.5" y2="24.5" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                <line x1="24.5" y1="11.5" x2="21.1" y2="14.9" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                <line x1="14.9" y1="21.1" x2="11.5" y2="24.5" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-size:28px; font-weight:500; color:#e8e8e8; letter-spacing:-0.5px; line-height:1.1">
                    h<span style="color:#7F77DD">A</span>r<span style="color:#7F77DD">I</span>
                </div>
                <div style="font-size:12px; color:#666; margin-top:2px">document intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_suggestion_pills():
    """Renders non-clickable suggestion pill chips."""
    st.markdown("""
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:24px">
            <span style="background:#141414; border:0.5px solid #2a2a2a; border-radius:20px; padding:5px 14px; font-size:12px; color:#999; font-family:monospace">
                <span style="color:#7F77DD; margin-right:4px">◈</span>summarise this doc
            </span>
            <span style="background:#141414; border:0.5px solid #2a2a2a; border-radius:20px; padding:5px 14px; font-size:12px; color:#999; font-family:monospace">
                <span style="color:#7F77DD; margin-right:4px">◈</span>key points
            </span>
            <span style="background:#141414; border:0.5px solid #2a2a2a; border-radius:20px; padding:5px 14px; font-size:12px; color:#999; font-family:monospace">
                <span style="color:#7F77DD; margin-right:4px">◈</span>analyse this CSV
            </span>
            <span style="background:#141414; border:0.5px solid #2a2a2a; border-radius:20px; padding:5px 14px; font-size:12px; color:#999; font-family:monospace">
                <span style="color:#7F77DD; margin-right:4px">◈</span>find all dates
            </span>
            <span style="background:#141414; border:0.5px solid #2a2a2a; border-radius:20px; padding:5px 14px; font-size:12px; color:#999; font-family:monospace">
                <span style="color:#7F77DD; margin-right:4px">◈</span>compare both files
            </span>
        </div>
    """, unsafe_allow_html=True)


def render_upload_section():
    """
    Renders file uploader and triggers ingest_file() for new uploads.
    Limits to MAX_FILES_UPLOAD files per session.
    """
    st.markdown(
        "<div style='font-size:10px; color:#999; letter-spacing:0.08em; margin-bottom:8px'>DOCUMENTS</div>",
        unsafe_allow_html=True
    )

    if len(st.session_state.processed_files) >= MAX_FILES_UPLOAD:
        st.markdown(
            f"<div style='font-size:12px; color:#555; font-family:monospace; padding:8px 0'>Max {MAX_FILES_UPLOAD} files reached.</div>",
            unsafe_allow_html=True
        )
        return

    uploaded = st.file_uploader(
        label="Upload documents",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=f"PDF · CSV · XLSX — up to {MAX_FILES_UPLOAD} files"
    )

    if uploaded:
        new_files = [
            f for f in uploaded
            if f.name not in st.session_state.processed_files
        ]
        for file in new_files:
            if len(st.session_state.processed_files) >= MAX_FILES_UPLOAD:
                st.warning(f"Max {MAX_FILES_UPLOAD} files allowed per session.")
                break
            with st.spinner(f"Indexing {file.name}..."):
                ingest_file(file)


def render_file_list():
    """
    Renders indexed file cards with colored dot, filename,
    page/row count, and remove button.
    """
    if not st.session_state.uploaded_files:
        return

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    for i, file_meta in enumerate(st.session_state.uploaded_files):
        fname     = file_meta["filename"]
        ftype     = file_meta["file_type"]
        dot_color = "#7F77DD" if ftype == "pdf" else "#1D9E75"
        detail    = file_meta.get("detail", "")

        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px; padding:7px 12px;
                            background:#111; border:0.5px solid #1e1e1e; border-radius:6px; margin-bottom:6px">
                    <div style="width:7px; height:7px; background:{dot_color}; border-radius:50%; flex-shrink:0"></div>
                    <span style="font-size:12px; color:#bbb; font-family:monospace; flex:1">{fname}</span>
                    <span style="font-size:10px; color:#555; font-family:monospace">{detail}</span>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✕", key=f"remove_{i}_{fname}"):
                remove_file(fname)
                st.rerun()


def render_status_bar():
    """
    Renders chunk count + mode badge (pdf only / csv only / mixed mode).
    Only shown when session is ready.
    """
    if not st.session_state.session_ready:
        return

    chunk_count = st.session_state.vector_store.collection_count()

    has_pdf = any(f["file_type"] == "pdf"              for f in st.session_state.uploaded_files)
    has_csv = any(f["file_type"] in ["csv", "excel"]   for f in st.session_state.uploaded_files)

    if has_pdf and has_csv:
        mode_label, mode_color, mode_bg, mode_border = "mixed mode", "#534AB7", "#1a1429", "#2a2050"
    elif has_pdf:
        mode_label, mode_color, mode_bg, mode_border = "pdf mode",   "#534AB7", "#1a1429", "#2a2050"
    else:
        mode_label, mode_color, mode_bg, mode_border = "csv mode",   "#1D9E75", "#0f1f14", "#1a3a22"

    file_count = len(st.session_state.uploaded_files)
    col_left, col_right = st.columns([8, 2])
    with col_left:
        st.markdown(
            f'<p style="font-size:11px;color:#639922;font-family:monospace;margin:10px 0 16px">'
            f'● {file_count} file(s) ready</p>',
            unsafe_allow_html=True
        )
    with col_right:
        st.markdown(
            f'<p style="font-size:10px;background:{mode_bg};color:{mode_color};'
            f'border:0.5px solid {mode_border};border-radius:4px;padding:2px 8px;'
            f'font-family:monospace;text-align:center;margin:10px 0 16px">{mode_label}</p>',
            unsafe_allow_html=True
        )


def render_chat_history():
    """
    Renders full chat history.
    User messages right-aligned, AI messages left with source citations.
    """
    if not st.session_state.chat_history:
        if st.session_state.session_ready:
            st.markdown("""
                <div style="text-align:center; padding:40px 0; color:#444; font-size:12px; font-family:monospace">
                    ◈ ask anything about your documents
                </div>
            """, unsafe_allow_html=True)
        return

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; margin-bottom:12px">
                    <div style="background:#1e1a3a; border:0.5px solid #2a2450; border-radius:10px 10px 2px 10px;
                                padding:9px 14px; font-size:13px; color:#AFA9EC; max-width:78%; font-family:monospace">
                        {html.escape(msg["content"])}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # avatar row
            st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px">
                    <div style="width:22px; height:22px; background:#1a1a1a; border-radius:50%;
                                display:flex; align-items:center; justify-content:center; flex-shrink:0">
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                            <circle cx="5" cy="5" r="2" fill="#7F77DD"/>
                            <line x1="5" y1="1" x2="5" y2="3" stroke="#534AB7" stroke-width="0.8" stroke-linecap="round"/>
                            <line x1="5" y1="7" x2="5" y2="9" stroke="#534AB7" stroke-width="0.8" stroke-linecap="round"/>
                            <line x1="1" y1="5" x2="3" y2="5" stroke="#534AB7" stroke-width="0.8" stroke-linecap="round"/>
                            <line x1="7" y1="5" x2="9" y2="5" stroke="#534AB7" stroke-width="0.8" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div style="font-size:11px; color:#555; font-family:monospace">hArI</div>
                </div>
            """, unsafe_allow_html=True)


            # AI response content — pure st.markdown so markdown syntax renders
            _, content_col = st.columns([0.08, 0.92])
            with content_col:
                st.markdown(msg["content"])

            # sources — only show first unique source, not all pages
            if msg.get("sources"):
                raw_sources = msg["sources"]
                # deduplicate: keep only unique filenames, drop duplicate page refs
                seen_files = []
                unique_refs = []
                for ref in raw_sources.split("  |  "):
                    fname = ref.split(" · ")[0].strip()
                    if fname not in seen_files:
                        seen_files.append(fname)
                        unique_refs.append(ref.strip())
                clean_sources = "  |  ".join(unique_refs)

                st.markdown(f"""
                    <div style="font-size:10px; color:#7F77DD; margin-bottom:12px;
                                margin-left:32px; font-family:monospace">
                        ↳ {html.escape(clean_sources)}
                    </div>
                """, unsafe_allow_html=True)


def render_chat_input():
    """
    Renders query input + send button.
    Disabled until session_ready is True.
    """
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    if not st.session_state.session_ready:
        st.markdown("""
            <div style="background:#111; border:0.5px solid #1a1a1a; border-radius:10px;
                        padding:12px 16px; font-size:12px; color:#444; font-family:monospace; text-align:center">
                upload a document to start chatting
            </div>
        """, unsafe_allow_html=True)
        return

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            query = st.text_input(
                label="query",
                placeholder="Ask anything about your documents...",
                label_visibility="collapsed"
            )
        with col2:
            send = st.form_submit_button("→", use_container_width=True)

    if send and query.strip():
        placeholder = st.empty()
        handle_query(query.strip(), placeholder)
        st.rerun()


def render_action_buttons():
    """Renders clear chat and reset session buttons."""
    if not st.session_state.chat_history and not st.session_state.session_ready:
        return

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("clear chat", use_container_width=True):
            clear_chat()
            st.rerun()
    with col2:
        if st.button("reset session", use_container_width=True):
            reset_session()
            st.rerun()

