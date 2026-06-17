"""
ui/components.py
----------------
All Streamlit UI render functions for hArI.
"""

import streamlit as st
import html 
from core.utils import get_supabase_client

from ui.handlers import ingest_file, remove_file, handle_query, clear_chat, reset_session
from config import MAX_FILES_UPLOAD, SUPPORTED_EXTENSIONS


def render_header():
    """Renders hArI logo + title."""
    st.markdown("""
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:24px;">
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


def render_status_bar():
    """Renders file count + mode badge. Only shown when session is ready."""
    if not st.session_state.session_ready:
        return

    has_pdf = any(f["file_type"] == "pdf"            for f in st.session_state.uploaded_files)
    has_csv = any(f["file_type"] in ["csv", "excel"] for f in st.session_state.uploaded_files)

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
    """Renders full chat history. User right-aligned, AI left with citations."""
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

            _, content_col = st.columns([0.08, 0.92])
            with content_col:
                st.markdown(msg["content"])

            # if msg.get("sources"):
            #     raw_sources = msg["sources"]
            #     seen_files  = []
            #     unique_refs = []
            #     for ref in raw_sources.split("  |  "):
            #         fname = ref.split(" · ")[0].strip()
            #         if fname not in seen_files:
            #             seen_files.append(fname)
            #             unique_refs.append(ref.strip())
            #     clean_sources = "  |  ".join(unique_refs)
            #     st.markdown(f"""
            #         <div style="font-size:10px; color:#7F77DD; margin-bottom:12px;
            #                     margin-left:32px; font-family:monospace">
            #             ↳ {html.escape(clean_sources)}
            #         </div>
            #     """, unsafe_allow_html=True)

            if msg.get("sources"):
                sources_data = msg["sources"]
                
                # If it's a LIST, it came from the PDF pipeline (raw chunks)
                if isinstance(sources_data, list):
                    with st.expander("📚 View Retrieved Sources", expanded=False):
                        for i, chunk in enumerate(sources_data):
                            # Get the page number and filename
                            page = chunk.get("metadata", {}).get("page", "?")
                            fname = chunk.get("metadata", {}).get("filename", "Document")
                            
                            st.markdown(f"**Source {i+1}** · {fname} (Page {page})")
                            # Show the actual markdown text that Groq read!
                            st.info(chunk.get("content", ""))
                            if i < len(sources_data) - 1:
                                st.divider()
                                
                # If it's a STRING, it came from the CSV pipeline
                else:
                    st.markdown(f"""
                        <div style="font-size:10px; color:#7F77DD; margin-bottom:12px;
                                    margin-left:32px; font-family:monospace">
                            ↳ {html.escape(str(sources_data))}
                        </div>
                    """, unsafe_allow_html=True)

            # Add Feedback Widget below Assistant Responses
            if msg["role"] == "assistant":
                # Get the index of this message to create a unique key
                idx = st.session_state.chat_history.index(msg)
                
                # The callback that runs the millisecond a user clicks a thumb!
                def log_feedback(message_idx):
                    val = st.session_state[f"fb_{message_idx}"]
                    
                    # Log to Supabase
                    supabase = get_supabase_client()
                    
                    # Find the user's question (it's the message right before the assistant's)
                    user_query = st.session_state.chat_history[message_idx - 1]["content"]
                    ai_response = st.session_state.chat_history[message_idx]["content"]
                    
                    try:
                        supabase.table("telemetry").insert({
                            "session_id": st.session_state.current_session_id,
                            "query": user_query,
                            "response": ai_response,
                            "feedback": val
                        }).execute()
                    except Exception as e:
                        print(f"Telemetry failed: {e}")
                        
                    st.toast("Thank you for your feedback!" if val == 1 else "Thanks, we will improve this.", icon="✅")

                # Streamlit's native beautiful feedback widget!
                st.feedback("thumbs", key=f"fb_{idx}", on_change=log_feedback, args=[idx])


def render_upload_section():
    """Renders file uploader and triggers ingest on new uploads."""
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

    current_filenames = [f.name for f in uploaded] if uploaded else []
    for old_file in list(st.session_state.processed_files):
        if old_file not in current_filenames:
            remove_file(old_file)

    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.processed_files]
        for file in new_files:
            if len(st.session_state.processed_files) >= MAX_FILES_UPLOAD:
                st.warning(f"Max {MAX_FILES_UPLOAD} files allowed per session.")
                break
            with st.spinner(f"Indexing {file.name}..."):
                ingest_file(file)


def render_file_list():
    pass


def render_chat_input():
    """Renders bottom-pinned chat input. Disabled until session_ready."""
    if not st.session_state.session_ready:
        st.chat_input("Upload a document to start chatting...", disabled=True)
        return

    query = st.chat_input("Ask anything about your documents...")

    if query:
        import html as html_lib
        st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin-bottom:12px">
                <div style="background:#1e1a3a; border:0.5px solid #2a2450; border-radius:10px 10px 2px 10px;
                            padding:9px 14px; font-size:13px; color:#AFA9EC; max-width:78%; font-family:monospace">
                    {html_lib.escape(query)}
                </div>
            </div>
        """, unsafe_allow_html=True)

        placeholder = st.empty()
        handle_query(query.strip(), placeholder)
        st.rerun()


def render_chat_history_sidebar():
    """Fetches and displays past chat sessions in the sidebar."""
    supabase = get_supabase_client()
    user_id  = st.session_state["user"].id

    # ── Branding ──
    st.sidebar.markdown("""
        <div style="font-size:15px; font-weight:500; color:#e8e8e8;
                    font-family:monospace; margin-bottom:16px;
                    padding:0 0 12px 0; border-bottom:0.5px solid #1a1a1a">
            h<span style="color:#7F77DD">A</span>r<span style="color:#7F77DD">I</span>
            <span style="font-size:10px; color:#333; margin-left:6px">v1.0</span>
        </div>
    """, unsafe_allow_html=True)

    # ── New Chat ──
    if st.sidebar.button("+ New Chat", key="new_chat_btn", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.chat_history = []
        st.rerun()

    st.sidebar.markdown(
        "<div style='font-size:10px; color:#444; letter-spacing:0.08em; "
        "margin-top:24px; margin-bottom:8px; font-family:monospace'>RECENT RESEARCH</div>",
        unsafe_allow_html=True
    )

    # ── Fetch sessions ──
    try:
        response = supabase.table("chat_sessions").select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        sessions = response.data
    except Exception:
        st.sidebar.error("Could not load history.")
        return

    if not sessions:
        st.sidebar.markdown(
            "<div style='font-size:11px; color:#333; font-family:monospace; padding:4px 8px'>No past conversations.</div>",
            unsafe_allow_html=True
        )
        return

    for session in sessions:
        is_active = (session["id"] == st.session_state.current_session_id)
        # active session gets a green dot prefix, others plain
        prefix    = "● " if is_active else ""
        label     = f"{prefix}{session['title']}"

        btn_style = "color:#7F77DD !important;" if is_active else ""

        if st.sidebar.button(label, key=f"session_{session['id']}", use_container_width=True):
            st.session_state.current_session_id = session["id"]
            try:
                msg_response = supabase.table("messages").select("*")\
                    .eq("session_id", session["id"])\
                    .order("created_at", desc=False)\
                    .execute()
                st.session_state.chat_history = [
                    {"role": row["role"], "content": row["content"]}
                    for row in msg_response.data
                ]
            except Exception:
                st.sidebar.error("Failed to load messages.")
            st.rerun()


def render_action_buttons():
    """Renders settings/action buttons in the sidebar."""

    st.sidebar.markdown(
        "<div style='margin-top:30px'></div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<div style='font-size:10px; color:#444; letter-spacing:0.08em; "
        "margin-bottom:8px; font-family:monospace'>SETTINGS</div>",
        unsafe_allow_html=True
    )

    # ── Download Chat ──
    if st.session_state.chat_history:
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}:\n{msg['content']}"
            for msg in st.session_state.chat_history
        ])
        st.sidebar.download_button(
            label="Download Chat",
            data=chat_text,
            file_name=f"hari_chat_{st.session_state.current_session_id}.txt"
                        if st.session_state.current_session_id else "hari_chat.txt",
            mime="text/plain",
            key="download_chat_btn",
            use_container_width=True
        )

    # ── Delete This Chat ──
    if st.session_state.current_session_id:
        if st.sidebar.button("Delete Chat", key="delete_chat_btn", use_container_width=True ):
            supabase = get_supabase_client()
            try:
                supabase.table("chat_sessions").delete()\
                    .eq("id", st.session_state.current_session_id).execute()
            except Exception:
                pass
            st.session_state.current_session_id = None
            st.session_state.chat_history = []
            st.rerun()

    # ── Clear / Reset ──
    if st.session_state.chat_history or st.session_state.session_ready:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.sidebar.button("Clear", key="clear_btn", use_container_width=True):
                clear_chat()
                st.rerun()
        with col2:
            if st.sidebar.button("Reset", key="reset_btn", use_container_width=True):
                reset_session()
                st.rerun()

    # ── Log Out ──
    st.sidebar.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    if st.sidebar.button("Log Out", key="logout_btn", use_container_width=True):
        supabase = get_supabase_client()
        supabase.auth.sign_out()
        st.session_state["user"] = None
        st.session_state.pop("access_token", None)
        st.session_state.pop("refresh_token", None)
        st.rerun()