"""
ui/styles.py
------------
Global CSS injection for hArI.
"""

import streamlit as st


def inject_styles():
    st.markdown("""
    <style>
        /* ── Hide default Streamlit chrome ── */
        #MainMenu, footer { display: none !important; }
        .stDeployButton { display: none !important; }
        [data-testid="stMainMenuPopover"] { display: none !important; }
        .stAppDeployButton { display: none !important; }
        [data-testid="stHeader"] {
            background-color: transparent !important;
            background: transparent !important;
        }
        [data-testid="stDecoration"] { display: none !important; }

        /* ── App base ── */
        .stApp { background-color: #0a0a0a; }
        .block-container {
            max-width: 720px !important;
            padding: 5rem 1.5rem 2rem 1.5rem !important;
        }

        /* ── Chat input bottom bar ── */
        [data-testid="stBottomBlockContainer"] {
            background-color: #0a0a0a !important;
            background: #0a0a0a !important;
            border-top: 0.5px solid #1a1a1a !important;
        }
        [data-testid="stBottomBlockContainer"] > div {
            background-color: #0a0a0a !important;
            background: #0a0a0a !important;
        }
        /* The chat input inner wrapper */
        [data-testid="stChatInput"] {
            background-color: #0a0a0a !important;
        }
        [data-testid="stChatInput"] > div {
            background-color: #111 !important;
            border: 0.5px solid #2a2a2a !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            outline: none !important;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: #534AB7 !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInputTextArea"] {
            background-color: #111 !important;
            color: #c8c8c8 !important;
            font-family: monospace !important;
            font-size: 13px !important;
        }
        [data-testid="stChatInputTextArea"]::placeholder {
            color: #444 !important;
        }

        /* ── Text input ── */
        .stTextInput > div > div > input {
            background-color: #141414 !important;
            border: 0.5px solid #2a2a2a !important;
            border-radius: 10px !important;
            color: #c8c8c8 !important;
            font-family: monospace !important;
            font-size: 13px !important;
        }
        .stTextInput > div > div > input::placeholder { color: #555 !important; }
        .stTextInput > div > div > input:focus {
            border-color: #534AB7 !important;
            box-shadow: none !important;
        }

        /* ── File uploader ── */
        .stFileUploader > div {
            background-color: #111 !important;
            border: 0.5px dashed #2a2a2a !important;
            border-radius: 8px !important;
        }

        /* ── Global buttons (main area only) ── */
        .stButton > button {
            background-color: transparent !important;
            border: 0.5px solid #222 !important;
            color: #888 !important;
            border-radius: 6px !important;
            font-family: monospace !important;
            font-size: 12px !important;
        }
        .stButton > button:hover {
            border-color: #555 !important;
            color: #bbb !important;
        }

        
        [data-testid="stSidebar"] > div:first-child {
            padding: 20px 12px !important;
        }

        /* ── Sidebar collapse arrow ── */
        [data-testid="stSidebarCollapseButton"] button {
            background-color: #0d0d0d !important;
            border: 0.5px solid #222 !important;
            color: #555 !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover {
            border-color: #534AB7 !important;
            color: #7F77DD !important;
        }

        /* ── Sidebar buttons — base style ── */
        [data-testid="stSidebar"] .stButton > button {
            background-color: transparent !important;
            border: 0.5px solid transparent !important;
            color: #b0b5c1 !important; /* <--- CHANGED: Beautiful bright silver text! */
            border-radius: 5px !important;
            font-family: monospace !important;
            font-size: 13px !important; /* <--- CHANGED: Slightly bigger to read easily */
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 5px 8px !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #1e2233 !important; /* <--- CHANGED: Beautiful lighter navy hover */
            border-color: #1e2233 !important;
            color: #ffffff !important; /* <--- CHANGED: Turns pure white when hovered */
        }

        /* ── New Chat button — purple ── */
        .st-key-new_chat_btn > button {
            border-color: #2a2450 !important;
            color: #7F77DD !important;
        }
        .st-key-new_chat_btn > button:hover {
            background-color: #1a1429 !important;
            border-color: #534AB7 !important;
            color: #AFA9EC !important;
        }

        /* ── Download button — force override Streamlit default ── */
        .st-key-download_chat_btn button,
        .st-key-download_chat_btn > button {
            background-color: transparent !important;
            background: transparent !important;
            border: 0.5px solid transparent !important;
            color: #555 !important;
            font-family: monospace !important;
            font-size: 12px !important;
            border-radius: 5px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            box-shadow: none !important;
        }
        .st-key-download_chat_btn button:hover,
        .st-key-download_chat_btn > button:hover {
            background-color: #141414 !important;
            border-color: #1e1e1e !important;
            color: #AFA9EC !important;
            box-shadow: none !important;
        }

        /* ── Delete Chat button — subtle red ── */
        .st-key-delete_chat_btn > button {
            color: #7a3030 !important;
            border-color: transparent !important;
        }
        .st-key-delete_chat_btn > button:hover {
            background-color: #1a0808 !important;
            border-color: #3a1010 !important;
            color: #e05555 !important;
        }

        /* ── Clear button ── */
        .st-key-clear_btn > button {
            color: #555 !important;
            border-color: transparent !important;
        }
        .st-key-clear_btn > button:hover {
            background-color: #141414 !important;
            color: #AFA9EC !important;
        }

        /* ── Reset button — red ── */
        .st-key-reset_btn > button {
            color: #7a3030 !important;
            border-color: transparent !important;
        }
        .st-key-reset_btn > button:hover {
            background-color: #1a0808 !important;
            border-color: #3a1010 !important;
            color: #e05555 !important;
        }

        /* ── Log Out button ── */
        .st-key-logout_btn > button {
            color: #444 !important;
            border-color: transparent !important;
        }
        .st-key-logout_btn > button:hover {
            color: #888 !important;
            border-color: #1e1e1e !important;
        }

        /* ── Sidebar scrollbar ── */
        [data-testid="stSidebar"] ::-webkit-scrollbar { width: 2px; }
        [data-testid="stSidebar"] ::-webkit-scrollbar-track { background: #0d0d0d; }
        [data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: #1e1e1e; }

        /* ── Misc ── */
        .stSpinner > div { border-top-color: #7F77DD !important; }
        hr { border-color: #1a1a1a !important; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0a0a; }
        ::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)