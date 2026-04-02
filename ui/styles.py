"""
ui/styles.py
------------
Global CSS injection for hArI.

Responsibilities:
    - Inject app-wide Streamlit style overrides
    - Define base theme: dark background, monospace fonts, purple accents
"""

import streamlit as st


def inject_styles():
    st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        .stApp { background-color: #0a0a0a; }
        .block-container {
            max-width: 720px !important;
            padding: 2rem 1.5rem !important;
        }
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
        .stFileUploader > div {
            background-color: #111 !important;
            border: 0.5px dashed #2a2a2a !important;
            border-radius: 8px !important;
        }
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
        .stSpinner > div { border-top-color: #7F77DD !important; }
        hr { border-color: #1a1a1a !important; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0a0a; }
        ::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)

