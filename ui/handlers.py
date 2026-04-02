"""
ui/handlers.py
--------------
Business logic handlers for file ingestion and query pipelines.

Responsibilities:
    - Ingest uploaded files into ChromaDB or session dataframes
    - Route queries to PDF (RAG) or CSV (pandas) pipeline
    - Manage session reset including ChromaDB reinitialization
    - Remove individual files from session and vector store
"""

from groq import Groq
import streamlit as st
from core.memory import (
    add_message,
    get_context,
    should_summarize,
    summarize,
    trim_buffer
)

from core.file_processor import load_file
from core.query_intent import get_intent
from core.responser import get_response
from core.utils import strip_thinking


from rag.embedder import split_docs
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

from ui.state import clear_chat


from config import GROQ_API_KEY, ANALYSIS_MODEL


def ingest_file(file):
    """
    Processes a single uploaded file and indexes it.
    PDF      -> chunks -> embeddings -> ChromaDB
    CSV/Excel -> DataFrame -> session state
    """
    try:
        result = load_file(file)
        fname  = result["filename"]
        ftype  = result["file_type"]

        if ftype == "pdf":
            documents = result["documents"]
            chunks, total_chunks, _ = split_docs(documents)
            texts = [c.page_content for c in chunks]
            embeddings, _, _ = st.session_state.embedding_manager.generate_embeddings(texts)
            st.session_state.vector_store.add_documents(chunks, embeddings)
            detail = f"{result['metadata']['total_pages']} pages"

        else:  # csv or excel
            df       = result["dataframe"]
            metadata = result["metadata"]
            st.session_state.dataframes[fname] = (df, metadata)
            detail = f"{metadata['rows']} rows · {len(metadata['columns'])} cols"

        st.session_state.processed_files.add(fname)
        st.session_state.uploaded_files.append({
            "filename"  : fname,
            "file_type" : ftype,
            "detail"    : detail,
            "size"      : result["metadata"].get("file_size", "")
        })
        st.session_state.session_ready = True
        st.toast(f"✓ {fname} ready", icon="✅")

    except Exception as e:
        st.error(f"Failed to index {file.name}: {e}")


def handle_query(query: str, placeholder=None):
    """
    Handles a user query end-to-end.
    1. Add user message to memory + chat history
    2. Check if memory needs summarization
    3. Detect intent (pdf or csv)
    4. Run correct pipeline
    5. Add AI response to memory + chat history
    """
    # add user message
    add_message("user", query)
    st.session_state.chat_history.append({
        "role"   : "user",
        "content": query
    })

    # check memory summarization
    if should_summarize():
        with st.spinner("Compressing memory..."):
            summary = summarize()
            trim_buffer(summary)

    # detect intent
    intent = get_intent(
        st.session_state.uploaded_files,
        query
    )

    # run correct pipeline
    if intent == "pdf":
        response, sources = run_pdf_pipeline(query, placeholder)
    else:
        response, sources = run_csv_pipeline(query, placeholder)

    # add AI response to memory + chat history
    add_message("assistant", response)
    st.session_state.chat_history.append({
        "role"   : "assistant",
        "content": response,
        "sources": sources
    })


def run_pdf_pipeline(query: str, placeholder=None) -> tuple[str, str]:
    """
    Runs RAG retrieval for PDF queries.
    Returns (response_string, sources_string)
    """
    # retrieve top-k chunks
    chunks = st.session_state.retriever.retrieve(query)

    # build sources citation string for UI
    sources = ""
    if chunks:
        seen = []
        for c in chunks:
            src  = c["metadata"].get("source", "")
            page = c["metadata"].get("page", "")
            ref  = f"{src} · p{page}"
            if ref not in seen:
                seen.append(ref)
        sources = "  |  ".join(seen)

    # get LLM response
    response = get_response(
        query    = query,
        intent   = "pdf",
        context  = chunks,
        memory   = get_context(),
        sections = st.session_state.prompt_sections,
        placeholder = placeholder
    )

    response = strip_thinking(response)
    return response, sources


def run_csv_pipeline(query: str, placeholder=None) -> tuple[str, str]:
    """
    Runs pandas analysis for CSV/Excel queries.
    1. Build pandas code via Groq
    2. Execute safely
    3. Pass result to LLM for formatting
    Returns (response_string, sources_string)
    """
    if not st.session_state.dataframes:
        return "No CSV or Excel file loaded in this session.", ""

    # use first available dataframe (or detected one)
    fname    = list(st.session_state.dataframes.keys())[0]
    df, meta = st.session_state.dataframes[fname]

    # build schema string for code gen prompt
    schema = "\n".join(
        f"  - {col}: {meta['dtypes'].get(col, '?')}"
        for col in meta["columns"]
    )
    preview = str(df.head(3).to_dict(orient="records"))

    # ask Groq to generate pandas code
    code_prompt = (
        f"You are a pandas expert. Write Python code to answer this query.\n"
        f"DataFrame variable name is 'df'.\n"
        f"Schema:\n{schema}\n"
        f"Preview:\n{preview}\n\n"
        f"Query: {query}\n\n"
        f"Rules:\n"
        f"- Use only pandas and numpy\n"
        f"- Store final result in a variable called 'result'\n"
        f"- result must be a string, number, or DataFrame\n"
        f"- No imports needed, df is already loaded\n"
        f"- Return ONLY the Python code, no explanation"
    )

    try:
        client   = Groq(api_key=GROQ_API_KEY)
        code_res = client.chat.completions.create(
            model    = ANALYSIS_MODEL,
            messages = [{"role": "user", "content": code_prompt}],
            max_tokens = 500
        )
        code = code_res.choices[0].message.content.strip()

        # strip markdown code fences + thinking tags robustly
        import re
        code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL).strip()
        code = re.sub(r"^```[\w]*\n?", "", code).strip()
        code = re.sub(r"\n?```$", "", code).strip()

        # safe execution — only pandas/numpy allowed
        safe_globals = {"df": df, "__builtins__": {}}
        try:
            import pandas as pd
            import numpy as np
            safe_globals["pd"] = pd
            safe_globals["np"] = np
            exec(code, safe_globals)
            analysis_result = str(safe_globals.get("result", "No result produced."))
        except Exception as exec_err:
            analysis_result = f"Execution error: {exec_err}\nGenerated code:\n{code}"

    except Exception as e:
        analysis_result = f"Code generation failed: {e}"

    # pass result to LLM for final formatting
    context = {
        "metadata"       : meta,
        "analysis_result": analysis_result
    }

    response = get_response(
        query    = query,
        intent   = "csv",
        context  = context,
        memory   = get_context(),
        sections = st.session_state.prompt_sections,
        placeholder = placeholder
    )

    response = strip_thinking(response)
    sources = f"{fname} · {meta['rows']} rows"
    return response, sources


def reset_session():
    """Full reset — clears everything including indexed files and ChromaDB."""
    clear_chat()
    st.session_state.uploaded_files  = []
    st.session_state.processed_files = set()
    st.session_state.dataframes      = {}
    st.session_state.session_ready   = False
    st.session_state.vector_store    = VectorStore()
    st.session_state.retriever       = RAGRetriever(
        st.session_state.vector_store,
        st.session_state.embedding_manager
    )


def remove_file(filename: str):
    """Removes a file from session state and ChromaDB."""
    st.session_state.uploaded_files  = [
        f for f in st.session_state.uploaded_files
        if f["filename"] != filename
    ]
    st.session_state.processed_files.discard(filename)
    st.session_state.dataframes.pop(filename, None)
    st.session_state.vector_store.delete_source(filename)

    if not st.session_state.uploaded_files:
        st.session_state.session_ready = False

