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


from config import GROQ_API_KEY, ANALYSIS_MODEL, SCORE_THRESHOLD, TITLE_MODEL


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


def run_pdf_pipeline(query: str, placeholder=None) -> tuple[str, str]:
    """
    Runs RAG retrieval for PDF queries.
    Returns (response_string, sources_string)
    """
    # retrieve top-k chunks
    chunks = st.session_state.retriever.retrieve(query, score_threshold=SCORE_THRESHOLD)

    # early return if no relevant chunks found
    if not chunks:
        return (
            "I couldn't find anything relevant to that in the uploaded document. "
            "Try rephrasing your question or ask something specific about the document.",
            None
        )

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
    return response, chunks


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


def handle_query(query: str, placeholder=None):
    """
    Handles a user query end-to-end and saves it to the Supabase database.
    """

    from core.utils import get_supabase_client
    supabase = get_supabase_client()
    user_id = st.session_state["user"].id

    # --- 1. CREATE A SESSION IF NEEDED ---
    if st.session_state.current_session_id is None:
        # Create a title based on the first few words of the query
        title = _generate_chat_title(query)

        # Insert new session into database
        try:
            # Try to insert new session
            response = supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "title": title
            }).execute()
            st.session_state.current_session_id = response.data[0]["id"]
            
        except Exception as e:
            # Force Streamlit to show us the error!
            st.error(f"🚨 DATABASE ERROR: {e}")
            st.stop()
            return  # Stop the code here so it doesn't refresh!

    session_id = st.session_state.current_session_id

    # --- 2. SAVE USER MESSAGE ---
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": query
        }).execute()
    except Exception as e:
        st.error(f"🚨 MESSAGE SAVE ERROR: {e}")
        st.stop()
        return
    add_message("user", query)
    st.session_state.chat_history.append({
        "role"   : "user",
        "content": query
    })

    # --- 3. RUN THE PIPELINE ---
    if should_summarize():
        with st.spinner("Compressing memory..."):
            summary = summarize()
            trim_buffer(summary)

    intent = get_intent(st.session_state.uploaded_files, query)

    if intent == "pdf":
        response_text, sources = run_pdf_pipeline(query, placeholder)
    else:
        response_text, sources = run_csv_pipeline(query, placeholder)

    # --- 4. SAVE AI RESPONSE ---
    supabase.table("messages").insert({
        "session_id": session_id,
        "role": "assistant",
        "content": response_text
    }).execute()
    add_message("assistant", response_text)
    st.session_state.chat_history.append({
        "role"   : "assistant",
        "content": response_text,
        "sources": sources
    })


def _generate_chat_title(query: str) -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=TITLE_MODEL,
            messages=[{
                "role": "user",
                "content": f"Give a 4-5 word title for a conversation starting with: '{query}'. Return ONLY the title, no quotes, no punctuation."
            }],
            max_tokens=15,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except:
        return query[:30] + "..." if len(query) > 30 else query


def run_csv_pipeline(query: str, placeholder=None) -> tuple[str, str]:
    """
    Runs explicitly-registered DuckDB SQL analysis for CSV/Excel queries.
    Supports MULTIPLE files for cross-table JOINs!
    """
    import duckdb
    import re
    from groq import Groq
    
    if not st.session_state.dataframes:
        return "No CSV or Excel file loaded in this session.", ""

    # 1. Open an explicit connection
    con = duckdb.connect()

    # 2. Register EVERY loaded file as its own table for cross-file JOINs!
    schema_parts = []
    file_names = []
    
    for raw_fname, (df, meta) in st.session_state.dataframes.items():
        # Sanitize filename into a valid SQL table name (e.g. "my data.csv" -> "my_data_csv")
        table_name = re.sub(r'\W|^(?=\d)', '_', raw_fname).lower()
        file_names.append(f"{raw_fname} ({meta['rows']} rows)")
        
        # Explicitly register the DataFrame into the connection
        con.register(table_name, df)
        
        # Build schema block for this specific table
        table_schema = f"Table Name: {table_name}\nColumns:\n"
        table_schema += "\n".join(f"  - {col}: {meta['dtypes'].get(col, '?')}" for col in meta["columns"])
        schema_parts.append(table_schema)

    combined_schema = "\n\n".join(schema_parts)
    client = Groq(api_key=GROQ_API_KEY)
    
    max_retries = 3
    sql_query = ""
    analysis_result = ""
    
    # The Prompt now dynamically supports multiple tables
    messages = [
        {"role": "system", "content": "You are a senior PostgreSQL and DuckDB expert."},
        {"role": "user", "content": (
            f"Write a SQL query to answer this request.\n\n"
            f"Available Tables & Schemas:\n{combined_schema}\n\n"
            f"Request: {query}\n\n"
            f"Rules:\n"
            f"- Return ONLY the raw SQL query\n"
            f"- Do NOT wrap it in markdown block quotes\n"
            f"- Do NOT add any explanations."
        )}
    ]

    for attempt in range(max_retries):
        try:
            sql_res = client.chat.completions.create(
                model    = ANALYSIS_MODEL,
                messages = messages,
                max_tokens = 500,
                temperature = 0.1
            )
            sql_query = sql_res.choices[0].message.content.strip()

            sql_query = re.sub(r"<think>.*?</think>", "", sql_query, flags=re.DOTALL).strip()
            sql_query = re.sub(r"^```[sS][qQ][lL]?\n?", "", sql_query).strip()
            sql_query = re.sub(r"\n?```$", "", sql_query).strip()

            # 3. Execute explicitly against our managed connection
            result_df = con.execute(sql_query).df()
            analysis_result = result_df.to_string()
            break
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                # Self-heal loop
                messages.append({"role": "assistant", "content": sql_query})
                messages.append({"role": "user", "content": f"That SQL failed with this error: {error_msg}\nFix the SQL and return ONLY the corrected SQL query."})
            else:
                analysis_result = f"Failed to run SQL after {max_retries} attempts.\nLast Error: {error_msg}\nLast SQL: {sql_query}"

    # Close the connection to free memory
    con.close()

    # Pass result to LLM for final formatting
    # Note: We just pass the first file's meta for basic context, but analysis_result is what matters
    first_fname = list(st.session_state.dataframes.keys())[0]
    _, first_meta = st.session_state.dataframes[first_fname]
    
    context = {
        "metadata"       : first_meta,
        "analysis_result": analysis_result
    }

    from core.responser import get_response
    response = get_response(
        query    = query,
        intent   = "csv",
        context  = context,
        memory   = get_context(),
        sections = st.session_state.prompt_sections,
        placeholder = placeholder
    )

    from core.utils import strip_thinking
    response = strip_thinking(response)
    
    # Sources now shows all files that were queried!
    sources = " | ".join(file_names)
    return response, sources


