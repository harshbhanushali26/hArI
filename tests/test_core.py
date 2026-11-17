"""
tests/test_core.py
------------------
Automated test suite for hArI's core data processing, chunking, 
routing, and utility functions.

Executes locally with zero network calls and zero API cost.
"""

import pytest
import pandas as pd
from core.file_processor import Document, _clean_dataframe, _build_df_metadata
from core.query_intent import query_classifier
from core.utils import strip_thinking
from rag.embedder import split_docs


# ── 1. Document Chunking & Metadata Tests ─────────────────────────────────────

def test_split_docs_retains_metadata():
    """Verifies that split_docs retains source, page, and adds chunk_index."""
    sample_text = (
        "Retrieval-Augmented Generation (RAG) optimizes the output of an LLM. "
        "It references an authoritative knowledge base outside of its training data. "
        "This ensures factual grounding."
    )
    doc = Document(
        page_content=sample_text,
        metadata={"source": "report.pdf", "page": 1, "file_type": "pdf"}
    )
    
    chunks, total_chunks, total_docs = split_docs([doc])
    
    assert total_docs == 1
    assert total_chunks >= 1
    assert len(chunks) == total_chunks
    
    # Check metadata propagation
    first_chunk = chunks[0]
    assert first_chunk.metadata["source"] == "report.pdf"
    assert first_chunk.metadata["page"] == 1
    assert first_chunk.metadata["file_type"] == "pdf"
    assert "chunk_index" in first_chunk.metadata
    assert first_chunk.metadata["chunk_index"] == 0


def test_split_docs_empty_input():
    """split_docs should raise ValueError if no documents are provided."""
    with pytest.raises(ValueError, match="No documents provided"):
        split_docs([])


# ── 2. DataFrame Cleaning & Metadata Tests ────────────────────────────────────

def test_clean_dataframe():
    """Verifies that whitespace in column names and completely empty rows are removed."""
    dirty_data = {
        "  customer_id  ": [1, 2, None],
        " amount ": [100.5, None, None],
        " empty_col ": [None, None, None]
    }
    df = pd.DataFrame(dirty_data)
    
    cleaned_df = _clean_dataframe(df)
    
    # Assert column names have no leading/trailing spaces
    assert "customer_id" in cleaned_df.columns
    assert "amount" in cleaned_df.columns
    # Assert fully empty column was dropped
    assert "empty_col" not in cleaned_df.columns
    # Assert fully empty rows were dropped
    assert len(cleaned_df) == 2


def test_build_df_metadata():
    """Verifies schema metadata generation for tabular files."""
    df = pd.DataFrame({
        "product": ["Widget A", "Widget B"],
        "price": [19.99, 29.99]
    })
    
    meta = _build_df_metadata(df, filename="products.csv", file_type="csv", file_size="1.5 KB")
    
    assert meta["source"] == "products.csv"
    assert meta["rows"] == 2
    assert meta["columns"] == ["product", "price"]
    assert "preview" in meta
    assert len(meta["preview"]) == 2


# ── 3. Intent Routing Tests ───────────────────────────────────────────────────

def test_query_classifier_only_pdf():
    """Single file type (PDF) should route deterministically to 'pdf'."""
    files = [{"filename": "notes.pdf", "file_type": "pdf"}]
    assert query_classifier(files) == "pdf"


def test_query_classifier_only_csv():
    """Single file type (CSV) should route deterministically to 'csv'."""
    files = [{"filename": "sales.csv", "file_type": "csv"}]
    assert query_classifier(files) == "csv"


def test_query_classifier_mixed_files():
    """Mixed file types should route to 'mix' for LLM classification."""
    files = [
        {"filename": "notes.pdf", "file_type": "pdf"},
        {"filename": "sales.csv", "file_type": "csv"}
    ]
    assert query_classifier(files) == "mix"


# ── 4. Reasoning Token Stripping Tests ────────────────────────────────────────

def test_strip_thinking_removes_tags():
    """Verifies that <think>...</think> blocks from reasoning models are removed."""
    raw_output = "<think>\nThinking about the query...\nStep 1: Check table.\n</think>\nFinal Answer."
    clean_output = strip_thinking(raw_output)
    assert clean_output == "Final Answer."


def test_strip_thinking_without_tags():
    """Regular responses without thinking tags should return untouched."""
    raw_output = "Standard response text."
    assert strip_thinking(raw_output) == "Standard response text."