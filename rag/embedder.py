"""
embedder.py
-----------
Handles document chunking and embedding generation for hArI's RAG pipeline.

Responsibilities:
    - Split raw Document objects into smaller chunks using LangChain's
        RecursiveCharacterTextSplitter (only LangChain component used in hArI)
    - Generate dense vector embeddings using sentence-transformers (all-MiniLM-L6-v2)
    - Preserve source metadata (filename, page, file_type) across all chunks

Design decisions:
    - Uses split_text() instead of split_documents() to stay compatible with
        our custom Document class (not LangChain's Document)
    - EmbeddingManager loads model once and reuses — expensive to reload per query
    - Model loaded at init time, not lazily — fail fast if model unavailable
    - transformers logging set to ERROR only — suppresses noisy HuggingFace output

Public API:
    split_docs(docs)                        -> tuple[list[Document], int, int]
    EmbeddingManager._load_model()          -> None
    EmbeddingManager.generate_embeddings()  -> tuple[np.ndarray, tuple, int]
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBED_MODEL, CHUNK_OVERLAP, CHUNK_SIZE
from core.file_processor import Document

transformers_logging.set_verbosity_error()


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_docs(docs: List[Document]) -> tuple[List[Document], int, int]:
    """
    Splits a list of Document objects into smaller chunks.

    Uses RecursiveCharacterTextSplitter with split_text() instead of
    split_documents() to stay compatible with our custom Document class.
    Metadata (source, page, file_type) is preserved on every chunk.

    Args:
        docs: list of Document objects from file_processor.load_pdf_file()

    Returns:
        chunks      : list of chunked Document objects with metadata preserved
        total_chunks: total number of chunks produced
        total_docs  : number of original documents passed in
    """
    if not docs:
        raise ValueError("No documents provided to split.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc.page_content)
        for i, text in enumerate(texts):
            chunks.append(Document(
                page_content=text,
                metadata={
                    **doc.metadata,         # preserve source, page, file_type
                    "chunk_index": i        # track chunk position within page
                }
            ))

    return chunks, len(chunks), len(docs)


# ── Embedding Manager ─────────────────────────────────────────────────────────

class EmbeddingManager:
    """
    Manages sentence-transformer model loading and embedding generation.

    Loads the model once at init and reuses across all encode() calls.
    Designed to be stored in st.session_state or st.cache_resource
    to avoid reloading on every Streamlit rerun.

    Attributes:
        model_name : str           — model identifier from config
        model      : SentenceTransformer | None
    """


    def __init__(self):
        self.model_name = EMBED_MODEL
        self.model = None
        self._load_model()


    def _load_model(self) -> None:
        """
        Loads the SentenceTransformer model from HuggingFace.
        Raises RuntimeError if loading fails — fail fast at startup.
        """
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")


    def generate_embeddings(self, texts: List[str]) -> tuple[np.ndarray, tuple, int]:
        """
        Generates dense vector embeddings for a list of text strings.

        Args:
            texts: list of strings to embed (chunk page_content values)

        Returns:
            embeddings : np.ndarray of shape (n_texts, embedding_dim)
            shape      : tuple — (n_texts, embedding_dim) for logging/debugging
            count      : number of texts embedded

        Raises:
            ValueError  if model not loaded or texts list is empty
        """
        if not self.model:
            raise ValueError("Embedding model not loaded. Call _load_model() first.")

        if not texts:
            raise ValueError("No texts provided for embedding.")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,  # cleaner in Streamlit UI
            convert_to_numpy=True
        )

        return embeddings, embeddings.shape, len(texts)

