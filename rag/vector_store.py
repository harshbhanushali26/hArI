"""
vector_store.py
---------------
Manages ChromaDB vector store operations for hArI's RAG pipeline.

Responsibilities:
    - Initialize and persist a ChromaDB collection on disk
    - Add document chunks + embeddings via deterministic upsert (no duplicates)
    - Track which source files are already indexed
    - Delete chunks by source filename (when user removes a file)
    - Expose collection count for UI status display

Design decisions:
    - Uses PersistentClient — collection survives app restarts
    - Deterministic IDs via _make_doc_id() — upsert is idempotent,
        re-uploading same file never creates duplicates
    - No embedding logic here — embeddings come from embedder.py    
    - No search logic here — querying handled by retriever.py
    - print() only on errors — status info returned to caller, not printed

Public API:
    VectorStore.add_documents(documents, embeddings)  -> None
    VectorStore.get_ingested_sources()                -> set[str]
    VectorStore.delete_source(filename)               -> None
    VectorStore.collection_count()                    -> int

Internal:
    VectorStore._initialize_vector_store()            -> None
    VectorStore._make_doc_id(doc, index)              -> str
"""

import os
import numpy as np
import chromadb

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR


class VectorStore:
    """
    Wrapper around ChromaDB PersistentClient for hArI.

    Handles storage, deduplication, and deletion of document chunks.
    Designed to be instantiated once and stored in st.session_state
    or st.cache_resource to avoid re-initializing on every Streamlit rerun.
    """

    def __init__(self):
        self.collection_name = CHROMA_COLLECTION_NAME
        self.persist_directory = CHROMA_PERSIST_DIR
        self.client = None
        self.collection = None
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """
        Creates persist directory if needed, initializes ChromaDB
        PersistentClient and gets or creates the collection.

        Raises RuntimeError if initialization fails.
        """
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Document embeddings for hArI RAG",
                    "hnsw:space": "cosine"
                    }
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {e}")

    def add_documents(self, documents, embeddings: np.ndarray) -> None:
        """
        Adds document chunks and their embeddings to the ChromaDB collection.

        Uses deterministic IDs + upsert — re-uploading the same file
        overwrites existing chunks instead of creating duplicates.

        Args:
            documents  : list of Document objects (from embedder.split_docs)
            embeddings : np.ndarray of shape (n_chunks, embedding_dim)

        Raises:
            ValueError if documents and embeddings lengths don't match.
            RuntimeError if ChromaDB upsert fails.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) count mismatch."
            )

        ids             = []
        metadatas       = []
        documents_text  = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = self._make_doc_id(doc, i)
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"]      = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents_text,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upsert documents into ChromaDB: {e}")

    def get_ingested_sources(self) -> set:
        """
        Returns set of source filenames already stored in the collection.
        Used by app.py to avoid re-processing already indexed files
        and to display indexed file status in the UI.

        Returns:
            set of filename strings e.g. {"report.pdf", "sales.csv"}
            Empty set if collection is empty.
        """
        if self.collection.count() == 0:
            return set()

        results = self.collection.get(include=["metadatas"])
        sources = {
            m.get("source")
            for m in results["metadatas"]
            if m.get("source")
        }
        return sources

    def delete_source(self, filename: str) -> None:
        """
        Deletes all chunks belonging to a specific source file.
        Called when user removes a file from the session.

        Args:
            filename: source filename to delete e.g. "report.pdf"
        """
        results = self.collection.get(include=["metadatas"])

        ids_to_delete = [
            id for id, meta in zip(results["ids"], results["metadatas"])
            if meta.get("source") == filename
        ]

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

    def collection_count(self) -> int:
        """
        Returns total number of chunks currently stored in the collection.
        Used by app.py for UI status display e.g. '324 chunks indexed'.
        """
        return self.collection.count()

    @staticmethod
    def _make_doc_id(doc, index: int) -> str:
        """
        Builds a deterministic chunk ID from source filename + page + index.
        Same file always produces same IDs — makes upsert idempotent.

        Format: {safe_source}_p{page}_c{index}
        Example: report_pdf_p3_c1

        Args:
            doc   : Document object with metadata["source"] and metadata["page"]
            index : chunk position in the full documents list
        """
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", 0)
        safe_source = (
            source
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(".", "_")
        )
        return f"{safe_source}_p{page}_c{index}"