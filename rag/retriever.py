import streamlit as st
from typing import List, Dict, Any
from config import TOP_K
from rag.vector_store import VectorStore
from rag.embedder import EmbeddingManager
from core.utils import get_supabase_client

class RAGRetriever:
    """Handles semantic & hybrid search using Supabase pgvector with active session scoping."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        # 1. Identify which PDF files are currently active in the user's session
        active_filenames = {
            f["filename"] for f in st.session_state.get("uploaded_files", []) 
            if f.get("file_type") == "pdf"
        }

        supabase = get_supabase_client()
        query_embedding = self.embedding_manager.generate_embeddings([query])[0][0]

        # 2. Fetch a deeper candidate pool (e.g. 25 candidates) so old documents 
        # in the database don't crowd out the currently active ones
        candidate_count = max(top_k * 5, 25)

        try:
            response = supabase.rpc(
                "hybrid_search",
                {
                    "query_text": query,
                    "query_embedding": query_embedding.tolist(),
                    "match_count": candidate_count
                }
            ).execute()
        except Exception as e:
            raise RuntimeError(f"Supabase vector search failed: {e}")

        retrieved_docs = []
        seen_contents  = set()
        if not response.data:
            return []

        # 3. Filter candidates so ONLY chunks from the currently uploaded files are accepted
        for row in response.data:
            meta = row.get("metadata") or {}
            doc_name = meta.get("filename") or meta.get("source") or row.get("filename")

            # 🛡️ Discard chunks from documents uploaded in previous/other sessions
            if active_filenames and doc_name not in active_filenames:
                continue

            text = row["content"]
            similarity = row["similarity"]

            content_key = text[:200].strip()
            if content_key in seen_contents:
                continue

            seen_contents.add(content_key)
            retrieved_docs.append({
                "id"              : str(row["id"]),
                "content"         : text,
                "metadata"        : meta,
                "similarity_score": round(similarity, 4),
                "rank"            : len(retrieved_docs) + 1,
            })

            if len(retrieved_docs) >= top_k:
                break

        return retrieved_docs