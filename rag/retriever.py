import streamlit as st
from typing import List, Dict, Any
from config import TOP_K
from rag.vector_store import VectorStore
from rag.embedder import EmbeddingManager
from core.utils import get_supabase_client

class RAGRetriever:
    """Handles semantic search using Supabase pgvector."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = TOP_K, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        supabase = get_supabase_client() # GET FRESH CONNECTION
        query_embedding = self.embedding_manager.generate_embeddings([query])[0][0]

        try:
            response = supabase.rpc(
                "hybrid_search",
                {
                    "query_text": query,  # We now pass the raw text for keyword matching!
                    "query_embedding": query_embedding.tolist(),
                    "match_count": top_k * 2 
                }
            ).execute()
        except Exception as e:
            raise RuntimeError(f"Supabase vector search failed: {e}")

        retrieved_docs = []
        seen_contents  = set()
        if not response.data:
            return []

        for row in response.data:
            text = row["content"]
            similarity = row["similarity"]

            content_key = text[:200].strip()
            if content_key in seen_contents:
                continue

            seen_contents.add(content_key)
            retrieved_docs.append({
                "id"              : str(row["id"]),
                "content"         : text,
                "metadata"        : row["metadata"],
                "similarity_score": round(similarity, 4),
                "rank"            : len(retrieved_docs) + 1,
            })

            if len(retrieved_docs) >= top_k:
                break

        return retrieved_docs