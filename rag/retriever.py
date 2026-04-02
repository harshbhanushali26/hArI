"""
retriever.py
------------
Handles semantic search and result retrieval for hArI's RAG pipeline.

Responsibilities:
    - Embed the user query using EmbeddingManager
    - Query ChromaDB collection for top-k similar chunks
    - Deduplicate results based on content prefix
    - Filter results below a similarity score threshold
    - Return ranked list of retrieved chunks with metadata

Design decisions:
    - RAGRetriever takes VectorStore + EmbeddingManager as dependencies
        (injected, not created here) — both live in st.session_state
    - Directly queries vector_store.collection — retriever owns query logic,
        vector_store owns storage logic
    - generate_embeddings returns (embeddings, shape, count) tuple —
        we unpack index [0] to get the np.ndarray, then [0] again for first row
    - Deduplication on first 200 chars — avoids near-duplicate chunks from
        overlapping windows returning redundant context to LLM
    - score_threshold defaults to 0.0 — caller (responser/app) can tighten
    - Raises RuntimeError on failure instead of returning empty list silently

Public API:
    RAGRetriever.retrieve(query, top_k, score_threshold) -> list[dict]
"""

from typing import List, Dict, Any

from config import TOP_K
from rag.vector_store import VectorStore
from rag.embedder import EmbeddingManager


class RAGRetriever:
    """
    Handles semantic search over the ChromaDB collection.

    Embeds the query, queries the collection, deduplicates and
    ranks results by similarity score.

    Attributes:
        vector_store      : VectorStore instance (from st.session_state)
        embedding_manager : EmbeddingManager instance (from st.session_state)
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store      = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Finds the most semantically similar chunks for a given query.

        Flow:
            1. Embed query string → query vector
            2. Query ChromaDB collection → raw results
            3. Deduplicate by content prefix (handles chunk overlap)
            4. Filter by similarity score threshold
            5. Return ranked list of result dicts

        Args:
            query           : user's question string
            top_k           : max number of chunks to return (default from config)
            score_threshold : minimum similarity score to include (0.0 = no filter)

        Returns:
            list of dicts, each containing:
            {
                "id"              : str    — ChromaDB chunk ID
                "content"         : str    — chunk text
                "metadata"        : dict   — source, page, file_type, etc.
                "similarity_score": float  — cosine similarity (0.0 to 1.0)
                "rank"            : int    — position in result list (1-indexed)
            }
            Returns empty list if collection is empty or no results found.

        Raises:
            ValueError    if query is empty
            RuntimeError  if ChromaDB query fails
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if self.vector_store.collection_count() == 0:
            return []

        # 1. Embed query — generate_embeddings returns (np.ndarray, shape, count)
        query_embedding = self.embedding_manager.generate_embeddings([query])[0][0]

        # 2. Query ChromaDB
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 2, self.vector_store.collection_count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB query failed: {e}")

        # 3. Parse, deduplicate and filter results
        retrieved_docs = []
        seen_contents  = set()

        if not results["documents"] or not results["documents"][0]:
            return []

        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - dist

            # filter below threshold
            if similarity < score_threshold:
                continue

            # deduplicate on content prefix (handles overlapping chunks)
            content_key = text[:200].strip()
            if content_key in seen_contents:
                continue

            seen_contents.add(content_key)
            retrieved_docs.append({
                "id"              : doc_id,
                "content"         : text,
                "metadata"        : meta,
                "similarity_score": round(similarity, 4),
                "rank"            : len(retrieved_docs) + 1,
            })

            if len(retrieved_docs) >= top_k:
                break

        return retrieved_docs