import streamlit as st
from core.utils import get_supabase_client

class VectorStore:
    """Supabase pgvector implementation for hArI."""
    
    def add_documents(self, chunks, embeddings):
        """Inserts document chunks and their embeddings into Supabase."""
        if not chunks:
            return

        supabase = get_supabase_client() # GET FRESH CONNECTION
        user_id = st.session_state["user"].id
        filename = chunks[0].metadata.get("source", "unknown")

        supabase.table("documents").delete().eq("user_id", user_id).eq("filename", filename).execute()

        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "user_id": user_id,
                "filename": filename,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embeddings[i].tolist() 
            })
            
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("documents").insert(batch).execute()

    def delete_source(self, filename: str):
        if "user" not in st.session_state or not st.session_state["user"]:
            return
            
        supabase = get_supabase_client() # GET FRESH CONNECTION
        user_id = st.session_state["user"].id

        try:
            # First try to delete using the filename column
            del_response = supabase.table("documents").delete().eq("user_id", user_id).eq("filename", filename).execute()
            st.toast(f"🗑️ Deleted {len(del_response.data)} old chunks!", icon="🗑️")
        except Exception as e:
            st.error(f"🚨 DELETE ERROR: {e}")
            st.stop()

    def collection_count(self) -> int:
        if "user" not in st.session_state or not st.session_state["user"]:
            return 0
            
        supabase = get_supabase_client() # GET FRESH CONNECTION
        user_id = st.session_state["user"].id
        response = supabase.table("documents").select("id", count="exact").eq("user_id", user_id).execute()
        return response.count if response.count else 0