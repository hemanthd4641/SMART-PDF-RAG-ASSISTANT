import streamlit as st
from utils.config import is_configured, GROQ_MODEL, PINECONE_INDEX_NAME
from database.database import clear_chat_history
from utils.helpers import get_logger

logger = get_logger("ui_sidebar")

def render_sidebar() -> None:
    """Renders configuration settings and statuses inside the sidebar."""
    with st.sidebar:
        st.title("⚙️ RAG Settings")
        st.markdown("---")
        
        # Connection status check indicators
        st.subheader("System Status")
        
        # Check API status configuration
        if is_configured():
            st.success("API Keys Loaded")
            st.markdown(
                "**Pipeline Features (Active)**:\n"
                "- 🎯 Cross-Encoder Re-Ranking: `Active`\n"
                "- 🔄 Query Expansion: `Active`\n"
                "- 🧹 Chunk Deduplication: `Active`"
            )
        else:
            st.warning("API Keys Missing")
            st.info("Ensure GROQ_API_KEY and PINECONE_API_KEY are configured in .env file.")

            
        st.markdown("---")
        
        # Details about connected services
        st.subheader("Configuration Details")
        st.text(f"LLM Model: {GROQ_MODEL}")
        st.text(f"Vector Index: {PINECONE_INDEX_NAME}")
        st.text("Embedding: all-MiniLM-L6-v2")
        
        st.markdown("---")
        
        # Database Statistics
        st.subheader("Database Statistics")
        try:
            from database.database import fetch_all_documents, get_total_chunks, get_chunks_count_by_type
            docs = fetch_all_documents()
            total_chunks = get_total_chunks()
            type_counts = get_chunks_count_by_type()
            
            st.markdown(
                f"- **Documents Processed**: `{len(docs)}`\n"
                f"- **Total Chunks**: `{total_chunks}`\n"
                f"  - 📝 Text Chunks: `{type_counts.get('text', 0)}`\n"
                f"  - 📊 Table Chunks: `{type_counts.get('table', 0)}`"
            )
        except Exception as stats_err:
            logger.error(f"Failed to render sidebar stats: {stats_err}")
            st.text("Failed to load database stats.")
            
        st.markdown("---")
        
        # Utility button
        if st.button("Clear Chat History", use_container_width=True):
            try:
                clear_chat_history()
                logger.info("Chat history cleared from database by user.")
                st.rerun()
            except Exception as e:
                logger.error(f"Failed to clear chat history: {e}")
                st.error("Failed to clear chat history in database.")

