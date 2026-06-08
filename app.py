import streamlit as st
from database.database import init_db
from components.sidebar import render_sidebar
from components.upload import render_upload_section
from components.chat import render_chat_interface
from utils.helpers import get_logger

logger = get_logger("app")

# Set up page configurations
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Relational Database Schema
try:
    init_db()
except Exception as e:
    st.error(f"Failed to initialize metadata database: {e}")

# Render Sidebar Component
render_sidebar()

# Main Application Layout
st.title("🧠 Document-Aware RAG Assistant")
st.markdown(
    """
    Welcome to the RAG Assistant. This application enables you to upload PDF documents, 
    automatically parse and index them, and conduct conversational retrieval over their contents.
    """
)

st.divider()

# Layout: Operations Panel (Left) & Inspector Panel (Right)
col1, col2 = st.columns([0.60, 0.40], gap="large")

with col1:
    tab1, tab2 = st.tabs(["💬 Chat Interface", "📤 Upload Documents"])
    with tab1:
        render_chat_interface()
    with tab2:
        render_upload_section()

with col2:
    st.subheader("🔍 Inspector Panel")
    inspect_tab1, inspect_tab2 = st.tabs(["📋 Document Summaries", "📄 Source Preview"])
    
    with inspect_tab1:
        st.write("### Indexed Documents Summary")
        from database.database import fetch_all_documents
        try:
            docs = fetch_all_documents()
        except Exception as e:
            docs = []
            
        if not docs:
            st.info("No documents uploaded yet. Go to the Upload tab to add files.")
        else:
            doc_names = [d["document_name"] for d in docs]
            selected_doc_name = st.selectbox("Select document:", doc_names, key="inspect_doc_selector")
            selected_doc = next((d for d in docs if d["document_name"] == selected_doc_name), None)
            
            if selected_doc:
                # Show summary
                st.markdown(f"#### 📄 Summary")
                st.info(selected_doc.get("summary") or "No summary available.")
                
                # Show key topics
                st.markdown(f"#### 🏷️ Key Topics")
                topics = selected_doc.get("key_topics", "")
                if topics:
                    # Render topic pills
                    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
                    badge_html = " ".join([
                        f'<span style="background-color:#1e3d59; color:#f5f0e1; padding:4px 10px; border-radius:12px; margin-right:5px; font-size:0.85em; display:inline-block; margin-bottom:5px;">{t}</span>'
                        for t in topic_list
                    ])
                    st.markdown(badge_html, unsafe_allow_html=True)
                else:
                    st.text("No key topics extracted.")
                    
                # Show Metadata
                st.markdown("---")
                st.markdown(f"#### ⚙️ Metadata")
                st.write(f"- **Pages**: `{selected_doc['page_count']}`")
                st.write(f"- **Uploaded**: `{selected_doc['upload_timestamp']}`")
                st.write(f"- **MD5 Hash**: `{selected_doc['id']}`")

    with inspect_tab2:
        st.write("### Reconstructed Page Citation Preview")
        last_chunks = st.session_state.get("last_retrieved_chunks", [])
        
        if not last_chunks:
            st.info("No query citations available yet. Ask a question in the Chat Interface to view sources.")
        else:
            # Let the user select a citation from the last RAG turn
            options = []
            for i, c in enumerate(last_chunks):
                doc = c.get("document_name", "unknown")
                pg = c.get("page_number", 1)
                score = c.get("score", 0.0)
                options.append(f"Match {i+1}: {doc} (Page {pg}) [Score: {score:.4f}]")
                
            selected_option = st.selectbox("Select citation to preview:", options, key="citation_preview_selector")
            selected_idx = options.index(selected_option)
            selected_chunk = last_chunks[selected_idx]
            
            doc_name = selected_chunk.get("document_name")
            page_num = selected_chunk.get("page_number")
            retrieved_text = selected_chunk.get("chunk_text", "")
            
            # Fetch all chunks for this page to reconstruct the full page text!
            from database.database import fetch_page_chunks
            page_chunks = fetch_page_chunks(doc_name, page_num)
            
            if not page_chunks:
                st.warning(f"Could not reconstruct text for {doc_name} (Page {page_num}) from database.")
                st.markdown("**Retrieved Chunk Text:**")
                st.info(retrieved_text)
            else:
                # Concatenate all chunk texts ordered by their chunk_id/offset
                full_page_text = "\n".join([ch["chunk_text"] for ch in page_chunks])
                
                # Highlight the retrieved chunk text inside the full page text safely
                import html
                escaped_full = html.escape(full_page_text)
                escaped_retrieved = html.escape(retrieved_text)
                
                if escaped_retrieved in escaped_full:
                    highlighted_html = escaped_full.replace(
                        escaped_retrieved,
                        f'<mark style="background-color: #ffd166; color: black; padding: 2px 4px; border-radius: 4px; font-weight: bold;">{escaped_retrieved}</mark>'
                    )
                else:
                    highlighted_html = escaped_full + f'<br/><br/><hr/><p>⚠️ Exact match highlighting bypassed. <b>Retrieved text:</b></p><mark style="background-color: #ffd166; color: black;">{escaped_retrieved}</mark>'
                
                st.markdown(f"**Reconstructed Page {page_num} Text:**")
                st.markdown(
                    f'<div style="background-color: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333; height: 400px; overflow-y: scroll; font-family: monospace; white-space: pre-wrap; font-size: 0.9em; line-height: 1.5;">'
                    f'{highlighted_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Document-Aware RAG Assistant © 2026. Built with Streamlit, Pinecone, and Groq."
    "</div>",
    unsafe_allow_html=True
)
