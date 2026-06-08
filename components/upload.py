import streamlit as st
import os
import hashlib
from database.database import (
    fetch_all_documents, 
    delete_document, 
    insert_document, 
    insert_chunks,
    get_total_chunks
)
from services.parser import parse_document
from services.chunker import DocumentChunker
from utils.config import is_configured
from utils.helpers import get_logger, format_file_size

logger = get_logger("ui_upload")

# Constants
TEMP_UPLOADS_DIR = "data/temp_uploads"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _process_single_file(
    file_path: str,
    file_name: str,
    configured: bool,
    embedding_service,
    pinecone_store,
    chunker: DocumentChunker,
) -> bool:
    """
    Runs the full ingestion pipeline (parse → summarize → chunk → embed → upsert)
    for a single file. Returns True on success, False on failure.
    """
    doc_id = hashlib.md5(file_name.encode("utf-8")).hexdigest()

    status_placeholder = st.empty()

    def ocr_log(p_num):
        status_placeholder.info(f"Running OCR on Page {p_num}...")

    try:
        # Step A: Parse
        with st.spinner(f"Extracting text from **{file_name}**..."):
            parsed_doc = parse_document(file_path, progress_callback=ocr_log)
        status_placeholder.empty()

        page_count = len(parsed_doc["pages"])
        native_page_count = sum(
            1 for p in parsed_doc["pages"] if p.get("extraction_method", "native") == "native"
        )
        ocr_page_count = sum(
            1 for p in parsed_doc["pages"] if p.get("extraction_method", "native") == "ocr"
        )

        # Step B: Document summary
        summary_info = {
            "summary": "Demo summary: API keys not configured.",
            "key_topics": "Demo, Document",
        }
        if configured:
            with st.spinner(f"Generating summary for **{file_name}**..."):
                try:
                    from services.llm import LLMService
                    llm_srv = LLMService()
                    doc_text = "\n".join([p.get("text", "") for p in parsed_doc.get("pages", [])])
                    summary_info = llm_srv.generate_summary(doc_text)
                except Exception as sum_err:
                    logger.warning(f"Failed to generate document summary: {sum_err}")
                    summary_info = {"summary": "Failed to generate summary.", "key_topics": "Error"}

        # Step C: Chunk
        with st.spinner(f"Chunking **{file_name}**..."):
            chunks = chunker.chunk_document(parsed_doc)

        # Step D: Save to SQLite
        insert_document(
            doc_id=doc_id,
            document_name=file_name,
            page_count=page_count,
            summary=summary_info.get("summary"),
            key_topics=summary_info.get("key_topics"),
            native_page_count=native_page_count,
            ocr_page_count=ocr_page_count,
        )

        num_text, num_table = 0, 0
        db_chunks = []
        for chunk in chunks:
            c_type = chunk["chunk_type"]
            num_text += c_type == "text"
            num_table += c_type == "table"
            db_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": doc_id,
                "page_number": chunk["page_number"],
                "chunk_text": chunk["chunk_text"],
                "chunk_type": c_type,
            })
        insert_chunks(db_chunks)

        # Step E: Embed + Pinecone
        if configured and embedding_service and pinecone_store:
            with st.spinner(f"Generating embeddings for **{file_name}** ({len(chunks)} chunks)..."):
                chunk_texts = [c["chunk_text"] for c in chunks]
                embeddings = embedding_service.generate_embeddings(chunk_texts)

            with st.spinner(f"Uploading vectors to Pinecone for **{file_name}**..."):
                vectors_payload = [
                    {
                        "id": chunk["chunk_id"],
                        "values": emb,
                        "metadata": {
                            "document_name": file_name,
                            "page_number": chunk["page_number"],
                            "text": chunk["chunk_text"],
                            "chunk_type": chunk["chunk_type"],
                        },
                    }
                    for chunk, emb in zip(chunks, embeddings)
                ]
                pinecone_store.upsert_vectors(vectors_payload)
        else:
            st.info(f"[Demo Mode] Embeddings and Pinecone bypassed for '{file_name}'.")

        # Cleanup temp file
        try:
            os.remove(file_path)
        except Exception:
            pass

        # Success summary
        st.success(
            f"**{file_name}** indexed successfully!\n"
            f"- Pages: `{page_count}` (Native: `{native_page_count}`, OCR: `{ocr_page_count}`)\n"
            f"- Text Chunks: `{num_text}` | Table Chunks: `{num_table}`"
        )
        logger.info(
            f"Processed '{file_name}': {page_count} pages, {num_text} text, {num_table} table chunks."
        )
        return True

    except Exception as err:
        st.error(f"Indexing failed for **{file_name}**: {err}")
        logger.error(f"Failed to process '{file_name}': {err}")
        return False


def render_upload_section() -> None:
    """Renders the document upload section with fully automatic processing and indexing."""
    st.subheader("📄 Document Upload & Processing")
    st.write(
        "Upload **PDF**, **TXT**, or **DOCX** documents below. "
        "Processing, chunking, embedding, and indexing will start "
        "**automatically** as soon as your file is uploaded. Maximum file size: 50 MB."
    )

    # Ensure temp folder exists
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

    # Track which files have already been processed this session
    if "processed_file_hashes" not in st.session_state:
        st.session_state["processed_file_hashes"] = set()

    # --- 1. File Uploader ---
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        configured = is_configured()

        # Load vector services once for the whole batch
        embedding_service = None
        pinecone_store = None
        if configured:
            try:
                from services.embeddings import EmbeddingService
                from services.pinecone_store import PineconeStore
                embedding_service = EmbeddingService()
                pinecone_store = PineconeStore()
            except Exception as service_err:
                st.error(f"Failed to load vector services: {service_err}")
                return

        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        any_new = False

        for file in uploaded_files:
            file_name = file.name
            file_size = file.size
            _, ext = os.path.splitext(file_name)
            ext = ext.lower()

            # Build a unique key for this file (name + size) to avoid re-processing
            file_key = hashlib.md5(f"{file_name}:{file_size}".encode()).hexdigest()

            # Skip if already processed in this session
            if file_key in st.session_state["processed_file_hashes"]:
                continue

            # Validate extension
            if ext not in ALLOWED_EXTENSIONS:
                st.error(
                    f"**{file_name}** rejected: Unsupported format `{ext}`. "
                    f"Only PDF (.pdf), TXT (.txt), and DOCX (.docx) files are supported."
                )
                continue

            # Validate size
            if file_size > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"**{file_name}** rejected: Exceeds 50 MB limit "
                    f"(Size: {format_file_size(file_size)})."
                )
                continue

            # Check if document is already in the database (duplicate guard)
            doc_id = hashlib.md5(file_name.encode("utf-8")).hexdigest()
            existing_docs = fetch_all_documents()
            existing_ids = {d["id"] for d in existing_docs}
            if doc_id in existing_ids:
                st.warning(
                    f"**{file_name}** is already indexed. "
                    "Delete the existing entry first if you want to re-index it."
                )
                st.session_state["processed_file_hashes"].add(file_key)
                continue

            # Save to temp folder
            save_path = os.path.join(TEMP_UPLOADS_DIR, file_name)
            try:
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                logger.info(f"Saved upload: {save_path} ({file_size} bytes)")
            except Exception as e:
                st.error(f"Failed to save **{file_name}**: {e}")
                continue

            # --- Auto-trigger full pipeline ---
            st.markdown(f"---\n#### Processing: `{file_name}`")
            success = _process_single_file(
                file_path=save_path,
                file_name=file_name,
                configured=configured,
                embedding_service=embedding_service,
                pinecone_store=pinecone_store,
                chunker=chunker,
            )

            if success:
                st.session_state["processed_file_hashes"].add(file_key)
                any_new = True

        # Rebuild BM25 index once after all new files are processed
        if any_new:
            try:
                from components.chat import get_rag_pipeline
                retriever, _ = get_rag_pipeline()
                retriever.bm25_retriever.build_index()
                logger.info("BM25 index rebuilt after auto-ingestion.")
            except Exception as rebuild_err:
                logger.warning(f"BM25 rebuild failed: {rebuild_err}")

            st.rerun()

    # --- 2. Statistics & Managed Documents ---
    st.divider()
    st.write("### Indexed Documents & Statistics")

    try:
        documents = fetch_all_documents()
        total_chunks = get_total_chunks()
    except Exception as e:
        logger.error(f"Error fetching document metrics: {e}")
        st.error("Failed to load document statistics.")
        documents = []
        total_chunks = 0

    if not documents:
        st.info("No documents indexed yet. Upload a file above to begin.")
    else:
        from database.database import get_chunks_count_by_type
        type_counts = get_chunks_count_by_type()

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Indexed Documents", len(documents))

        total_native = sum(d.get("native_page_count", 0) for d in documents)
        total_ocr = sum(d.get("ocr_page_count", 0) for d in documents)
        stat_col2.metric("Pages (Native / OCR)", f"{total_native} / {total_ocr}")
        stat_col3.metric("Text Chunks", type_counts.get("text", 0))
        stat_col4.metric("Table Chunks", type_counts.get("table", 0))

        st.write("Documents currently indexed:")
        display_data = [
            {
                "File Name": doc["document_name"],
                "Pages (Native/OCR)": f"{doc['page_count']} ({doc.get('native_page_count', 0)} / {doc.get('ocr_page_count', 0)})",
                "Uploaded At": doc["upload_timestamp"],
            }
            for doc in documents
        ]
        st.table(display_data)

        # Deletion utility
        st.write("#### Remove Document")
        doc_names = {doc["document_name"]: doc["id"] for doc in documents}
        selected_doc_name = st.selectbox(
            "Select document to remove:", list(doc_names.keys())
        )

        if st.button("Delete Document", type="secondary", use_container_width=True):
            target_id = doc_names[selected_doc_name]
            try:
                delete_document(target_id)

                if is_configured():
                    try:
                        from services.pinecone_store import PineconeStore
                        PineconeStore().delete_document_vectors(selected_doc_name)
                    except Exception as pine_err:
                        logger.warning(f"Pinecone cleanup failed for '{selected_doc_name}': {pine_err}")
                        st.warning("Deleted from local database, but Pinecone cleanup failed. See logs.")

                try:
                    from components.chat import get_rag_pipeline
                    retriever, _ = get_rag_pipeline()
                    retriever.bm25_retriever.build_index()
                    logger.info("BM25 index rebuilt after deletion.")
                except Exception as rebuild_err:
                    logger.warning(f"BM25 rebuild failed after deletion: {rebuild_err}")

                st.success(f"Deleted '{selected_doc_name}' from the RAG system.")
                logger.info(f"Deleted document '{selected_doc_name}'.")
                st.rerun()
            except Exception as e:
                logger.error(f"Failed to delete document: {e}")
                st.error(f"Failed to delete document: {e}")
