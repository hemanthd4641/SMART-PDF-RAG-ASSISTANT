import streamlit as st
import time
from database.database import fetch_chat_history, fetch_recent_chat_history, insert_chat_history, fetch_all_documents
from utils.config import is_configured
from utils.helpers import get_logger

logger = get_logger("ui_chat")

# Suggested conversation starters shown on an empty chat
SUGGESTED_QUESTIONS = [
    "📋 What is this document about?",
    "👥 What roles or people are mentioned?",
    "📝 Summarize the key points",
    "🔍 What are the main topics covered?",
]


@st.cache_resource
def get_rag_pipeline():
    """Caches RAG pipeline connections to keep model loads and client connections in memory."""
    from services.embeddings import EmbeddingService
    from services.pinecone_store import PineconeStore
    from services.retriever import RAGRetriever
    from services.llm import LLMService

    logger.info("Loading RAG services into memory...")
    embeddings = EmbeddingService()
    vector_store = PineconeStore()
    retriever = RAGRetriever(embeddings, vector_store)
    llm = LLMService()

    return retriever, llm


def _render_suggested_questions() -> str | None:
    """
    Renders clickable suggestion chips below the welcome message.
    Returns the selected question text if a chip was clicked, else None.
    """
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, suggestion in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(suggestion, use_container_width=True, key=f"suggest_{suggestion}"):
                return suggestion.split(" ", 1)[1]   # strip the leading emoji
    return None


def render_chat_interface() -> None:
    """Renders a fully conversational RAG chat interface with multi-turn memory."""

    # ── Configuration check ──────────────────────────────────────────────────
    configured = is_configured()
    if not configured:
        st.warning(
            "⚠️ **API keys missing** — running in Demo Mode. "
            "Set `GROQ_API_KEY` and `PINECONE_API_KEY` in `.env` to enable live responses."
        )

    # ── Load full chat history from SQLite for display ───────────────────────
    try:
        history = fetch_chat_history()
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        history = []

    # ── Welcome message (always visible at top) ──────────────────────────────
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(
            "👋 **Hey there! I'm your Document Assistant.**\n\n"
            "I've read your uploaded documents and I'm ready to chat about them. "
            "You can ask me anything — factual questions, summaries, comparisons, follow-ups, "
            "or just have a conversation. I'll always tell you where I got my information from.\n\n"
            "*What would you like to know?*"
        )

        # Show suggestion chips only when there's no history yet
        if not history:
            clicked = _render_suggested_questions()
            if clicked:
                st.session_state["prefill_query"] = clicked
                st.rerun()

    # ── Render all past messages ─────────────────────────────────────────────
    for message in history:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(message["user_question"])
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(message["assistant_answer"])

    # ── Smart Document Filter ─────────────────────────────────────────────────
    try:
        all_docs = fetch_all_documents()
        doc_names = [d["document_name"] for d in all_docs]
    except Exception:
        doc_names = []

    if doc_names:
        selected_docs = st.multiselect(
            "🔍 Filter by document (leave empty to search all)",
            options=doc_names,
            default=[],
            key="doc_filter_multiselect",
            help="Restrict retrieval to one or more specific documents. Leave empty to query across all indexed documents.",
        )
    else:
        selected_docs = []

    # ── Handle pre-filled query from suggestion chip ─────────────────────────
    prefill = st.session_state.pop("prefill_query", None)

    # ── Chat input ───────────────────────────────────────────────────────────
    user_query = st.chat_input("Ask me anything about your documents…") or prefill

    if not user_query:
        return

    # Display the user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    # ── RAG pipeline execution ───────────────────────────────────────────────
    answer = ""
    retrieved_chunks = []

    if configured:
        try:
            retriever, llm = get_rag_pipeline()

            # Fetch last 6 turns (3 exchanges) for conversation memory
            recent_history = fetch_recent_chat_history(limit=6)

            use_reranker = st.session_state.get("use_reranker", False)
            use_query_expansion = st.session_state.get("use_query_expansion", False)
            use_deduplication = st.session_state.get("use_deduplication", False)
            doc_filter = selected_docs if selected_docs else None

            with st.spinner(""):
                # Step 1 — Retrieve relevant chunks
                retrieved_chunks = retriever.retrieve(
                    user_query,
                    top_k=5,
                    use_reranker=use_reranker,
                    use_query_expansion=use_query_expansion,
                    use_deduplication=use_deduplication,
                    document_filter=doc_filter,
                )

                # Step 2 — Generate conversational response with history
                answer = llm.generate_response(
                    user_question=user_query,
                    retrieved_chunks=retrieved_chunks,
                    conversation_history=recent_history,
                )

        except Exception as e:
            logger.error(f"RAG execution failure: {e}")
            st.error(f"Something went wrong: {e}")
            return

    else:
        # Demo mode
        with st.spinner("Thinking…"):
            time.sleep(1.0)
        retrieved_chunks = [
            {
                "chunk_text": "Demo Mode: This is a simulated document context block.",
                "document_name": "demo_document.pdf",
                "page_number": 1,
                "score": 0.91,
                "chunk_type": "text",
            }
        ]
        from services.llm import generate_citations_block
        mock_answer = (
            "Sure! In Demo Mode, I'm simulating a response based on your uploaded documents. "
            "When you configure your API keys, I'll provide real answers grounded in your actual content."
        )
        citations = generate_citations_block(retrieved_chunks)
        answer = f"{mock_answer}\n\n{citations}" if citations else mock_answer

    # Store retrieved chunks for Source Preview tab
    st.session_state["last_retrieved_chunks"] = retrieved_chunks

    # ── Render assistant response ─────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(answer)

        # Source chunks expander
        if retrieved_chunks:
            with st.expander("📎 View source chunks", expanded=False):
                stats = st.session_state.get("retrieval_stats")
                if stats:
                    queries_used = stats.get("queries_used", 1)
                    expansion_label = f" · Queries: **{queries_used}**" if queries_used > 1 else ""
                    st.caption(
                        f"🔍 Pinecone Dense: **{stats['dense_count']}** | "
                        f"BM25 Sparse: **{stats['sparse_count']}** | "
                        f"RRF Fused: **{stats['fused_count']}**"
                        f"{expansion_label}"
                    )
                    st.markdown("---")

                for idx, chunk in enumerate(retrieved_chunks):
                    score_val  = chunk.get("score", 0.0)
                    doc_name   = chunk.get("document_name", "unknown")
                    page_num   = chunk.get("page_number", 1)
                    chunk_text = chunk.get("chunk_text", "")
                    chunk_type = chunk.get("chunk_type", "text")
                    type_icon  = "📊" if chunk_type == "table" else "📝"

                    st.markdown(
                        f"**#{idx+1}** &nbsp; {type_icon} `{chunk_type}` &nbsp;·&nbsp; "
                        f"Score `{score_val:.4f}` &nbsp;·&nbsp; "
                        f"📄 **{doc_name}** — Page `{page_num}`"
                    )
                    st.markdown(
                        f'<div style="background:#0f0f1a; border-left:3px solid #4a9eff; '
                        f'padding:10px 14px; border-radius:6px; font-size:0.85em; '
                        f'font-family:monospace; white-space:pre-wrap; color:#d0d0e0; '
                        f'margin-bottom:10px;">{chunk_text}</div>',
                        unsafe_allow_html=True,
                    )

    # ── Persist to SQLite ────────────────────────────────────────────────────
    try:
        insert_chat_history(user_query, answer)
        st.rerun()
    except Exception as e:
        logger.error(f"Failed to write chat history: {e}")
        st.error("Failed to save chat history.")
