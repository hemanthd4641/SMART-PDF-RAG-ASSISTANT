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


def get_dynamic_query_suggestions(all_docs: list = None) -> list:
    """
    Generates context-aware query suggestions based on indexed document metadata
    (summary, key topics, content category) while preserving baseline suggestions.
    """
    baseline = list(SUGGESTED_QUESTIONS)
    if not all_docs:
        return baseline

    combined_text = " ".join([
        f"{d.get('summary', '')} {d.get('key_topics', '')}".lower()
        for d in all_docs
    ])

    topic_suggestions = []

    # HR / Policy / Employee topic detection
    if any(k in combined_text for k in ["leave", "policy", "employee", "hr", "benefit", "handbook", "salary", "work"]):
        topic_suggestions.extend([
            "🌴 What is the leave policy?",
            "🎁 What benefits are available?",
            "👥 What are employee responsibilities?",
        ])

    # Finance / Business / Revenue / Funding topic detection
    if any(k in combined_text for k in ["financial", "revenue", "funding", "investment", "risk", "profit", "contract", "partnership", "corp"]):
        topic_suggestions.extend([
            "💰 What was the revenue or funding?",
            "⚠️ What risks or agreements are mentioned?",
            "📈 What are the major financial highlights?",
        ])

    # Topic-specific chips from key_topics
    for doc in all_docs:
        topics_str = doc.get("key_topics", "")
        if topics_str:
            for top in [t.strip() for t in topics_str.split(",") if t.strip()]:
                if len(top) > 2 and top.lower() not in ["demo", "document", "error", "metadata", "general"]:
                    sug = f"📌 Details on {top}"
                    if sug not in topic_suggestions:
                        topic_suggestions.append(sug)

    # Combine baseline + topic suggestions without duplicates
    all_suggestions = []
    for sug in baseline + topic_suggestions:
        if sug not in all_suggestions:
            all_suggestions.append(sug)

    return all_suggestions[:6]


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


def _render_suggested_questions(all_docs: list = None) -> str | None:
    """
    Renders clickable suggestion chips below the welcome message.
    Returns the selected question text if a chip was clicked, else None.
    """
    suggestions = get_dynamic_query_suggestions(all_docs or [])
    
    # Render chips in balanced columns
    row1 = suggestions[:3]
    row2 = suggestions[3:6]
    
    if row1:
        cols1 = st.columns(len(row1))
        for col, suggestion in zip(cols1, row1):
            with col:
                if st.button(suggestion, use_container_width=True, key=f"suggest_{suggestion}"):
                    parts = suggestion.split(" ", 1)
                    return parts[1] if len(parts) > 1 else suggestion
                    
    if row2:
        cols2 = st.columns(len(row2))
        for col, suggestion in zip(cols2, row2):
            with col:
                if st.button(suggestion, use_container_width=True, key=f"suggest_{suggestion}"):
                    parts = suggestion.split(" ", 1)
                    return parts[1] if len(parts) > 1 else suggestion
                    
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

    # ── Fetch documents for smart filter and dynamic suggestions ─────────────
    try:
        all_docs = fetch_all_documents()
        doc_names = [d["document_name"] for d in all_docs]
    except Exception:
        all_docs = []
        doc_names = []

    # ── Load full chat history from Supabase for display ────────────────────
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
            clicked = _render_suggested_questions(all_docs)
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
            help="Select 2 or more documents to compare their contents directly. Leave empty to query across all indexed documents.",
        )
        if len(selected_docs) >= 2:
            st.caption(f"📊 **Document Comparison Mode Active** — comparing `{len(selected_docs)}` documents: **{', '.join(selected_docs)}**")
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
            is_comp = len(selected_docs) >= 2

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

                # Step 2 — Evidence Gate Evaluation
                from services.evidence_gate import EvidenceGate
                evidence_eval = EvidenceGate.evaluate(
                    user_query,
                    retrieved_chunks,
                    use_reranker=use_reranker,
                )
                st.session_state["evidence_eval"] = evidence_eval

                if not evidence_eval["is_sufficient"]:
                    # Evidence is insufficient: BYPASS LLM completely!
                    logger.info(f"Evidence Gate blocked LLM call for query: '{user_query}'")
                    answer = (
                        "**🔴 Evidence: Insufficient**\n\n"
                        "I couldn't find enough relevant information in the uploaded documents to answer this reliably."
                    )
                    retrieved_chunks = []
                else:
                    # Evidence is sufficient (Strong or Moderate): Generate LLM response
                    answer = llm.generate_response(
                        user_question=user_query,
                        retrieved_chunks=retrieved_chunks,
                        conversation_history=recent_history,
                        evidence_level=evidence_eval.get("level", "Strong Evidence"),
                        is_comparison=is_comp,
                    )

        except Exception as e:
            logger.error(f"RAG execution failure: {e}")
            st.error(
                "Something went wrong while processing your question. "
                "Please try again or check the application logs."
            )
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
        st.session_state["retrieval_stats"] = {
            "dense_count": 1,
            "sparse_count": 1,
            "fused_count": 1,
            "queries_used": 1,
        }
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
                ev_eval = st.session_state.get("evidence_eval")
                ev_label = f" | Evidence Level: **{ev_eval['level']}**" if ev_eval and "level" in ev_eval else ""
                if stats:
                    queries_used = stats.get("queries_used", 1)
                    expansion_label = f" · Queries: **{queries_used}**" if queries_used > 1 else ""
                    st.caption(
                        f"🔍 Pinecone Dense: **{stats['dense_count']}** | "
                        f"BM25 Sparse: **{stats['sparse_count']}** | "
                        f"RRF Fused: **{stats['fused_count']}**"
                        f"{expansion_label}"
                        f"{ev_label}"
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

    # ── Persist to Supabase ──────────────────────────────────────────────────
    try:
        insert_chat_history(user_query, answer)
        st.rerun()
    except Exception as e:
        logger.error(f"Failed to write chat history: {e}")
        st.error("Failed to save chat history.")
