from typing import List, Dict, Any, Optional
from services.embeddings import EmbeddingService
from services.pinecone_store import PineconeStore
from services.bm25_retriever import BM25Retriever
from services.hybrid_retriever import reciprocal_rank_fusion
from utils.helpers import get_logger, time_it

logger = get_logger("retriever")

class RAGRetriever:
    """Orchestrates query embedding generation, Pinecone similarity searches, BM25 matches, and Hybrid RRF fusion."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: PineconeStore):
        """
        Initializes the retriever.
        
        Args:
            embedding_service: Instance of EmbeddingService.
            vector_store: Instance of PineconeStore.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever()
        self._cross_encoder = None
        self._query_expander = None
        logger.info("RAGRetriever successfully initialized.")

    def get_cross_encoder(self):
        """Lazy-loads the Cross-Encoder model to preserve startup memory."""
        if self._cross_encoder is None:
            logger.info("Loading CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
            from sentence_transformers import CrossEncoder
            # Set device="cpu" or auto-detect if GPU/MPS is available
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def get_query_expander(self):
        """Lazy-loads the QueryExpander to avoid import cost at startup."""
        if self._query_expander is None:
            from services.query_expander import QueryExpander
            self._query_expander = QueryExpander()
        return self._query_expander

    def _retrieve_single_query(
        self,
        query: str,
        fetch_k: int,
        filter_dict: Optional[dict],
        document_filter: Optional[List[str]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs one dense + sparse retrieval pass for a single query string.

        Returns:
            Tuple of (dense_results, sparse_results).
        """
        # Dense: Pinecone
        dense_results = []
        try:
            query_embeddings = self.embedding_service.generate_embeddings([query])
            if query_embeddings:
                query_vector = query_embeddings[0]
                matches = self.vector_store.query_vectors(
                    query_vector=query_vector,
                    top_k=fetch_k,
                    filter_dict=filter_dict,
                )
                for match in matches:
                    metadata = match.get("metadata", {})
                    dense_results.append({
                        "chunk_id": match.get("id"),
                        "chunk_text": metadata.get("text", ""),
                        "document_name": metadata.get("document_name", ""),
                        "page_number": int(metadata.get("page_number", 1)),
                        "score": float(match.get("score", 0.0)),
                        "chunk_type": metadata.get("chunk_type", "text"),
                    })
        except Exception as e:
            logger.error(f"Pinecone dense retrieval failed for query '{query}': {e}")

        # Sparse: BM25
        sparse_results = []
        try:
            sparse_results = self.bm25_retriever.retrieve_bm25(
                query, top_k=fetch_k, document_filter=document_filter
            )
        except Exception as e:
            logger.error(f"BM25 sparse retrieval failed for query '{query}': {e}")

        return dense_results, sparse_results

    @time_it
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[List[str]] = None,
        use_reranker: bool = False,
        use_query_expansion: bool = False,
        use_deduplication: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Runs hybrid search (Pinecone dense + BM25 sparse) merged with Reciprocal Rank Fusion (RRF).
        Optionally applies Query Expansion, Cross-Encoder re-ranking, and Chunk Deduplication.
        
        Workflow:
            Question -> [Query Expansion] -> (Dense retrieve & Sparse retrieve) per query
                     -> RRF Merge -> [Rerank] -> [Dedup] -> Top K results
            
        Args:
            query:               User prompt question.
            top_k:               Number of relevant chunks to retrieve.
            document_filter:     List of document names to filter search results (optional).
                                 If None or empty, all documents are searched.
            use_reranker:        If True, fetches more candidates and re-ranks them using Cross-Encoder.
            use_query_expansion: If True, generates 2 additional query variants via Groq LLM to
                                 run extra retrieval passes and broaden semantic coverage.
            use_deduplication:   If True, drops near-duplicate chunks (≥85% Jaccard n-gram
                                 similarity) from the final result set before returning.
            
        Returns:
            List of dictionaries matching format:
                [
                    {
                        "chunk_id": str,
                        "chunk_text": str,
                        "document_name": str,
                        "page_number": int,
                        "score": float,
                        "chunk_type": str
                    }
                ]
        """
        logger.info(
            f"Initiating retrieval for: '{query}' "
            f"(top_k={top_k}, reranker={use_reranker}, "
            f"expansion={use_query_expansion}, dedup={use_deduplication})"
        )
        
        # Guard against empty query inputs
        if not query.strip():
            logger.warning("Empty query string received. Returning empty matches.")
            return []

        # Determine fetch size (larger candidate pool for re-ranking)
        fetch_k = 20 if use_reranker else 10

        # Build Pinecone metadata filter dict from document_filter list
        filter_dict = None
        if document_filter:
            if len(document_filter) == 1:
                filter_dict = {"document_name": {"$eq": document_filter[0]}}
                logger.info(f"Applying single-document Pinecone filter: '{document_filter[0]}'")
            else:
                filter_dict = {"document_name": {"$in": document_filter}}
                logger.info(f"Applying multi-document Pinecone filter: {document_filter}")

        # ── Step 1: Collect queries (original + expansions if enabled) ────────
        queries: List[str] = [query]
        if use_query_expansion:
            try:
                expander = self.get_query_expander()
                queries = expander.expand(query, num_variants=2)
                logger.info(f"Using {len(queries)} queries (original + {len(queries)-1} expansion(s)).")
            except Exception as exp_err:
                logger.warning(f"Query expansion error: {exp_err}. Using original query only.")

        # ── Step 2: Dense + Sparse retrieval for each query ───────────────────
        all_dense_lists: List[List[Dict]] = []
        all_sparse_lists: List[List[Dict]] = []

        for q in queries:
            d, s = self._retrieve_single_query(q, fetch_k, filter_dict, document_filter)
            all_dense_lists.append(d)
            all_sparse_lists.append(s)

        # Flatten and report
        total_dense = sum(len(d) for d in all_dense_lists)
        total_sparse = sum(len(s) for s in all_sparse_lists)
        logger.info(f"Raw retrieved: {total_dense} dense + {total_sparse} sparse across {len(queries)} query/ies.")

        # ── Step 3: Hybrid RRF Fusion ─────────────────────────────────────────
        # When query expansion is used, treat each expansion's dense/sparse results
        # as independent rank lists and fuse all of them together.
        fusion_top_n = fetch_k if use_reranker else top_k

        if len(queries) == 1:
            # Single query: standard two-list RRF
            fused_chunks = reciprocal_rank_fusion(
                dense_results=all_dense_lists[0],
                sparse_results=all_sparse_lists[0],
                k=60,
                top_n=fusion_top_n,
            )
        else:
            # Multi-query (expanded): fuse all dense lists first, then all sparse lists,
            # then do a final two-side fusion for the combined candidate pool.
            from services.hybrid_retriever import reciprocal_rank_fusion as rrf

            # Combine all dense results via multi-list RRF (treat each dense list as a rank list)
            combined_dense = _multi_list_rrf(all_dense_lists, k=60, top_n=fetch_k * 2)
            combined_sparse = _multi_list_rrf(all_sparse_lists, k=60, top_n=fetch_k * 2)

            fused_chunks = rrf(
                dense_results=combined_dense,
                sparse_results=combined_sparse,
                k=60,
                top_n=fusion_top_n,
            )

        # Log retrieval stats for Streamlit display
        try:
            import streamlit as st
            st.session_state["retrieval_stats"] = {
                "dense_count": total_dense,
                "sparse_count": total_sparse,
                "fused_count": len(fused_chunks),
                "queries_used": len(queries),
            }
        except Exception:
            pass

        # ── Step 4: Cross-Encoder Re-ranking (optional) ───────────────────────
        if use_reranker and fused_chunks:
            try:
                logger.info("Re-ranking fused chunks using CrossEncoder...")
                cross_encoder = self.get_cross_encoder()
                pairs = [[query, chunk["chunk_text"]] for chunk in fused_chunks]
                scores = cross_encoder.predict(pairs)
                for idx, score in enumerate(scores):
                    fused_chunks[idx]["score"] = float(score)
                fused_chunks.sort(key=lambda x: x["score"], reverse=True)
                fused_chunks = fused_chunks[:top_k]
                logger.info(f"Re-ranking completed. Retained top {len(fused_chunks)} chunks.")
            except Exception as rerank_err:
                logger.error(f"Failed during re-ranking: {rerank_err}. Falling back to RRF order.")
                fused_chunks = fused_chunks[:top_k]
        else:
            fused_chunks = fused_chunks[:top_k]

        # ── Step 5: Chunk Deduplication (optional) ───────────────────────────
        if use_deduplication and fused_chunks:
            from services.deduplicator import deduplicate_chunks
            before_count = len(fused_chunks)
            fused_chunks = deduplicate_chunks(fused_chunks, similarity_threshold=0.85)
            logger.info(
                f"Deduplication: {before_count} chunks in → {len(fused_chunks)} chunks out."
            )

        logger.info(f"Final retrieval complete: returning {len(fused_chunks)} chunk(s).")
        return fused_chunks


# ── Helper: multi-list RRF ─────────────────────────────────────────────────────

def _multi_list_rrf(
    rank_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Applies Reciprocal Rank Fusion across an arbitrary number of ranked chunk lists.

    Each list contributes 1/(rank + k) to the RRF score of each chunk.

    Args:
        rank_lists: List of ranked chunk lists (each list is one retrieval pass).
        k:          RRF smoothing constant (default 60).
        top_n:      Number of top results to return.

    Returns:
        Fused and sorted list of chunk dicts.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank_list in rank_lists:
        for rank, chunk in enumerate(rank_list):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + 1 + k)

    if not rrf_scores:
        return []

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    result = []
    for cid in sorted_ids[:top_n]:
        c = chunk_map[cid].copy()
        c["score"] = rrf_scores[cid]
        result.append(c)
    return result
