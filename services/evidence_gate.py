"""
services/evidence_gate.py
-------------------------
Evidence Gate for RAG Assistant.

Evaluates retrieval signals (re-ranker scores, dense cosine scores, BM25 scores,
RRF ranking, chunk count, and source coverage) to categorize evidence into:
  - Strong Evidence
  - Moderate Evidence
  - Insufficient Evidence

If evidence is Insufficient, the gate blocks the LLM call completely, preventing
hallucination, fabricated citations, or wasted API usage, and returns a controlled,
grounded message:
  "I couldn't find enough relevant information in the uploaded documents to answer this reliably."
"""

from typing import List, Dict, Any
from utils.helpers import get_logger

logger = get_logger("evidence_gate")

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough relevant information in the uploaded documents to answer this reliably."
)


class EvidenceGate:
    """Evaluates candidate retrieval chunks and determines evidence sufficiency."""

    @staticmethod
    def evaluate(
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        use_reranker: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates the retrieved chunks against multi-signal evidence thresholds.

        Args:
            query:            The user question string.
            retrieved_chunks: List of retrieved chunk dictionaries after deduplication.
            use_reranker:     Whether Cross-Encoder re-ranking was active.

        Returns:
            Dict matching format:
            {
                "level": "Strong Evidence" | "Moderate Evidence" | "Insufficient Evidence",
                "is_sufficient": bool,
                "top_score": float,
                "chunk_count": int,
                "unique_sources": int,
                "message": str or None,  # Controlled fallback message if insufficient
                "reason": str
            }
        """
        if not retrieved_chunks:
            logger.info(f"Evidence Gate: 0 chunks retrieved for '{query}' -> Insufficient Evidence")
            return {
                "level": "Insufficient Evidence",
                "is_sufficient": False,
                "top_score": 0.0,
                "chunk_count": 0,
                "unique_sources": 0,
                "message": INSUFFICIENT_EVIDENCE_MESSAGE,
                "reason": "No matching document chunks were found.",
            }

        chunk_count = len(retrieved_chunks)
        top_chunk = retrieved_chunks[0]
        top_score = float(top_chunk.get("score", 0.0))

        # Track unique source (document, page) pairs
        sources = {(c.get("document_name"), c.get("page_number")) for c in retrieved_chunks}
        unique_sources = len(sources)

        # Signal 1: Re-ranker Logits (CrossEncoder ms-marco-MiniLM-L-6-v2)
        if use_reranker:
            # MS-MARCO CrossEncoder logit thresholds:
            #   >= 0.0        : Very strong semantic match
            #   -1.5 to 0.0   : Strong match
            #   -3.5 to -1.5  : Moderate match
            #   < -4.2        : Weak / irrelevant match
            if top_score >= 0.0 or (top_score >= -1.5 and chunk_count >= 2):
                level = "Strong Evidence"
                is_sufficient = True
                reason = f"High Cross-Encoder logit ({top_score:.2f}) across {chunk_count} chunk(s)."
            elif top_score >= -3.5 or (top_score >= -4.2 and chunk_count >= 2):
                level = "Moderate Evidence"
                is_sufficient = True
                reason = f"Moderate Cross-Encoder logit ({top_score:.2f}) across {chunk_count} chunk(s)."
            else:
                level = "Insufficient Evidence"
                is_sufficient = False
                reason = f"Low Cross-Encoder logit ({top_score:.2f} < -4.2) indicates irrelevant content."

        # Signal 2: RRF / Dense / Sparse signals (when re-ranker is off)
        else:
            dense_score = float(top_chunk.get("dense_score", top_score))
            sparse_score = float(top_chunk.get("sparse_score", 0.0))

            if top_score >= 0.025 or (top_score >= 0.015 and chunk_count >= 2):
                level = "Strong Evidence"
                is_sufficient = True
                reason = f"High RRF score ({top_score:.4f}) with dense={dense_score:.2f}, sparse={sparse_score:.2f}."
            elif top_score >= 0.010 or dense_score >= 0.35:
                level = "Moderate Evidence"
                is_sufficient = True
                reason = f"Moderate retrieval score ({top_score:.4f}) with dense={dense_score:.2f}."
            else:
                level = "Insufficient Evidence"
                is_sufficient = False
                reason = f"Low retrieval score ({top_score:.4f}) indicates weak relevance."

        logger.info(
            f"Evidence Gate: Query '{query}' -> Level='{level}', "
            f"Sufficient={is_sufficient}, TopScore={top_score:.4f}, Chunks={chunk_count} ({reason})"
        )

        return {
            "level": level,
            "is_sufficient": is_sufficient,
            "top_score": top_score,
            "chunk_count": chunk_count,
            "unique_sources": unique_sources,
            "message": None if is_sufficient else INSUFFICIENT_EVIDENCE_MESSAGE,
            "reason": reason,
        }
