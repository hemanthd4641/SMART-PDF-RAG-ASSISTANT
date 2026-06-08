import string
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from database.database import fetch_all_chunks
from utils.helpers import get_logger

logger = get_logger("bm25")

def tokenize(text: str) -> List[str]:
    """Tokenize and clean text for BM25 indexing by converting to lowercase and removing punctuation."""
    if not text:
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Split by whitespace
    return [word for word in text.split(" ") if word]

class BM25Retriever:
    """Manages the BM25 keyword index over all chunks currently stored in SQLite database."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.build_index()

    def build_index(self) -> None:
        """Loads all chunks from SQLite database and builds the BM25Okapi index."""
        try:
            logger.info("Building BM25 index from SQLite chunks corpus...")
            self.chunks = fetch_all_chunks()
            if not self.chunks:
                logger.warning("No chunks found in SQLite database. BM25 index is empty.")
                self.bm25 = None
                return
                
            tokenized_corpus = [tokenize(c["chunk_text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Successfully indexed {len(self.chunks)} chunks in BM25.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
            self.chunks = []

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs BM25 query against the index and returns the top_k ranked match dictionaries.

        Args:
            query:           User search string.
            top_k:           Number of chunks to retrieve.
            document_filter: Optional list of document names to restrict retrieval to.
                             If None or empty, all documents are searched.

        Returns:
            List of chunks with scores and metadata.
        """
        if not query.strip():
            logger.warning("Empty query string received for BM25 retrieval.")
            return []
            
        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 retrieval bypassed: Index is empty or uninitialized.")
            return []
            
        # Normalise filter to a set of names for O(1) lookup
        filter_set: set = set(document_filter) if document_filter else set()

        try:
            tokenized_query = tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Map scores to their chunks
            results_with_scores = []
            for idx, score in enumerate(scores):
                if score > 0.0:  # Return only chunks with at least one matching keyword token
                    chunk = self.chunks[idx]
                    # Apply multi-document name filter if specified
                    if filter_set and chunk.get("document_name") not in filter_set:
                        continue
                    results_with_scores.append({
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["chunk_text"],
                        "document_name": chunk["document_name"],
                        "page_number": chunk["page_number"],
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "score": float(score)
                    })
            
            # Sort descending by BM25 relevance score
            results_with_scores.sort(key=lambda x: x["score"], reverse=True)
            return results_with_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error executing BM25 retrieval query: {e}")
            return []
