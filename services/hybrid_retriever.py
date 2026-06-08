from typing import List, Dict, Any
from utils.helpers import get_logger

logger = get_logger("hybrid")

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Merges dense vector search results and sparse BM25 results using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF_score = sum( 1 / (rank + k) ) for each document in both lists.
    
    Args:
        dense_results: List of retrieved chunks from Pinecone.
        sparse_results: List of retrieved chunks from BM25.
        k: Smoothing constant parameter (default: 60).
        top_n: Final number of merged results to return.
        
    Returns:
        List of fused and re-ranked chunks.
    """
    rrf_scores = {}
    chunk_mapping = {}
    
    # 1. Process dense vector results (ranked order is preserved by index in list)
    for rank, chunk in enumerate(dense_results):
        chunk_id = chunk["chunk_id"]
        chunk_mapping[chunk_id] = chunk
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + 1 + k)
        
    # 2. Process sparse keyword results (ranked order is preserved by index in list)
    for rank, chunk in enumerate(sparse_results):
        chunk_id = chunk["chunk_id"]
        chunk_mapping[chunk_id] = chunk
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + 1 + k)
        
    # If no results returned from either retriever, return empty list
    if not rrf_scores:
        return []
        
    # 3. Sort chunk IDs by final fusion score in descending order
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # 4. Compile final results mapping
    merged_results = []
    for chunk_id in sorted_chunk_ids[:top_n]:
        chunk = chunk_mapping[chunk_id].copy()
        chunk["score"] = rrf_scores[chunk_id] # Update chunk score to the merged RRF score
        merged_results.append(chunk)
        
    logger.debug(f"Merged {len(dense_results)} dense & {len(sparse_results)} sparse results using RRF. Retained top {len(merged_results)}.")
    return merged_results
