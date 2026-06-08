import sys
import os
from unittest.mock import MagicMock, patch

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hybrid_retriever import reciprocal_rank_fusion
from services.bm25_retriever import BM25Retriever, tokenize

def run_tests():
    print("Starting Hybrid Search and RRF Fusion unit tests...")

    # 1. Mock SQLite chunks corpus
    mock_chunks = [
        {
            "chunk_id": "contract.pdf#page_1#text_chunk_0",
            "document_id": "doc1",
            "page_number": 1,
            "chunk_text": "This page explains the official termination conditions and employee policies.",
            "document_name": "contract.pdf"
        },
        {
            "chunk_id": "contract.pdf#page_2#text_chunk_1",
            "document_id": "doc1",
            "page_number": 2,
            "chunk_text": "The unique reference billing key is INV-2024-8891 for invoice processing.",
            "document_name": "contract.pdf"
        },
        {
            "chunk_id": "contract.pdf#page_3#text_chunk_2",
            "document_id": "doc1",
            "page_number": 3,
            "chunk_text": "The agreement contract duration lasts for 24 months with renewal options.",
            "document_name": "contract.pdf"
        }
    ]

    # --- Test 1: Tokenization ---
    print("\n--- Running Test 1: Tokenizer ---")
    tokens = tokenize("contract duration, INV-2024-8891!")
    print(f"Tokens: {tokens}")
    assert "contract" in tokens
    assert "duration" in tokens
    assert "inv20248891" in tokens # punctuation removed
    print("Test 1 Passed!")

    # --- Test 2: BM25 Retriever standalone ---
    print("\n--- Running Test 2: BM25 Index & Retrieval ---")
    with patch("services.bm25_retriever.fetch_all_chunks") as mock_fetch:
        mock_fetch.return_value = mock_chunks
        
        # Build index
        bm25_retriever = BM25Retriever()
        assert len(bm25_retriever.chunks) == 3, "Corpus size mismatch"
        assert bm25_retriever.bm25 is not None, "BM25 index not built"
        
        # Query 1: Exact match code
        results_code = bm25_retriever.retrieve_bm25("INV-2024-8891", top_k=2)
        print(f"Results for 'INV-2024-8891': {[r['chunk_id'] for r in results_code]}")
        assert len(results_code) >= 1
        assert "page_2" in results_code[0]["chunk_id"], "Should retrieve chunk 2 first due to exact code match"
        
        # Query 2: Semantic keywords
        results_term = bm25_retriever.retrieve_bm25("termination conditions", top_k=2)
        print(f"Results for 'termination conditions': {[r['chunk_id'] for r in results_term]}")
        assert len(results_term) >= 1
        assert "page_1" in results_term[0]["chunk_id"], "Should retrieve chunk 1 first due to keyword match"
        
        print("Test 2 Passed!")

    # --- Test 3: RRF Fusion Logic (Test 1, Test 2, and Test 3 criteria) ---
    print("\n--- Running Test 3: RRF Fusion Ordering ---")
    
    # Mock Dense (Pinecone) Results (ranked chunk 3 first, then chunk 1)
    dense_results = [
        {
            "chunk_id": "contract.pdf#page_3#text_chunk_2",
            "chunk_text": "The agreement contract duration lasts for 24 months with renewal options.",
            "document_name": "contract.pdf",
            "page_number": 3,
            "score": 0.88,
            "chunk_type": "text"
        },
        {
            "chunk_id": "contract.pdf#page_1#text_chunk_0",
            "chunk_text": "This page explains the official termination conditions and employee policies.",
            "document_name": "contract.pdf",
            "page_number": 1,
            "score": 0.72,
            "chunk_type": "text"
        }
    ]
    
    # Mock Sparse (BM25) Results (ranked chunk 2 first, then chunk 3)
    sparse_results = [
        {
            "chunk_id": "contract.pdf#page_2#text_chunk_1",
            "chunk_text": "The unique reference billing key is INV-2024-8891 for invoice processing.",
            "document_name": "contract.pdf",
            "page_number": 2,
            "score": 12.5,
            "chunk_type": "text"
        },
        {
            "chunk_id": "contract.pdf#page_3#text_chunk_2",
            "chunk_text": "The agreement contract duration lasts for 24 months with renewal options.",
            "document_name": "contract.pdf",
            "page_number": 3,
            "score": 8.1,
            "chunk_type": "text"
        }
    ]
    
    # Case A: Exact Match Query - BM25 contributes heavily
    fused_a = reciprocal_rank_fusion(dense_results=[], sparse_results=sparse_results, top_n=2)
    print(f"RRF Fusion A (Exact Match): {[r['chunk_id'] for r in fused_a]}")
    assert fused_a[0]["chunk_id"] == "contract.pdf#page_2#text_chunk_1", "BM25 fallback failed"
    
    # Case B: Mixed Query - Both systems contribute
    # Chunk 3 is rank 1 in dense, rank 2 in sparse -> RRF score should be 1/(0+60) + 1/(1+60)
    # Chunk 2 is rank 1 in sparse -> RRF score should be 1/(0+60)
    fused_b = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=3)
    print("\nRRF Fusion B (Mixed Query):")
    for idx, r in enumerate(fused_b):
        print(f"Rank {idx+1}: {r['chunk_id']} | score: {r['score']:.5f}")
        
    assert fused_b[0]["chunk_id"] == "contract.pdf#page_3#text_chunk_2", "Chunk 3 should win due to contributions from both dense and sparse"
    print("Test 3 Passed!")

    # --- Test 4: Error Handling Fallback ---
    print("\n--- Running Test 4: Fallback Strategy ---")
    
    # Fallback to sparse if dense is empty
    fused_sparse_fallback = reciprocal_rank_fusion(dense_results=[], sparse_results=sparse_results, top_n=5)
    assert len(fused_sparse_fallback) == 2
    assert fused_sparse_fallback[0]["chunk_id"] == "contract.pdf#page_2#text_chunk_1"
    
    # Fallback to dense if sparse is empty
    fused_dense_fallback = reciprocal_rank_fusion(dense_results=dense_results, sparse_results=[], top_n=5)
    assert len(fused_dense_fallback) == 2
    assert fused_dense_fallback[0]["chunk_id"] == "contract.pdf#page_3#text_chunk_2"
    
    print("Test 4 Passed!")
    
    print("\nAll Hybrid Search and RRF Fusion unit tests completed successfully!")

if __name__ == "__main__":
    run_tests()
