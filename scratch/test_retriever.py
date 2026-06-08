import sys
import os
from unittest.mock import MagicMock

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retriever import RAGRetriever

def run_tests():
    print("Starting RAG retriever service unit tests...")
    
    # 1. Initialize Mock Dependencies
    mock_embeddings = MagicMock()
    mock_pinecone = MagicMock()
    
    # Patch fetch_all_chunks to prevent real SQLite access during test
    from unittest.mock import patch
    with patch("services.bm25_retriever.fetch_all_chunks") as mock_fetch:
        mock_fetch.return_value = [
            {
                "chunk_id": "intro.pdf#page_2#chunk_0",
                "document_id": "doc1",
                "page_number": 2,
                "chunk_text": "This is retrieved context paragraph.",
                "document_name": "intro.pdf"
            }
        ]
        retriever = RAGRetriever(embedding_service=mock_embeddings, vector_store=mock_pinecone)
    
    # 2. Setup mock returns for normal execution
    mock_embeddings.generate_embeddings.return_value = [[0.2] * 384]
    
    # Setup mock Pinecone query output
    mock_pinecone.query_vectors.return_value = [
        {
            "id": "intro.pdf#page_2#chunk_0",
            "score": 0.89,
            "metadata": {
                "document_name": "intro.pdf",
                "page_number": 2,
                "text": "This is retrieved context paragraph."
            }
        },
        {
            "id": "intro.pdf#page_1#chunk_1",
            "score": 0.76,
            "metadata": {
                "document_name": "intro.pdf",
                "page_number": 1,
                "text": "This is another secondary match paragraph."
            }
        }
    ]
    
    # Test 1: Standard retrieval mapping
    print("\n--- Running Test 1: Retrieval Mapping ---")
    query = "What is retrieved context?"
    results = retriever.retrieve(query, top_k=2)
    
    # Assertions
    mock_embeddings.generate_embeddings.assert_called_once_with([query])
    mock_pinecone.query_vectors.assert_called_once_with(
        query_vector=[0.2] * 384,
        top_k=10,
        filter_dict=None
    )
    
    # RRF fuses matches and returns top_k=2
    assert len(results) == 2, "Should return 2 context matches"
    
    # Assert key matching
    for idx, item in enumerate(results):
        print(f"Match {idx+1}: {item}")
        assert "chunk_text" in item
        assert "document_name" in item
        assert "page_number" in item
        assert "score" in item
        assert isinstance(item["score"], float)
        assert isinstance(item["page_number"], int)
        
    assert results[0]["chunk_text"] == "This is retrieved context paragraph."
    print("Test 1 (Mapping) Passed!")
    
    # Test 2: Document filtering
    print("\n--- Running Test 2: Retrieval Document Filtering ---")
    mock_embeddings.generate_embeddings.reset_mock()
    mock_pinecone.query_vectors.reset_mock()
    
    results_filtered = retriever.retrieve(query, top_k=5, document_filter="intro.pdf")
    mock_pinecone.query_vectors.assert_called_once_with(
        query_vector=[0.2] * 384,
        top_k=10,
        filter_dict={"document_name": {"$eq": "intro.pdf"}}
    )
    print("Test 2 (Filtering syntax) Passed!")

    # Test 3: Empty query inputs
    print("\n--- Running Test 3: Empty Query Handling ---")
    results_empty = retriever.retrieve("")
    assert len(results_empty) == 0, "Empty query should return empty results list"
    print("Test 3 (Empty input) Passed!")
    
    # Test 4: Graceful error handling
    print("\n--- Running Test 4: Error Handling Fallback ---")
    mock_embeddings.generate_embeddings.side_effect = RuntimeError("Embedding service connection lost")
    results_err = retriever.retrieve("Faulty query")
    assert len(results_err) == 0, "Retrieve should return empty list on exception"
    print("Test 4 (Error fallback) Passed!")
    
    print("\nAll retriever unit tests passed successfully!")

if __name__ == "__main__":
    run_tests()
