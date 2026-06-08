import sys
import os
from unittest.mock import MagicMock

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retriever import RAGRetriever

def run_tests():
    print("Starting RAG Retriever Re-ranking unit tests...")
    
    # 1. Initialize Mock Dependencies
    mock_embeddings = MagicMock()
    mock_pinecone = MagicMock()
    
    retriever = RAGRetriever(embedding_service=mock_embeddings, vector_store=mock_pinecone)
    
    # Setup mock returns
    mock_embeddings.generate_embeddings.return_value = [[0.1] * 384]
    
    # Return 4 mock chunks from Pinecone.
    # Notice that the one that has the most relevant answer ("Google DeepMind team built Antigravity")
    # has a lower vector score (0.60) compared to the noisy/irrelevant one (0.90).
    mock_pinecone.query_vectors.return_value = [
        {
            "id": "doc.pdf#page_1#chunk_0",
            "score": 0.90,
            "metadata": {
                "document_name": "doc.pdf",
                "page_number": 1,
                "text": "The weather today is sunny with mild winds from the east."
            }
        },
        {
            "id": "doc.pdf#page_2#chunk_1",
            "score": 0.82,
            "metadata": {
                "document_name": "doc.pdf",
                "page_number": 2,
                "text": "Python is a high-level programming language known for readability."
            }
        },
        {
            "id": "doc.pdf#page_3#chunk_2",
            "score": 0.60,
            "metadata": {
                "document_name": "doc.pdf",
                "page_number": 3,
                "text": "The Google DeepMind team built the Antigravity coding assistant in 2026."
            }
        }
    ]
    
    # Test 1: Cross-Encoder Lazy Loading
    print("\n--- Running Test 1: Cross-Encoder Lazy Loading ---")
    from unittest.mock import patch
    
    with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder_class:
        mock_cross_encoder_instance = MagicMock()
        # Mock scores: first chunk gets 0.10, second gets 0.20, third (relevant one) gets 0.95
        mock_cross_encoder_instance.predict.return_value = [0.10, 0.20, 0.95]
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance

        assert retriever._cross_encoder is None, "Cross-Encoder should not be loaded on init"
        encoder = retriever.get_cross_encoder()
        assert encoder is not None, "Lazy loading should instantiate the Cross-Encoder"
        assert retriever._cross_encoder is not None, "Internal state should store the Cross-Encoder"
        mock_cross_encoder_class.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Test 1 Passed!")
    
        # Test 2: Re-ranking Output Reordering
        print("\n--- Running Test 2: Re-ranking Output Reordering ---")
        query = "Who built the Antigravity coding assistant?"
        
        # Retrieve with re-ranking enabled
        results = retriever.retrieve(query, top_k=2, use_reranker=True)
        
        # Check that Pinecone was queried with fetch_k = max(top_k * 4, 20) = 20
        mock_pinecone.query_vectors.assert_called_once_with(
            query_vector=[0.1] * 384,
            top_k=20,
            filter_dict=None
        )
        
        # We requested top_k=2, so we expect exactly 2 results returned
        assert len(results) == 2, f"Should return top_k (2) results, got {len(results)}"
        
        print("\nRe-ranked results in order:")
        for idx, item in enumerate(results):
            print(f"Rank {idx+1}: {item['document_name']} (Page {item['page_number']}) | Score: {item['score']:.4f} | Preview: '{item['chunk_text']}'")
    
        # The most relevant block should be ranked #1
        assert "Google DeepMind" in results[0]["chunk_text"], "Re-ranker failed to rank the most relevant context at position 0"
        print("Test 2 Passed!")
        
    print("\nAll retriever re-ranking unit tests passed successfully!")

if __name__ == "__main__":
    run_tests()
