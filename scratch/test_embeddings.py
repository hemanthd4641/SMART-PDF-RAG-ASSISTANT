import sys
import os

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embeddings import EmbeddingService

def run_tests():
    print("Starting embedding service tests...")
    
    # 1. Initialize service
    service = EmbeddingService()
    
    # Assert lazy loading (model shouldn't be loaded yet)
    print("\nChecking lazy loading model initialization...")
    assert service.model is None, "Model should start as None before call"
    print("Lazy loading check passed: Model initialized as None.")
    
    # 2. Test generation for valid strings
    test_chunks = [
        "Attention is all you need is a seminal paper in machine learning.",
        "Generative pre-trained transformers are trained on large text corpora.",
        "Vector databases like Pinecone enable fast nearest-neighbor similarity searches."
    ]
    
    print(f"\nGenerating embeddings for {len(test_chunks)} mock chunks...")
    vectors = service.generate_embeddings(test_chunks, batch_size=2)
    
    # Assert model is loaded now
    assert service.model is not None, "Model should be instantiated after call"
    
    # Assert output shape
    print(f"Generated {len(vectors)} embedding vectors.")
    assert len(vectors) == 3, "Output count must match input count"
    
    # Assert dimensions (all-MiniLM-L6-v2 dimension should be 384)
    print("Checking dimensions and types...")
    for idx, vec in enumerate(vectors):
        dim = len(vec)
        print(f"Vector {idx+1} dimension: {dim}")
        assert dim == 384, f"Vector dimension mismatch (expected 384, got {dim})"
        assert isinstance(vec, list), "Vector should be returned as a list"
        assert all(isinstance(val, float) for val in vec), "Vector values should all be floats"
        
    print("Vector dimensions and types verified!")
    
    # 3. Test empty inputs
    print("\nTesting empty list input handling...")
    empty_vectors = service.generate_embeddings([])
    assert len(empty_vectors) == 0, "Empty list input should return empty list output"
    print("Empty list check passed!")
    
    print("\nAll embedding tests completed successfully!")

if __name__ == "__main__":
    run_tests()
