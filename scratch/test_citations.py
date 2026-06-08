import sys
import os
from unittest.mock import MagicMock, patch

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm import LLMService, generate_citations_block

def test_citations_block_generation():
    print("Testing citation formatting and ordering...")
    
    # 1. Chunks with duplicates and mixed orders
    chunks = [
        {"document_name": "contract.pdf", "page_number": 4},
        {"document_name": "contract.pdf", "page_number": 7},
        {"document_name": "contract.pdf", "page_number": 4},  # Duplicate
        {"document_name": "policy.txt", "page_number": 1},
        {"document_name": "contract.pdf", "page_number": 7}   # Duplicate
    ]
    
    citations_text = generate_citations_block(chunks)
    print(f"\nGenerated Citations Text:\n{citations_text}")
    
    # Assert layout structure
    expected_structure = (
        "Sources:\n\n"
        "* contract.pdf (Page 4)\n"
        "* contract.pdf (Page 7)\n"
        "* policy.txt (Page 1)"
    )
    assert citations_text.strip() == expected_structure.strip(), "Citation block structure or ordering mismatch"
    print("Citations block generation test passed!")

def test_citations_in_response():
    print("\nTesting LLMService citation integrations...")
    
    service = LLMService()
    service.api_key = "mock-key"
    
    retrieved_chunks = [
        {"document_name": "attention.pdf", "page_number": 3, "chunk_text": "Transformer attention details."}
    ]
    
    with patch("services.llm.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        # Test Case A: Valid answer should attach citations
        mock_completion_valid = MagicMock()
        mock_choice_valid = MagicMock()
        mock_message_valid = MagicMock()
        mock_message_valid.content = "The Transformer is awesome."
        mock_choice_valid.message = mock_message_valid
        mock_completion_valid.choices = [mock_choice_valid]
        mock_client.chat.completions.create.return_value = mock_completion_valid
        
        ans_valid = service.generate_response("Question?", retrieved_chunks)
        print(f"\nResponse Valid:\n{ans_valid}")
        assert "The Transformer is awesome." in ans_valid
        assert "Sources:" in ans_valid
        assert "attention.pdf (Page 3)" in ans_valid
        print("Test Case A (Valid response adds citations) Passed!")
        
        # Test Case B: Refusal response should NOT attach citations
        mock_completion_refuse = MagicMock()
        mock_choice_refuse = MagicMock()
        mock_message_refuse = MagicMock()
        mock_message_refuse.content = "I could not find this information in the uploaded documents."
        mock_choice_refuse.message = mock_message_refuse
        mock_completion_refuse.choices = [mock_choice_refuse]
        mock_client.chat.completions.create.return_value = mock_completion_refuse
        
        ans_refuse = service.generate_response("Question?", retrieved_chunks)
        print(f"\nResponse Refusal:\n{ans_refuse}")
        assert ans_refuse == "I could not find this information in the uploaded documents."
        assert "Sources:" not in ans_refuse
        assert "attention.pdf" not in ans_refuse
        print("Test Case B (Refusal excludes citations) Passed!")

if __name__ == "__main__":
    test_citations_block_generation()
    test_citations_in_response()
    print("\nAll citation integration tests completed successfully!")
