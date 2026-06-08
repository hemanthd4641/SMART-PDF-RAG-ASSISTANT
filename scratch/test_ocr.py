import sys
import os
import io
import fitz  # PyMuPDF
from unittest.mock import MagicMock, patch

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.parser import parse_document
from services.retriever import RAGRetriever

def create_test_pdfs():
    """Programmatically generates test PDFs for the OCR fallback test suite."""
    os.makedirs("test_files", exist_ok=True)
    
    # 1. Text-based PDF (Native only)
    doc1 = fitz.open()
    p1 = doc1.new_page()
    p1.insert_text((50, 50), "This page contains native text layer. No OCR fallback should be triggered here.")
    doc1.save("test_files/test_native.pdf")
    doc1.close()
    
    # 2. Scanned PDF (Low text/Empty pages)
    doc2 = fitz.open()
    doc2.new_page() # Page 1: Completely empty
    doc2.save("test_files/test_scanned.pdf")
    doc2.close()
    
    # 3. Mixed PDF (Page 1 native text, Page 2 empty)
    doc3 = fitz.open()
    p_native = doc3.new_page()
    p_native.insert_text((50, 50), "Page 1: Native text is here.")
    doc3.new_page() # Page 2: empty
    doc3.save("test_files/test_mixed_ocr.pdf")
    doc3.close()

def run_tests():
    print("Starting OCR Fallback unit tests...")
    create_test_pdfs()
    
    try:
        # Patch easyocr.Reader to prevent network calls to Hugging Face
        with patch("easyocr.Reader") as mock_reader_class:
            mock_reader_instance = MagicMock()
            
            # Setup mock readtext output for scanned text:
            # bbox, text, confidence
            mock_reader_instance.readtext.return_value = [
                ([0, 0, 10, 10], "Invoice Number INV-1234 Amount 500 USD", 0.98)
            ]
            mock_reader_class.return_value = mock_reader_instance
            
            # Import services.ocr to bind the mock reader
            from services.ocr import ocr_service
            ocr_service._reader = mock_reader_instance

            # --- Test 1: Text-based PDF (Native) ---
            print("\n--- Running Test 1: Text-based PDF (Native Only) ---")
            result_native = parse_document("test_files/test_native.pdf")
            
            assert len(result_native["pages"]) == 1
            page = result_native["pages"][0]
            print(f"Page 1 Method: {page['extraction_method']}")
            print(f"Page 1 Text Preview: '{page['text'][:40]}...'")
            
            assert page["extraction_method"] == "native"
            assert "native text layer" in page["text"]
            # Verify EasyOCR was NEVER called
            mock_reader_instance.readtext.assert_not_called()
            print("Test 1 Passed!")

            # --- Test 2: Scanned PDF (OCR) ---
            print("\n--- Running Test 2: Scanned PDF (OCR Triggered) ---")
            mock_reader_instance.readtext.reset_mock()
            
            result_scanned = parse_document("test_files/test_scanned.pdf")
            
            assert len(result_scanned["pages"]) == 1
            page_scanned = result_scanned["pages"][0]
            print(f"Page 1 Method: {page_scanned['extraction_method']}")
            print(f"Page 1 Text Preview: '{page_scanned['text'][:40]}...'")
            
            assert page_scanned["extraction_method"] == "ocr"
            assert "INV-1234" in page_scanned["text"]
            # Verify EasyOCR was called
            mock_reader_instance.readtext.assert_called_once()
            print("Test 2 Passed!")

            # --- Test 3: Mixed PDF (Native + OCR) ---
            print("\n--- Running Test 3: Mixed PDF (Native + OCR) ---")
            mock_reader_instance.readtext.reset_mock()
            
            result_mixed = parse_document("test_files/test_mixed_ocr.pdf")
            
            assert len(result_mixed["pages"]) == 2
            p1 = result_mixed["pages"][0]
            p2 = result_mixed["pages"][1]
            
            print(f"Page 1 (Native) Method: {p1['extraction_method']} | Text: '{p1['text']}'")
            print(f"Page 2 (OCR) Method: {p2['extraction_method']} | Text: '{p2['text'][:40]}...'")
            
            assert p1["extraction_method"] == "native"
            assert p2["extraction_method"] == "ocr"
            assert "INV-1234" in p2["text"]
            # Verify EasyOCR was called exactly once (for page 2)
            mock_reader_instance.readtext.assert_called_once()
            print("Test 3 Passed!")

            # --- Test 4: Question Answering over OCR Retrieval ---
            print("\n--- Running Test 4: QA Retrieval Grounding ---")
            # We mock the retrieval of the OCR chunk
            mock_embeddings = MagicMock()
            mock_pinecone = MagicMock()
            
            # Setup retriever returning the OCR-extracted chunk
            retriever = RAGRetriever(embedding_service=mock_embeddings, vector_store=mock_pinecone)
            mock_embeddings.generate_embeddings.return_value = [[0.1] * 384]
            mock_pinecone.query_vectors.return_value = [
                {
                    "id": "invoice.pdf#page_1#text_chunk_0",
                    "score": 0.95,
                    "metadata": {
                        "document_name": "invoice.pdf",
                        "page_number": 1,
                        "text": "Invoice Number INV-1234 Amount 500 USD",
                        "chunk_type": "text"
                    }
                }
            ]
            
            from services.llm import LLMService
            llm = LLMService()
            
            with patch("groq.Groq") as mock_groq_class:
                mock_groq_client = MagicMock()
                mock_groq_class.return_value = mock_groq_client
                
                # Mock Groq chat completions response
                mock_completions = MagicMock()
                mock_completions.choices = [
                    MagicMock(message=MagicMock(content="The invoice number is INV-1234."))
                ]
                mock_groq_client.chat.completions.create.return_value = mock_completions
                llm.client = mock_groq_client
                
                # Query RAG
                retrieved_chunks = retriever.retrieve("What is the invoice number?", top_k=1)
                answer = llm.generate_response("What is the invoice number?", retrieved_chunks)
                
                print(f"Retrieval Text: '{retrieved_chunks[0]['chunk_text']}'")
                print(f"QA Response: '{answer}'")
                
                assert "INV-1234" in retrieved_chunks[0]["chunk_text"]
                assert "INV-1234" in answer
                print("Test 4 Passed!")

    finally:
        # Cleanup temp test files
        if os.path.exists("test_files"):
            for f in os.listdir("test_files"):
                try:
                    os.remove(os.path.join("test_files", f))
                except:
                    pass
            try:
                os.rmdir("test_files")
            except:
                pass
                
    print("\nAll OCR fallback tests passed successfully!")

if __name__ == "__main__":
    run_tests()
