import sys
import os
import fitz
from unittest.mock import MagicMock, patch

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.parser import parse_document
from services.chunker import DocumentChunker
from services.llm import LLMService

TEST_DIR = "test_tables_temp"

def make_simple_table_pdf(filename: str):
    pdf_path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    
    # Draw table bounding grid lines
    # Horizontal lines
    page.draw_line((50, 50), (250, 50))
    page.draw_line((50, 75), (250, 75))
    page.draw_line((50, 100), (250, 100))
    # Vertical lines
    page.draw_line((50, 50), (50, 100))
    page.draw_line((150, 50), (150, 100))
    page.draw_line((250, 50), (250, 100))
    
    # Insert text in grid cells
    page.insert_text((55, 67), "Employee")
    page.insert_text((155, 67), "Salary")
    page.insert_text((55, 92), "Alice")
    page.insert_text((155, 92), "60000")
    
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def make_multi_table_pdf(filename: str):
    pdf_path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    
    # Table 1: Employee/Salary
    page.draw_line((50, 50), (250, 50))
    page.draw_line((50, 75), (250, 75))
    page.draw_line((50, 100), (250, 100))
    page.draw_line((50, 50), (50, 100))
    page.draw_line((150, 50), (150, 100))
    page.draw_line((250, 50), (250, 100))
    page.insert_text((55, 67), "Employee")
    page.insert_text((155, 67), "Salary")
    page.insert_text((55, 92), "Alice")
    page.insert_text((155, 92), "60000")
    
    # Table 2: Project/Budget
    page.draw_line((50, 150), (250, 150))
    page.draw_line((50, 175), (250, 175))
    page.draw_line((50, 200), (250, 200))
    page.draw_line((50, 150), (50, 200))
    page.draw_line((150, 150), (150, 200))
    page.draw_line((250, 150), (250, 200))
    page.insert_text((55, 167), "Project")
    page.insert_text((155, 167), "Budget")
    page.insert_text((55, 192), "Alpha")
    page.insert_text((155, 192), "12000")
    
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def make_text_only_pdf(filename: str):
    pdf_path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is narrative text only. There are no tables in this document.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def make_mixed_pdf(filename: str):
    pdf_path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 30), "Report summary text details are displayed below:")
    
    # Add table grid
    page.draw_line((50, 50), (250, 50))
    page.draw_line((50, 75), (250, 75))
    page.draw_line((50, 100), (250, 100))
    page.draw_line((50, 50), (50, 100))
    page.draw_line((150, 50), (150, 100))
    page.draw_line((250, 50), (250, 100))
    page.insert_text((55, 67), "Employee")
    page.insert_text((155, 67), "Salary")
    page.insert_text((55, 92), "Alice")
    page.insert_text((155, 92), "60000")
    
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def setup_files():
    os.makedirs(TEST_DIR, exist_ok=True)
    f1 = make_simple_table_pdf("test_simple.pdf")
    f2 = make_multi_table_pdf("test_multi.pdf")
    f3 = make_text_only_pdf("test_text.pdf")
    f4 = make_mixed_pdf("test_mixed.pdf")
    return f1, f2, f3, f4

def cleanup():
    import shutil
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def run_tests():
    f1, f2, f3, f4 = setup_files()
    
    # Test 1: Simple table extraction
    print("\n--- Test 1: Simple Table Extraction ---")
    parsed_1 = parse_document(f1)
    page_1 = parsed_1["pages"][0]
    print(f"Extracted tables count: {len(page_1['tables'])}")
    assert len(page_1["tables"]) == 1
    print(f"Table output:\n{page_1['tables'][0]}")
    assert "Alice" in page_1["tables"][0]
    assert "60000" in page_1["tables"][0]
    print("Test 1 Passed!")
    
    # Test 2: Multiple tables on a page
    print("\n--- Test 2: Multiple Tables on Page ---")
    parsed_2 = parse_document(f2)
    page_2 = parsed_2["pages"][0]
    print(f"Extracted tables count: {len(page_2['tables'])}")
    assert len(page_2["tables"]) == 2
    print(f"Table 1:\n{page_2['tables'][0]}")
    print(f"Table 2:\n{page_2['tables'][1]}")
    assert "Budget" in page_2["tables"][1]
    assert "Alpha" in page_2["tables"][1]
    print("Test 2 Passed!")

    # Test 3: Text-only PDF
    print("\n--- Test 3: Text-only PDF ---")
    parsed_3 = parse_document(f3)
    page_3 = parsed_3["pages"][0]
    print(f"Extracted tables count: {len(page_3['tables'])}")
    assert len(page_3["tables"]) == 0
    assert "There are no tables" in page_3["text"]
    print("Test 3 Passed!")
    
    # Test 4: Mixed text and tables
    print("\n--- Test 4: Mixed Text and Tables ---")
    parsed_4 = parse_document(f4)
    page_4 = parsed_4["pages"][0]
    print(f"Text: '{page_4['text']}'")
    print(f"Extracted tables count: {len(page_4['tables'])}")
    assert len(page_4["tables"]) == 1
    assert "Report summary text" in page_4["text"]
    print("Test 4 Passed!")

    # Test 5: Question answering from table content
    print("\n--- Test 5: Question Answering on Table Chunks ---")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(parsed_4)
    
    print(f"Generated {len(chunks)} chunks.")
    # Chunks should include a text chunk and a table chunk
    types = [c["chunk_type"] for c in chunks]
    print(f"Chunk types: {types}")
    assert "text" in types
    assert "table" in types
    
    # Retrieve matching table chunk
    table_chunk = next(c for c in chunks if c["chunk_type"] == "table")
    print(f"Found table chunk content: '{table_chunk['chunk_text']}'")
    
    # Mock LLM generation call
    service = LLMService()
    service.api_key = "mock-api"
    with patch("services.llm.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Alice's salary is 60000."
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Test QA query passing the table chunk
        response = service.generate_response("What is Alice's salary?", [table_chunk])
        print(f"QA Response: '{response}'")
        assert "60000" in response
        assert "Sources:" in response
        assert "test_mixed.pdf (Page 1)" in response
        
    print("Test 5 Passed!")

if __name__ == "__main__":
    try:
        run_tests()
    finally:
        cleanup()
    print("\nAll Table Support tests passed successfully!")
