import sys
import os
import fitz

# Append parent directory of scratch so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.parser import parse_document, DocumentParsingError

def create_test_files():
    print("Creating programmatically generated test files...")
    os.makedirs("test_files", exist_ok=True)
    
    # 1. Create a valid text file
    txt_path = os.path.join("test_files", "test_valid.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("This is a valid test text document content.\nLine two of the document.")
    print(f"Created: {txt_path}")
        
    # 2. Create a valid PDF file using PyMuPDF
    pdf_path = os.path.join("test_files", "test_valid.pdf")
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is page one text content.")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is page two text content.")
    
    doc.save(pdf_path)
    doc.close()
    print(f"Created: {pdf_path}")
    
    # 3. Create an empty PDF (0 bytes)
    empty_pdf_path = os.path.join("test_files", "test_empty.pdf")
    with open(empty_pdf_path, "wb") as f:
        pass
    print(f"Created: {empty_pdf_path}")
    
    # 4. Create a corrupted PDF file (invalid file signature headers)
    corrupt_pdf_path = os.path.join("test_files", "test_corrupt.pdf")
    with open(corrupt_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 but corrupted with random binary bytes \x00\xFF\x99\x11")
    print(f"Created: {corrupt_pdf_path}")
    
    return txt_path, pdf_path, empty_pdf_path, corrupt_pdf_path

def cleanup_test_files():
    print("\nCleaning up test files...")
    import shutil
    if os.path.exists("test_files"):
        shutil.rmtree("test_files")
        print("Removed test_files directory.")

def run_tests():
    txt_path, pdf_path, empty_pdf_path, corrupt_pdf_path = create_test_files()
    
    # Test 1: Valid TXT Parsing
    print("\n--- Running Test 1: Valid TXT Parsing ---")
    res_txt = parse_document(txt_path)
    print(f"Result TXT keys: {res_txt.keys()}")
    print(f"Result TXT Pages Count: {len(res_txt['pages'])}")
    assert res_txt["document_name"] == "test_valid.txt"
    assert len(res_txt["pages"]) == 1
    assert "Line two of the document." in res_txt["pages"][0]["text"]
    print("Test 1 Passed!")
    
    # Test 2: Valid PDF Parsing
    print("\n--- Running Test 2: Valid PDF Parsing ---")
    res_pdf = parse_document(pdf_path)
    print(f"Result PDF keys: {res_pdf.keys()}")
    print(f"Result PDF Pages Count: {len(res_pdf['pages'])}")
    assert res_pdf["document_name"] == "test_valid.pdf"
    assert len(res_pdf["pages"]) == 2
    assert res_pdf["pages"][0]["page_number"] == 1
    assert "page one text" in res_pdf["pages"][0]["text"].lower()
    assert res_pdf["pages"][1]["page_number"] == 2
    assert "page two text" in res_pdf["pages"][1]["text"].lower()
    print("Test 2 Passed!")
    
    # Test 3: Empty PDF (0 bytes) Parsing
    print("\n--- Running Test 3: Empty PDF Handling ---")
    try:
        parse_document(empty_pdf_path)
        assert False, "Should have raised DocumentParsingError"
    except DocumentParsingError as e:
        print(f"Successfully caught DocumentParsingError for empty PDF: '{e}'")
    print("Test 3 Passed!")
        
    # Test 4: Corrupted PDF Parsing
    print("\n--- Running Test 4: Corrupted PDF Handling ---")
    try:
        parse_document(corrupt_pdf_path)
        assert False, "Should have raised DocumentParsingError"
    except DocumentParsingError as e:
        print(f"Successfully caught DocumentParsingError for corrupted PDF: '{e}'")
    print("Test 4 Passed!")
    
    print("\nAll Parser tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    finally:
        cleanup_test_files()
