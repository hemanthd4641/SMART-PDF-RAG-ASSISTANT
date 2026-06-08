"""
Creates test_company.pdf for RAG pipeline testing.
Uses PyMuPDF (fitz) which is already a project dependency.
"""
import fitz  # PyMuPDF
import os

# Output path — save inside the rag-assistant data directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_company.pdf")

# Document content
CONTENT = """ABC Technologies Pvt Ltd

Founded: 2020

CEO: Sarah Johnson

Headquarters: Bangalore, India

Number of Employees: 500

Main Products:
1. AI Legal Assistant
2. Contract Analyzer
3. Document Search Platform

The company signed a strategic partnership agreement with XYZ Corp in March 2024.

The partnership agreement is valid for 3 years.

The confidentiality period under the agreement is 5 years.

Customer support is available Monday to Friday from 9 AM to 6 PM.

The company received funding of $10 million in 2023.
"""

def create_pdf(content: str, output_path: str) -> None:
    """Creates a single-page searchable PDF from plain text content."""
    doc = fitz.open()  # new empty document
    page = doc.new_page(width=595, height=842)  # A4 page size in points

    # Define text rect with margins
    rect = fitz.Rect(60, 60, 535, 780)

    # Insert the text block
    page.insert_textbox(
        rect,
        content,
        fontsize=13,
        fontname="helv",
        color=(0, 0, 0),
        align=0,  # left-aligned
    )

    doc.save(output_path)
    doc.close()
    print(f"[OK] PDF created successfully at: {output_path}")

if __name__ == "__main__":
    create_pdf(CONTENT, OUTPUT_PATH)
