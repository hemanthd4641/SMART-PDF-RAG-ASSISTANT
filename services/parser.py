import fitz  # PyMuPDF
import pdfplumber
import os
from typing import List, Dict, Any
from utils.helpers import get_logger, time_it

logger = get_logger("parser")

class DocumentParsingError(Exception):
    """Exception raised when a document cannot be parsed due to corruption, invalid formats, or empty content."""
    pass

def table_to_markdown(table: List[List[Any]]) -> str:
    """
    Converts a raw table list-of-lists extracted by pdfplumber into a clean Markdown table string.
    Handles empty cells and pads rows to ensure valid Markdown table structure.
    
    Args:
        table: Raw list-of-lists representing the table rows.
        
    Returns:
        A Markdown formatted table string, or empty string if table is empty.
    """
    if not table:
        return ""
        
    clean_table = []
    for row in table:
        if row is None:
            continue
        # Replace None cells with empty string and clean surrounding whitespace
        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
        # Only keep rows that are not completely empty (filters empty rows)
        if any(clean_row):
            clean_table.append(clean_row)
            
    if not clean_table:
        return ""
        
    # Extract headers and body rows
    headers = clean_table[0]
    rows = clean_table[1:]
    
    # Construct Markdown segments
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    row_lines = []
    for row in rows:
        # Match row cell length to headers count to prevent column distortions
        row_padded = row + [""] * (len(headers) - len(row))
        row_lines.append("| " + " | ".join(row_padded[:len(headers)]) + " |")
        
    return "\n".join([header_line, separator_line] + row_lines)

@time_it
def parse_pdf(file_path: str, progress_callback=None) -> Dict[str, Any]:
    """
    Parses a PDF file extracting regular text using PyMuPDF and tables using pdfplumber.
    Merges both outputs on a page-by-page basis.
    
    Args:
        file_path: Absolute path to the PDF document.
        
    Returns:
        Structured output containing:
            {
                "document_name": "filename.pdf",
                "pages": [
                    {
                        "page_number": 1,
                        "text": "regular extracted page text...",
                        "tables": ["markdown table 1", "markdown table 2"]
                    }
                ]
            }
            
    Raises:
        DocumentParsingError: If file is corrupted, invalid, or empty.
    """
    document_name = os.path.basename(file_path)
    pages = []
    
    try:
        # 1. Base check for empty files
        if os.path.getsize(file_path) == 0:
            raise DocumentParsingError("The PDF file is empty (0 bytes).")
            
        # 2. Extract page texts using PyMuPDF (fast, highly reliable text layer) with OCR Fallback
        fitz_pages = []
        extraction_methods = []
        doc = fitz.open(file_path)
        if doc.is_closed:
            raise DocumentParsingError("PDF file structure is closed or corrupted.")
            
        page_count = len(doc)
        for page_idx in range(page_count):
            page_num = page_idx + 1
            method = "native"
            text = ""
            try:
                page = doc.load_page(page_idx)
                raw_text = page.get_text()
                text = raw_text.strip() if raw_text else ""
                
                # If extracted text is empty or very short (< 20 characters), run OCR fallback
                if len(text) < 20:
                    logger.info(f"Page {page_num} of '{document_name}' has low native text ({len(text)} chars). Triggering EasyOCR fallback...")
                    if progress_callback:
                        progress_callback(page_num)
                    try:
                        # Render page to Pixmap (150 DPI is standard for text extraction speed/accuracy balance)
                        pix = page.get_pixmap(dpi=150)
                        
                        import io
                        from PIL import Image
                        img_data = pix.tobytes("png")
                        pil_img = Image.open(io.BytesIO(img_data))
                        
                        from services.ocr import extract_text_from_image
                        ocr_text = extract_text_from_image(pil_img)
                        
                        if ocr_text:
                            text = ocr_text
                            method = "ocr"
                            logger.info(f"Page {page_num} successfully processed with OCR fallback ({len(text)} chars).")
                        else:
                            logger.warning(f"OCR fallback on page {page_num} returned empty results. Sticking with native text.")
                    except Exception as ocr_err:
                        logger.error(f"OCR fallback failed on page {page_num} of '{document_name}': {ocr_err}. Continuing with native text.")
            except Exception as e:
                logger.error(f"PyMuPDF failed to read text on page {page_num} of '{document_name}': {e}")
                
            fitz_pages.append(text)
            extraction_methods.append(method)
            
        doc.close()
    except fitz.FileDataError as e:
        logger.error(f"PyMuPDF FileDataError parsing PDF '{file_path}': {e}")
        raise DocumentParsingError(f"Corrupted or invalid PDF file layout: {e}")
    except DocumentParsingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected PyMuPDF error parsing PDF '{file_path}': {e}")
        raise DocumentParsingError(f"PyMuPDF parsing failed: {e}")

    # 3. Extract tables using pdfplumber (isolated inside exception blocks to prevent crashing)
    plumber_tables_by_page = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_tables = []
                try:
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            try:
                                md_table = table_to_markdown(tbl)
                                if md_table:
                                    page_tables.append(md_table)
                            except Exception as format_err:
                                logger.error(
                                    f"Failed to format table row cells on page {page_idx + 1} of '{document_name}': {format_err}"
                                )
                except Exception as table_extract_err:
                    logger.error(
                        f"pdfplumber table extraction error on page {page_idx + 1} of '{document_name}': {table_extract_err}"
                    )
                plumber_tables_by_page[page_idx] = page_tables
    except Exception as e:
        # If pdfplumber fails entirely (e.g. invalid PDF elements), log the failure and fall back to empty tables list
        logger.error(f"pdfplumber failed to extract tables from '{file_path}': {e}. Continuing with text extraction only.")

    # 4. Merge results page-by-page
    total_pages = max(len(fitz_pages), len(plumber_tables_by_page))
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_text = fitz_pages[page_idx] if page_idx < len(fitz_pages) else ""
        page_tables = plumber_tables_by_page.get(page_idx, [])
        method = extraction_methods[page_idx] if page_idx < len(extraction_methods) else "native"
        
        pages.append({
            "page_number": page_num,
            "text": page_text,
            "tables": page_tables,
            "extraction_method": method
        })
        
    return {
        "document_name": document_name,
        "pages": pages
    }

@time_it
def parse_txt(file_path: str) -> Dict[str, Any]:
    """
    Parses a TXT file using standard Python file reading.
    
    Args:
        file_path: Absolute path to the TXT document.
        
    Returns:
        A dictionary matching the RAG parsed format with an empty 'tables' array.
    """
    document_name = os.path.basename(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text_content = f.read()
            
        return {
            "document_name": document_name,
            "pages": [
                {
                    "page_number": 1,
                    "text": text_content.strip(),
                    "tables": [],  # Standard text files contain no tables
                    "extraction_method": "native"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to parse TXT file '{file_path}': {e}")
        raise DocumentParsingError(f"Failed to read TXT file: {e}")

def parse_document(file_path: str, progress_callback=None) -> Dict[str, Any]:
    """
    Orchestrates parsing based on file extension.

    Supported formats:
        - .pdf  → PyMuPDF text + pdfplumber tables + EasyOCR fallback
        - .txt  → Standard Python file reader
        - .docx → python-docx paragraph + table extraction

    Args:
        file_path: Absolute path to the document.
        progress_callback: Optional callable for OCR progress updates (PDF only).

    Returns:
        Structured parsed dictionary matching requested output format:
        {
            "document_name": str,
            "pages": [{"page_number": int, "text": str, "tables": list, "extraction_method": str}]
        }

    Raises:
        FileNotFoundError: If the file does not exist.
        DocumentParsingError: If the format is unsupported or parsing fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return parse_pdf(file_path, progress_callback)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".docx":
        from services.docx_parser import extract_docx_text, DocxParsingError as _DocxError
        try:
            return extract_docx_text(file_path)
        except _DocxError as e:
            raise DocumentParsingError(str(e))
    else:
        raise DocumentParsingError(
            f"Unsupported document format: '{ext}'. "
            f"Supported formats are: PDF (.pdf), TXT (.txt), DOCX (.docx)."
        )
