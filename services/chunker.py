from typing import List, Dict, Any
import os
from utils.helpers import get_logger

logger = get_logger("chunker")

class RecursiveTextSplitter:
    """
    Splits text recursively using a sequence of separators, falling back
    to finer splits (e.g. paragraphs -> sentences -> words -> characters)
    to keep chunk sizes below the configured maximum.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default recursive splitting separators
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap size must be strictly smaller than chunk size.")

    def split_text(self, text: str) -> List[str]:
        """Public entrypoint to split a text block recursively."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        # If the string fits in chunk size, no further splitting needed
        if len(text) <= self.chunk_size:
            return [text]

        # If we ran out of separators, fallback to character-level slicing with overlap
        if not separators:
            step = self.chunk_size - self.chunk_overlap
            return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

        separator = separators[0]
        next_separators = separators[1:]

        # If separator is not in text, skip it and go to the next separator
        if separator != "" and separator not in text:
            return self._split_text(text, next_separators)

        # Split on current separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current_doc = []
        current_len = 0

        for split in splits:
            # Re-attach the separator character to split segments (except for empty char splits)
            part = split + separator if separator != "" else split
            part_len = len(part)

            if part_len > self.chunk_size:
                # Flush what we accumulated so far
                if current_doc:
                    chunks.append("".join(current_doc))
                    current_doc = []
                    current_len = 0
                
                # Recursively split the oversized sub-part using finer separators
                sub_splits = self._split_text(split, next_separators)
                chunks.extend(sub_splits)
            else:
                if current_len + part_len > self.chunk_size:
                    # Flush current accumulator
                    chunks.append("".join(current_doc))
                    
                    # Backtrack to build the overlap segment
                    overlap_doc = []
                    overlap_len = 0
                    for item in reversed(current_doc):
                        if overlap_len + len(item) <= self.chunk_overlap:
                            overlap_doc.insert(0, item)
                            overlap_len += len(item)
                        else:
                            break
                    current_doc = overlap_doc
                    current_len = overlap_len
                
                current_doc.append(part)
                current_len += part_len

        # Flush final remaining buffer
        if current_doc:
            chunks.append("".join(current_doc))

        # Filter out empty strings and strip whitespace
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class DocumentChunker:
    """Orchestrates page-by-page document chunking with metadata and table extraction preservation."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"Initialized DocumentChunker with size={chunk_size}, overlap={chunk_overlap}")

    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits parsed document pages into smaller text and table chunks.
        
        Args:
            parsed_doc: Dictionary format:
                {
                    "document_name": "filename.pdf",
                    "pages": [
                        {
                            "page_number": 1, 
                            "text": "...",
                            "tables": ["markdown table 1", "markdown table 2"]
                        }
                    ]
                }
                
        Returns:
            A list of chunk dictionaries containing:
                - chunk_id (str)
                - page_number (int)
                - document_name (str)
                - chunk_text (str)
                - content (str)
                - chunk_type (str): "text" or "table"
        """
        document_name = parsed_doc.get("document_name", "unknown_document")
        pages = parsed_doc.get("pages", [])
        
        logger.info(f"Chunking document '{document_name}' containing {len(pages)} pages.")
        all_chunks = []
        
        for page in pages:
            page_num = page.get("page_number", 1)
            text_content = page.get("text", "")
            tables = page.get("tables", [])
            
            # 1. Process regular page narrative text
            if text_content.strip():
                page_splits = self.splitter.split_text(text_content)
                for index, chunk_text in enumerate(page_splits):
                    chunk_id = f"{document_name}#page_{page_num}#text_chunk_{index}"
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "document_name": document_name,
                        "chunk_text": chunk_text,
                        "content": chunk_text,
                        "chunk_type": "text"
                    })
                    
            # 2. Process page tabular text blocks (kept intact to protect grid layout relations)
            for index, table_text in enumerate(tables):
                if table_text.strip():
                    chunk_id = f"{document_name}#page_{page_num}#table_chunk_{index}"
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "document_name": document_name,
                        "chunk_text": table_text,
                        "content": table_text,
                        "chunk_type": "table"
                    })
                    
        logger.info(f"Generated {len(all_chunks)} chunks for document '{document_name}'.")
        return all_chunks
