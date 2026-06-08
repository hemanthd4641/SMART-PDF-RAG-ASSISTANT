import sys
import os

# Append parent directory of scratch so we can import database and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import (
    init_db,
    insert_document,
    fetch_document_by_id,
    insert_chunks,
    fetch_page_chunks,
    get_connection
)
from utils.config import SQLITE_DB_PATH

def run_tests():
    print("Starting database summary and page preview unit tests...")
    
    # 1. Clean up old test database file if it exists
    if os.path.exists(SQLITE_DB_PATH):
        try:
            os.remove(SQLITE_DB_PATH)
            print(f"Removed existing database file at {SQLITE_DB_PATH}")
        except Exception as e:
            print(f"Failed to remove existing db file: {e}")
            
    # 2. Initialize Database Schema
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "database", "schema.sql"
    )
    init_db(schema_path=schema_path)
    
    # Verify migration: columns summary and key_topics should exist in the documents table
    with get_connection() as conn:
        cursor = conn.execute("PRAGMA table_info(documents);")
        columns = [row["name"] for row in cursor.fetchall()]
        assert "summary" in columns, "Migration failed: summary column is missing"
        assert "key_topics" in columns, "Migration failed: key_topics column is missing"
    print("Database migration check passed: summary and key_topics columns successfully added.")

    # 3. Test insert and fetch document with summary & key topics
    doc_name = "test_summary.pdf"
    import hashlib
    doc_id = hashlib.md5(doc_name.encode('utf-8')).hexdigest()
    page_count = 10
    summary = "This document is a comprehensive guide to building neural networks."
    key_topics = "AI, Deep Learning, PyTorch, Neural Networks"
    
    insert_document(doc_id, doc_name, page_count, summary, key_topics)
    
    doc = fetch_document_by_id(doc_id)
    assert doc is not None
    assert doc["summary"] == summary, f"Summary mismatch: {doc['summary']}"
    assert doc["key_topics"] == key_topics, f"Key topics mismatch: {doc['key_topics']}"
    print("Document CRUD check passed: Summary and key topics successfully inserted and retrieved.")
    
    # 4. Test chunk retrieval for page reconstruction
    chunks = [
        {
            "chunk_id": f"{doc_id}#page_4#text_chunk_0",
            "document_id": doc_id,
            "page_number": 4,
            "chunk_text": "Section 1. Introduction to CNNs. Convolutional Neural Networks are designed for spatial data.",
            "chunk_type": "text"
        },
        {
            "chunk_id": f"{doc_id}#page_4#text_chunk_1",
            "document_id": doc_id,
            "page_number": 4,
            "chunk_text": "Section 2. Max Pooling Layers. Max pooling layers reduce spatial dimensions of feature maps.",
            "chunk_type": "text"
        },
        {
            "chunk_id": f"{doc_id}#page_5#text_chunk_0",
            "document_id": doc_id,
            "page_number": 5,
            "chunk_text": "Section 3. Backpropagation. Backpropagation computes gradients using the chain rule.",
            "chunk_type": "text"
        }
    ]
    insert_chunks(chunks)
    
    # Retrieve page 4 chunks
    page_4_chunks = fetch_page_chunks(doc_name, 4)
    assert len(page_4_chunks) == 2, f"Expected 2 chunks for page 4, got {len(page_4_chunks)}"
    assert page_4_chunks[0]["chunk_text"].startswith("Section 1"), "Chunks are not ordered by chunk_id"
    assert page_4_chunks[1]["chunk_text"].startswith("Section 2"), "Chunks are not ordered by chunk_id"
    
    # Retrieve page 5 chunks
    page_5_chunks = fetch_page_chunks(doc_name, 5)
    assert len(page_5_chunks) == 1, f"Expected 1 chunk for page 5, got {len(page_5_chunks)}"
    
    # Reconstruct text
    reconstructed_page_4 = "\n".join([c["chunk_text"] for c in page_4_chunks])
    expected_reconstructed = (
        "Section 1. Introduction to CNNs. Convolutional Neural Networks are designed for spatial data.\n"
        "Section 2. Max Pooling Layers. Max pooling layers reduce spatial dimensions of feature maps."
    )
    assert reconstructed_page_4 == expected_reconstructed, "Reconstructed text mismatch"
    print("Page reconstruction check passed: Chunks retrieved are complete, correct, and ordered by chunk_id.")
    
    print("\nAll database summary and page preview tests passed successfully!")

if __name__ == "__main__":
    run_tests()
