import sys
import os

# Append parent directory of scratch so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import (
    init_db,
    insert_document,
    fetch_all_documents,
    fetch_document_by_id,
    delete_document,
    insert_chunks,
    fetch_chunks_for_document,
    insert_chat_history,
    fetch_chat_history,
    clear_chat_history
)
from utils.config import SQLITE_DB_PATH

def run_tests():
    print("Starting SQLite database tests...")
    
    # 1. Clean up old test database file if it exists
    if os.path.exists(SQLITE_DB_PATH):
        try:
            os.remove(SQLITE_DB_PATH)
            print(f"Removed existing database file at {SQLITE_DB_PATH}")
        except Exception as e:
            print(f"Failed to remove existing db file: {e}")
            
    # 2. Init DB schema
    print("\n--- Initializing Database Schema ---")
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "database", "schema.sql"
    )
    init_db(schema_path=schema_path)
    
    # 3. Test insert and fetch documents
    print("\n--- Testing Document CRUD ---")
    doc_id = "test-doc-123"
    document_name = "Attention_Is_All_You_Need.pdf"
    page_count = 15
    
    insert_document(doc_id, document_name, page_count)
    
    docs = fetch_all_documents()
    print(f"Fetch All Documents count: {len(docs)}")
    assert len(docs) == 1, "Documents count should be 1"
    print(f"Fetched Document: {docs[0]}")
    
    doc = fetch_document_by_id(doc_id)
    assert doc is not None, "Fetched doc should not be None"
    assert doc["document_name"] == document_name, "Name mismatch"
    
    # 4. Test insert and fetch chunks
    print("\n--- Testing Chunks CRUD ---")
    chunks = [
        {
            "chunk_id": "test-doc-123#0",
            "document_id": doc_id,
            "page_number": 1,
            "chunk_text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
        },
        {
            "chunk_id": "test-doc-123#1",
            "document_id": doc_id,
            "page_number": 2,
            "chunk_text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
        }
    ]
    
    insert_chunks(chunks)
    
    fetched_chunks = fetch_chunks_for_document(doc_id)
    print(f"Fetch Chunks count: {len(fetched_chunks)}")
    assert len(fetched_chunks) == 2, "Chunks count should be 2"
    print(f"Chunk 1: {fetched_chunks[0]}")
    print(f"Chunk 2: {fetched_chunks[1]}")
    
    # 5. Test chat history
    print("\n--- Testing Chat History CRUD ---")
    insert_chat_history("What is the Transformer architecture based on?", "It is based solely on attention mechanisms.")
    insert_chat_history("What models were dominant before?", "Complex recurrent or convolutional neural networks.")
    
    history = fetch_chat_history()
    print(f"Chat History count: {len(history)}")
    assert len(history) == 2, "Chat history should contain 2 messages"
    for item in history:
        print(f"[{item['timestamp']}] Q: {item['user_question']} -> A: {item['assistant_answer']}")
        
    # 6. Test Cascade delete
    print("\n--- Testing Cascade Delete ---")
    delete_document(doc_id)
    
    # Verify document is gone
    docs = fetch_all_documents()
    assert len(docs) == 0, "Document list should be empty after deletion"
    
    # Verify chunks are gone (cascaded deletion)
    remaining_chunks = fetch_chunks_for_document(doc_id)
    assert len(remaining_chunks) == 0, "Chunks list should be empty after document deletion"
    print("Cascade delete verified: document deletion also deleted all associated chunks.")
    
    # 7. Test clear chat history
    print("\n--- Testing Clear Chat ---")
    clear_chat_history()
    history = fetch_chat_history()
    assert len(history) == 0, "Chat history should be empty after clearing"
    print("Chat history cleared successfully.")

    print("\nAll database tests passed successfully!")

if __name__ == "__main__":
    run_tests()
