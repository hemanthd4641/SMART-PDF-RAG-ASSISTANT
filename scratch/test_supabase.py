"""
scratch/test_supabase.py
--------------------------
Integration test suite for the Supabase database layer.

Tests cover:
  1. Supabase client connectivity
  2. Document insert / fetch / delete (with cascade chunk deletion)
  3. Chunk insert / fetch / count / type breakdown
  4. Chat history insert / fetch / recent / clear
  5. Error handling for invalid credentials

Usage:
    python scratch/test_supabase.py

All test data is cleaned up automatically at the end of each test.
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from database.database import (
    init_db,
    insert_document,
    fetch_all_documents,
    fetch_document_by_id,
    delete_document,
    insert_chunks,
    fetch_chunks_for_document,
    fetch_all_chunks,
    get_total_chunks,
    get_chunks_count_by_type,
    insert_chat_history,
    fetch_chat_history,
    fetch_recent_chat_history,
    clear_chat_history,
    fetch_page_chunks,
)
from database.supabase_client import get_client, SupabaseConnectionError, reset_client

# ── Test constants ────────────────────────────────────────────────────────────
TEST_DOC_ID   = "test_migration_abc123def456"
TEST_DOC_NAME = "test_rag_document.pdf"
PASS = "✓ PASS"
FAIL = "✗ FAIL"

results = {"passed": 0, "failed": 0}


def _ok(test_name: str):
    print(f"  {PASS}: {test_name}")
    results["passed"] += 1


def _fail(test_name: str, reason: str):
    print(f"  {FAIL}: {test_name}")
    print(f"         Reason: {reason}")
    results["failed"] += 1


def _cleanup():
    """Remove test data from Supabase."""
    try:
        client = get_client()
        client.table("documents").delete().eq("id", TEST_DOC_ID).execute()
        client.table("chat_history").delete().like("user_question", "[TEST]%").execute()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Connectivity
# ═══════════════════════════════════════════════════════════════════════════════

def test_connectivity():
    print("\n── Test 1: Supabase Connectivity ───────────────────────")
    try:
        init_db()
        _ok("init_db() reached Supabase without error")
    except Exception as e:
        _fail("init_db() connectivity check", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Document CRUD
# ═══════════════════════════════════════════════════════════════════════════════

def test_document_crud():
    print("\n── Test 2: Document Insert / Fetch / Delete ────────────")

    # Insert
    try:
        insert_document(
            doc_id=TEST_DOC_ID,
            document_name=TEST_DOC_NAME,
            page_count=10,
            summary="A test document about AI.",
            key_topics="AI, Testing, Supabase",
            native_page_count=8,
            ocr_page_count=2,
        )
        _ok("insert_document() succeeded")
    except Exception as e:
        _fail("insert_document()", str(e))
        return

    # Fetch all
    try:
        docs = fetch_all_documents()
        found = any(d["id"] == TEST_DOC_ID for d in docs)
        if found:
            _ok(f"fetch_all_documents() returned test document (total: {len(docs)})")
        else:
            _fail("fetch_all_documents()", "Test document not found in result")
    except Exception as e:
        _fail("fetch_all_documents()", str(e))

    # Fetch by ID
    try:
        doc = fetch_document_by_id(TEST_DOC_ID)
        assert doc is not None, "Returned None"
        assert doc["document_name"] == TEST_DOC_NAME, f"Name mismatch: {doc['document_name']}"
        assert doc["page_count"] == 10, f"Page count mismatch: {doc['page_count']}"
        assert doc["summary"] == "A test document about AI.", "Summary mismatch"
        _ok("fetch_document_by_id() returned correct record")
    except Exception as e:
        _fail("fetch_document_by_id()", str(e))

    # Fetch non-existent
    try:
        missing = fetch_document_by_id("nonexistent_id_xyz")
        assert missing is None, f"Expected None, got: {missing}"
        _ok("fetch_document_by_id() returns None for missing ID")
    except Exception as e:
        _fail("fetch_document_by_id() missing ID", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Chunk CRUD
# ═══════════════════════════════════════════════════════════════════════════════

def test_chunk_crud():
    print("\n── Test 3: Chunk Insert / Fetch / Count ────────────────")

    test_chunks = [
        {
            "chunk_id": f"{TEST_DOC_ID}#page_1#text_chunk_0",
            "document_id": TEST_DOC_ID,
            "page_number": 1,
            "chunk_text": "Attention mechanisms allow models to focus on relevant parts of the input.",
            "chunk_type": "text",
        },
        {
            "chunk_id": f"{TEST_DOC_ID}#page_1#text_chunk_1",
            "document_id": TEST_DOC_ID,
            "page_number": 1,
            "chunk_text": "The Transformer architecture relies entirely on self-attention.",
            "chunk_type": "text",
        },
        {
            "chunk_id": f"{TEST_DOC_ID}#page_2#table_chunk_0",
            "document_id": TEST_DOC_ID,
            "page_number": 2,
            "chunk_text": "| Model | BLEU | Params |\n| --- | --- | --- |\n| Transformer | 28.4 | 65M |",
            "chunk_type": "table",
        },
    ]

    # Insert
    try:
        insert_chunks(test_chunks)
        _ok(f"insert_chunks() inserted {len(test_chunks)} chunks")
    except Exception as e:
        _fail("insert_chunks()", str(e))
        return

    # Fetch for document
    try:
        fetched = fetch_chunks_for_document(TEST_DOC_ID)
        assert len(fetched) == 3, f"Expected 3 chunks, got {len(fetched)}"
        _ok(f"fetch_chunks_for_document() returned {len(fetched)} chunks")
    except Exception as e:
        _fail("fetch_chunks_for_document()", str(e))

    # Fetch page chunks
    try:
        page1 = fetch_page_chunks(TEST_DOC_NAME, 1)
        assert len(page1) == 2, f"Expected 2 chunks on page 1, got {len(page1)}"
        _ok(f"fetch_page_chunks() returned {len(page1)} chunks for page 1")
    except Exception as e:
        _fail("fetch_page_chunks()", str(e))

    # Fetch all chunks (with document_name join)
    try:
        all_chunks = fetch_all_chunks()
        test_chunks_in_result = [c for c in all_chunks if c["document_id"] == TEST_DOC_ID]
        assert len(test_chunks_in_result) == 3, f"Expected 3, got {len(test_chunks_in_result)}"
        assert test_chunks_in_result[0]["document_name"] == TEST_DOC_NAME, "document_name not joined"
        _ok(f"fetch_all_chunks() returned chunks with document_name (total: {len(all_chunks)})")
    except Exception as e:
        _fail("fetch_all_chunks()", str(e))

    # Count total chunks
    try:
        total = get_total_chunks()
        assert total >= 3, f"Expected ≥3 chunks, got {total}"
        _ok(f"get_total_chunks() returned {total}")
    except Exception as e:
        _fail("get_total_chunks()", str(e))

    # Count by type
    try:
        counts = get_chunks_count_by_type()
        assert counts.get("text", 0) >= 2, f"Expected ≥2 text chunks, got {counts}"
        assert counts.get("table", 0) >= 1, f"Expected ≥1 table chunk, got {counts}"
        _ok(f"get_chunks_count_by_type() → {counts}")
    except Exception as e:
        _fail("get_chunks_count_by_type()", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Cascade Delete
# ═══════════════════════════════════════════════════════════════════════════════

def test_cascade_delete():
    print("\n── Test 4: Document Delete (Cascade to Chunks) ─────────")
    try:
        delete_document(TEST_DOC_ID)
        # Verify document gone
        doc = fetch_document_by_id(TEST_DOC_ID)
        assert doc is None, "Document still exists after delete"
        # Verify chunks gone
        chunks = fetch_chunks_for_document(TEST_DOC_ID)
        assert len(chunks) == 0, f"Expected 0 chunks after cascade, got {len(chunks)}"
        _ok("delete_document() removed document and cascaded to chunks")
    except Exception as e:
        _fail("delete_document() cascade", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Chat History
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_history():
    print("\n── Test 5: Chat History CRUD ───────────────────────────")

    # Clear first to start fresh
    try:
        clear_chat_history()
        _ok("clear_chat_history() initial clear succeeded")
    except Exception as e:
        _fail("clear_chat_history() initial", str(e))

    # Insert
    qa_pairs = [
        ("[TEST] What is the Transformer?", "It is a model based on self-attention."),
        ("[TEST] Who wrote the paper?", "Vaswani et al., 2017."),
        ("[TEST] What task was it designed for?", "Sequence-to-sequence tasks like translation."),
    ]
    try:
        for q, a in qa_pairs:
            insert_chat_history(q, a)
        _ok(f"insert_chat_history() inserted {len(qa_pairs)} records")
    except Exception as e:
        _fail("insert_chat_history()", str(e))
        return

    # Fetch all
    try:
        history = fetch_chat_history()
        assert len(history) == len(qa_pairs), f"Expected {len(qa_pairs)}, got {len(history)}"
        assert history[0]["user_question"] == qa_pairs[0][0], "Order wrong"
        _ok(f"fetch_chat_history() returned {len(history)} records in correct order")
    except Exception as e:
        _fail("fetch_chat_history()", str(e))

    # Fetch recent (last 2)
    try:
        recent = fetch_recent_chat_history(limit=2)
        assert len(recent) == 2, f"Expected 2, got {len(recent)}"
        # Should be in chronological order (oldest of the recent 2 first)
        assert recent[0]["user_question"] == qa_pairs[1][0], f"Wrong order: {recent[0]}"
        _ok(f"fetch_recent_chat_history(limit=2) returned {len(recent)} records in order")
    except Exception as e:
        _fail("fetch_recent_chat_history()", str(e))

    # Clear
    try:
        clear_chat_history()
        after = fetch_chat_history()
        assert len(after) == 0, f"Expected 0 after clear, got {len(after)}"
        _ok("clear_chat_history() removed all records")
    except Exception as e:
        _fail("clear_chat_history()", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — Error Handling (invalid credentials)
# ═══════════════════════════════════════════════════════════════════════════════

def test_error_handling():
    print("\n── Test 6: Error Handling (Bad Credentials) ────────────")
    import os

    # Temporarily override env vars
    original_url = os.environ.get("SUPABASE_URL", "")
    original_key = os.environ.get("SUPABASE_KEY", "")

    try:
        os.environ["SUPABASE_URL"] = ""
        os.environ["SUPABASE_KEY"] = ""
        reset_client()

        # Re-import to pick up new env
        import utils.config as cfg
        orig_url = cfg.SUPABASE_URL
        orig_key = cfg.SUPABASE_KEY
        cfg.SUPABASE_URL = ""
        cfg.SUPABASE_KEY = ""

        try:
            get_client()
            _fail("get_client() with empty credentials", "Should have raised SupabaseConnectionError")
        except SupabaseConnectionError:
            _ok("get_client() raises SupabaseConnectionError when credentials are empty")
        finally:
            cfg.SUPABASE_URL = orig_url
            cfg.SUPABASE_KEY = orig_key
    finally:
        os.environ["SUPABASE_URL"] = original_url
        os.environ["SUPABASE_KEY"] = original_key
        reset_client()  # Re-init with real credentials for subsequent use


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("=" * 60)
    print("RAG Assistant — Supabase Integration Tests")
    print("=" * 60)

    _cleanup()  # Remove any leftover test data from previous runs

    test_connectivity()
    test_document_crud()
    test_chunk_crud()
    test_cascade_delete()
    test_chat_history()
    test_error_handling()

    _cleanup()  # Final cleanup

    print("\n" + "=" * 60)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    print("=" * 60)

    if results["failed"] > 0:
        sys.exit(1)
    else:
        print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
