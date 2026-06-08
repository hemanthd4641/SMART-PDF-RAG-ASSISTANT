"""
database/database.py
---------------------
Supabase PostgreSQL persistence layer for the RAG Assistant.

All public function signatures are identical to the original SQLite version so
that no upstream code (app.py, components/, services/) needs any changes.

Every function:
  1. Obtains the shared Supabase client via get_client().
  2. Executes the operation via the Supabase PostgREST SDK.
  3. Returns data in the same dict/list format as before.
  4. Catches ALL exceptions, logs them, and re-raises a clean RuntimeError
     so no raw stack traces leak into the Streamlit UI.
"""

import hashlib
from typing import List, Dict, Any, Optional

from database.supabase_client import get_client, SupabaseConnectionError
from utils.helpers import get_logger

logger = get_logger("database")

# ── Batch size for chunk upserts (keeps request payload manageable) ───────────
_CHUNK_BATCH_SIZE = 200


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(schema_path: str = "database/schema.sql") -> None:
    """
    Verifies Supabase connectivity by pinging the documents table.

    The `schema_path` argument is accepted for API compatibility with the old
    SQLite version but is intentionally ignored — the schema must be applied
    once via the Supabase Dashboard SQL Editor using database/schema.sql.

    Raises:
        SupabaseConnectionError: If credentials are missing.
        RuntimeError: If the connectivity check fails.
    """
    try:
        client = get_client()
        client.table("documents").select("id").limit(1).execute()
        logger.info("Supabase connectivity verified successfully.")
    except SupabaseConnectionError:
        raise
    except Exception as e:
        logger.error(f"Supabase connectivity check failed: {e}")
        raise RuntimeError(
            f"Cannot reach Supabase. Check SUPABASE_URL and SUPABASE_KEY. Detail: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Document Operations
# ═══════════════════════════════════════════════════════════════════════════════

def insert_document(
    doc_id: str,
    document_name: str,
    page_count: int,
    summary: Optional[str] = None,
    key_topics: Optional[str] = None,
    native_page_count: int = 0,
    ocr_page_count: int = 0,
) -> None:
    """
    Inserts (or upserts) a document metadata record into Supabase.

    Uses upsert so that re-running ingestion on the same file is idempotent.
    """
    payload = {
        "id": doc_id,
        "document_name": document_name,
        "page_count": page_count,
        "summary": summary,
        "key_topics": key_topics,
        "native_page_count": native_page_count,
        "ocr_page_count": ocr_page_count,
    }
    try:
        client = get_client()
        client.table("documents").upsert(payload).execute()
        logger.info(f"Document '{document_name}' ({doc_id}) upserted into Supabase.")
    except Exception as e:
        logger.error(f"Failed to insert document '{document_name}': {e}")
        raise RuntimeError(f"Document insert failed: {e}")


def fetch_all_documents() -> List[Dict[str, Any]]:
    """
    Retrieves all document metadata records ordered by upload timestamp.
    """
    try:
        client = get_client()
        response = (
            client.table("documents")
            .select(
                "id, document_name, upload_timestamp, page_count, "
                "summary, key_topics, native_page_count, ocr_page_count"
            )
            .order("upload_timestamp", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch all documents: {e}")
        raise RuntimeError(f"Document fetch failed: {e}")


def fetch_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single document metadata record by its ID.
    Returns None if the document does not exist.
    """
    try:
        client = get_client()
        response = (
            client.table("documents")
            .select(
                "id, document_name, upload_timestamp, page_count, "
                "summary, key_topics, native_page_count, ocr_page_count"
            )
            .eq("id", doc_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.error(f"Failed to fetch document with ID '{doc_id}': {e}")
        raise RuntimeError(f"Document fetch by ID failed: {e}")


def fetch_page_chunks(document_name: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Fetches all chunks for a specific page of a document.
    Resolves document_id from the document_name via MD5 hash (same as upload.py).
    """
    doc_id = hashlib.md5(document_name.encode("utf-8")).hexdigest()
    try:
        client = get_client()
        response = (
            client.table("chunks")
            .select("chunk_id, chunk_text, chunk_type")
            .eq("document_id", doc_id)
            .eq("page_number", page_number)
            .order("chunk_id", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch page chunks for '{document_name}' page {page_number}: {e}")
        return []


def delete_document(doc_id: str) -> None:
    """
    Deletes a document from Supabase. The ON DELETE CASCADE FK constraint
    automatically removes all associated chunks.
    """
    try:
        client = get_client()
        client.table("documents").delete().eq("id", doc_id).execute()
        logger.info(f"Document '{doc_id}' and its chunks deleted from Supabase.")
    except Exception as e:
        logger.error(f"Failed to delete document '{doc_id}': {e}")
        raise RuntimeError(f"Document deletion failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk Operations
# ═══════════════════════════════════════════════════════════════════════════════

def insert_chunks(chunks_list: List[Dict[str, Any]]) -> None:
    """
    Batch-upserts document chunks into Supabase in batches of 200.

    Each dict must contain: chunk_id, document_id, page_number, chunk_text.
    chunk_type defaults to 'text' if not present.
    """
    if not chunks_list:
        return

    # Normalise: ensure chunk_type is always set
    normalized = []
    for chunk in chunks_list:
        c = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page_number": chunk["page_number"],
            "chunk_text": chunk["chunk_text"],
            "chunk_type": chunk.get("chunk_type", "text"),
        }
        normalized.append(c)

    try:
        client = get_client()
        # Process in batches to stay within Supabase request size limits
        for i in range(0, len(normalized), _CHUNK_BATCH_SIZE):
            batch = normalized[i : i + _CHUNK_BATCH_SIZE]
            client.table("chunks").upsert(batch).execute()
            logger.debug(f"Upserted chunk batch {i // _CHUNK_BATCH_SIZE + 1}: {len(batch)} chunks.")

        logger.info(f"Successfully upserted {len(normalized)} chunks into Supabase.")
    except Exception as e:
        logger.error(f"Failed to insert chunks batch: {e}")
        raise RuntimeError(f"Chunk insert failed: {e}")


def fetch_chunks_for_document(doc_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all chunks for a specific document ID, ordered by page then chunk_id.
    """
    try:
        client = get_client()
        response = (
            client.table("chunks")
            .select("chunk_id, document_id, page_number, chunk_text, chunk_type")
            .eq("document_id", doc_id)
            .order("page_number", desc=False)
            .order("chunk_id", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch chunks for document '{doc_id}': {e}")
        raise RuntimeError(f"Chunk fetch failed: {e}")


def fetch_all_chunks() -> List[Dict[str, Any]]:
    """
    Retrieves all chunks with their associated document_name.

    Performs two queries (chunks + documents) and merges in Python,
    which is reliable regardless of PostgREST foreign-key join syntax.
    """
    try:
        client = get_client()

        # 1. Fetch all documents to build id → name mapping
        docs_resp = client.table("documents").select("id, document_name").execute()
        doc_map: Dict[str, str] = {
            d["id"]: d["document_name"] for d in (docs_resp.data or [])
        }

        # 2. Fetch all chunks
        chunks_resp = (
            client.table("chunks")
            .select("chunk_id, document_id, page_number, chunk_text, chunk_type")
            .order("document_id", desc=False)
            .order("page_number", desc=False)
            .order("chunk_id", desc=False)
            .execute()
        )

        # 3. Merge document_name into each chunk record
        result = []
        for row in (chunks_resp.data or []):
            row["document_name"] = doc_map.get(row["document_id"], "unknown")
            result.append(row)

        return result

    except Exception as e:
        logger.error(f"Failed to fetch all chunks: {e}")
        raise RuntimeError(f"All-chunks fetch failed: {e}")


def get_total_chunks() -> int:
    """
    Returns the total number of chunks stored in Supabase.
    Uses Supabase's server-side count to avoid fetching all rows.
    """
    try:
        client = get_client()
        response = (
            client.table("chunks")
            .select("chunk_id", count="exact")
            .execute()
        )
        return response.count or 0
    except Exception as e:
        logger.error(f"Failed to get total chunk count: {e}")
        return 0


def get_chunks_count_by_type() -> Dict[str, int]:
    """
    Returns a breakdown of chunk counts by type: {"text": N, "table": M}.
    """
    results = {"text": 0, "table": 0}
    try:
        client = get_client()
        for chunk_type in ("text", "table"):
            response = (
                client.table("chunks")
                .select("chunk_id", count="exact")
                .eq("chunk_type", chunk_type)
                .execute()
            )
            results[chunk_type] = response.count or 0
        return results
    except Exception as e:
        logger.error(f"Failed to get chunk counts by type: {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Chat History Operations
# ═══════════════════════════════════════════════════════════════════════════════

def insert_chat_history(user_question: str, assistant_answer: str) -> None:
    """
    Persists a single question–answer exchange to the chat_history table.
    The UUID primary key and timestamp are auto-generated by PostgreSQL.
    """
    payload = {
        "user_question": user_question,
        "assistant_answer": assistant_answer,
    }
    try:
        client = get_client()
        client.table("chat_history").insert(payload).execute()
        logger.debug("Chat history record inserted.")
    except Exception as e:
        logger.error(f"Failed to insert chat history: {e}")
        raise RuntimeError(f"Chat history insert failed: {e}")


def fetch_chat_history() -> List[Dict[str, Any]]:
    """
    Retrieves all chat history rows in chronological order (oldest first).
    """
    try:
        client = get_client()
        response = (
            client.table("chat_history")
            .select("id, user_question, assistant_answer, timestamp")
            .order("timestamp", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch chat history: {e}")
        raise RuntimeError(f"Chat history fetch failed: {e}")


def fetch_recent_chat_history(limit: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent N chat turns for multi-turn conversation context.
    Returns turns in chronological order (oldest first) so the LLM sees them
    in the correct conversation order.

    Args:
        limit: Maximum number of recent turns to fetch (default 6 = last 3 exchanges).
    """
    try:
        client = get_client()
        response = (
            client.table("chat_history")
            .select("user_question, assistant_answer")
            .order("timestamp", desc=True)  # newest first
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        # Reverse to get chronological order (oldest first)
        return list(reversed(rows))
    except Exception as e:
        logger.error(f"Failed to fetch recent chat history: {e}")
        return []


def clear_chat_history() -> None:
    """
    Deletes all rows from the chat_history table.

    Uses a timestamp filter covering all realistic records to satisfy
    Supabase's requirement that DELETE operations include a WHERE clause.
    """
    try:
        client = get_client()
        # Delete all rows with timestamp ≤ far future (effectively all rows)
        client.table("chat_history").delete().lte(
            "timestamp", "2999-12-31T23:59:59Z"
        ).execute()
        logger.info("Chat history cleared from Supabase.")
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise RuntimeError(f"Chat history clear failed: {e}")
