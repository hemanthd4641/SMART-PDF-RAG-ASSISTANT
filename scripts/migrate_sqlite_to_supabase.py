"""
scripts/migrate_sqlite_to_supabase.py
---------------------------------------
One-shot migration utility: reads all existing SQLite data and upserts it
into Supabase PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_supabase.py

Safe to re-run — all operations use upsert, so duplicate rows are silently
skipped rather than causing errors.

Requirements:
    - SQLite database file must exist at the path below (or override via env).
    - SUPABASE_URL and SUPABASE_KEY must be set in .env.
"""

import sys
import os
import sqlite3
import logging

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from database.supabase_client import get_client, SupabaseConnectionError

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migration")

# ── SQLite source path ────────────────────────────────────────────────────────
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/rag_metadata.db")
CHUNK_BATCH_SIZE = 200


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sqlite_connection() -> sqlite3.Connection:
    if not os.path.exists(SQLITE_DB_PATH):
        raise FileNotFoundError(
            f"SQLite database not found at: {SQLITE_DB_PATH}\n"
            "Nothing to migrate — the file may have already been removed."
        )
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows) -> list:
    return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Migration functions
# ═══════════════════════════════════════════════════════════════════════════════

def migrate_documents(sqlite_conn: sqlite3.Connection, supabase_client) -> int:
    """Migrate all document records from SQLite to Supabase. Returns migrated count."""
    logger.info("── Migrating documents ──────────────────────────────")
    try:
        cursor = sqlite_conn.execute(
            "SELECT id, document_name, upload_timestamp, page_count, "
            "summary, key_topics, native_page_count, ocr_page_count "
            "FROM documents"
        )
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not read 'documents' table from SQLite: {e}")
        return 0

    rows = _rows_to_dicts(cursor.fetchall())
    if not rows:
        logger.info("No document records found in SQLite. Skipping.")
        return 0

    # Normalise None values and type conversions
    payloads = []
    for row in rows:
        payloads.append({
            "id":                 row.get("id") or "",
            "document_name":      row.get("document_name") or "",
            "upload_timestamp":   row.get("upload_timestamp"),
            "page_count":         int(row.get("page_count") or 0),
            "summary":            row.get("summary"),
            "key_topics":         row.get("key_topics"),
            "native_page_count":  int(row.get("native_page_count") or 0),
            "ocr_page_count":     int(row.get("ocr_page_count") or 0),
        })

    try:
        supabase_client.table("documents").upsert(payloads).execute()
        logger.info(f"✓ Migrated {len(payloads)} document(s) to Supabase.")
        return len(payloads)
    except Exception as e:
        logger.error(f"✗ Document migration failed: {e}")
        return 0


def migrate_chunks(sqlite_conn: sqlite3.Connection, supabase_client) -> int:
    """Migrate all chunk records in batches. Returns migrated count."""
    logger.info("── Migrating chunks ─────────────────────────────────")
    try:
        cursor = sqlite_conn.execute(
            "SELECT chunk_id, document_id, page_number, chunk_text, chunk_type "
            "FROM chunks"
        )
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not read 'chunks' table from SQLite: {e}")
        return 0

    rows = _rows_to_dicts(cursor.fetchall())
    if not rows:
        logger.info("No chunk records found in SQLite. Skipping.")
        return 0

    # Normalise chunk_type default
    payloads = []
    for row in rows:
        payloads.append({
            "chunk_id":    row["chunk_id"],
            "document_id": row["document_id"],
            "page_number": int(row.get("page_number") or 1),
            "chunk_text":  row.get("chunk_text") or "",
            "chunk_type":  row.get("chunk_type") or "text",
        })

    migrated = 0
    failed = 0
    total_batches = (len(payloads) + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE

    for i in range(0, len(payloads), CHUNK_BATCH_SIZE):
        batch = payloads[i : i + CHUNK_BATCH_SIZE]
        batch_num = i // CHUNK_BATCH_SIZE + 1
        try:
            supabase_client.table("chunks").upsert(batch).execute()
            migrated += len(batch)
            logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} chunks migrated.")
        except Exception as e:
            failed += len(batch)
            logger.error(f"  Batch {batch_num}/{total_batches} failed: {e}")

    logger.info(f"✓ Chunks migration complete: {migrated} migrated, {failed} failed.")
    return migrated


def migrate_chat_history(sqlite_conn: sqlite3.Connection, supabase_client) -> int:
    """Migrate all chat history records. Returns migrated count."""
    logger.info("── Migrating chat history ───────────────────────────")
    try:
        cursor = sqlite_conn.execute(
            "SELECT user_question, assistant_answer, timestamp FROM chat_history"
        )
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not read 'chat_history' table from SQLite: {e}")
        return 0

    rows = _rows_to_dicts(cursor.fetchall())
    if not rows:
        logger.info("No chat history records found in SQLite. Skipping.")
        return 0

    # Chat history uses UUID PKs in Supabase (auto-generated) — don't pass id
    payloads = []
    for row in rows:
        payloads.append({
            "user_question":    row.get("user_question") or "",
            "assistant_answer": row.get("assistant_answer") or "",
            "timestamp":        row.get("timestamp"),
        })

    try:
        supabase_client.table("chat_history").insert(payloads).execute()
        logger.info(f"✓ Migrated {len(payloads)} chat history record(s) to Supabase.")
        return len(payloads)
    except Exception as e:
        logger.error(f"✗ Chat history migration failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("RAG Assistant — SQLite → Supabase Migration")
    logger.info("=" * 60)

    # 1. Validate Supabase connection
    logger.info("Step 1: Connecting to Supabase...")
    try:
        client = get_client()
        # Ping the documents table
        client.table("documents").select("id").limit(1).execute()
        logger.info("✓ Supabase connection verified.")
    except SupabaseConnectionError as e:
        logger.error(f"✗ Cannot connect to Supabase: {e}")
        logger.error("Ensure SUPABASE_URL and SUPABASE_KEY are set in your .env file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Supabase ping failed: {e}")
        sys.exit(1)

    # 2. Open SQLite source
    logger.info(f"Step 2: Opening SQLite database at: {SQLITE_DB_PATH}")
    try:
        sqlite_conn = _get_sqlite_connection()
        logger.info("✓ SQLite database opened.")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # 3. Run migrations in dependency order (documents → chunks → chat)
    logger.info("Step 3: Running migrations...")
    doc_count   = migrate_documents(sqlite_conn, client)
    chunk_count = migrate_chunks(sqlite_conn, client)
    chat_count  = migrate_chat_history(sqlite_conn, client)

    sqlite_conn.close()

    # 4. Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"  Documents migrated  : {doc_count}")
    logger.info(f"  Chunks migrated     : {chunk_count}")
    logger.info(f"  Chat turns migrated : {chat_count}")
    logger.info("=" * 60)
    logger.info("Migration complete. You can now run the app with Supabase.")


if __name__ == "__main__":
    main()
