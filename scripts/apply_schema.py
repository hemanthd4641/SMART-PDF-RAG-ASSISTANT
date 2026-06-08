"""
scripts/apply_schema.py
------------------------
Applies the Supabase PostgreSQL schema by executing each DDL statement
through the Supabase REST API using the `sql` endpoint (requires service role key
OR by running individual table-creation via a helper RPC).

ALTERNATIVE (recommended if this script fails):
    Copy and paste the contents of database/schema.sql into:
    Supabase Dashboard → Your Project → SQL Editor → New query → Run

Usage:
    python scripts/apply_schema.py

This script uses the supabase-py client and a workaround via the
Supabase Management API to execute raw SQL statements.
"""

import sys
import os
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Extract project ref from URL (e.g. https://lfrvophizljxrbhtjsbl.supabase.co → lfrvophizljxrbhtjsbl)
PROJECT_REF = SUPABASE_URL.replace("https://", "").split(".")[0] if SUPABASE_URL else ""

DDL_STATEMENTS = [
    # documents table
    """
    CREATE TABLE IF NOT EXISTS documents (
        id                TEXT        PRIMARY KEY,
        document_name     TEXT        NOT NULL,
        upload_timestamp  TIMESTAMPTZ DEFAULT NOW(),
        page_count        INTEGER     NOT NULL,
        summary           TEXT,
        key_topics        TEXT,
        native_page_count INTEGER     DEFAULT 0,
        ocr_page_count    INTEGER     DEFAULT 0
    )
    """,
    # chunks table
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id    TEXT    PRIMARY KEY,
        document_id TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        chunk_text  TEXT    NOT NULL,
        chunk_type  TEXT    DEFAULT 'text'
    )
    """,
    # chunks index
    """
    CREATE INDEX IF NOT EXISTS idx_chunks_document_page
        ON chunks (document_id, page_number)
    """,
    # chat_history table
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        user_question    TEXT        NOT NULL,
        assistant_answer TEXT        NOT NULL,
        timestamp        TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    # chat_history index
    """
    CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp
        ON chat_history (timestamp ASC)
    """,
]


def apply_via_management_api():
    """
    Uses the Supabase Management REST API to execute SQL.
    Requires a valid service_role or personal access token.
    """
    if not PROJECT_REF:
        print("ERROR: Could not extract project ref from SUPABASE_URL")
        return False

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "apikey": SUPABASE_KEY,
    }

    # Try the Supabase SQL endpoint (available via pg_net or direct db query)
    sql_endpoint = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"

    all_sql = ";\n".join(stmt.strip() for stmt in DDL_STATEMENTS) + ";"

    try:
        resp = httpx.post(
            sql_endpoint,
            headers=headers,
            json={"query": all_sql},
            timeout=30,
        )
        if resp.status_code in (200, 201):
            print("✓ Schema applied via Management API.")
            return True
        else:
            print(f"Management API returned {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"Management API call failed: {e}")
        return False


def apply_via_rpc_workaround():
    """
    Alternative: uses the Supabase client to execute each DDL via a custom
    SQL function. This requires the `exec_sql` function to exist in Supabase —
    which typically doesn't by default. Prints instructions if not available.
    """
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    success_count = 0
    for i, stmt in enumerate(DDL_STATEMENTS):
        stmt = stmt.strip()
        try:
            # Try via rpc if you have a helper function defined
            client.rpc("exec_sql", {"sql": stmt}).execute()
            print(f"  ✓ Statement {i+1} executed via RPC.")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Statement {i+1} failed via RPC: {e}")

    return success_count == len(DDL_STATEMENTS)


def print_manual_instructions():
    print()
    print("=" * 70)
    print("MANUAL SETUP INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Automatic schema application requires a service_role key.")
    print("Please apply the schema manually:")
    print()
    print("1. Open your Supabase Dashboard:")
    print("   https://supabase.com/dashboard/project/lfrvophizljxrbhtjsbl")
    print()
    print("2. Click 'SQL Editor' in the left sidebar.")
    print()
    print("3. Click '+ New query'.")
    print()
    print("4. Paste the following SQL and click 'Run':")
    print()
    print("-" * 70)

    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "database", "schema.sql"
    )
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            print(f.read())
    else:
        print("(schema.sql not found — see database/schema.sql)")

    print("-" * 70)
    print()
    print("5. Once the schema is applied, run the app:")
    print("   streamlit run app.py")
    print()


def main():
    print("=" * 60)
    print("RAG Assistant — Supabase Schema Setup")
    print("=" * 60)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    print(f"Project: {SUPABASE_URL}")
    print()

    print("Attempting Management API approach...")
    if apply_via_management_api():
        print("Schema applied successfully!")
        return

    print("Management API unavailable (anon key). Printing manual instructions...")
    print_manual_instructions()


if __name__ == "__main__":
    main()
