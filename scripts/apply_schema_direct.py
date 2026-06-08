"""
scripts/apply_schema_direct.py
--------------------------------
Applies the Supabase PostgreSQL schema using a direct psycopg2 connection.
Supabase provides a direct PostgreSQL connection on port 5432 (or 6543 for pooler).

Usage:
    python scripts/apply_schema_direct.py

The DB password is your Supabase database password (set when you created the project,
or found under: Supabase Dashboard → Project Settings → Database → Password).

Set DB_PASSWORD in your .env file:
    DB_PASSWORD=your_supabase_db_password_here
"""

import sys
import os
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ── Connection config ─────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
PROJECT_REF = SUPABASE_URL.replace("https://", "").split(".")[0] if SUPABASE_URL else ""

DB_HOST     = f"aws-0-ap-south-1.pooler.supabase.com"   # Supabase pooler (transaction mode)
DB_PORT     = 6543
DB_NAME     = "postgres"
DB_USER     = f"postgres.{PROJECT_REF}"
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database", "schema.sql"
)


def main():
    print("=" * 60)
    print("RAG Assistant — Applying Supabase Schema (Direct Connection)")
    print("=" * 60)
    print(f"Host    : {DB_HOST}")
    print(f"Port    : {DB_PORT}")
    print(f"User    : {DB_USER}")
    print(f"DB      : {DB_NAME}")
    print()

    if not DB_PASSWORD:
        print("ERROR: DB_PASSWORD is not set in .env")
        print("Add: DB_PASSWORD=your_supabase_database_password")
        print()
        print("Find your password at:")
        print("  Supabase Dashboard → Project Settings → Database → Database Password")
        sys.exit(1)

    if not os.path.exists(SCHEMA_PATH):
        print(f"ERROR: Schema file not found at {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode="require",
            connect_timeout=10,
        )
        conn.autocommit = True
        cur = conn.cursor()

        print("Connected to Supabase PostgreSQL.")
        print("Applying schema...")
        cur.execute(schema_sql)
        print()
        print("✓ Schema applied successfully!")
        print()
        print("Tables created:")
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        for row in cur.fetchall():
            print(f"  - {row[0]}")

        cur.close()
        conn.close()

    except psycopg2.OperationalError as e:
        print(f"✗ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check DB_PASSWORD in .env")
        print("  2. Verify your Supabase project is active")
        print("  3. Check if your IP is allowed in Supabase → Settings → Database → Connection Pooling")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Schema application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
