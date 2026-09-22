import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from utils.config import (
    GROQ_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY,
    GROQ_MODEL, PINECONE_INDEX_NAME, is_configured, is_db_configured
)

print("--- Credential presence check ---")
print(f"GROQ_API_KEY    : {'OK (' + GROQ_API_KEY[:8] + '...)' if GROQ_API_KEY else 'MISSING'}")
print(f"PINECONE_API_KEY: {'OK (' + PINECONE_API_KEY[:8] + '...)' if PINECONE_API_KEY else 'MISSING'}")
print(f"SUPABASE_URL    : {'OK (' + SUPABASE_URL[:30] + '...)' if SUPABASE_URL else 'MISSING'}")
print(f"SUPABASE_KEY    : {'OK (JWT present)' if SUPABASE_KEY else 'MISSING'}")
print(f"GROQ_MODEL      : {GROQ_MODEL}")
print(f"PINECONE_INDEX  : {PINECONE_INDEX_NAME}")
print()
print(f"is_configured() = {is_configured()}")
print(f"is_db_configured() = {is_db_configured()}")

if not is_configured() or not is_db_configured():
    print("\n[FAIL] One or more credentials are missing!")
    sys.exit(1)

print("\n--- Testing Supabase connectivity ---")
try:
    from database.database import init_db
    init_db()
    print("[PASS] Supabase connected successfully.")
except Exception as e:
    print(f"[FAIL] Supabase connection failed: {e}")
    sys.exit(1)

print("\n--- Testing Pinecone connectivity ---")
try:
    from services.pinecone_store import PineconeStore
    store = PineconeStore()
    store.connect()
    print(f"[PASS] Pinecone connected to index '{PINECONE_INDEX_NAME}'.")
except Exception as e:
    print(f"[FAIL] Pinecone connection failed: {e}")
    sys.exit(1)

print("\n--- Testing Groq LLM connectivity ---")
try:
    from services.llm import LLMService
    llm = LLMService()
    llm.connect()
    print("[PASS] Groq client connected successfully.")
except Exception as e:
    print(f"[FAIL] Groq connection failed: {e}")
    sys.exit(1)

print("\n=== All services connected successfully ===")
