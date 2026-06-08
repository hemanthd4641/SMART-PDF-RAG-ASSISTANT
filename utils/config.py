import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# App Settings
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
PORT = int(os.getenv("PORT", 8501))

# Groq Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Pinecone Settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp-free")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-assistant-index")

# Supabase Settings (replaces SQLite)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Embeddings Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

def is_configured() -> bool:
    """Helper check to verify if Groq and Pinecone API keys are configured."""
    return bool(GROQ_API_KEY and PINECONE_API_KEY)

def is_db_configured() -> bool:
    """Helper check to verify if Supabase credentials are configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)
