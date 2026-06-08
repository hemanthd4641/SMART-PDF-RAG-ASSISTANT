"""
Supabase Client Singleton
--------------------------
Provides a single, reusable Supabase client instance for the entire app.
All database operations import get_client() from this module.

Usage:
    from database.supabase_client import get_client, SupabaseConnectionError
    client = get_client()
    response = client.table("documents").select("*").execute()
"""

from supabase import create_client, Client
from utils.config import SUPABASE_URL, SUPABASE_KEY
from utils.helpers import get_logger

logger = get_logger("supabase_client")

# ── Module-level singleton ────────────────────────────────────────────────────
_client: Client = None


class SupabaseConnectionError(Exception):
    """Raised when the Supabase client cannot be initialized or connected."""
    pass


def get_client() -> Client:
    """
    Returns the shared Supabase client instance, initializing it on first call.

    The client is cached at module level so connection overhead is paid once
    per process lifetime (compatible with Streamlit's @st.cache_resource pattern).

    Raises:
        SupabaseConnectionError: If SUPABASE_URL or SUPABASE_KEY are missing,
                                 or if the Supabase SDK raises during initialization.
    """
    global _client

    # Return cached client if already initialized
    if _client is not None:
        return _client

    # Validate credentials before attempting connection
    if not SUPABASE_URL or not SUPABASE_KEY:
        msg = (
            "Supabase credentials are missing. "
            "Set SUPABASE_URL and SUPABASE_KEY in your .env file. "
            "See .env.example for the required format."
        )
        logger.error(msg)
        raise SupabaseConnectionError(msg)

    try:
        logger.info(f"Initializing Supabase client → {SUPABASE_URL}")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
        return _client

    except SupabaseConnectionError:
        raise  # Re-raise our own error class unchanged

    except Exception as e:
        msg = f"Failed to initialize Supabase client: {e}"
        logger.error(msg)
        raise SupabaseConnectionError(msg) from e


def reset_client() -> None:
    """
    Resets the cached client singleton.
    Primarily useful in test suites that need to re-initialize with different credentials.
    """
    global _client
    _client = None
    logger.info("Supabase client singleton reset.")
