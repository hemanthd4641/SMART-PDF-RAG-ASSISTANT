-- ============================================================
-- RAG Assistant -- Supabase PostgreSQL Schema
-- Run this SQL once in your Supabase Dashboard -> SQL Editor
-- ============================================================

-- documents table
-- id uses TEXT (MD5 hex string) to match the existing upload.py
-- duplicate-guard logic. Do NOT change to UUID.
CREATE TABLE IF NOT EXISTS documents (
    id                TEXT        PRIMARY KEY,
    document_name     TEXT        NOT NULL,
    upload_timestamp  TIMESTAMPTZ DEFAULT NOW(),
    page_count        INTEGER     NOT NULL,
    summary           TEXT,
    key_topics        TEXT,
    native_page_count INTEGER     DEFAULT 0,
    ocr_page_count    INTEGER     DEFAULT 0
);

-- chunks table
-- ON DELETE CASCADE automatically removes chunks when the parent
-- document is deleted.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT    PRIMARY KEY,
    document_id TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    chunk_type  TEXT    DEFAULT 'text'
);

-- Index for fast lookup of chunks by document + page
CREATE INDEX IF NOT EXISTS idx_chunks_document_page
    ON chunks (document_id, page_number);

-- chat_history table
CREATE TABLE IF NOT EXISTS chat_history (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_question    TEXT        NOT NULL,
    assistant_answer TEXT        NOT NULL,
    timestamp        TIMESTAMPTZ DEFAULT NOW()
);

-- Index for chronological retrieval
CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp
    ON chat_history (timestamp ASC);

-- Row Level Security
-- Disable RLS on all tables so the anon key can perform full
-- CRUD operations. This app uses the anon key server-side only
-- (Streamlit backend), so this is safe for this architecture.
ALTER TABLE documents    DISABLE ROW LEVEL SECURITY;
ALTER TABLE chunks       DISABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history DISABLE ROW LEVEL SECURITY;
