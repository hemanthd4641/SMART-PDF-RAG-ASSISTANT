# 🏗️ System Architecture & Data Flow

This document details the architectural design, component interactions, data pipelines, and persistence models of the **Smart PDF/RAG Assistant**.

---

## 1. End-to-End System Pipeline (Ordered Flow)

```
Document Upload (PDF / DOCX / TXT)
  ↓
File Validation / MD5 Duplicate Detection
  ↓
Adaptive Extraction (PyMuPDF / EasyOCR / pdfplumber / python-docx)
  ↓
Document Summary & Key Topic Extraction (Groq LLM)
  ↓
Recursive Chunking (500 chars, 100 overlap, tables preserved)
  ↓
Dual Indexing: Dense Vectors (Pinecone) + Sparse Tokens (BM25) + Relational (Supabase)
  ↓
User Query / Comparison Request (Optional Groq Query Expansion)
  ↓
Hybrid Retrieval (Parallel Dense Cosine + Sparse BM25)
  ↓
Reciprocal Rank Fusion (RRF k=60)
  ↓
Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2)
  ↓
Trigram Deduplication (Drop >= 85% overlap)
  ↓
Evidence / Answerability Gate
  ├── 🟢 Strong / 🟡 Moderate → Groq LLM (Llama 3.3 70B) → Answer + Citations
  └── 🔴 Insufficient Evidence → Controlled Refusal (No Hallucination)
  ↓
Streamlit UI Display + Source Inspector (Yellow Highlight on Full Page)
```

```mermaid
flowchart TD
    %% ─────────────────────────────────────────────────────────────
    %% STAGE 1: INGESTION & EXTRACTION
    %% ─────────────────────────────────────────────────────────────
    subgraph STAGE1 ["1. Document Ingestion & Extraction"]
        direction TB
        A1["📄 User Uploads PDF / DOCX / TXT"] --> A2["🔍 Validation & MD5 Duplicate Check"]
        A2 --> A3["⚙️ Text & Table Extraction\n• PyMuPDF (Native text)\n• EasyOCR (Scanned fallback)\n• pdfplumber (Tables to Markdown)\n• python-docx (Word documents)"]
        A3 --> A4["🧠 Groq LLM: Auto-Summary & Key Topics JSON"]
    end

    %% ─────────────────────────────────────────────────────────────
    %% STAGE 2: CHUNKING & DUAL STORAGE
    %% ─────────────────────────────────────────────────────────────
    subgraph STAGE2 ["2. Chunking & Dual Storage Indexing"]
        direction TB
        B1["✂️ Recursive Text Chunker\n(500 chars, 100 overlap, tables kept intact)"]
        B1 --> B2["🔢 Dense Embeddings\n(all-MiniLM-L6-v2, 384-dim)"]
        B1 --> B3[("🗄️ Supabase PostgreSQL\n(documents & chunks tables)")]
        B2 --> B4[("🌲 Pinecone Vector Index\n(serverless cosine search)")]
        B3 --> B5["📚 In-Memory BM25Okapi Index\n(tokenized chunk corpus)"]
    end

    %% ─────────────────────────────────────────────────────────────
    %% STAGE 3: QUERY & HYBRID RETRIEVAL
    %% ─────────────────────────────────────────────────────────────
    subgraph STAGE3 ["3. Query & Hybrid Retrieval Pipeline"]
        direction TB
        C1["👤 User Query / Multi-Doc Comparison"] --> C2["🔄 Query Expansion\n(Optional Groq 2-variant generator)"]
        C2 --> C3["🌲 Dense Vector Search (Pinecone)"]
        C2 --> C4["📚 Sparse Keyword Search (BM25)"]
        C3 --> C5["🔀 Reciprocal Rank Fusion (RRF k=60)"]
        C4 --> C5
        C5 --> C6["🎯 Cross-Encoder Re-Ranking\n(ms-marco-MiniLM on top-20)"]
        C6 --> C7["🧹 Trigram Jaccard Deduplication\n(Drop >= 85% overlap)"]
    end

    %% ─────────────────────────────────────────────────────────────
    %% STAGE 4: EVIDENCE GATE & GENERATION
    %% ─────────────────────────────────────────────────────────────
    subgraph STAGE4 ["4. Evidence Gating & LLM Generation"]
        direction TB
        D1{"🛡️ Evidence / Answerability Gate\n(Relevance & Coverage Score)"}
        D1 -->|"🟢 Strong / 🟡 Moderate"| D2["🤖 Groq LLM: Llama 3.3 70B\n(Context blocks + 6-turn sliding memory)"]
        D1 -->|"🔴 Insufficient Evidence"| D3["🛑 Controlled Refusal\n(Safely decline without hallucinating)"]
        D2 --> D4["📋 Format Grounded Response\n(Answer / Table + [Doc, Page] Citations)"]
    end

    %% ─────────────────────────────────────────────────────────────
    %% STAGE 5: UI DISPLAY & ATTRIBUTION
    %% ─────────────────────────────────────────────────────────────
    subgraph STAGE5 ["5. UI Display & Source Attribution"]
        direction TB
        E1["🖥️ Streamlit Interactive UI"]
        E1 --> E2["🔍 Source Inspector Panel\n(Full-page text + Yellow passage highlight)"]
        E1 --> E3[("🗄️ Supabase chat_history")]
    end

    %% ─────────────────────────────────────────────────────────────
    %% ORDERED PIPELINE FLOW
    %% ─────────────────────────────────────────────────────────────
    A4 --> B1
    STAGE2 --> C1
    C7 --> D1
    D4 --> E1
    D3 --> E1
```

---

## 2. Architectural Subsystems

### Stage 1: Ingestion & Document Processing Pipeline
1. **File Validation**: Enforces extension allowlist (`.pdf`, `.docx`, `.txt`) and 50MB file size limit.
2. **MD5 Deduplication**: Hashes file content to prevent re-indexing identical files.
3. **Adaptive Text Extraction**:
   - **PyMuPDF (`fitz`)**: Fast native text layer extraction.
   - **EasyOCR Fallback**: Automatically renders scanned pages at 150 DPI when native text is under 20 characters.
   - **`pdfplumber` Table Extraction**: Isolates tabular grids and outputs clean Markdown tables.
   - **`python-docx`**: Extracts paragraphs and row-major tables from Word documents.
4. **Summary & Topic Extraction**: Groq LLM creates 2–3 sentence summaries and JSON key topics.

### Stage 2: Chunking, Indexing & Storage
- **Recursive Chunking**: `chunk_size=500` with `100` character overlap; table blocks preserved intact.
- **Dense Embedding**: `sentence-transformers/all-MiniLM-L6-v2` encodes 384-dimensional vectors.
- **Pinecone (Serverless)**: Cosine similarity vector index storing embeddings with metadata (`document_name`, `document_id`, `page_number`, `chunk_type`, `chunk_text`).
- **Supabase (PostgreSQL)**: Relational cloud database storing `documents`, `chunks`, and `chat_history`.
- **BM25 Index**: In-memory `BM25Okapi` index built from the Supabase chunks corpus.

### Stage 3: Hybrid Retrieval & Fusion
- **Dense Vector Search**: Pinecone metadata-filtered cosine search.
- **Sparse BM25 Search**: `rank-bm25` (BM25Okapi) over chunk corpus with tokenization and punctuation stripping.
- **Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum \frac{1}{k + r(d)}$ with $k=60$.
- **Cross-Encoder Re-Ranking**: `ms-marco-MiniLM-L-6-v2` joint cross-attention scoring on top-20 candidates.
- **Trigram Deduplication**: Jaccard similarity filter dropping near-duplicate chunks ($\ge 85\%$ overlap).

### Stage 4: Evidence Gating & Response Generation
- **Evidence Gate**: Heuristic 3-tier gate (🟢 Strong, 🟡 Moderate, 🔴 Insufficient).
- **Refusal Flow**: When evidence is insufficient, bypasses generation with a polite refusal.
- **Sliding Memory**: Injects last 6 conversation turns into Groq prompt.
- **Multi-Document Comparison**: Generates cross-tabulated Markdown comparison tables when comparing 2+ documents.

### Stage 5: Output, Attribution & History
- **Source Preview**: Inspector panel reconstructs full page text and highlights retrieved passages with `<mark>` tags.
- **Cloud History**: Chat exchanges persisted to Supabase `chat_history` table.

---

## 3. Component to Source File Mapping

| Diagram Component | Source File | Key Class / Function |
|---|---|---|
| **Streamlit Web UI** | [`app.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/app.py) | `main()` layout and session state orchestration |
| **Upload & Ingestion** | [`components/upload.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/components/upload.py) | `render_upload_section()`, `process_uploaded_file()` |
| **PDF/TXT Parsing & OCR** | [`services/parser.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/parser.py) | `parse_document()`, `parse_pdf()`, `table_to_markdown()` |
| **DOCX Parsing** | [`services/docx_parser.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/docx_parser.py) | `extract_docx_text()` |
| **OCR Engine** | [`services/ocr.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/ocr.py) | `extract_text_from_image()`, `EasyOCRService` |
| **Chunking** | [`services/chunker.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/chunker.py) | `DocumentChunker`, `RecursiveTextSplitter` |
| **Embeddings** | [`services/embeddings.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/embeddings.py) | `EmbeddingService.embed_chunks()`, `embed_query()` |
| **Pinecone Vector Store** | [`services/pinecone_store.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/pinecone_store.py) | `PineconeStore.upsert_chunks()`, `query_similar_chunks()` |
| **BM25 Keyword Search** | [`services/bm25_retriever.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/bm25_retriever.py) | `BM25Retriever.retrieve_bm25()`, `build_index()` |
| **RRF Fusion** | [`services/hybrid_retriever.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/hybrid_retriever.py) | `reciprocal_rank_fusion()` |
| **Cross-Encoder Re-Ranking** | [`services/reranker.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/reranker.py) | `RerankerService.rerank()` |
| **Query Expansion** | [`services/query_expander.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/query_expander.py) | `QueryExpander.expand_query()` |
| **Chunk Deduplication** | [`services/deduplicator.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/deduplicator.py) | `deduplicate_chunks()`, `jaccard_similarity()` |
| **Evidence Gate** | [`services/evidence_gate.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/evidence_gate.py) | `EvidenceGate.evaluate()` |
| **LLM Generation & Citations** | [`services/llm.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/services/llm.py) | `LLMService.generate_response()`, `generate_citations_block()` |
| **Database Persistence** | [`database/database.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/database/database.py) | `save_document()`, `save_chunks()`, `save_chat_turn()` |
| **Inspector & Source Preview** | [`components/sidebar.py`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/components/sidebar.py) | `render_inspector_panel()`, `render_source_preview_tab()` |
