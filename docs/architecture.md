# 🏗️ System Architecture & Data Flow

This document details the architectural design, component interactions, data pipelines, and persistence models of the **Smart PDF/RAG Assistant**.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    User([👤 User / Client]) -->|Interacts with| UI[🖥️ Streamlit Web Interface]

    %% ─────────────────────────────────────────────────────────────
    %% INGESTION FLOW
    %% ─────────────────────────────────────────────────────────────
    subgraph IngestionFlow ["📥 1. Ingestion & Document Processing Flow"]
        UI -->|Upload PDF / DOCX / TXT| FileVal[🔍 File Validation & MD5 Duplicate Check\ncomponents/upload.py]
        FileVal --> FormatRouter{File Format?}
        
        FormatRouter -->|PDF| ParserPDF[PyMuPDF fitz Parser\nservices/parser.py]
        FormatRouter -->|DOCX| ParserDOCX[python-docx Parser\nservices/docx_parser.py]
        FormatRouter -->|TXT| ParserTXT[Python Text Reader\nservices/parser.py]
        
        ParserPDF --> ScannedCheck{Native Text\n< 20 chars?}
        ScannedCheck -->|Yes: Scanned PDF| EasyOCR[EasyOCR 150 DPI Fallback\nservices/ocr.py]
        ScannedCheck -->|No: Digital PDF| TableExtract[pdfplumber Table Extractor\nservices/parser.py]
        EasyOCR --> TableExtract
        
        TableExtract --> DocSummary[Groq LLM: Auto-Summary & Key Topics JSON\nservices/llm.py]
        ParserDOCX --> DocSummary
        ParserTXT --> DocSummary
        
        DocSummary --> Chunking[Recursive Character Splitter\nchunk_size=500, overlap=100\nPreserves Tables Intact\nservices/chunker.py]
        
        Chunking --> EmbeddingEngine[SentenceTransformer\nall-MiniLM-L6-v2 384-dim\nservices/embeddings.py]
    end

    %% ─────────────────────────────────────────────────────────────
    %% PERSISTENCE LAYER
    %% ─────────────────────────────────────────────────────────────
    subgraph PersistenceLayer ["💾 2. Persistence Layer"]
        DocSummary -->|Save document metadata| DB_Docs[(Supabase: documents table)]
        Chunking -->|Save text & table chunks| DB_Chunks[(Supabase: chunks table)]
        EmbeddingEngine -->|Upsert 384-dim vectors + metadata| PineconeDB[(Pinecone Serverless: Vector Index)]
        DB_Chunks -.->|Load corpus chunks| BM25Index[In-Memory BM25Okapi Index\nservices/bm25_retriever.py]
        ChatHistoryStore[(Supabase: chat_history table)]
    end

    %% ─────────────────────────────────────────────────────────────
    %% QUERY & RETRIEVAL FLOW
    %% ─────────────────────────────────────────────────────────────
    subgraph RetrievalFlow ["🔍 3. Query & Hybrid Retrieval Flow"]
        UI -->|Question / Comparison Query| ScopeFilter{Document Scope Filter\ncomponents/chat.py}
        ScopeFilter -->|All or Filtered Docs| QueryExpCheck{Query Expansion\nEnabled?}
        
        QueryExpCheck -->|Yes| Expander[Groq LLM: Generate 2 Query Variants\nservices/query_expander.py]
        QueryExpCheck -->|No| SingleQuery[Original Query]
        Expander --> SearchEngine
        SingleQuery --> SearchEngine

        subgraph SearchEngine ["Parallel Hybrid Search Execution"]
            DenseSearch[Dense Search: Embed Query \u2192 Pinecone Cosine Search\nservices/pinecone_store.py]
            SparseSearch[Sparse Search: BM25 Token Search\nservices/bm25_retriever.py]
        end

        SearchEngine --> RRFFusion[Reciprocal Rank Fusion RRF\nRRF score = sum 1 / rank + 60\nservices/hybrid_retriever.py]
        
        RRFFusion --> RerankCheck{Re-ranking\nEnabled?}
        RerankCheck -->|Yes| CrossEncoder[Cross-Encoder ms-marco-MiniLM\nScore Query-Chunk Pairs\nservices/reranker.py]
        RerankCheck -->|No| TopRRF[Top Candidates by RRF Score]
        
        CrossEncoder --> DedupCheck{Chunk Deduplication\nEnabled?}
        TopRRF --> DedupCheck
        
        DedupCheck -->|Yes| TrigramDedup[Jaccard Trigram Deduplication\nDrop >= 85% Overlap\nservices/deduplicator.py]
        DedupCheck -->|No| FinalCandidates[Final Top Chunks]
        TrigramDedup --> FinalCandidates
    end

    %% ─────────────────────────────────────────────────────────────
    %% GENERATION & EVIDENCE GATE
    %% ─────────────────────────────────────────────────────────────
    subgraph GenerationFlow ["🛡️ 4. Generation & Evidence Gate Flow"]
        FinalCandidates --> EvidenceGate{Evidence Gate\nCheck Similarity / Logit Thresholds\nservices/evidence_gate.py}
        
        EvidenceGate -->|Score < Threshold OR Empty Chunks| ControlledRefusal["🔴 Evidence: Insufficient\nControlled Refusal: No Hallucination"]
        
        EvidenceGate -->|Score >= Threshold| ContextAssembly["Context Block Assembly\n[Context Block N | Doc, Page]\nservices/llm.py"]
        
        ContextAssembly --> MemoryInject[Inject Last 6 Conversation Turns\nservices/llm.py]
        
        MemoryInject --> GroqLLM[Groq LLM Inference\nLlama 3.3 70B Versatile\nservices/llm.py]
        
        GroqLLM --> ModeCheck{Comparison Mode?}
        ModeCheck -->|Yes| MarkdownTable[Format Structured Markdown Comparison Table]
        ModeCheck -->|No| AnswerGen[Format Grounded Conversational Answer]
        
        MarkdownTable --> CitationAppender[Append Evidence Badge & Sources\nservices/llm.py]
        AnswerGen --> CitationAppender
    end

    %% ─────────────────────────────────────────────────────────────
    %% OUTPUT & ATTRIBUTION
    %% ─────────────────────────────────────────────────────────────
    ControlledRefusal --> UI
    CitationAppender -->|Persist chat turn| ChatHistoryStore
    CitationAppender --> UI
    
    UI -->|Click Cited Page in Source Preview| HighlightViewer["Inspector: Source Preview\nReconstruct Page + Yellow Highlighting\ncomponents/sidebar.py"]
    DB_Chunks -.->|Fetch full page chunks| HighlightViewer
```

---

## 2. Architectural Subsystems

### A. Ingestion & Document Processing Pipeline
1. **File Validation**: Enforces extension allowlist (`.pdf`, `.docx`, `.txt`) and 50MB file size limit.
2. **MD5 Deduplication**: Hashes file content to prevent re-indexing identical files.
3. **Adaptive Text Extraction**:
   - **PyMuPDF (`fitz`)**: Fast native text layer extraction.
   - **EasyOCR Fallback**: Automatically renders scanned pages at 150 DPI when native text is under 20 characters.
   - **`pdfplumber` Table Extraction**: Isolates tabular grids and outputs clean Markdown tables.
   - **`python-docx`**: Extracts paragraphs and row-major tables from Word documents.
4. **Summary & Topic Extraction**: Groq LLM creates 2–3 sentence summaries and JSON key topics.
5. **Recursive Chunking**: `chunk_size=500` with `100` character overlap; table blocks preserved intact.
6. **Dense Embedding**: `sentence-transformers/all-MiniLM-L6-v2` encodes 384-dimensional vectors.

### B. Persistence Layer
- **Pinecone (Serverless)**: Cosine similarity vector index storing embeddings with metadata (`document_name`, `document_id`, `page_number`, `chunk_type`, `chunk_text`).
- **Supabase (PostgreSQL)**: Relational cloud database with 3 tables:
  - `documents`: Document metadata, MD5 hash, summaries, page counts, OCR tracking.
  - `chunks`: Chunk text, page numbers, chunk type, foreign key with cascade deletion.
  - `chat_history`: Conversation questions, answers, timestamps.

### C. Hybrid Retrieval & Re-Ranking Engine
- **Dense Vector Search**: Pinecone metadata-filtered cosine search.
- **Sparse BM25 Search**: `rank-bm25` (BM25Okapi) over Supabase chunk corpus with tokenization and punctuation stripping.
- **Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum \frac{1}{k + r(d)}$ with $k=60$.
- **Cross-Encoder Re-Ranking**: `ms-marco-MiniLM-L-6-v2` joint cross-attention scoring on top-20 candidates.
- **Trigram Deduplication**: Jaccard similarity filter dropping near-duplicate chunks ($\ge 85\%$ overlap).

### D. Evidence Gate & Generation
- **Evidence Gate**: Heuristic 3-tier gate (🟢 Strong, 🟡 Moderate, 🔴 Insufficient).
- **Refusal Flow**: When evidence is insufficient, bypasses generation with a polite refusal.
- **Sliding Memory**: Injects last 6 conversation turns into Groq prompt.
- **Multi-Document Comparison**: Generates cross-tabulated Markdown comparison tables when comparing 2+ documents.
- **Source Preview**: Inspector panel reconstructs full page text and highlights retrieved passages with `<mark>` tags.

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
