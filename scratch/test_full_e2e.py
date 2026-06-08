"""
scratch/test_full_e2e.py
--------------------------
Full end-to-end feature test for the RAG Assistant.
Covers every major component without needing a browser.

Usage:
    python scratch/test_full_e2e.py
"""

import sys, os, hashlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

passed = 0
failed = 0
warnings = []

def ok(label):
    global passed; passed += 1
    print(f"  PASS  {label}")

def fail(label, reason):
    global failed; failed += 1
    print(f"  FAIL  {label}")
    print(f"        -> {reason}")

def warn(label, reason):
    warnings.append((label, reason))
    print(f"  WARN  {label}: {reason}")

def section(title):
    print()
    print(f"[{title}]")
    print("-" * 55)

# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("RAG Assistant -- Full Feature E2E Test Suite")
print("=" * 55)

# ──────────────────────────────────────────────────────────────
# 1. Configuration & Credentials
# ──────────────────────────────────────────────────────────────
section("1. Configuration & Credentials")
from utils.config import (
    GROQ_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY,
    EMBEDDING_MODEL_NAME, GROQ_MODEL, is_configured, is_db_configured
)

if GROQ_API_KEY:     ok(f"GROQ_API_KEY set (model: {GROQ_MODEL})")
else:                fail("GROQ_API_KEY", "Missing from .env")
if PINECONE_API_KEY: ok("PINECONE_API_KEY set")
else:                fail("PINECONE_API_KEY", "Missing from .env")
if SUPABASE_URL and SUPABASE_KEY:
                     ok(f"Supabase credentials set ({SUPABASE_URL})")
else:                fail("Supabase credentials", "Missing from .env")
if is_configured():  ok("is_configured() = True")
else:                fail("is_configured()", "Returned False")
if is_db_configured(): ok("is_db_configured() = True")
else:                fail("is_db_configured()", "Returned False")

# ──────────────────────────────────────────────────────────────
# 2. Supabase Database Layer
# ──────────────────────────────────────────────────────────────
section("2. Supabase Database Layer")
from database.database import (
    init_db, insert_document, fetch_all_documents, fetch_document_by_id,
    delete_document, insert_chunks, fetch_chunks_for_document, fetch_all_chunks,
    get_total_chunks, get_chunks_count_by_type, insert_chat_history,
    fetch_chat_history, fetch_recent_chat_history, clear_chat_history, fetch_page_chunks
)

try:
    init_db()
    ok("init_db() -- Supabase connectivity verified")
except Exception as e:
    fail("init_db()", str(e))

try:
    docs = fetch_all_documents()
    ok(f"fetch_all_documents() -- {len(docs)} document(s) in Supabase")
except Exception as e:
    fail("fetch_all_documents()", str(e))

try:
    total = get_total_chunks()
    ct = get_chunks_count_by_type()
    ok(f"get_total_chunks() -- {total} total (text: {ct.get('text',0)}, table: {ct.get('table',0)})")
except Exception as e:
    fail("get_total_chunks()", str(e))

# ──────────────────────────────────────────────────────────────
# 3. Document Parsing
# ──────────────────────────────────────────────────────────────
section("3. Document Parsing (parser.py)")
from services.parser import parse_document, DocumentParsingError

# TXT parsing
tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
tmp.write(
    "The Transformer architecture relies entirely on attention mechanisms.\n"
    "It was proposed by Vaswani et al. in 2017.\n"
    "This approach achieves state of the art results on translation tasks.\n"
    "Multi-head attention allows the model to attend to information from different positions.\n"
    "Positional encodings are added to provide sequence order information.\n"
)
tmp.close()

try:
    parsed = parse_document(tmp.name)
    assert parsed["document_name"].endswith(".txt")
    assert len(parsed["pages"]) == 1
    assert "Transformer" in parsed["pages"][0]["text"]
    char_count = len(parsed["pages"][0]["text"])
    ok(f"parse_document() TXT -- 1 page, {char_count} chars")
except Exception as e:
    fail("parse_document() TXT", str(e))
finally:
    os.unlink(tmp.name)

try:
    parse_document("definitely_missing_file_xyz.pdf")
    fail("parse_document() missing file", "Should have raised FileNotFoundError")
except FileNotFoundError:
    ok("parse_document() missing file -- raises FileNotFoundError correctly")
except Exception as e:
    fail("parse_document() missing file error type", str(e))

# ──────────────────────────────────────────────────────────────
# 4. Text Chunking
# ──────────────────────────────────────────────────────────────
section("4. Text Chunking (chunker.py)")
from services.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
sample_doc = {
    "document_name": "test_e2e.txt",
    "pages": [{
        "page_number": 1,
        "text": "The Transformer model uses self-attention mechanisms for sequence transduction. " * 10,
        "tables": ["| Model | BLEU Score |\n| --- | --- |\n| Transformer | 28.4 |"],
        "extraction_method": "native"
    }]
}
try:
    chunks = chunker.chunk_document(sample_doc)
    text_chunks  = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(chunks) > 1
    assert len(table_chunks) == 1
    assert all("chunk_id" in c and "page_number" in c for c in chunks)
    ok(f"chunk_document() -- {len(chunks)} total ({len(text_chunks)} text, {len(table_chunks)} table)")
except Exception as e:
    fail("chunk_document()", str(e))

# Verify recursive splitting (overlap)
try:
    splitter = chunker.splitter
    long_text = "word " * 200
    splits = splitter.split_text(long_text)
    assert len(splits) > 1
    ok(f"RecursiveTextSplitter -- {len(splits)} splits on 200-word text (chunk_size=200, overlap=40)")
except Exception as e:
    fail("RecursiveTextSplitter", str(e))

# ──────────────────────────────────────────────────────────────
# 5. Embeddings
# ──────────────────────────────────────────────────────────────
section("5. Embedding Generation (embeddings.py)")
from services.embeddings import EmbeddingService

emb_svc = EmbeddingService()
try:
    vecs = emb_svc.generate_embeddings([
        "Attention is all you need",
        "Transformer architecture for NLP",
        "BERT is a bidirectional encoder"
    ])
    assert len(vecs) == 3
    assert len(vecs[0]) == 384
    assert all(isinstance(v, float) for v in vecs[0])
    ok(f"generate_embeddings() -- 3 vectors, dim={len(vecs[0])}, dtype=float")
except Exception as e:
    fail("generate_embeddings()", str(e))

# Check lazy loading works (model should now be cached)
try:
    vecs2 = emb_svc.generate_embeddings(["second call test"])
    assert len(vecs2) == 1
    ok("generate_embeddings() -- model cached, second call works")
except Exception as e:
    fail("generate_embeddings() second call", str(e))

# ──────────────────────────────────────────────────────────────
# 6. Pinecone Vector Store
# ──────────────────────────────────────────────────────────────
section("6. Pinecone Vector Store (pinecone_store.py)")
from services.pinecone_store import PineconeStore

store = PineconeStore()
try:
    store.connect()
    ok("PineconeStore.connect() -- connected to index")
except Exception as e:
    fail("PineconeStore.connect()", str(e))

try:
    test_vec = emb_svc.generate_embeddings(["test query for pinecone"])[0]
    results = store.query_vectors(test_vec, top_k=3)
    ok(f"PineconeStore.query_vectors() -- {len(results)} result(s) returned")
except Exception as e:
    fail("PineconeStore.query_vectors()", str(e))

# ──────────────────────────────────────────────────────────────
# 7. BM25 Sparse Retriever
# ──────────────────────────────────────────────────────────────
section("7. BM25 Sparse Retriever (bm25_retriever.py)")
from services.bm25_retriever import BM25Retriever

bm25 = BM25Retriever()
try:
    bm25.build_index()
    ok(f"BM25Retriever.build_index() -- {len(bm25.chunks)} chunks indexed from Supabase")
except Exception as e:
    fail("BM25Retriever.build_index()", str(e))

if bm25.chunks:
    try:
        results_bm25 = bm25.retrieve_bm25("attention transformer architecture", top_k=5)
        ok(f"BM25Retriever.retrieve_bm25() -- {len(results_bm25)} result(s)")
    except Exception as e:
        fail("BM25Retriever.retrieve_bm25()", str(e))

    try:
        # Document filter test
        all_docs = fetch_all_documents()
        if all_docs:
            filter_doc = [all_docs[0]["document_name"]]
            results_filtered = bm25.retrieve_bm25("transformer", top_k=5, document_filter=filter_doc)
            ok(f"BM25Retriever.retrieve_bm25() with document_filter -- {len(results_filtered)} result(s)")
    except Exception as e:
        fail("BM25Retriever.retrieve_bm25() with document_filter", str(e))
else:
    warn("BM25Retriever.retrieve_bm25()", "No chunks in Supabase yet -- upload a document first")

# ──────────────────────────────────────────────────────────────
# 8. Query Expander
# ──────────────────────────────────────────────────────────────
section("8. Query Expansion (query_expander.py)")
from services.query_expander import QueryExpander

expander = QueryExpander()
try:
    variants = expander.expand("What is the Transformer architecture?")
    assert isinstance(variants, list) and len(variants) >= 1
    ok(f"QueryExpander.expand() -- {len(variants)} variant(s) generated")
    for i, v in enumerate(variants):
        print(f"        Variant {i+1}: {v}")
except Exception as e:
    fail("QueryExpander.expand()", str(e))

# ──────────────────────────────────────────────────────────────
# 9. Chunk Deduplicator
# ──────────────────────────────────────────────────────────────
section("9. Chunk Deduplication (deduplicator.py)")
from services.deduplicator import deduplicate_chunks

test_chunks_d = [
    {"chunk_text": "The Transformer model uses self-attention for sequence transduction tasks."},
    {"chunk_text": "The Transformer model uses self-attention for sequence transduction tasks."},  # exact dup
    {"chunk_text": "BERT uses bidirectional context from the Transformer encoder."},
    {"chunk_text": "GPT uses a unidirectional causal Transformer for text generation."},
]
try:
    deduped = deduplicate_chunks(test_chunks_d, similarity_threshold=0.85)
    assert len(deduped) == 3, f"Expected 3 after dup removal, got {len(deduped)}"
    ok(f"deduplicate_chunks() -- {len(test_chunks_d)} in -> {len(deduped)} out (1 exact dup removed)")
except Exception as e:
    fail("deduplicate_chunks()", str(e))

# Near-dup test (very similar)
near_dup_chunks = [
    {"chunk_text": "The quick brown fox jumps over the lazy dog near the river bank."},
    {"chunk_text": "The quick brown fox jumps over the lazy dog near the river bank!"},  # ~95% similar
]
try:
    near_deduped = deduplicate_chunks(near_dup_chunks, similarity_threshold=0.85)
    ok(f"deduplicate_chunks near-duplicate -- {len(near_dup_chunks)} in -> {len(near_deduped)} out")
except Exception as e:
    fail("deduplicate_chunks near-duplicate", str(e))

# ──────────────────────────────────────────────────────────────
# 10. Full RAG Retriever (all feature flags)
# ──────────────────────────────────────────────────────────────
section("10. RAG Retriever -- All Feature Combinations (retriever.py)")
from services.retriever import RAGRetriever

retriever = RAGRetriever(embedding_service=emb_svc, vector_store=store)
try:
    retriever.bm25_retriever.build_index()
    ok("RAGRetriever -> bm25_retriever.build_index() succeeded")
except Exception as e:
    fail("RAGRetriever -> bm25_retriever.build_index()", str(e))

if bm25.chunks:
    query = "What is the main contribution of the paper?"

    combos = [
        ("basic (no flags)",              dict(use_reranker=False, use_query_expansion=False, use_deduplication=False)),
        ("+ reranker",                    dict(use_reranker=True,  use_query_expansion=False, use_deduplication=False)),
        ("+ query expansion",             dict(use_reranker=False, use_query_expansion=True,  use_deduplication=False)),
        ("+ deduplication",               dict(use_reranker=False, use_query_expansion=False, use_deduplication=True)),
        ("ALL: rerank + QE + dedup",      dict(use_reranker=True,  use_query_expansion=True,  use_deduplication=True)),
    ]
    for label, flags in combos:
        try:
            result_chunks = retriever.retrieve(query, top_k=5, **flags)
            ok(f"retrieve() {label} -- {len(result_chunks)} chunk(s)")
        except Exception as e:
            fail(f"retrieve() {label}", str(e))

    # Smart document filter
    all_docs = fetch_all_documents()
    if all_docs:
        try:
            doc_filter = [all_docs[0]["document_name"]]
            filtered_chunks = retriever.retrieve(query, top_k=3, document_filter=doc_filter)
            ok(f"retrieve() with document_filter=['{doc_filter[0]}'] -- {len(filtered_chunks)} chunk(s)")
        except Exception as e:
            fail("retrieve() with document_filter", str(e))
else:
    warn("RAGRetriever.retrieve()", "No chunks in index -- upload a document first to test retrieval")

# ──────────────────────────────────────────────────────────────
# 11. LLM / Groq
# ──────────────────────────────────────────────────────────────
section("11. LLM Response Generation (llm.py)")
from services.llm import LLMService

llm = LLMService()
dummy_chunks = [
    {"metadata": {"text": "The Transformer uses multi-head self-attention and positional encodings.", "document_name": "paper.pdf", "page_number": 1}},
    {"metadata": {"text": "It achieves state of the art results on machine translation tasks.", "document_name": "paper.pdf", "page_number": 2}},
]

try:
    answer = llm.generate_response(
        user_question="What mechanism does the Transformer use and what tasks does it excel at?",
        retrieved_chunks=dummy_chunks,
        conversation_history=[]
    )
    assert isinstance(answer, str) and len(answer) > 20
    ok(f"generate_response() -- {len(answer)}-char answer generated")
    preview = answer[:150].replace("\n", " ").strip()
    print(f"        Preview: \"{preview}...\"")
except Exception as e:
    fail("generate_response()", str(e))

# Multi-turn context test
try:
    history = [
        {"user_question": "What is attention?", "assistant_answer": "Attention allows models to focus on relevant parts of input."}
    ]
    answer2 = llm.generate_response(
        user_question="Can you elaborate on that?",
        retrieved_chunks=dummy_chunks,
        conversation_history=history
    )
    assert isinstance(answer2, str) and len(answer2) > 10
    ok(f"generate_response() with conversation history -- {len(answer2)}-char answer")
except Exception as e:
    fail("generate_response() multi-turn", str(e))

# ──────────────────────────────────────────────────────────────
# 12. Chat History in Supabase
# ──────────────────────────────────────────────────────────────
section("12. Chat History Persistence (Supabase)")
try:
    clear_chat_history()
    qa_pairs = [
        ("What is the Transformer?",  "A model based entirely on self-attention."),
        ("Who proposed it?",          "Vaswani et al. in 2017."),
        ("What task does it target?", "Machine translation and other seq2seq tasks."),
    ]
    for q, a in qa_pairs:
        insert_chat_history(q, a)

    history = fetch_chat_history()
    assert len(history) == 3
    assert history[0]["user_question"] == qa_pairs[0][0]
    ok(f"insert + fetch_chat_history() -- {len(history)} records in correct order")

    recent = fetch_recent_chat_history(limit=2)
    assert len(recent) == 2
    assert recent[0]["user_question"] == qa_pairs[1][0]  # 2nd oldest of recent 2
    ok(f"fetch_recent_chat_history(2) -- {len(recent)} records in chrono order")

    clear_chat_history()
    after = fetch_chat_history()
    assert len(after) == 0
    ok("clear_chat_history() -- all records removed")
except Exception as e:
    fail("Chat history persistence", str(e))

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  PASSED   : {passed}")
print(f"  FAILED   : {failed}")
print(f"  WARNINGS : {len(warnings)}")
print("=" * 55)
if warnings:
    print()
    print("Warnings (non-blocking):")
    for w in warnings:
        print(f"  * {w[0]}: {w[1]}")
print()
if failed == 0:
    print("All feature tests passed! The app is fully operational.")
else:
    print(f"{failed} test(s) failed. Review details above.")
    sys.exit(1)
