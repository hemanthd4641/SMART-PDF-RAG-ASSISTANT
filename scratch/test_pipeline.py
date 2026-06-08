"""
End-to-end pipeline test for test_company.pdf.
Tests: PDF parsing, table extraction, chunking, OCR fallback check, BM25 tokenization.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- 1. Test: PDF Parsing ----
print("=" * 60)
print("TEST 1: PDF Parsing (PyMuPDF + pdfplumber)")
print("=" * 60)

from services.parser import parse_document

pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_company.pdf")
parsed = parse_document(pdf_path)

pages = parsed.get("pages", [])
print(f"[OK] Document name: {parsed['document_name']}")
print(f"[OK] Pages extracted: {len(pages)}")
for p in pages:
    text_len = len(p.get("text", ""))
    method = p.get("extraction_method", "native")
    tables = p.get("tables", [])
    print(f"     Page {p['page_number']}: {text_len} chars | method={method} | tables={len(tables)}")

# Check key terms in extracted text
full_text = " ".join(p.get("text", "") for p in pages)
checks = [
    "ABC Technologies",
    "Sarah Johnson",
    "Bangalore",
    "XYZ Corp",
    "March 2024",
    "10 million",
    "2020",
    "500",
]
print()
print("Key term extraction checks:")
all_ok = True
for term in checks:
    found = term.lower() in full_text.lower()
    status = "[PASS]" if found else "[FAIL]"
    if not found:
        all_ok = False
    print(f"  {status} '{term}'")

print()

# ---- 2. Test: Chunking ----
print("=" * 60)
print("TEST 2: Chunking (RecursiveTextSplitter)")
print("=" * 60)

from services.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(parsed)

print(f"[OK] Total chunks generated: {len(chunks)}")
text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
print(f"     Text chunks: {len(text_chunks)}")
print(f"     Table chunks: {len(table_chunks)}")

print()
print("Sample chunk content (first 200 chars):")
for i, c in enumerate(chunks[:3]):
    print(f"  Chunk {i+1} [{c['chunk_type']}] | Page {c['page_number']}")
    print(f"    {c['chunk_text'][:200]!r}")
    print()

# ---- 3. Test: BM25 Keyword Search (offline, using parsed chunks directly) ----
print("=" * 60)
print("TEST 3: BM25 Tokenization")
print("=" * 60)

from services.bm25_retriever import tokenize
from rank_bm25 import BM25Okapi

corpus = [c["chunk_text"] for c in chunks]
tokenized_corpus = [tokenize(t) for t in corpus]
bm25 = BM25Okapi(tokenized_corpus)

test_queries = [
    "Who is the CEO?",
    "partnership agreement XYZ Corp",
    "funding received million",
    "customer support hours",
    "confidentiality period years",
]

print(f"BM25 indexed {len(corpus)} chunks offline.")
print()
for q in test_queries:
    scores = bm25.get_scores(tokenize(q))
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    best_chunk = corpus[best_idx][:120]
    print(f"  Query: '{q}'")
    print(f"  Best Score: {best_score:.4f} | Chunk: {best_chunk!r}")
    print()

print("=" * 60)
if all_ok:
    print("[ALL PARSING CHECKS PASSED]")
else:
    print("[WARNING] Some key terms not found in extracted text!")
print("=" * 60)
