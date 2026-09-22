"""
Integration test for Evidence Gate with live retriever.
Tests:
  1. Answerable question against document context.
  2. Unrelated/unanswerable question -> Evidence Gate returns Insufficient Evidence, bypassing LLM call.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.evidence_gate import EvidenceGate, INSUFFICIENT_EVIDENCE_MESSAGE
from services.parser import parse_document
from services.chunker import DocumentChunker

PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 60)
print("RUNNING EVIDENCE GATE INTEGRATION TESTS")
print("=" * 60)

# Parse test_company.pdf to get candidate chunks
pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_company.pdf")
parsed = parse_document(pdf_path)
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(parsed)
print(f"Loaded {len(chunks)} chunk(s) from test_company.pdf")

# Mock/Simulate CrossEncoder scoring on answerable vs unanswerable query
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Scenario A: Answerable Question ---
q_answerable = "Who is the CEO of ABC Technologies?"
pairs_a = [[q_answerable, c["chunk_text"]] for c in chunks]
scores_a = cross_encoder.predict(pairs_a)

chunks_a = [c.copy() for c in chunks]
for idx, s in enumerate(scores_a):
    chunks_a[idx]["score"] = float(s)
chunks_a.sort(key=lambda x: x["score"], reverse=True)

eval_a = EvidenceGate.evaluate(q_answerable, chunks_a[:5], use_reranker=True)

print(f"\n[Answerable Question]: '{q_answerable}'")
print(f"  Level: {eval_a['level']}")
print(f"  Is Sufficient: {eval_a['is_sufficient']}")
print(f"  Top Score: {eval_a['top_score']:.4f}")
print(f"  Reason: {eval_a['reason']}")

assert eval_a["is_sufficient"] == True, "Answerable question should have sufficient evidence!"
assert eval_a["level"] in ("Strong Evidence", "Moderate Evidence"), "Level should be Strong or Moderate!"
print(f"{PASS} Scenario A Passed: Answerable question granted LLM execution access.")

# --- Scenario B: Unrelated / Unanswerable Question ---
q_unanswerable = "What is the company stock ticker symbol and quarterly revenue dividend rate?"
pairs_b = [[q_unanswerable, c["chunk_text"]] for c in chunks]
scores_b = cross_encoder.predict(pairs_b)

chunks_b = [c.copy() for c in chunks]
for idx, s in enumerate(scores_b):
    chunks_b[idx]["score"] = float(s)
chunks_b.sort(key=lambda x: x["score"], reverse=True)

eval_b = EvidenceGate.evaluate(q_unanswerable, chunks_b[:5], use_reranker=True)

print(f"\n[Unanswerable Question]: '{q_unanswerable}'")
print(f"  Level: {eval_b['level']}")
print(f"  Is Sufficient: {eval_b['is_sufficient']}")
print(f"  Top Score: {eval_b['top_score']:.4f}")
print(f"  Controlled Message: {eval_b['message']}")
print(f"  Reason: {eval_b['reason']}")

assert eval_b["is_sufficient"] == False, "Unanswerable question must be marked Insufficient!"
assert eval_b["level"] == "Insufficient Evidence", "Level must be Insufficient Evidence!"
assert eval_b["message"] == INSUFFICIENT_EVIDENCE_MESSAGE, "Controlled message mismatch!"
print(f"{PASS} Scenario B Passed: Unanswerable question correctly blocked LLM execution.")

print("\n" + "=" * 60)
print("ALL EVIDENCE GATE INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
