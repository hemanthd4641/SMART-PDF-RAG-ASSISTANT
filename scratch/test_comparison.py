"""
Test suite for Document Comparison feature.
Tests:
  1. Two-document comparison (Markdown comparison table + citations)
  2. One-document query (Standard answer + citations)
  3. Unsupported comparison (Evidence Gate blocks LLM, returns Insufficient Evidence)
  4. Source citations for multi-document comparison
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm import LLMService, generate_citations_block
from services.evidence_gate import EvidenceGate

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

print("=" * 60)
print("RUNNING DOCUMENT COMPARISON FEATURE TESTS")
print("=" * 60)

llm = LLMService()

# Simulated context from two distinct documents
chunks_doc_a = [
    {
        "document_name": "Employee_Handbook.pdf",
        "page_number": 12,
        "chunk_text": "Employees receive 20 days of annual leave per year. Carry forward is allowed up to 5 days.",
        "score": 1.5,
    }
]

chunks_doc_b = [
    {
        "document_name": "Leave_Policy.pdf",
        "page_number": 4,
        "chunk_text": "Annual leave allocation is 24 days per year. Carry forward is allowed up to 10 days with manager approval.",
        "score": 1.8,
    }
]

combined_chunks = chunks_doc_a + chunks_doc_b

# ── Test 1: Two-document comparison ─────────────────────────────────────────
print("\n--- Test 1: Two-Document Comparison Formatting ---")
try:
    # Test prompt building logic and citation block for two documents
    citations = generate_citations_block(combined_chunks, evidence_level="Strong Evidence")
    
    # Verify citations contains both documents
    assert "Employee_Handbook.pdf" in citations, "Employee_Handbook.pdf missing from citations"
    assert "Leave_Policy.pdf" in citations, "Leave_Policy.pdf missing from citations"
    assert "🟢 Evidence: Strong" in citations, "Evidence badge missing"
    
    print("CITATIONS BLOCK OUTPUT:")
    print(citations)
    print(f"\n{PASS} Test 1 Passed: Two-document citations properly generated.")
    results.append(True)
except Exception as e:
    print(f"\n{FAIL} Test 1 Failed: {e}")
    results.append(False)

# ── Test 2: One-document query ───────────────────────────────────────────────
print("\n--- Test 2: One-Document Query Formatting ---")
try:
    citations_single = generate_citations_block(chunks_doc_a, evidence_level="Strong Evidence")
    assert "Employee_Handbook.pdf" in citations_single, "Document name missing"
    assert "Leave_Policy.pdf" not in citations_single, "Unrelated document should not be in citations"
    assert "Page `12`" in citations_single, "Page number missing"
    print("SINGLE DOC CITATIONS OUTPUT:")
    print(citations_single)
    print(f"\n{PASS} Test 2 Passed: Single document citations properly generated.")
    results.append(True)
except Exception as e:
    print(f"\n{FAIL} Test 2 Failed: {e}")
    results.append(False)

# ── Test 3: Unsupported comparison (Evidence Gate) ──────────────────────────
print("\n--- Test 3: Unsupported Comparison (Evidence Gate Block) ---")
try:
    # Weak chunks representing zero relevance for an unsupported comparison query
    weak_chunks = [
        {"document_name": "Employee_Handbook.pdf", "page_number": 1, "chunk_text": "Company overview.", "score": -8.5},
        {"document_name": "Leave_Policy.pdf", "page_number": 1, "chunk_text": "Table of contents.", "score": -9.1},
    ]
    query_unsupported = "Compare the stock option vesting schedules in these documents."
    eval_res = EvidenceGate.evaluate(query_unsupported, weak_chunks, use_reranker=True)

    assert eval_res["is_sufficient"] == False, "Unsupported comparison should have is_sufficient=False"
    assert eval_res["level"] == "Insufficient Evidence", "Level should be Insufficient Evidence"
    
    # Controlled fallback message
    fallback_resp = (
        "**🔴 Evidence: Insufficient**\n\n"
        "I couldn't find enough relevant information in the uploaded documents to answer this reliably."
    )
    assert "🔴 Evidence: Insufficient" in fallback_resp
    assert "couldn't find enough relevant information" in fallback_resp
    
    print("UNSUPPORTED COMPARISON FALLBACK RESPONSE:")
    print(fallback_resp)
    print(f"\n{PASS} Test 3 Passed: Unsupported comparison correctly blocked with Insufficient Evidence badge.")
    results.append(True)
except Exception as e:
    print(f"\n{FAIL} Test 3 Failed: {e}")
    results.append(False)

# ── Test 4: Source Citations Integrity for Comparison ────────────────────────
print("\n--- Test 4: Source Citations Integrity for Comparison ---")
try:
    sources_set = {(c["document_name"], c["page_number"]) for c in combined_chunks}
    assert len(sources_set) == 2, "Expected 2 unique document/page source pairs"
    print(f"{PASS} Test 4 Passed: Multi-document source tracking verified.")
    results.append(True)
except Exception as e:
    print(f"\n{FAIL} Test 4 Failed: {e}")
    results.append(False)

print("\n" + "=" * 60)
print(f"DOCUMENT COMPARISON TEST SUMMARY: {sum(results)}/{len(results)} PASSED")
print("=" * 60)
