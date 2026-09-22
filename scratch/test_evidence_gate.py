"""
Test suite for EvidenceGate service.
Verifies categorical evidence level classification and LLM bypass rules.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.evidence_gate import EvidenceGate, INSUFFICIENT_EVIDENCE_MESSAGE

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

print("=" * 60)
print("RUNNING EVIDENCE GATE UNIT TESTS")
print("=" * 60)

# Test 1: Empty chunk list
try:
    res = EvidenceGate.evaluate("What is the stock price?", [], use_reranker=True)
    assert res["level"] == "Insufficient Evidence", f"Expected Insufficient Evidence, got {res['level']}"
    assert res["is_sufficient"] == False, "is_sufficient should be False"
    assert res["message"] == INSUFFICIENT_EVIDENCE_MESSAGE, "Message mismatch"
    print(f"{PASS} Test 1: Empty chunks correctly classified as Insufficient Evidence.")
    results.append(True)
except Exception as e:
    print(f"{FAIL} Test 1 failed: {e}")
    results.append(False)

# Test 2: Low CrossEncoder score (unrelated content)
try:
    weak_chunks = [
        {"chunk_id": "c1", "chunk_text": "Company policy regarding leaves.", "score": -7.8, "document_name": "policy.pdf", "page_number": 1},
        {"chunk_id": "c2", "chunk_text": "Working hours are 9 to 5.", "score": -8.2, "document_name": "policy.pdf", "page_number": 2},
    ]
    res = EvidenceGate.evaluate("What is the stock price?", weak_chunks, use_reranker=True)
    assert res["level"] == "Insufficient Evidence", f"Expected Insufficient Evidence, got {res['level']}"
    assert res["is_sufficient"] == False, "is_sufficient should be False for weak logits"
    assert res["message"] == INSUFFICIENT_EVIDENCE_MESSAGE, "Message mismatch"
    print(f"{PASS} Test 2: Low Cross-Encoder logits (-7.8) correctly classified as Insufficient Evidence.")
    results.append(True)
except Exception as e:
    print(f"{FAIL} Test 2 failed: {e}")
    results.append(False)

# Test 3: High CrossEncoder score (Strong Evidence)
try:
    strong_chunks = [
        {"chunk_id": "c1", "chunk_text": "Employees receive 20 days of annual leave.", "score": 2.4, "document_name": "leave_policy.pdf", "page_number": 1},
        {"chunk_id": "c2", "chunk_text": "Sick leave entitlement is 10 days.", "score": 1.1, "document_name": "leave_policy.pdf", "page_number": 1},
    ]
    res = EvidenceGate.evaluate("How many annual leaves do employees get?", strong_chunks, use_reranker=True)
    assert res["level"] == "Strong Evidence", f"Expected Strong Evidence, got {res['level']}"
    assert res["is_sufficient"] == True, "is_sufficient should be True"
    assert res["message"] is None, "Message should be None for sufficient evidence"
    print(f"{PASS} Test 3: High Cross-Encoder logits (2.4) correctly classified as Strong Evidence.")
    results.append(True)
except Exception as e:
    print(f"{FAIL} Test 3 failed: {e}")
    results.append(False)

# Test 4: Moderate CrossEncoder score (Moderate Evidence)
try:
    mod_chunks = [
        {"chunk_id": "c1", "chunk_text": "Working hours are from 9 AM to 6 PM Monday to Friday.", "score": -2.8, "document_name": "policy.pdf", "page_number": 3},
    ]
    res = EvidenceGate.evaluate("What are the working hours?", mod_chunks, use_reranker=True)
    assert res["level"] == "Moderate Evidence", f"Expected Moderate Evidence, got {res['level']}"
    assert res["is_sufficient"] == True, "is_sufficient should be True"
    print(f"{PASS} Test 4: Moderate Cross-Encoder logits (-2.8) correctly classified as Moderate Evidence.")
    results.append(True)
except Exception as e:
    print(f"{FAIL} Test 4 failed: {e}")
    results.append(False)

# Test 5: Low RRF score (re-ranker off)
try:
    low_rrf_chunks = [
        {"chunk_id": "c1", "chunk_text": "Random text fragment.", "score": 0.005, "dense_score": 0.12, "sparse_score": 0.0, "document_name": "policy.pdf", "page_number": 1}
    ]
    res = EvidenceGate.evaluate("Unrelated query", low_rrf_chunks, use_reranker=False)
    assert res["level"] == "Insufficient Evidence", f"Expected Insufficient Evidence, got {res['level']}"
    assert res["is_sufficient"] == False, "is_sufficient should be False"
    print(f"{PASS} Test 5: Low RRF score (0.005) correctly classified as Insufficient Evidence.")
    results.append(True)
except Exception as e:
    print(f"{FAIL} Test 5 failed: {e}")
    results.append(False)

print("=" * 60)
print(f"EVIDENCE GATE TEST SUMMARY: {sum(results)}/{len(results)} PASSED")
print("=" * 60)
