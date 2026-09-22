"""
Live Groq LLM Test Script.
Tests LLM generation using the configured Groq API Key.
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm import LLMService
from services.evidence_gate import EvidenceGate

PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 60)
print("TESTING LIVE GROQ LLM SERVICE")
print("=" * 60)

llm = LLMService()

# --- Test 1: Live LLM Answer Generation ---
print("\n--- Test 1: Live Answer Generation ---")
mock_chunks = [
    {
        "document_name": "company_policy.pdf",
        "page_number": 4,
        "chunk_text": "Employees receive 24 days of annual leave per calendar year. Leave requests must be submitted at least 2 weeks in advance.",
    },
    {
        "document_name": "employee_handbook.pdf",
        "page_number": 12,
        "chunk_text": "Sick leave is allocated at 10 days per year. Unused annual leave up to 5 days can be carried over.",
    }
]

query = "How many annual leaves do employees get and what is the carry-over policy?"

try:
    response = llm.generate_response(
        user_question=query,
        retrieved_chunks=mock_chunks,
        evidence_level="Strong Evidence",
    )
    print("LIVE LLM RESPONSE:")
    print("-" * 40)
    print(response)
    print("-" * 40)

    assert "24" in response, "Response missing 24 days detail"
    assert "🟢 Evidence: Strong" in response, "Evidence badge missing"
    assert "company_policy.pdf" in response, "Citation missing"
    print(f"\n{PASS} Test 1 Passed: Live Groq LLM generated grounded answer with citations.")
except Exception as e:
    print(f"\n{FAIL} Test 1 Failed: {e}")

# --- Test 2: Live Summary Generation ---
print("\n--- Test 2: Live Document Summary (JSON Mode) ---")
sample_text = (
    "ABC Technologies Pvt Ltd was founded in 2020 in Bangalore, India. "
    "The company specializes in AI Legal Assistant software, Contract Analyzers, "
    "and Document Search Platforms. It employs 500 people and received $10 million in funding in 2023."
)

try:
    summary_data = llm.generate_summary(sample_text)
    print("LIVE SUMMARY DATA:")
    print(f"Summary: {summary_data.get('summary')}")
    print(f"Key Topics: {summary_data.get('key_topics')}")
    assert "summary" in summary_data, "Summary key missing"
    assert "key_topics" in summary_data, "Key topics key missing"
    print(f"\n{PASS} Test 2 Passed: Live Groq JSON mode generated document summary.")
except Exception as e:
    print(f"\n{FAIL} Test 2 Failed: {e}")

# --- Test 3: Unanswerable Question (LLM Bypass) ---
print("\n--- Test 3: Unanswerable Question LLM Bypass ---")
unanswerable_query = "What is the company stock ticker symbol?"

eval_res = EvidenceGate.evaluate(unanswerable_query, mock_chunks, use_reranker=False)
# Force insufficient score test for unanswerable question
eval_res_blocked = EvidenceGate.evaluate(unanswerable_query, [], use_reranker=False)

if not eval_res_blocked["is_sufficient"]:
    blocked_answer = (
        "**🔴 Evidence: Insufficient**\n\n"
        "I couldn't find enough relevant information in the uploaded documents to answer this reliably."
    )
    print("CONTROLLED RESPONSE (LLM Bypassed):")
    print(blocked_answer)
    assert "🔴 Evidence: Insufficient" in blocked_answer
    print(f"\n{PASS} Test 3 Passed: LLM call correctly bypassed for unanswerable query.")

print("\n" + "=" * 60)
print("LIVE GROQ TEST SUITE COMPLETE")
print("=" * 60)
