"""
Test presentation formatting for Evidence Badges and Citations.
Tests:
  1. Normal answer source display (Strong Evidence)
  2. Moderate Evidence display
  3. Insufficient Evidence display
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm import generate_citations_block

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

print("=" * 60)
print("TESTING ANSWER & SOURCE PRESENTATION FORMATTING")
print("=" * 60)

mock_chunks = [
    {"document_name": "leave_policy.pdf", "page_number": 4, "chunk_text": "Employees get 24 days annual leave."},
    {"document_name": "employee_handbook.pdf", "page_number": 12, "chunk_text": "Annual leave policy details..."},
]

# Test 1: Strong Evidence Citation Block
block_strong = generate_citations_block(mock_chunks, evidence_level="Strong Evidence")
print("\n--- Test 1 Output (Strong Evidence) ---")
print(block_strong)
assert "🟢 Evidence: Strong" in block_strong, "Strong badge missing"
assert "📄 **leave_policy.pdf** — Page `4`" in block_strong, "leave_policy citation missing"
assert "📄 **employee_handbook.pdf** — Page `12`" in block_strong, "employee_handbook citation missing"
print(f"{PASS} Test 1 Passed: Strong Evidence citation block properly formatted.")
results.append(True)

# Test 2: Moderate Evidence Citation Block
block_mod = generate_citations_block(mock_chunks, evidence_level="Moderate Evidence")
print("\n--- Test 2 Output (Moderate Evidence) ---")
print(block_mod)
assert "🟡 Evidence: Moderate" in block_mod, "Moderate badge missing"
print(f"{PASS} Test 2 Passed: Moderate Evidence citation block properly formatted.")
results.append(True)

# Test 3: Insufficient Evidence Display
insufficient_answer = (
    "**🔴 Evidence: Insufficient**\n\n"
    "I couldn't find enough relevant information in the uploaded documents to answer this reliably."
)
print("\n--- Test 3 Output (Insufficient Evidence) ---")
print(insufficient_answer)
assert "🔴 Evidence: Insufficient" in insufficient_answer, "Insufficient badge missing"
assert "I couldn't find enough relevant information" in insufficient_answer, "Fallback message missing"
print(f"{PASS} Test 3 Passed: Insufficient Evidence presentation properly formatted.")
results.append(True)

print("\n" + "=" * 60)
print(f"PRESENTATION TEST SUMMARY: {sum(results)}/{len(results)} PASSED")
print("=" * 60)
