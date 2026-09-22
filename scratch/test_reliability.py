"""
scratch/test_reliability.py
----------------------------
Reliability and code-quality pass verification tests.
Tests error handling for:
- Empty / zero-byte documents
- Whitespace-only TXT files
- Unsupported file formats
- Empty query strings
- Missing metadata fields in retrieved chunks
- Corrupted/missing API keys (config.py helpers)
"""

import os
import sys
import tempfile

# Force UTF-8 stdout for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def test(name: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    results.append((name, status, detail))
    line = f"  {status} -- {name}"
    if detail:
        line += f": {detail}"
    print(line)


print("\n=== Reliability & Code-Quality Verification ===\n")


# ── 1. Empty TXT file (0 bytes) raises DocumentParsingError ──────────────────
print("1. Empty TXT file (0 bytes) validation")
try:
    from services.parser import parse_txt, DocumentParsingError
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("")
        tmp_empty = f.name
    try:
        parse_txt(tmp_empty)
        test("Empty TXT raises DocumentParsingError", False, "No error raised")
    except DocumentParsingError as e:
        test("Empty TXT raises DocumentParsingError", True, str(e))
    finally:
        try:
            os.unlink(tmp_empty)
        except OSError:
            pass
except Exception as e:
    test("Empty TXT import/setup", False, str(e))


# ── 2. Whitespace-only TXT raises DocumentParsingError ───────────────────────
print("\n2. Whitespace-only TXT file validation")
try:
    from services.parser import parse_txt, DocumentParsingError
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("   \n\n\t  \n")
        tmp_ws = f.name
    try:
        parse_txt(tmp_ws)
        test("Whitespace-only TXT raises DocumentParsingError", False, "No error raised")
    except DocumentParsingError as e:
        test("Whitespace-only TXT raises DocumentParsingError", True, str(e))
    finally:
        try:
            os.unlink(tmp_ws)
        except OSError:
            pass
except Exception as e:
    test("Whitespace TXT import/setup", False, str(e))


# ── 3. Valid TXT file parses correctly ───────────────────────────────────────
print("\n3. Valid TXT file parsing")
try:
    from services.parser import parse_txt
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write("Hello, this is a test document.\nIt has two lines.")
        tmp_valid = f.name
    result = parse_txt(tmp_valid)
    ok = (
        result.get("document_name", "").endswith(".txt")
        and len(result.get("pages", [])) == 1
        and "Hello" in result["pages"][0]["text"]
    )
    test("Valid TXT parses successfully", ok, f"pages={len(result.get('pages', []))}")
    os.unlink(tmp_valid)
except Exception as e:
    test("Valid TXT parsing", False, str(e))


# ── 4. Unsupported file format raises DocumentParsingError ───────────────────
print("\n4. Unsupported file format")
try:
    from services.parser import parse_document, DocumentParsingError
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as f:
        f.write("some content")
        tmp_unsup = f.name
    try:
        parse_document(tmp_unsup)
        test("Unsupported format raises DocumentParsingError", False, "No error raised")
    except DocumentParsingError as e:
        test("Unsupported format raises DocumentParsingError", True, str(e)[:80])
    finally:
        try:
            os.unlink(tmp_unsup)
        except OSError:
            pass
except Exception as e:
    test("Unsupported format test setup", False, str(e))


# ── 5. Empty query to BM25Retriever returns [] without crashing ──────────────
print("\n5. Empty query to BM25Retriever")
try:
    from unittest.mock import patch
    from services.bm25_retriever import BM25Retriever
    # Patch fetch_all_chunks at the bm25_retriever module level to avoid real DB call
    with patch("services.bm25_retriever.fetch_all_chunks", return_value=[]):
        bm25 = BM25Retriever()
        result = bm25.retrieve_bm25("   ")
        test("Empty query (spaces) returns []", result == [], f"Got: {result}")
        result2 = bm25.retrieve_bm25("")
        test("Empty query (blank) returns []", result2 == [], f"Got: {result2}")
        # Also verify that a non-empty query against empty index returns []
        result3 = bm25.retrieve_bm25("what is the leave policy")
        test("Non-empty query on empty index returns []", result3 == [], f"Got: {result3}")
except Exception as e:
    test("Empty BM25 query", False, str(e))


# ── 6. EvidenceGate with 0 chunks -> Insufficient Evidence ───────────────────
print("\n6. EvidenceGate with zero chunks")
try:
    from services.evidence_gate import EvidenceGate
    ev = EvidenceGate.evaluate("What is the leave policy?", [], use_reranker=False)
    test("0 chunks -> Insufficient Evidence", ev["level"] == "Insufficient Evidence", f"level={ev['level']}")
    test("0 chunks -> is_sufficient=False", not ev["is_sufficient"])
    test("0 chunks -> controlled message set", bool(ev.get("message")))
    test("0 chunks -> chunk_count=0", ev["chunk_count"] == 0)
except Exception as e:
    test("EvidenceGate empty chunks", False, str(e))


# ── 7. EvidenceGate with low-score chunks -> Insufficient ────────────────────
print("\n7. EvidenceGate with weak retrieval scores (no reranker)")
try:
    from services.evidence_gate import EvidenceGate
    weak_chunks = [
        {
            "chunk_id": "c1",
            "chunk_text": "unrelated text about nothing",
            "document_name": "doc.pdf",
            "page_number": 1,
            "score": 0.001,
        }
    ]
    ev = EvidenceGate.evaluate("What is revenue?", weak_chunks, use_reranker=False)
    test(
        "Low-score chunks -> Insufficient Evidence",
        ev["level"] == "Insufficient Evidence",
        f"level={ev['level']}, score={ev['top_score']}",
    )
except Exception as e:
    test("EvidenceGate low-score", False, str(e))


# ── 8. EvidenceGate with strong-score chunks -> Strong Evidence ───────────────
print("\n8. EvidenceGate with strong retrieval scores (no reranker)")
try:
    from services.evidence_gate import EvidenceGate
    strong_chunks = [
        {
            "chunk_id": "c1",
            "chunk_text": "Revenue was $50M in FY2024.",
            "document_name": "report.pdf",
            "page_number": 3,
            "score": 0.030,
        },
        {
            "chunk_id": "c2",
            "chunk_text": "Total income including other sources.",
            "document_name": "report.pdf",
            "page_number": 4,
            "score": 0.028,
        },
    ]
    ev = EvidenceGate.evaluate("What was the revenue?", strong_chunks, use_reranker=False)
    test(
        "Strong-score chunks -> Strong Evidence",
        ev["level"] == "Strong Evidence",
        f"level={ev['level']}, score={ev['top_score']}",
    )
    test("Strong Evidence -> is_sufficient=True", ev["is_sufficient"])
except Exception as e:
    test("EvidenceGate strong-score", False, str(e))


# ── 9. config.py is_configured() detects missing keys correctly ──────────────
print("\n9. config.py missing Groq/Pinecone key detection")
try:
    from unittest.mock import patch
    import utils.config as cfg

    with patch.object(cfg, "GROQ_API_KEY", ""), patch.object(cfg, "PINECONE_API_KEY", ""):
        test("is_configured() False when keys empty", not cfg.is_configured())

    with patch.object(cfg, "GROQ_API_KEY", "fake_groq_key"), patch.object(cfg, "PINECONE_API_KEY", "fake_pine_key"):
        test("is_configured() True when keys present", cfg.is_configured())
except Exception as e:
    test("config is_configured check", False, str(e))


# ── 10. config.py is_db_configured() detects missing Supabase keys ───────────
print("\n10. config.py Supabase credential detection")
try:
    from unittest.mock import patch
    import utils.config as cfg

    with patch.object(cfg, "SUPABASE_URL", ""), patch.object(cfg, "SUPABASE_KEY", ""):
        test("is_db_configured() False when keys empty", not cfg.is_db_configured())

    with patch.object(cfg, "SUPABASE_URL", "https://abc.supabase.co"), patch.object(cfg, "SUPABASE_KEY", "service_key"):
        test("is_db_configured() True when keys present", cfg.is_db_configured())
except Exception as e:
    test("config is_db_configured check", False, str(e))


# ── 11. generate_citations_block handles missing metadata gracefully ──────────
print("\n11. generate_citations_block with partial/missing metadata")
try:
    from services.llm import generate_citations_block

    partial_chunks = [
        {"chunk_id": "c1"},                                                      # no doc_name, no page
        {"chunk_id": "c2", "document_name": "doc.pdf"},                          # no page
        {"chunk_id": "c3", "document_name": "doc.pdf", "page_number": 5},        # complete
    ]
    block = generate_citations_block(partial_chunks, evidence_level="Moderate Evidence")
    test("generate_citations_block does not crash on partial metadata", bool(block))
    test("Evidence badge present in output", "Moderate" in block)
except Exception as e:
    test("Missing metadata in citations", False, str(e))


# ── 12. format_file_size edge cases ──────────────────────────────────────────
print("\n12. format_file_size utility edge cases")
try:
    from utils.helpers import format_file_size
    test("0 bytes", format_file_size(0) == "0.00 B")
    test("1023 bytes -> B", "B" in format_file_size(1023))
    test("1024 bytes -> KB", "KB" in format_file_size(1024))
    test("50 MB", "MB" in format_file_size(50 * 1024 * 1024))
except Exception as e:
    test("format_file_size edge cases", False, str(e))


# ── 13. table_to_markdown robustness ─────────────────────────────────────────
print("\n13. table_to_markdown robustness")
try:
    from services.parser import table_to_markdown
    test("Empty table -> empty string", table_to_markdown([]) == "")
    test("All-None row filtered out", table_to_markdown([[None, None]]) == "")
    result = table_to_markdown([["Header A", "Header B"], ["Val 1", "Val 2"]])
    test("Valid 2-col table produces markdown", "Header A" in result and "---" in result)
    # Row shorter than header — should not crash
    result2 = table_to_markdown([["Col1", "Col2", "Col3"], ["only_one"]])
    test("Short row padded without crash", "Col1" in result2)
except Exception as e:
    test("table_to_markdown robustness", False, str(e))


# ── 14. .env is excluded from git ────────────────────────────────────────────
print("\n14. .env git exclusion")
try:
    import subprocess
    r = subprocess.run(
        ["git", "check-ignore", "-q", ".env"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        capture_output=True,
    )
    test(".env is gitignored", r.returncode == 0)
except Exception as e:
    test(".env gitignore check", False, str(e))


# ── 15. No hardcoded secrets in source code ──────────────────────────────────
print("\n15. No hardcoded secrets in source Python files")
try:
    import glob
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_files = glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
    suspicious_patterns = [
        "sk-", "gsk_", "eyJh",  # common API key prefixes (Groq starts with gsk_)
    ]
    # We allow the test script itself and scratch scripts; focus on services/components/database
    core_files = [f for f in py_files if any(
        d in f for d in ["services", "components", "database", "utils"]
    )]
    leaks = []
    for fpath in core_files:
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            for pat in suspicious_patterns:
                if pat in content:
                    leaks.append(f"{os.path.basename(fpath)} contains '{pat}'")
        except Exception:
            pass
    test("No hardcoded API key prefixes in core source files", len(leaks) == 0,
         "; ".join(leaks) if leaks else "Clean")
except Exception as e:
    test("Hardcoded secret check", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
total = len(results)
print(f"\n{'=' * 52}")
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed:
    print("\nFailed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  {FAIL} {name}: {detail}")
print("=" * 52)
sys.exit(0 if failed == 0 else 1)
