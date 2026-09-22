# ⏱️ Development Time Log

This document provides a summary of the approximate 8-hour development allocation for the **Smart PDF/RAG Assistant** take-home assessment.

---

## Time Allocation Breakdown

| Area | Approx. Time |
|---|---:|
| Assessment analysis and existing project review | 0.5 hours |
| Document processing & ingestion (PDF/DOCX/OCR/Table extraction) | 1.5 hours |
| RAG & hybrid retrieval pipeline (Dense + BM25 + RRF + Cross-Encoder) | 2.0 hours |
| Evidence & hallucination handling (3-tier Evidence Gate & Refusals) | 1.0 hours |
| UI & creative features (Source Preview Highlighting, Comparison Table, Ask This Doc) | 1.0 hours |
| Testing & debugging (Automated test suites: 30 reliability tests, 9 assessment scenarios) | 1.0 hours |
| Documentation & architecture specification (`README.md`, `docs/architecture.md`) | 0.5 hours |
| Demo preparation (Sample document package, sample README, and demo checklist) | 0.5 hours |
| **Total** | **~8.0 hours** |

---

## Notes on Estimation

This log represents an approximate distribution of engineering effort across the project lifecycle. Time was prioritized toward building a grounded, multi-stage hybrid retrieval pipeline (Pinecone dense search + BM25 sparse search merged via Reciprocal Rank Fusion and Cross-Encoder re-ranking), implementing heuristic evidence validation to eliminate ungrounded hallucinations, and creating transparent source attribution tools (including in-app yellow text highlighting and structured multi-document comparisons).
