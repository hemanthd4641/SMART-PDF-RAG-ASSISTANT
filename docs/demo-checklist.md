# 🎬 5–8 Minute Assessment Demo Checklist & Script

This checklist provides the exact chronological script and action items for recording or delivering the live take-home assessment demonstration.

---

## ⏱️ Timeline & Script Overview

```
┌──────────────┬─────────────────────────────────────────────────────────────┐
│ Time Window  │ Demo Phase & Actions                                        │
├──────────────┼─────────────────────────────────────────────────────────────┤
│ 00:00–01:00  │ 1. Problem & Architectural Approach                         │
│ 01:00–02:00  │ 2. Document Upload & Auto-Summarization                     │
│ 02:00–03:15  │ 3. Answerable Question & Page Citations                     │
│ 03:15–04:15  │ 4. Source Preview & In-App Yellow Highlighting              │
│ 04:15–05:00  │ 5. Follow-Up Question & Conversation Memory                 │
│ 05:00–06:00  │ 6. Creative Feature: Multi-Document Comparison Table        │
│ 06:00–07:00  │ 7. Unanswerable Question: Evidence Gate & Refusal           │
│ 07:00–08:00  │ 8. Architecture Summary & 2-Day Future Roadmap              │
└──────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 📋 Step-by-Step Demonstration Actions

### 1. Problem & Approach (00:00 – 01:00)
- **Talking Point**: *"Standard RAG systems hallucinate when evidence is missing and fail to provide verifiable citations. I built a document-aware assistant featuring hybrid retrieval (Dense + BM25), Cross-Encoder re-ranking, and a 3-tier Evidence Gate to guarantee grounded answers."*
- **Visual**: Show the landing page and the Mermaid diagram in `docs/architecture.md`.

---

### 2. Document Upload & Summaries (01:00 – 02:00)
- **Action**: In the sidebar uploader, select and upload:
  - `sample_documents/Employee_Handbook_2026.pdf`
  - `sample_documents/Leave_Policy_2026.pdf`
  - `sample_documents/Contractor_Agreement_Guidelines_2026.docx`
- **Showcase**:
  - Point out batch ingestion progress bars.
  - Open the **Inspector Panel** → **Document Summaries** tab to show the AI-extracted 2–3 sentence summary and key topic pills.
  - Mention the MD5 content hash deduplication guard.

---

### 3. Answerable Question & Citations (02:00 – 03:15)
- **Prompt**: Type *"What is the annual leave allocation and how many days can be carried forward into the next year?"*
- **Showcase**:
  - Show the fast response generated via Groq (Llama 3.3 70B).
  - Point out the **`🟢 Evidence: Strong`** badge.
  - Highlight the exact source citation: `📄 Leave_Policy_2026.pdf — Page 1`.
  - Expand **Retrieved Chunks & Diagnostics** to show dense vs. sparse score fusion.

---

### 4. Source Attribution & Interactive Yellow Highlighting (03:15 – 04:15)
- **Action**: Navigate to the **Inspector Panel** → **Source Preview** tab.
- **Showcase**:
  - Select `Leave_Policy_2026.pdf` and `Page 1`.
  - Show the full page text reconstructed from Supabase.
  - Demonstrate the exact retrieved sentence wrapped in yellow highlight (`<mark style="background-color: #fef08a">`).
  - Explain why this improves human verifiability.

---

### 5. Follow-Up Question & Conversation Memory (04:15 – 05:00)
- **Prompt**: Type *"What about sick leave?"*
- **Showcase**:
  - Demonstrate that the model maintains context without needing to restate *"What is the policy for sick leave?"*.
  - Show that the response answers 10 days of paid sick leave and cites `📄 Leave_Policy_2026.pdf — Page 1`.

---

### 6. Creative Feature: Structured Multi-Document Comparison (05:00 – 06:00)
- **Action**: In the chat header multiselect filter, select all three documents.
- **Prompt**: Type *"Compare the notice period, annual leave, and probation policies between full-time employees and contractors."*
- **Showcase**:
  - Point out how the model synthesizes information from `Employee_Handbook_2026.pdf`, `Leave_Policy_2026.pdf`, and `Contractor_Agreement_Guidelines_2026.docx`.
  - Show the resulting **Markdown Comparison Table** (e.g., 30 days notice for employees vs. 14 days for contractors).
  - Show multi-document citations appended cleanly at the bottom.

---

### 7. Unanswerable Question: Evidence Gate & Refusal (06:00 – 07:00)
- **Prompt**: Type *"What was the company's Q4 financial revenue and stock price in 2025?"*
- **Showcase**:
  - Point out the **`🔴 Evidence: Insufficient`** badge.
  - Show the controlled polite refusal: *"I couldn't find enough relevant information in the uploaded documents to answer this reliably."*
  - Expand the diagnostics to show how low similarity/cross-encoder scores triggered the Evidence Gate.
  - Emphasize that **zero hallucinated market data** was produced.

---

### 8. Architecture Summary & Future Improvements (07:00 – 08:00)
- **Summary**:
  - Hybrid search combines semantic intent with BM25 lexical precision.
  - RRF ($k=60$) seamlessly merges differing score scales.
  - Cross-Encoder re-ranking provides high-precision top-5 context.
- **Two-Day Future Improvements**:
  1. *Unify vector and full-text search directly inside Supabase using `pgvector` and PostgreSQL `tsvector`.*
  2. *Add multimodal visual RAG (e.g., Llama 3.2 Vision) for PDF chart and infographic interpretation.*
  3. *Implement streaming token generation in the Streamlit UI.*
