# 📄 Sample Evaluation Documents

This directory contains pre-packaged sample documents for testing, evaluation, and live demonstration of the **Smart PDF/RAG Assistant**.

---

## 1. Document Catalog

| Document | Format | Pages | Content Overview |
|---|---|---|---|
| [`Employee_Handbook_2026.pdf`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/sample_documents/Employee_Handbook_2026.pdf) | PDF (Digital) | 2 | Company mission, working hours (9 AM–5 PM), 3-month probation period, healthcare benefits, and 30-day resignation notice. |
| [`Leave_Policy_2026.pdf`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/sample_documents/Leave_Policy_2026.pdf) | PDF (Digital) | 2 | 24 days annual leave, 5 days carry-forward before March 31, 10 days sick leave, 16 weeks maternity leave, and 11 public holidays. |
| [`Contractor_Agreement_Guidelines_2026.docx`](file:///c:/Users/heman/OneDrive/Desktop/rag%20assistant/SMART-PDF-RAG-ASSISTANT/sample_documents/Contractor_Agreement_Guidelines_2026.docx) | DOCX (Word) | 1 | SOW deliverables, Net 30 billing, 14-day termination notice, and summary comparison table. |

---

## 2. Recommended Documents to Upload

For the full evaluation walkthrough, upload all three documents:
1. `Employee_Handbook_2026.pdf`
2. `Leave_Policy_2026.pdf`
3. `Contractor_Agreement_Guidelines_2026.docx`

---

## 3. Demo Questions & Expected Behavior

### A. Answerable Questions (Single & Cross-Document)

#### Question 1 (Specific Fact & Page Citation):
> *"What is the annual leave allocation and how many days can be carried forward into the next year?"*
- **Expected Behavior**:
  - Direct, grounded answer stating 24 days annual leave and a maximum of 5 days carry-forward until March 31.
  - Evidence indicator: `🟢 Evidence: Strong`.
  - Citation: `📄 Leave_Policy_2026.pdf — Page 1`.

#### Question 2 (Multi-Turn Follow-Up / Memory):
> *"What about sick leave?"*
- **Expected Behavior**:
  - Recognizes conversational context from previous turn.
  - Answers that employees receive 10 paid sick leave days per year with medical certificate required for >2 days.
  - Citation: `📄 Leave_Policy_2026.pdf — Page 1`.

#### Question 3 (Health Benefits & Coverage):
> *"What health insurance benefits does the company provide?"*
- **Expected Behavior**:
  - Lists medical, dental, and vision coverage, 85% premium coverage for employees, 60% for dependents, and 2x base salary life insurance.
  - Citation: `📄 Employee_Handbook_2026.pdf — Page 2`.

---

### B. Creative Feature Demonstration Question (Document Comparison)

#### Question 4 (Multi-Document Comparison):
> *"Compare the notice period, annual leave, and probation policies between full-time employees and contractors."*
- **Expected Behavior**:
  - Automatically activates **Document Comparison Mode**.
  - Outputs a structured Markdown table comparing policies across `Employee_Handbook_2026.pdf`, `Leave_Policy_2026.pdf`, and `Contractor_Agreement_Guidelines_2026.docx`.
  - Cites all three source documents.
  - Visual passage highlight available in the **Source Preview** tab with yellow `<mark>` highlighting.

---

### C. Deliberately Unanswerable Question (Evidence Gate & Refusal)

#### Question 5 (Unknown Information / Hallucination Test):
> *"What was the company's Q4 financial revenue and stock price growth in 2025?"*
- **Expected Behavior**:
  - The **Evidence Gate** detects low retrieval similarity scores (`score < threshold`).
  - LLM generation is intercepted, returning the standardized refusal:
    > **🔴 Evidence: Insufficient**  
    > *I couldn't find enough relevant information in the uploaded documents to answer this reliably.*
  - **Zero hallucinated financial figures or stock tickers.**

---

## 4. Source Preview Highlighting Verification

1. After asking Question 1, open the **Inspector Panel** on the right.
2. Click the **Source Preview** tab.
3. Select `Leave_Policy_2026.pdf` and `Page 1`.
4. The system reconstructs the page text and encloses the retrieved sentence in yellow highlight (`<mark style="background-color: #fef08a">`).
