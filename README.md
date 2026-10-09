# FinTech Compliance Explainer Bot

A chatbot that helps users understand FinTech, payment systems, and compliance concepts through simple, AI-generated explanations grounded in reference documents.

## Problem Statement

Financial and payment compliance information can be difficult to understand because it is often spread across different sources and written in technical language.

The FinTech Compliance Explainer Bot aims to make this information more accessible by allowing users to ask questions in plain English and receive understandable explanations.

## How It Works

The project uses Retrieval-Augmented Generation (RAG) to combine document retrieval with AI-generated responses.

1. **Document Loading:** Loads reference documents containing relevant information.
2. **Text Chunking:** Splits documents into smaller sections for efficient retrieval.
3. **Embeddings:** Converts text chunks into numerical representations using a Hugging Face embedding model.
4. **Vector Search:** Uses FAISS to retrieve document chunks relevant to the user's question.
5. **Response Generation:** Uses Google's Gemini model to generate an explanation based on the question and retrieved context.
6. **Answer Delivery:** Returns the generated explanation to the user through the application interface.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Google Gemini | Natural-language response generation |
| LangChain | Integrating components of the RAG pipeline |
| Hugging Face | Text embeddings |
| FAISS | Similarity search and document retrieval |
| Streamlit | User interface, if used in the application |

## Knowledge Base

The RAG system indexes UTF-8 `.txt` files recursively from both `data/` and `knowledge_base/india_fintech/`. Its scope is Indian fintech products, infrastructure, software, operational workflows, and compliance. It excludes general personal finance, investing tutorials, standalone stock-market education, and unrelated corporate finance. WealthTech coverage is limited to product/system workflows.

- `upi_workflow.txt`: UPI push, QR and collect flows, statuses, security, and debited-but-not-credited cases.
- `neft_rtgs_imps_workflow.txt`: participant steps, availability, limits, transfer tracking, and differences among bank transfer rails.
- `card_payment_lifecycle.txt`: card authentication, authorization, capture, clearing, settlement, merchant payout, refunds, and disputes.
- `kyc_aml_and_transaction_controls.txt`: customer onboarding, ongoing due diligence, transaction controls, and what a generic review status can and cannot tell a user.
- `payment_failures_and_complaints.txt`: failure triage, rail-specific reversal timelines, and escalation steps.
- `payment_roles_and_lifecycle.txt`: actors and how initiation, authentication, authorization, clearing, settlement, posting, and confirmation differ.

The modular `knowledge_base/india_fintech/` collection adds banking/core integration, domestic payment rails, merchant/card processing, failures and recovery, settlement/reconciliation, APIs and software architecture, ledgers, digital lending, Account Aggregators, embedded finance, InsurTech, WealthTech operations, RegTech/KYC/AML, fraud/security, cross-border flows, public infrastructure, and data/observability. `MANIFEST.txt`, `GLOSSARY.txt`, `COVERAGE_MATRIX.txt`, and `SOURCE_REGISTER.txt` describe contents, terminology, coverage, and verification limits.

The loader strips metadata headers from model text while preserving title, domain, document ID, and relative filename for retrieval and citations. The index is rebuilt in memory at startup; restart Streamlit to re-index edits. Regulatory facts are date-sensitive: check `SOURCE_REGISTER.txt` for references and verification gaps. The bot has no access to a user's bank account or case status.

## Project Structure

- `app.py`: Streamlit chat UI.
- `answer_pipeline.py`, `rag_pipeline.py`, `safety.py`: answer generation, document retrieval, and safety handling.
- `data/` and `knowledge_base/india_fintech/`: UTF-8 FinTech documents indexed recursively by retrieval.
- `templates/index.html` and `index.html`: project overview/launch pages; the chat application runs in Streamlit.
- `requirements.txt`: Python dependencies.
- `tests/`: project setup and retrieval checks.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rupikagouri/Fintech-Compliance-Explainer-Bot.git
cd Fintech-Compliance-Explainer-Bot
