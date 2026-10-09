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

The RAG system indexes the UTF-8 `.txt` files in `data/`. The corpus is organized around practical payment workflows and customer questions, not general AI/ML definitions:

- `upi_workflow.txt`: UPI push, QR and collect flows, statuses, security, and debited-but-not-credited cases.
- `neft_rtgs_imps_workflow.txt`: participant steps, availability, limits, transfer tracking, and differences among bank transfer rails.
- `card_payment_lifecycle.txt`: card authentication, authorization, capture, clearing, settlement, merchant payout, refunds, and disputes.
- `kyc_aml_and_transaction_controls.txt`: customer onboarding, ongoing due diligence, transaction controls, and what a generic review status can and cannot tell a user.
- `payment_failures_and_complaints.txt`: failure triage, rail-specific reversal timelines, and escalation steps.
- `payment_roles_and_lifecycle.txt`: actors and how initiation, authentication, authorization, clearing, settlement, posting, and confirmation differ.

Each document includes source notes linking to official RBI, NPCI, Visa, or government material. Rules and timelines are tied to the named rail and failure case; they should not be generalized to every payment. The bot has no access to a user's bank account or case status.

## Project Structure

- `app.py`: Streamlit chat UI.
- `answer_pipeline.py`, `rag_pipeline.py`, `safety.py`: answer generation, document retrieval, and safety handling.
- `data/`: sourced FinTech workflow knowledge base indexed by retrieval.
- `templates/index.html` and `index.html`: project overview/launch pages; the chat application runs in Streamlit.
- `requirements.txt`: Python dependencies.
- `tests/`: project setup and retrieval checks.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rupikagouri/Fintech-Compliance-Explainer-Bot.git
cd Fintech-Compliance-Explainer-Bot
