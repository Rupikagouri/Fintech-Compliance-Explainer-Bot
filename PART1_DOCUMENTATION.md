# FinTech Knowledge Base and Application Notes

The assistant answers questions about Indian fintech products, infrastructure, software, operations, and compliance using UTF-8 documents in `data/` and `knowledge_base/india_fintech/`. The scope excludes general personal finance, investing tutorials, standalone stock-market education, and unrelated corporate finance. WealthTech coverage concerns fintech product and servicing workflows only.

## Retrieval sources

| File | Main coverage |
|---|---|
| `data/upi_workflow.txt` | UPI P2P/P2M, QR and collect payments, status interpretation, security, and debited-but-not-credited handling |
| `data/neft_rtgs_imps_workflow.txt` | NEFT batch flow, RTGS gross settlement, IMPS, limits, timing, tracking |
| `data/card_payment_lifecycle.txt` | Authentication, authorization, capture, clearing, settlement, merchant payout, refunds, and disputes |
| `data/kyc_aml_and_transaction_controls.txt` | Onboarding, customer due diligence, monitoring, and distinctions among compliance controls |
| `data/payment_failures_and_complaints.txt` | Failure triage, applicable reversal timelines, and complaint escalation |
| `data/payment_roles_and_lifecycle.txt` | Participants and the distinction among initiation, posting, clearing, settlement, and confirmation |

The modular collection covers payments, banking APIs/core systems, lending technology, Account Aggregators, embedded finance, InsurTech, WealthTech operations, RegTech, KYC/AML, fraud/security, settlement/reconciliation, ledgers, and fintech engineering. `knowledge_base/india_fintech/MANIFEST.txt`, `GLOSSARY.txt`, `SOURCE_REGISTER.txt`, and `COVERAGE_MATRIX.txt` provide inventory, terms, official sources, caveats, and topic mapping. Check the exact source, effective date, entity, and rail for time-sensitive rules.

## Application layout

- `app.py` renders the Streamlit chat experience.
- `rag_pipeline.py` loads `.txt` files recursively from `data/` and `knowledge_base/india_fintech/`, preserves metadata, splits and embeds them, and builds retrieval.
- `answer_pipeline.py` combines retrieved context with the Gemini model.
- `safety.py` applies the assistant's scope and response constraints.
- `index.html` and `templates/index.html` are project overview/launch pages; the chat UI itself is served by Streamlit.

The assistant is informational. It cannot see a customer's bank account, inspect a transaction, or determine why a regulated institution applied an internal control.

The FAISS index is in-memory and rebuilt when the application starts. Restart the app after changing documents. Full semantic retrieval and generation require the dependencies in `requirements.txt` and a server-side Gemini API key.
