# FinTech Knowledge Base and Application Notes

The assistant answers practical questions about payment workflows using the source documents in `data/`. Those documents describe how a transaction moves between a customer, app/provider, bank, payment network, merchant/acquirer, and settlement system; they include exception handling and source links. The knowledge base is not a collection of general AI, machine-learning, or economy definitions.

## Retrieval sources

| File | Main coverage |
|---|---|
| `data/upi_workflow.txt` | UPI P2P/P2M, QR and collect payments, status interpretation, security, and debited-but-not-credited handling |
| `data/neft_rtgs_imps_workflow.txt` | NEFT batch flow, RTGS gross settlement, IMPS, limits, timing, tracking |
| `data/card_payment_lifecycle.txt` | Authentication, authorization, capture, clearing, settlement, merchant payout, refunds, and disputes |
| `data/kyc_aml_and_transaction_controls.txt` | Onboarding, customer due diligence, monitoring, and distinctions among compliance controls |
| `data/payment_failures_and_complaints.txt` | Failure triage, applicable reversal timelines, and complaint escalation |
| `data/payment_roles_and_lifecycle.txt` | Participants and the distinction among initiation, posting, clearing, settlement, and confirmation |

The source documents link to primary RBI, NPCI, Visa, and government references. They identify the rail and transaction case for a timing or rule instead of presenting one deadline as universal. Because regulations and scheme rules can change, time-sensitive answers should be checked against the linked source and the user's bank/provider.

## Application layout

- `app.py` renders the Streamlit chat experience.
- `rag_pipeline.py` loads text documents from `data/`, splits and embeds them, and builds retrieval.
- `answer_pipeline.py` combines retrieved context with the Gemini model.
- `safety.py` applies the assistant's scope and response constraints.
- `index.html` and `templates/index.html` are project overview/launch pages; the chat UI itself is served by Streamlit.

The assistant is informational. It cannot see a customer's bank account, inspect a transaction, or determine why a regulated institution applied an internal control.
