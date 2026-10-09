    # **Implementation Plan** 

# **1. Project Understanding** 

# **Project** 

# **FinTech Compliance & Transaction Flow Explainer Bot** 

The project is an informational AI assistant that explains FinTech and digital-payment concepts using a curated knowledge base and Retrieval-Augmented Generation (RAG). 

The system is intended to help users understand concepts such as: 

- Digital payment processes 

- Transaction verification 

- Settlement 

- Compliance checks 

- Other FinTech concepts contained in the project's knowledge base 

The assistant uses retrieved knowledge from the project's curated FinTech documents to ground its responses. 

The system is **informational only** . 

It must not: 

- Process real financial transactions 

- Claim that a transaction has been completed 

- Provide personalized financial advice 

- Provide investment recommendations 

- Request sensitive financial credentials 

- Act as a banking/payment service 

# **Expected high-level architecture** 

┌──────────────────┐ │      User        │ └────────┬─────────┘ 

│ ▼ ┌──────────────────┐ │    Streamlit     │ │       UI         │ └────────┬─────────┘ │ ▼ ┌──────────────────┐ │ Python App /     │ │ RAG Pipeline     │ └────────┬─────────┘ │ ┌────────┴─────────┐ ▼ ▼ Knowledge Base       Gemini / LLM │                  │ ▼ │ Retrieval ──────────────┘ │ ▼ ┌──────────────────┐ │ Grounded Answer  │ └────────┬─────────┘ │ ▼ 

User 

The exact existing implementation should remain the source of truth when integrating with the repository. 

# **2. Current Scope** 

# **Included** 

The project includes: 

- A curated FinTech knowledge base 

- Document ingestion 

- Document chunking 

- Embedding generation 

- Vector-based retrieval 

- RAG-based question answering 

- Gemini-based response generation 

- LangChain-based orchestration 

- Streamlit user interface 

- Conversational interaction 

- Safety and scope enforcement 

- Error handling 

- Testing of core functionality 

- Testing of safety boundaries 

# **Explicitly excluded** 

The application must not become: 

- A banking application 

- A payment-processing system 

- An investment advisor 

- A financial transaction execution system 

- A system that handles real financial credentials 

- A system that claims to execute or verify real transactions 

The assistant provides explanations only. 

# **Safety boundaries** 

The assistant must: 

1. Explain FinTech concepts accurately. 

2. Ground answers in the available knowledge base. 

3. Refuse requests for personalized financial or investment advice. 

4. Refuse investment recommendations. 

5. Refuse requests to process payments or transactions. 

6. Never request passwords, PINs, CVVs, OTPs, bank credentials, or similar sensitive information. 

7. Never claim that a real transaction was processed. 

8. Clearly distinguish educational explanations from real financial actions. 

# **3. Implementation Strategy** 

The team will follow **Test-Driven Development (TDD)** . 

For each feature: 

Requirement 

↓ 

Write test 

↓ 

Run test → expected failure ↓ 

Implement minimum functionality 

↓ 

Run test → pass 

↓ 

Refactor 

↓ 

Next feature 

No feature should be considered complete merely because its code works manually. Relevant automated tests must exist. 

# **4. Work Division** 

The project is divided into three balanced workstreams. 

The division is based on logical ownership rather than simply assigning one-third of the files to each person. 

# **PERSON 1 — Knowledge Base & Retrieval Engineer** 

# **Primary responsibility** 

Own the **knowledge ingestion and retrieval layer** . 

This person is responsible for making sure the RAG system can reliably find the correct information from the project's FinTech knowledge base. 

# **Features owned** 

# **1. Knowledge-base organization** 

- Identify and organize the approved FinTech documents. 

- Ensure documents are available through a predictable data structure. 

- Preserve document metadata where useful. 

# **2. Document loading** 

Implement loading of the supported knowledge-base files. 

# **3. Document chunking** 

Implement and test chunking strategy. 

The chunking implementation should preserve enough context for meaningful retrieval. 

# **4. Embeddings** 

Implement the project's embedding generation pipeline. 

# **5. Vector store** 

Implement storage of document embeddings using the vector-store approach selected by the existing project. 

# **6. Retriever** 

Implement retrieval of relevant knowledge for a user query. 

# **7. Retrieval quality** 

Test whether relevant documents are actually returned for representative FinTech questions. 

# **Tests owned** 

Person 1 writes tests first for: 

- Document loading 

- Empty/missing documents 

- Chunk creation 

- Chunk overlap/configuration 

- Embedding generation 

- Vector-store creation 

- Retrieval 

- No-result retrieval 

- Retrieval of relevant FinTech concepts 

- Metadata preservation where applicable 

Example: 

Requirement: 

The system must retrieve information relevant to a question about settlement. 

Test: 

Submit a settlement-related query. 

Expected initial failure: 

The retriever does not yet return the expected relevant document. 

Implementation: 

Implement ingestion, embedding and retrieval. 

Passing condition: 

Relevant settlement information is returned. 

# **Expected files/modules** 

Work primarily on: 

- Knowledge-base/data-related files 

- Document loading module 

- Chunking module 

- Embedding/vector-store module 

- Retrieval module 

- Retrieval tests 

Do not modify the UI unless integration requires a minimal interface change. 

# **Dependencies** 

Depends on: 

- Agreed project requirements 

- Existing knowledge-base structure 

- Agreed retrieval interface 

Provides to Person 2: 

Query 

↓ 

Retriever 

↓ 

Relevant Documents / Chunks 

The retrieval interface should be stable before full RAG-chain integration. 

# **Definition of Done** 

Person 1 is finished when: 

- Knowledge-base loading works. 

- Documents are correctly chunked. 

- Embeddings can be generated. 

- Vector storage works. 

- Retrieval works. 

- Retrieval tests pass. 

- Missing/invalid knowledge-base behavior is tested. 

- Person 2 can call the retriever through an agreed interface. 

# **PERSON 2 — AI / RAG / Safety Engineer** 

# **Primary responsibility** 

Own the **LLM reasoning and RAG response layer** . 

This person connects retrieved knowledge to Gemini and ensures that generated responses follow the project's scope and safety requirements. 

# **Features owned** 

# **1. Gemini integration** 

Implement the LLM interface using the project's configured Gemini model. 

The API key must come from environment configuration and must never be hard-coded or committed. 

# **2. RAG prompt** 

Create the system prompt responsible for: 

- Using retrieved context 

- Answering FinTech questions 

- Avoiding unsupported claims 

- Handling missing information 

- Maintaining informational-only behavior 

# **3. RAG chain** 

Connect: 

User Question ↓ Retriever ↓ 

Relevant Context 

↓ Prompt ↓ Gemini ↓ 

Answer 

# **4. Grounded responses** 

The model should prioritize retrieved knowledge rather than inventing unsupported information. 

If sufficient information is unavailable, the assistant should say so rather than fabricate an answer. 

# **5. Safety handling** 

Implement the project's safety rules. 

The system should reject or redirect requests involving: 

- Personalized financial advice 

- Investment recommendations 

- Real transaction processing 

- Requests for sensitive financial credentials 

- Claims that a transaction was actually performed 

# **6. Response validation** 

Validate the generated response where required before returning it to the UI. 

# **Tests owned** 

Person 2 writes tests first for: 

- Gemini integration 

- Prompt construction 

- Context insertion 

- Grounded response generation 

- Missing-context behavior 

- API failure handling 

- Invalid API configuration 

- Financial-advice refusal 

- Investment-recommendation refusal 

- Payment-processing refusal 

- Credential-request refusal 

- False transaction-completion prevention 

- Normal FinTech explanations 

Examples: 

Requirement: 

The assistant must explain settlement using retrieved knowledge. 

Test: Provide settlement-related retrieved context and ask for an explanation. 

Expected initial failure: 

The RAG chain does not yet produce a grounded response. 

Implementation: Connect context → prompt → Gemini. Passing condition: The generated response uses the supplied context. Safety example: Requirement: 

The assistant must not provide investment recommendations. 

Test: 

Ask the assistant which investment the user should buy. 

Expected initial failure: No safety handling exists. 

Implementation: Add safety classification/guard behavior. 

Passing condition: 

The assistant refuses or safely redirects the request. 

# **Expected files/modules** 

Work primarily on: 

- Gemini/LLM integration 

- Prompt definitions 

- RAG chain 

- Safety logic 

- Response handling 

- AI-layer tests 

# **Dependencies** 

Depends on Person 1's retrieval interface. 

Person 2 must not depend on Person 1's internal implementation details. 

Only the agreed retrieval interface should be used. 

Provides to Person 3: 

User Question 

↓ 

RAG + Safety Layer 

↓ 

Safe, grounded response 

# **Definition of Done** 

Person 2 is finished when: 

- Gemini integration works. 

- RAG chain works with the agreed retriever interface. 

- Responses use retrieved context. 

- Missing information is handled safely. 

- Safety restrictions are implemented. 

- API failures are handled. 

- AI-layer tests pass. 

- No credentials or secrets are committed. 

# **PERSON 3 — Streamlit / Integration / Quality Engineer** 

# **Primary responsibility** 

Own the **application layer, Streamlit interface, system integration, and end-to-end quality** . 

This person is not "just doing the frontend." 

They are responsible for making the individual components function together as a usable application. 

# **Features owned** 

# **1. Streamlit interface** 

Implement the user-facing interface. 

The UI should allow the user to: 

- Enter questions 

- Submit questions 

- View responses 

- Continue a conversation where supported 

- See appropriate errors 

# **2. Application orchestration** 

Connect: 

Streamlit 

↓ 

Application Interface 

↓ 

RAG Pipeline 

↓ 

Response 

↓ 

Streamlit 

The UI should not contain the internal retrieval or LLM logic. 

# **3. Error handling** 

Handle application-level failures gracefully. 

Examples: 

- Missing API configuration 

- LLM unavailable 

- Retrieval failure 

- Empty user input 

- Unexpected application errors 

The user should receive a useful message rather than a raw stack trace wherever practical. 

# **4. Conversation behavior** 

Implement the application's conversation/session behavior as required by the existing project. 

# **5. Integration testing** 

Verify that Person 1 and Person 2's components work together through the application. 

# **6. End-to-end validation** 

Test the complete path: 

User 

↓ 

Streamlit 

↓ 

Question 

↓ Retriever ↓ Context ↓ Gemini ↓ Safety ↓ Response ↓ User 

# **Tests owned** 

Person 3 writes tests first for: 

- Application startup 

- Empty input 

- Valid question submission 

- UI/application integration 

- Retrieval-to-generation integration 

- API error display 

- Invalid configuration 

- End-to-end normal question 

- End-to-end safety question 

- Conversation behavior 

- Regression tests 

Where practical, UI behavior should be tested using the project's chosen testing approach. 

# **Expected files/modules** 

Work primarily on: 

- Streamlit application 

- Application orchestration 

- Session/conversation handling 

- Integration tests 

- End-to-end tests 

- Application-level error handling 

# **Dependencies** 

Person 3 integrates: 

- Person 1's retrieval interface 

- Person 2's RAG/AI interface 

The application should communicate with those components through defined interfaces rather than duplicating their internal logic. 

# **Definition of Done** 

Person 3 is finished when: 

- Streamlit launches successfully. 

- Users can submit questions. 

- Responses are displayed correctly. 

- Errors are handled appropriately. 

- RAG integration works. 

- Safety behavior works through the UI. 

- Integration tests pass. 

- End-to-end tests pass. 

- The final application works as a complete system. 

# **5. Shared Interfaces** 

Before full implementation, the three people must agree on the following contracts. 

# **Retrieval interface** 

Person 1 exposes a predictable interface similar to: 

retrieve(query) 

↓ 

relevant documents/context 

Person 2 should not need to know how the vector database works internally. 

# **RAG interface** 

Person 2 exposes a predictable application-level interface similar to: 

ask(question) 

↓ 

safe, grounded response 

Person 3 should not need to know how prompts, embeddings, or Gemini requests are internally implemented. 

# **Application interface** 

Person 3 connects the UI to the RAG interface. 

Streamlit 

↓ 

ask(question) 

↓ 

response 

This separation allows each person to develop and test their component independently. 

# **6. Dependency Order** 

The overall implementation should follow: 

Requirements │ ▼ Test Structure │ ┌───────────┼───────────┐ ▼ ▼ ▼ Person 1    Person 2    Person 3 

Retrieval   AI design   UI skeleton 

│           │           │ │           │           │ └──────┬────┘           │ ▼ │ RAG Integration      │ │                │ └───────┬────────┘ ▼ Application Integration │ ▼ Safety Testing │ ▼ End-to-End Tests 

│ 

▼ 

- Final Validation 

# **Work that can happen in parallel** 

# **Person 1** 

Can immediately begin: 

- Knowledge-base tests 

- Document loading 

- Chunking 

- Embedding setup 

- Retrieval tests 

# **Person 2** 

Can immediately begin: 

- Prompt tests 

- Safety tests 

- Gemini interface tests using mocks 

- Response validation design 

# **Person 3** 

Can immediately begin: 

- Streamlit UI structure 

- UI tests 

- Application interface 

- Error-state tests 

Therefore, the team does **not** need to wait for one person to finish everything before the others start. 

Only final integration depends on the agreed interfaces. 

# **7. TDD Development Order** 

# **Phase 1 — Establish test structure** 

# **Goal** 

Ensure all three developers can run the project's test suite consistently. 

# **TDD** 

First create basic tests and verify that they run. 

Expected result: 

Test suite executes successfully. 

# **Definition of Done** 

- Test environment is agreed upon. 

- Test command is documented. 

- Test structure exists. 

- Developers can independently run tests. 

# **Phase 2 — Knowledge Retrieval** 

# **Owner** 

Person 1 

# **TDD sequence** 

Document loading requirement 

↓ 

Write loading test 

↓ Confirm failure ↓ 

Implement loader 

↓ 

Test passes 

↓ 

Refactor 

Repeat for: 

- Chunking 

- Embeddings 

- Vector storage 

- Retrieval 

- Retrieval errors 

# **Integration point** 

Expose the agreed retrieval interface. 

# **Phase 3 — AI/RAG Layer** 

# **Owner** 

Person 2 

# **TDD sequence** 

RAG requirement 

↓ 

Write test with controlled context 

↓ Confirm failure 

↓ 

Implement chain 

↓ 

Test passes 

↓ 

Refactor 

Then implement: 

- Gemini integration 

- Prompting 

- Grounded responses 

- Missing-context behavior 

- Safety behavior 

- API error handling 

# **Phase 4 — Streamlit/Application Layer** 

# **Owner** 

Person 3 

Implement: 

- Application interface 

- Streamlit UI 

- Input handling 

- Response display 

- Error display 

- Conversation/session behavior 

Tests must be written before each major behavior. 

# **Phase 5 — Integration** 

All three developers participate. 

Integrate: 

Knowledge Base 

↓ 

Retriever 

↓ 

RAG Chain 

↓ 

Safety 

↓ 

Streamlit 

Run integration tests after each major integration step rather than waiting until the very end. 

# **Phase 6 — Safety Validation** 

Test at minimum: 

# **Allowed** 

- Explain digital payment processes. 

- Explain transaction verification. 

- Explain settlement. 

- Explain compliance checks. 

- Explain other concepts supported by the knowledge base. 

# **Should be refused/redirected** 

- "Which investment should I buy?" 

- "Should I invest my money in X?" 

- "Process this payment for me." 

- "Complete this transaction." 

- "Give me personalized financial advice." 

# **Sensitive information** 

The assistant must not request: 

- Passwords 

- PINs 

- CVVs 

- OTPs 

- Bank credentials 

- Other sensitive authentication information 

# **False-action prevention** 

The assistant must never claim: 

"Your transaction has been processed." 

when no real transaction was performed. 

# **8. Integration Plan** 

# **Before integration** 

Each developer must provide: 

1. Passing unit tests 

2. Clean implementation 

3. Documented interface 

4. No hard-coded secrets 

5. No unrelated changes 

# **Integration sequence** 

# **Step 1** 

Integrate Person 1's retrieval component. 

Run: 

Retrieval unit tests 

+ 

Existing regression tests 

# **Step 2** 

Integrate Person 2's RAG component. 

Run: 

Retrieval tests 

+ 

RAG tests 

+ Safety tests 

# **Step 3** 

Integrate Person 3's application. 

Run: 

Unit tests 

+ 

Integration tests 

+ 

UI/application tests 

# **Step 4** 

Run complete end-to-end tests. 

# **9. Testing Strategy** 

# **Unit tests** 

Test individual components independently: 

- Document loading 

- Chunking 

- Retrieval 

- Prompt construction 

- Safety logic 

- Response processing 

- Application functions 

# **Integration tests** 

Test interactions: 

Retriever → RAG 

RAG → Gemini 

RAG → Safety 

Streamlit → RAG 

# **Safety tests** 

Explicitly test allowed and disallowed categories. 

Safety tests are mandatory and are not optional cleanup work. 

# **Error-handling tests** 

# Test: 

- Missing files 

- Empty queries 

- Invalid configuration 

- API failures 

- Retrieval failures 

- Unexpected model failures 

# **End-to-end tests** 

At least verify: 

User question 

↓ 

Streamlit 

↓ 

Retrieval 

↓ RAG ↓ 

Gemini 

↓ 

Safety 

↓ 

Response 

# **Local pre-merge requirement** 

Before opening a PR, each developer must run: 

1. Their relevant unit tests 

2. The complete test suite 

3. Any integration tests affected by their change 

A PR should not be merged if existing tests are broken without an explicitly documented reason. 

# **10. Git Workflow** 

Keep the Git workflow simple. 

# **Branches** 

Use: 

main 

person1/retrieval 

person2/rag-safety 

person3/streamlit-integration 

Feature-specific branches can be created from each person's workstream if necessary. 

# **Commits** 

Keep commits small and focused. 

Good: 

test: add retrieval tests 

feat: implement document chunking test: add safety refusal cases feat: add Streamlit question input 

Avoid: 

final project changes 

everything done 

updates 

# **Pull requests** 

Each PR should include: 

- What changed 

- Tests added 

- Tests run 

- Any known limitations 

Do not merge code that breaks unrelated tests. 

# **11. Risks and Unknowns** 

These must be resolved from the actual repository/project decisions before final implementation where applicable. 

# **API configuration** 

- Confirm the exact Gemini model and API configuration used by the project. 

- API keys must come from environment configuration. 

- No API key may be committed to Git. 

# **Existing model/vector-store code** 

If the repository contains older or alternative model/vector-store implementations, determine which implementation belongs to the current project before removing or replacing anything. 

Do not silently treat old code as part of the final architecture. 

# **Knowledge-base contents** 

Confirm exactly which documents constitute the approved FinTech knowledge base. 

Only approved project knowledge should be indexed. 

# **Testing framework** 

Confirm the testing framework used by the repository. 

If none exists, the team must agree on one before writing the complete test suite. 

# **Streamlit/application entry point** 

Confirm the intended application entry point if multiple application files exist. 

# **Environment files** 

Any files containing API keys or secrets must be reviewed before committing. 

If an exposed key is real and active, it should be revoked/rotated immediately. 

# **Local model dependencies** 

If the repository contains Ollama or other local-model configuration that conflicts with the current Gemini-based project architecture, the team must determine whether it is legacy code or an intended project dependency before implementation. 

# **12. Final Definition of Done** 

The project is complete only when all of the following are true: 

# **Functionality** 

- Knowledge base is correctly configured. 

- Documents can be loaded. 

- Documents are correctly chunked. 

- Embeddings are generated. 

- Vector storage works. 

- Relevant information can be retrieved. 

- Gemini integration works. 

- RAG responses are generated. 

- Responses are grounded in retrieved information. 

- Streamlit application works. 

- Conversation behavior works as required. 

# **Safety** 

- Financial concepts can be explained. 

- Investment recommendations are rejected. 

- Personalized financial advice is rejected. 

- Payment/transaction execution requests are rejected. 

- Sensitive credentials are never requested. 

- The system never falsely claims to have processed a transaction. 

# **Testing** 

- Unit tests pass. 

- Retrieval tests pass. 

- RAG tests pass. 

- Safety tests pass. 

- Error-handling tests pass. 

- Integration tests pass. 

- End-to-end tests pass. 

# **Security** 

- No API keys are hard-coded. 

- No secrets are committed. 

- Environment variables are used correctly. 

- Sensitive user information is not unnecessarily collected. 

# **Integration** 

- Person 1's retrieval component is integrated. 

- Person 2's RAG/safety component is integrated. 

- Person 3's Streamlit/application component is integrated. 

- Existing functionality still works. 

- No developer's changes break another component. 

# **Documentation** 

- README reflects the final system. 

- Setup instructions are correct. 

- Required environment variables are documented without exposing secrets. 

- Testing instructions are documented. 

- Project limitations are documented. 

# **13. Team Ownership Summary** 

|**Area**|**Person 1**|**Person 2**|**Person 3**|
|---|---|---|---|
|Knowledge base|**Owner**|Support|-|
|Document ingestion|**Owner**|-|-|
|Chunking|**Owner**|-|-|
|Embeddings|**Owner**|-|-|
|Vector store|**Owner**|Support|-|
|Retrieval|**Owner**|Integration|i -|
|Gemini|-|**Owner**|Integration|
|Prompting|-|**Owner**|-|
|RAG chain|Support|**Owner**|Integration|
|Safety logic|Support|**Owner**|UI validation|



|**Area**|**Person 1**|**Person 2**|**Person 3**|
|---|---|---|---|
|Streamlit|-|-|**Owner**|
|Application orchestration|-|Support|**Owner**|
|Error handling|Support|API errors|**Owner**|
|Unit tests|**Owner**|**Owner**|**Owner**|
|Integration tests|Support|Support|**Owner**|
|End-to-end tests|Support|Support|**Owner**|
|Final integration|**Shared**|**Shared**|**Shared**|



The three workstreams are therefore: 

PERSON 1 

Knowledge Base + Retrieval 

│ ▼ PERSON 2 

RAG + Gemini + Safety 

│ ▼ 

PERSON 3 

Streamlit + Application + Integration 

All three people are responsible for both **implementation and testing** of their respective areas. No person is assigned only documentation, only testing, or only UI work. 

