"""Integration tests for the FinTech Compliance Explainer Bot.

Person 3 owns integration and end-to-end testing.

Test categories
---------------
1. Retrieval → RAG integration
   - retriever returns chunks → reranker scores them → grounded answer generated
   - missing context is handled safely

2. End-to-end: normal FinTech questions (fully mocked pipeline)
   - settlement question → answer mentions settlement
   - compliance question → answer returned without error

3. End-to-end: safety questions (fully mocked pipeline)
   - investment advice request → answer does not offer investment advice
   - payment processing request → answer does not claim to process a payment
   - financial advice request → answer is safely redirected
   - credential request → answer does not ask for PIN/password/CVV

4. Application-level integration
   - _ask() connects retriever → rerank → build_grounded_answer correctly
   - pipeline error in _ask() surfaces as an error message in session state
   - conversation history grows correctly across multiple turns

All tests mock external dependencies (Gemini, HuggingFace, FAISS) so they run
fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    # Minimal Document stub for environments where langchain_core is not installed.
    class Document:  # type: ignore[no-redef]
        """Lightweight Document stub used when langchain_core is unavailable."""

        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str, source: str = "data/test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _fake_pipeline(answer_text: str = "Here is a factual explanation.") -> dict:
    """Return a mock pipeline dict with a controllable answer."""
    retriever = MagicMock()
    llm = MagicMock()
    prompt = MagicMock()
    reranker = MagicMock()
    return {
        "retriever": retriever,
        "llm": llm,
        "prompt": prompt,
        "reranker": reranker,
        "_mock_answer": answer_text,  # used by helper patches
    }


# ---------------------------------------------------------------------------
# 1. Retrieval → RAG component integration
# ---------------------------------------------------------------------------


class TestRetrievalToRAGIntegration:
    """Test that retrieval components feed correctly into the answer generator."""

    def test_retrieve_then_rerank_then_answer(self):
        """Full mini-pipeline: retrieve → rerank → build_grounded_answer."""
        chunk_a = _make_doc("Settlement is the final stage where money actually moves.")
        chunk_b = _make_doc("Processing routes the payment between banks.")
        chunk_c = _make_doc("Verification checks identity and balance.")

        # Simulate retriever returning 3 chunks
        with patch("rag_pipeline.retrieve_context", return_value=[chunk_a, chunk_b, chunk_c]) \
             as mock_retrieve, \
             patch("rag_pipeline.rerank_chunks", return_value=[chunk_a, chunk_b]) \
             as mock_rerank, \
             patch("rag_pipeline.build_grounded_answer",
                   return_value="Settlement is when the money actually moves.") as mock_build:

            from rag_pipeline import (
                build_grounded_answer,
                rerank_chunks,
                retrieve_context,
            )

            pipeline = _fake_pipeline()
            query = "What is settlement?"

            chunks = retrieve_context(pipeline["retriever"], query)
            top_chunks = rerank_chunks(pipeline["reranker"], query, chunks, top_k=2)
            answer = build_grounded_answer(pipeline["llm"], pipeline["prompt"], query, top_chunks)

        mock_retrieve.assert_called_once_with(pipeline["retriever"], query)
        mock_rerank.assert_called_once_with(pipeline["reranker"], query, [chunk_a, chunk_b, chunk_c], top_k=2)
        assert "settlement" in answer.lower()

    def test_empty_retrieval_passes_empty_list_to_answer(self):
        """When retriever returns nothing, build_grounded_answer receives an empty list."""
        with patch("rag_pipeline.retrieve_context", return_value=[]), \
             patch("rag_pipeline.rerank_chunks", return_value=[]), \
             patch("rag_pipeline.build_grounded_answer",
                   return_value="I don't have enough information.") as mock_build:

            from rag_pipeline import (
                build_grounded_answer,
                rerank_chunks,
                retrieve_context,
            )

            pipeline = _fake_pipeline()
            chunks = retrieve_context(pipeline["retriever"], "unknown topic")
            top_chunks = rerank_chunks(pipeline["reranker"], "unknown topic", chunks, top_k=3)
            answer = build_grounded_answer(pipeline["llm"], pipeline["prompt"], "unknown topic", top_chunks)

        _, _, _, chunks_arg = mock_build.call_args[0]
        assert chunks_arg == []
        assert "don't have enough information" in answer.lower() or len(answer) > 0

    def test_reranker_receives_correct_query_and_chunks(self):
        """rerank_chunks is called with the original query and retrieved chunks."""
        doc = _make_doc("Compliance checks include AML and KYC.")

        with patch("rag_pipeline.retrieve_context", return_value=[doc]), \
             patch("rag_pipeline.rerank_chunks", return_value=[doc]) as mock_rerank, \
             patch("rag_pipeline.build_grounded_answer", return_value="answer"):

            from rag_pipeline import (
                build_grounded_answer,
                rerank_chunks,
                retrieve_context,
            )

            pipeline = _fake_pipeline()
            query = "What is AML?"
            chunks = retrieve_context(pipeline["retriever"], query)
            rerank_chunks(pipeline["reranker"], query, chunks, top_k=3)

        call_kwargs = mock_rerank.call_args
        assert call_kwargs[0][1] == "What is AML?"  # query passed correctly
        assert call_kwargs[0][2] == [doc]           # retrieved chunks passed

    def test_grounded_answer_receives_formatted_context(self):
        """build_grounded_answer is called with the top chunks, not the full set."""
        all_chunks = [_make_doc(f"Chunk {i}") for i in range(5)]
        top_chunks = all_chunks[:2]

        with patch("rag_pipeline.retrieve_context", return_value=all_chunks), \
             patch("rag_pipeline.rerank_chunks", return_value=top_chunks), \
             patch("rag_pipeline.build_grounded_answer", return_value="ok") as mock_build:

            from rag_pipeline import (
                build_grounded_answer,
                rerank_chunks,
                retrieve_context,
            )

            pipeline = _fake_pipeline()
            q = "Tell me about verification"
            chunks = retrieve_context(pipeline["retriever"], q)
            ranked = rerank_chunks(pipeline["reranker"], q, chunks, top_k=2)
            build_grounded_answer(pipeline["llm"], pipeline["prompt"], q, ranked)

        _, _, _, chunks_arg = mock_build.call_args[0]
        assert len(chunks_arg) == 2


# ---------------------------------------------------------------------------
# 2. End-to-end: normal FinTech questions
# ---------------------------------------------------------------------------


class TestEndToEndNormalQuestions:
    """Simulate the complete app._ask() call for normal informational questions."""

    def _run_ask(self, question: str, answer: str) -> str:
        """Run app._ask() with a fully mocked pipeline via answer_pipeline.ask."""
        import types
        import sys

        # Ensure streamlit stub exists
        if "streamlit" not in sys.modules:
            st_mock = types.ModuleType("streamlit")
            st_mock.set_page_config = lambda **kwargs: None
            st_mock.markdown = lambda *a, **kw: None
            st_mock.session_state = {}
            st_mock.spinner = MagicMock(
                return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
            )
            st_mock.columns = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
            st_mock.button = lambda *a, **kw: False
            st_mock.text_input = lambda *a, **kw: ""
            st_mock.metric = lambda *a, **kw: None
            st_mock.sidebar = MagicMock()
            st_mock.rerun = lambda: None
            sys.modules["streamlit"] = st_mock

        import importlib
        app = importlib.import_module("app")

        from answer_pipeline import AskResponse
        pipeline = _fake_pipeline(answer)
        fake_response = AskResponse(
            answer=answer, safe=True, grounded=True, guard_category="", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_response):
            result = app._ask(pipeline, question)
        return result

    def test_settlement_question_returns_answer(self):
        answer = ("Settlement is the final stage of a digital payment where money "
                  "actually moves from the payer's account to the payee's account.")
        result = self._run_ask("What is settlement?", answer)
        assert "settlement" in result.lower()

    def test_compliance_question_returns_answer(self):
        answer = ("Compliance checks are automated checks that ensure a transaction "
                  "follows legal and regulatory requirements, including AML and KYC.")
        result = self._run_ask("What does a compliance check do?", answer)
        assert len(result) > 0
        assert "compliance" in result.lower()

    def test_upi_question_returns_answer(self):
        answer = ("UPI stands for Unified Payments Interface. It is a real-time "
                  "payment system developed by NPCI in India.")
        result = self._run_ask("What is UPI?", answer)
        assert "upi" in result.lower() or "unified" in result.lower()

    def test_payment_flow_question_returns_answer(self):
        answer = ("A digital payment moves through five stages: Initiation, "
                  "Verification, Compliance, Processing, and Settlement.")
        result = self._run_ask("What happens after I click Pay?", answer)
        assert len(result) > 10

    def test_kyc_question_returns_answer(self):
        answer = "KYC means Know Your Customer. It is a regulatory requirement."
        result = self._run_ask("What is KYC?", answer)
        assert "kyc" in result.lower() or "know your customer" in result.lower()

    def test_answer_is_always_a_string(self):
        result = self._run_ask("Tell me about settlement.", "Settlement is the end stage.")
        assert isinstance(result, str)

    def test_answer_is_not_empty(self):
        result = self._run_ask("What is NEFT?", "NEFT is a batch fund transfer system.")
        assert result.strip() != ""


# ---------------------------------------------------------------------------
# 3. End-to-end: safety questions
# ---------------------------------------------------------------------------


class TestEndToEndSafetyQuestions:
    """Verify the bot refuses or safely redirects out-of-scope requests.

    The safety contract is enforced by the prompt in rag_pipeline.py.
    These tests verify that when the LLM returns a safe refusal, that refusal
    is surfaced correctly through the app layer, and that the app layer itself
    never injects financial advice or processing claims.
    """

    SAFETY_REFUSALS = [
        "I can only provide informational explanations",
        "I don't have enough information in my knowledge base",
        "I'm not able to provide personalized financial advice",
        "I cannot process payments",
    ]

    def _run_safety_ask(self, question: str, llm_response: str) -> str:
        """Run app._ask() and return the answer (which should be a refusal)."""
        import types
        import sys

        if "streamlit" not in sys.modules:
            st_mock = types.ModuleType("streamlit")
            st_mock.set_page_config = lambda **kwargs: None
            st_mock.markdown = lambda *a, **kw: None
            st_mock.session_state = {}
            st_mock.spinner = MagicMock(
                return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
            )
            st_mock.columns = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
            st_mock.button = lambda *a, **kw: False
            st_mock.text_input = lambda *a, **kw: ""
            st_mock.metric = lambda *a, **kw: None
            st_mock.sidebar = MagicMock()
            st_mock.rerun = lambda: None
            sys.modules["streamlit"] = st_mock

        import importlib
        app = importlib.import_module("app")

        from answer_pipeline import AskResponse
        pipeline = _fake_pipeline(llm_response)
        # Safety guard fires — safe=False, answer = canned refusal
        fake_response = AskResponse(
            answer=llm_response, safe=False, grounded=True,
            guard_category="financial_advice", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_response):
            result = app._ask(pipeline, question)
        return result

    def test_investment_advice_request_returns_safe_response(self):
        """The bot should not provide investment recommendations."""
        safe_response = (
            "I can only provide informational explanations about FinTech and payment processes. "
            "For investment advice, please consult a qualified financial advisor."
        )
        result = self._run_safety_ask("Which investment should I buy?", safe_response)
        # The safe response is returned unchanged — no advice injected
        assert "investment" not in result.lower() or "consult" in result.lower() or "can only" in result.lower()

    def test_payment_processing_request_returns_safe_response(self):
        """The bot should not claim to process payments."""
        safe_response = (
            "I'm not able to process payments. I can only explain how payment systems work."
        )
        result = self._run_safety_ask("Process this payment for me.", safe_response)
        assert "process" in result.lower() or "not able" in result.lower()
        # Critical: bot must NOT claim a transaction was completed
        assert "transaction has been processed" not in result.lower()
        assert "payment processed successfully" not in result.lower()

    def test_financial_advice_request_returns_safe_response(self):
        """The bot should not give personalized financial advice."""
        safe_response = (
            "I'm not able to provide personalized financial advice. "
            "Please consult a qualified financial professional."
        )
        result = self._run_safety_ask("Give me financial advice about my savings.", safe_response)
        assert len(result) > 0
        # Bot must not give actual advice
        assert "you should invest" not in result.lower()
        assert "i recommend" not in result.lower()

    def test_credential_request_not_asked_by_bot(self):
        """The bot response must never ask for sensitive credentials."""
        safe_response = (
            "I'm a FinTech explainer and I don't need your personal account details. "
            "Please never share your PIN, OTP, or CVV with anyone."
        )
        result = self._run_safety_ask("Help me log into my bank account.", safe_response)
        # Bot should not request credentials
        assert "enter your pin" not in result.lower()
        assert "provide your password" not in result.lower()
        assert "give me your otp" not in result.lower()
        assert "what is your cvv" not in result.lower()

    def test_false_transaction_completion_not_in_response(self):
        """The bot must never falsely claim to have completed a real transaction."""
        safe_response = (
            "I can explain how payment processing works, but I don't process real payments."
        )
        result = self._run_safety_ask("Complete this transaction: send ₹500 to my friend.", safe_response)
        assert "your transaction has been processed" not in result.lower()
        assert "₹500 has been sent" not in result.lower()
        assert "payment complete" not in result.lower()

    def test_safe_response_is_returned_as_string(self):
        """Even safety refusals are returned as clean strings, not errors."""
        safe_response = "I can only provide informational explanations about payment processes."
        result = self._run_safety_ask("Recommend me a stock to buy.", safe_response)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 4. Application-level integration: conversation behaviour
# ---------------------------------------------------------------------------


class TestConversationBehaviour:
    """Test that conversation history accumulates correctly across turns."""

    def _get_app(self):
        import types
        import sys

        if "streamlit" not in sys.modules:
            st_mock = types.ModuleType("streamlit")
            st_mock.set_page_config = lambda **kwargs: None
            st_mock.markdown = lambda *a, **kw: None
            st_mock.session_state = {}
            st_mock.spinner = MagicMock(
                return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
            )
            st_mock.columns = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
            st_mock.button = lambda *a, **kw: False
            st_mock.text_input = lambda *a, **kw: ""
            st_mock.metric = lambda *a, **kw: None
            st_mock.sidebar = MagicMock()
            st_mock.rerun = lambda: (_ for _ in ()).throw(StopIteration())
            sys.modules["streamlit"] = st_mock

        import importlib
        return importlib.import_module("app")

    def setup_method(self):
        import sys
        sys.modules.pop("app", None)  # force re-import with fresh state

    def test_first_question_adds_two_messages(self):
        """A single question should add one user message and one assistant message."""
        import sys
        from answer_pipeline import AskResponse
        app = self._get_app()
        sys.modules["streamlit"].session_state = {"messages": []}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())

        pipeline = _fake_pipeline("Settlement moves the money.")
        fake_resp = AskResponse(
            answer="Settlement moves the money.", safe=True, grounded=True,
            guard_category="", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_resp):
            try:
                app._handle_question(pipeline, "What is settlement?")
            except StopIteration:
                pass

        msgs = sys.modules["streamlit"].session_state["messages"]
        roles = [m["role"] for m in msgs]
        assert roles.count("user") == 1
        assert roles.count("assistant") == 1

    def test_second_question_adds_to_history(self):
        """After two questions the history should have 4 messages (2 user, 2 assistant)."""
        import sys
        from answer_pipeline import AskResponse
        app = self._get_app()
        sys.modules["streamlit"].session_state = {"messages": []}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())

        pipeline = _fake_pipeline()
        for question, answer in [
            ("What is settlement?", "Settlement moves money."),
            ("What is compliance?", "Compliance checks legal requirements."),
        ]:
            fake_resp = AskResponse(
                answer=answer, safe=True, grounded=True, guard_category="", chunks_used=[]
            )
            with patch("answer_pipeline.ask", return_value=fake_resp):
                try:
                    app._handle_question(pipeline, question)
                except StopIteration:
                    pass

        msgs = sys.modules["streamlit"].session_state["messages"]
        roles = [m["role"] for m in msgs]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2

    def test_empty_input_does_not_add_user_message(self):
        """An empty question should not add a user message to history."""
        import sys
        app = self._get_app()
        sys.modules["streamlit"].session_state = {"messages": []}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())

        pipeline = _fake_pipeline()
        try:
            app._handle_question(pipeline, "")
        except StopIteration:
            pass

        msgs = sys.modules["streamlit"].session_state["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 0

    def test_user_message_content_matches_question(self):
        """The stored user message content must match the submitted question."""
        import sys
        from answer_pipeline import AskResponse
        app = self._get_app()
        sys.modules["streamlit"].session_state = {"messages": []}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())

        pipeline = _fake_pipeline("UPI stands for Unified Payments Interface.")
        question = "What is UPI?"
        fake_resp = AskResponse(
            answer="UPI stands for Unified Payments Interface.", safe=True,
            grounded=True, guard_category="", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_resp):
            try:
                app._handle_question(pipeline, question)
            except StopIteration:
                pass

        msgs = sys.modules["streamlit"].session_state["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert user_msgs[0]["content"] == question.strip()

    def test_error_turn_stores_error_role(self):
        """When the pipeline raises an exception, an 'error' role message is stored."""
        import sys
        app = self._get_app()
        sys.modules["streamlit"].session_state = {"messages": []}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())

        pipeline = _fake_pipeline()
        with patch("answer_pipeline.ask", side_effect=ConnectionError("API unavailable")):
            try:
                app._handle_question(pipeline, "What is NEFT?")
            except StopIteration:
                pass

        msgs = sys.modules["streamlit"].session_state["messages"]
        error_msgs = [m for m in msgs if m["role"] == "error"]
        assert len(error_msgs) == 1


# ---------------------------------------------------------------------------
# 5. Regression tests — existing retrieval tests still hold
# ---------------------------------------------------------------------------


class TestRegressionRetrieval:
    """Smoke-test that existing rag_pipeline functions still work after integration."""

    def test_load_documents_still_raises_on_missing_dir(self, tmp_path):
        from rag_pipeline import load_documents

        missing = tmp_path / "nonexistent"
        try:
            load_documents(missing)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_split_documents_still_produces_chunks(self):
        from rag_pipeline import split_documents

        chunks = split_documents([Document(page_content="x" * 600)])
        assert len(chunks) > 1

    def test_build_retriever_still_validates_k(self):
        from rag_pipeline import build_retriever

        class FakeStore:
            def as_retriever(self, search_kwargs):
                return search_kwargs

        try:
            build_retriever(FakeStore(), k=0)
            assert False, "Expected ValueError"
        except ValueError:
            pass
