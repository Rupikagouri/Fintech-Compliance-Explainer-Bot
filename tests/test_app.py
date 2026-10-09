"""Unit tests for the Person 3 application layer (app.py).

Person 3 owns testing of:
- Application startup helpers
- Empty / whitespace input validation
- Session-state initialisation
- Pipeline error message formatting
- Chat message handling (user / assistant / error paths)
- _ask() routing through the pipeline components
- _is_empty() utility

These tests use mocks and monkeypatching so no real models, files, or API
calls are needed.  They can run offline and without any installed ML packages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — import only the pure-Python functions we can test without Streamlit
# ---------------------------------------------------------------------------


def _import_helpers():
    """Import the testable helpers from app.py.

    We import inside a function so that the module-level st.set_page_config()
    call does not execute during collection (it would fail outside a running
    Streamlit server).  We mock streamlit before importing.
    """
    import sys
    import types

    # Build a minimal streamlit stub so app.py can be imported in plain pytest.
    if "streamlit" not in sys.modules:
        st_mock = types.ModuleType("streamlit")
        st_mock.set_page_config = lambda **kwargs: None
        st_mock.markdown = lambda *a, **kw: None
        st_mock.session_state = {}
        st_mock.spinner = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
        st_mock.columns = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        st_mock.button = lambda *a, **kw: False
        st_mock.text_input = lambda *a, **kw: ""
        st_mock.metric = lambda *a, **kw: None
        st_mock.sidebar = MagicMock()
        st_mock.rerun = lambda: None
        sys.modules["streamlit"] = st_mock

    import importlib

    app = importlib.import_module("app")
    return app


# ---------------------------------------------------------------------------
# _is_empty
# ---------------------------------------------------------------------------


class TestIsEmpty:
    def setup_method(self):
        self.app = _import_helpers()

    def test_empty_string_is_empty(self):
        assert self.app._is_empty("") is True

    def test_whitespace_only_is_empty(self):
        assert self.app._is_empty("   ") is True
        assert self.app._is_empty("\t\n  ") is True

    def test_none_is_empty(self):
        assert self.app._is_empty(None) is True

    def test_normal_question_is_not_empty(self):
        assert self.app._is_empty("What is settlement?") is False

    def test_single_character_is_not_empty(self):
        assert self.app._is_empty("a") is False

    def test_leading_trailing_whitespace_not_empty(self):
        # Non-empty despite whitespace padding
        assert self.app._is_empty("  hello  ") is False


# ---------------------------------------------------------------------------
# _pipeline_error_message
# ---------------------------------------------------------------------------


class TestPipelineErrorMessage:
    def setup_method(self):
        self.app = _import_helpers()

    def test_api_key_missing_message(self):
        exc = EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        msg = self.app._pipeline_error_message(exc)
        assert "API key" in msg or "GEMINI_API_KEY" in msg

    def test_no_txt_files_message(self):
        exc = ValueError("No .txt files found in data directory: /some/path")
        msg = self.app._pipeline_error_message(exc)
        assert "knowledge base" in msg.lower() or "data" in msg.lower()

    def test_data_dir_not_found_message(self):
        exc = FileNotFoundError("Data directory not found: /missing")
        msg = self.app._pipeline_error_message(exc)
        assert "knowledge base" in msg.lower() or "not found" in msg.lower()

    def test_import_error_message(self):
        exc = ImportError("langchain_google_genai is not installed")
        msg = self.app._pipeline_error_message(exc)
        assert "dependency" in msg.lower() or "install" in msg.lower()

    def test_generic_error_still_returns_string(self):
        exc = RuntimeError("something exploded")
        msg = self.app._pipeline_error_message(exc)
        assert isinstance(msg, str)
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# _init_chat_history
# ---------------------------------------------------------------------------


class TestInitChatHistory:
    def setup_method(self):
        self.app = _import_helpers()

    def test_creates_messages_key_when_absent(self):
        import sys
        sys.modules["streamlit"].session_state = {}
        self.app._init_chat_history()
        assert "messages" in sys.modules["streamlit"].session_state

    def test_messages_initialised_to_empty_list(self):
        import sys
        sys.modules["streamlit"].session_state = {}
        self.app._init_chat_history()
        assert sys.modules["streamlit"].session_state["messages"] == []

    def test_does_not_overwrite_existing_messages(self):
        import sys
        existing = [{"role": "user", "content": "Hello"}]
        sys.modules["streamlit"].session_state = {"messages": existing}
        self.app._init_chat_history()
        assert sys.modules["streamlit"].session_state["messages"] is existing


# ---------------------------------------------------------------------------
# _ask
# ---------------------------------------------------------------------------


class TestAsk:
    """Tests for the _ask() routing function using a fully mocked pipeline."""

    def setup_method(self):
        self.app = _import_helpers()

    def _make_pipeline(self, answer: str = "Settlement is the final stage.") -> dict:
        pipeline = {
            "retriever": MagicMock(),
            "llm": MagicMock(),
            "prompt": MagicMock(),
            "reranker": MagicMock(),
        }
        return pipeline, answer

    def test_ask_returns_answer_string(self):
        """_ask() returns the answer text from answer_pipeline.ask()."""
        from answer_pipeline import AskResponse
        pipeline, expected = self._make_pipeline("Settlement is the final stage.")
        fake_response = AskResponse(
            answer=expected, safe=True, grounded=True, guard_category="", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_response):
            result = self.app._ask(pipeline, "What is settlement?")
        assert result == expected

    def test_ask_calls_answer_pipeline_ask(self):
        """_ask() delegates to answer_pipeline.ask with the question and pipeline."""
        from answer_pipeline import AskResponse
        pipeline, _ = self._make_pipeline()
        fake_response = AskResponse(
            answer="KYC is Know Your Customer.", safe=True,
            grounded=True, guard_category="", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_response) as mock_ask:
            self.app._ask(pipeline, "What is KYC?")
        mock_ask.assert_called_once_with("What is KYC?", pipeline)

    def test_ask_returns_safe_refusal_for_blocked_question(self):
        """When a safety guard fires, the canned refusal text is returned."""
        from answer_pipeline import AskResponse
        pipeline, _ = self._make_pipeline()
        refusal = "I can only provide informational explanations about FinTech."
        fake_response = AskResponse(
            answer=refusal, safe=False, grounded=True,
            guard_category="unsupported_topic", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_response):
            result = self.app._ask(pipeline, "Which stock should I buy?")
        assert result == refusal

    def test_ask_propagates_pipeline_exception(self):
        """_ask() lets unexpected exceptions propagate to the caller."""
        pipeline, _ = self._make_pipeline()
        with patch("answer_pipeline.ask", side_effect=RuntimeError("LLM down")):
            try:
                self.app._ask(pipeline, "Q")
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "LLM down" in str(exc)


# ---------------------------------------------------------------------------
# _handle_question — empty input path
# ---------------------------------------------------------------------------


class TestHandleQuestion:
    """Tests for _handle_question() focusing on empty-input and error paths."""

    def setup_method(self):
        self.app = _import_helpers()
        import sys
        sys.modules["streamlit"].session_state = {"messages": []}
        sys.modules["streamlit"].rerun = lambda: None

    def test_empty_question_appends_error_message(self):
        import sys
        pipeline = {"retriever": MagicMock(), "llm": MagicMock(),
                    "prompt": MagicMock(), "reranker": MagicMock()}
        # Patch rerun so we don't actually restart
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())
        try:
            self.app._handle_question(pipeline, "   ")
        except StopIteration:
            pass
        msgs = sys.modules["streamlit"].session_state["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "error"

    def test_whitespace_only_question_appends_error(self):
        import sys
        pipeline = {"retriever": MagicMock(), "llm": MagicMock(),
                    "prompt": MagicMock(), "reranker": MagicMock()}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())
        try:
            self.app._handle_question(pipeline, "\t\n ")
        except StopIteration:
            pass
        msgs = sys.modules["streamlit"].session_state["messages"]
        assert msgs[0]["role"] == "error"

    def test_valid_question_appends_user_then_assistant(self):
        import sys
        from answer_pipeline import AskResponse
        pipeline = {"retriever": MagicMock(), "llm": MagicMock(),
                    "prompt": MagicMock(), "reranker": MagicMock()}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())
        fake_resp = AskResponse(
            answer="Settlement is...", safe=True, grounded=True,
            guard_category="", chunks_used=[]
        )
        with patch("answer_pipeline.ask", return_value=fake_resp):
            try:
                self.app._handle_question(pipeline, "What is settlement?")
            except StopIteration:
                pass
        msgs = sys.modules["streamlit"].session_state["messages"]
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_pipeline_failure_appends_error_message(self):
        import sys
        pipeline = {"retriever": MagicMock(), "llm": MagicMock(),
                    "prompt": MagicMock(), "reranker": MagicMock()}
        sys.modules["streamlit"].rerun = lambda: (_ for _ in ()).throw(StopIteration())
        with patch("answer_pipeline.ask", side_effect=RuntimeError("API down")):
            try:
                self.app._handle_question(pipeline, "What is UPI?")
            except StopIteration:
                pass
        msgs = sys.modules["streamlit"].session_state["messages"]
        roles = [m["role"] for m in msgs]
        assert "error" in roles
        assert "user" in roles  # user message still stored


# ---------------------------------------------------------------------------
# _init_pipeline — session caching
# ---------------------------------------------------------------------------


class TestInitPipeline:
    def setup_method(self):
        self.app = _import_helpers()
        import sys
        # Clean session state before each test
        sys.modules["streamlit"].session_state = {}

    def test_returns_cached_pipeline_on_second_call(self):
        import sys
        fake_pipeline = {"retriever": "r", "llm": "l", "prompt": "p", "reranker": "re"}
        sys.modules["streamlit"].session_state = {"pipeline": fake_pipeline}
        result = self.app._init_pipeline()
        assert result is fake_pipeline

    def test_returns_none_when_prior_error_cached(self):
        import sys
        sys.modules["streamlit"].session_state = {"pipeline_error": "API key missing"}
        result = self.app._init_pipeline()
        assert result is None

    def test_caches_pipeline_on_success(self):
        """_init_pipeline() caches a successfully built pipeline in session_state.

        We stub the entire rag_pipeline module so faiss is never imported.
        """
        import sys
        import types

        # Build a minimal rag_pipeline stub to avoid importing the real module
        # (which has a top-level ``import faiss`` that fails in this environment).
        fake_pipeline = {"retriever": "r", "llm": "l", "prompt": "p", "reranker": "re"}
        rp_stub = types.ModuleType("rag_pipeline")
        rp_stub.build_pipeline = MagicMock(return_value=fake_pipeline)
        # Temporarily replace rag_pipeline in sys.modules
        original_rp = sys.modules.get("rag_pipeline")
        sys.modules["rag_pipeline"] = rp_stub

        try:
            # Clear session state so _init_pipeline attempts to build
            sys.modules["streamlit"].session_state = {}

            # Reload app to pick up the stubbed rag_pipeline
            import importlib
            app = importlib.import_module("app")

            result = app._init_pipeline()

            # The pipeline should have been built and cached
            assert result == fake_pipeline
            assert sys.modules["streamlit"].session_state.get("pipeline") == fake_pipeline
        finally:
            # Restore the original module (or remove stub if none existed)
            if original_rp is not None:
                sys.modules["rag_pipeline"] = original_rp
            else:
                sys.modules.pop("rag_pipeline", None)
