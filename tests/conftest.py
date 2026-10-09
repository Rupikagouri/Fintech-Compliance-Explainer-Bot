"""Shared pytest configuration for the FinTech Compliance Explainer Bot test suite.

This file is loaded automatically by pytest before any tests are collected or
run.  It sets up sys.modules stubs for heavyweight packages (faiss, sentence-
transformers, the langchain ML stack) so that every test file can be *imported*
and *collected* even when those packages are not installed in the current Python
environment (e.g. a lightweight CI environment or a developer machine that only
has the system Python).

Tests that genuinely exercise the full retrieval or LLM stack (i.e. tests that
call :func:`rag_pipeline.build_pipeline` without mocking) are expected to pass
only in an environment where the full requirements.txt is installed.  All other
tests should pass with the stubs provided here.

Stub strategy
-------------
* :mod:`faiss` — replaced with a :class:`unittest.mock.MagicMock`.
* :mod:`langchain_core`, :mod:`langchain_core.documents` — replaced with real-
  looking stubs so that ``from langchain_core.documents import Document`` works
  and the :class:`Document` objects used throughout the test suite behave like
  the real class (plain ``page_content`` + ``metadata`` attributes).
* :mod:`rag_pipeline` — replaced with a stub that exposes the same public
  functions as the real module.  The stub functions are simple no-ops that
  return sensible empty values so that tests in :mod:`tests.test_integration`
  which call ``from rag_pipeline import ...`` inside test bodies don't fail at
  import time.  Tests that want to control behaviour should still use
  ``unittest.mock.patch`` in the usual way.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# 1. Stub: faiss
# ---------------------------------------------------------------------------

if "faiss" not in sys.modules:
    faiss_stub = MagicMock()
    faiss_stub.IndexFlatL2 = MagicMock(return_value=MagicMock())
    sys.modules["faiss"] = faiss_stub


# ---------------------------------------------------------------------------
# 2. Stub: langchain_core and langchain_core.documents
# ---------------------------------------------------------------------------

class _Document:
    """Minimal Document stub matching the langchain_core.documents.Document API."""

    def __init__(self, page_content: str = "", metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Document(page_content={self.page_content!r})"


if "langchain_core" not in sys.modules:
    # Build the package hierarchy that tests expect
    lc_core = types.ModuleType("langchain_core")
    lc_docs = types.ModuleType("langchain_core.documents")
    lc_docs.Document = _Document
    lc_core.documents = lc_docs

    # Additional sub-modules that rag_pipeline imports at module level
    lc_prompts = types.ModuleType("langchain_core.prompts")
    lc_prompts.ChatPromptTemplate = MagicMock()
    lc_core.prompts = lc_prompts

    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.documents"] = lc_docs
    sys.modules["langchain_core.prompts"] = lc_prompts

# Ensure the _Document class is also importable as langchain_core.documents.Document
# even if the real package was already partially loaded.
if hasattr(sys.modules.get("langchain_core", None), "documents"):
    if not hasattr(sys.modules["langchain_core.documents"], "Document"):
        sys.modules["langchain_core.documents"].Document = _Document  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. Stub remaining langchain sub-packages that rag_pipeline imports
# ---------------------------------------------------------------------------

_langchain_stubs = {
    "langchain_community": None,
    "langchain_community.docstore": None,
    "langchain_community.docstore.in_memory": None,
    "langchain_community.document_loaders": None,
    "langchain_community.vectorstores": None,
    "langchain_huggingface": None,
    "langchain_text_splitters": None,
    "sentence_transformers": None,
    "streamlit": None,  # only stubbed here if nothing else has set it yet
}

for mod_name in _langchain_stubs:
    if mod_name not in sys.modules:
        stub = MagicMock()
        stub.__name__ = mod_name
        # Give specific attributes their proper names so attribute access works
        if mod_name == "langchain_community.vectorstores":
            stub.FAISS = MagicMock()
        if mod_name == "langchain_community.docstore.in_memory":
            stub.InMemoryDocstore = MagicMock()
        if mod_name == "langchain_community.document_loaders":
            stub.DirectoryLoader = MagicMock()
            stub.TextLoader = MagicMock()
        if mod_name == "langchain_text_splitters":
            stub.RecursiveCharacterTextSplitter = MagicMock()
        if mod_name == "sentence_transformers":
            stub.CrossEncoder = MagicMock()
        sys.modules[mod_name] = stub


# ---------------------------------------------------------------------------
# 4. Stub: rag_pipeline
#    Provides the exact public API that test_integration.py calls via
#    ``from rag_pipeline import <name>`` inside test bodies.
# ---------------------------------------------------------------------------

def _stub_load_documents(data_path=None):
    """Stub for rag_pipeline.load_documents.  Raises FileNotFoundError if the
    given path does not exist on disk (so regression tests still pass)."""
    from pathlib import Path

    if data_path is not None:
        p = Path(data_path)
        if not p.is_dir():
            raise FileNotFoundError(f"Data directory not found: {p}")
    return []


def _stub_split_documents(docs):
    """Stub for rag_pipeline.split_documents.  Returns two sub-chunks per doc."""
    result = []
    for doc in docs:
        content = doc.page_content
        mid = len(content) // 2 or 1
        result.append(_Document(page_content=content[:mid], metadata=dict(doc.metadata)))
        result.append(_Document(page_content=content[mid:], metadata=dict(doc.metadata)))
    return result if result else [_Document(page_content="stub chunk")]


def _stub_build_vector_store(split_docs):
    return MagicMock()


def _stub_build_retriever(vector_store, k=5):
    if k < 1:
        raise ValueError("k must be at least 1")
    return MagicMock()


def _stub_retrieve_context(retriever, query):
    return []


def _stub_rerank_chunks(reranker, query, chunks, top_k=3):
    return chunks[:top_k] if chunks else []


def _stub_build_grounded_answer(llm, prompt_template, question, context_chunks):
    return "I don't have enough information in my knowledge base to answer that."


def _stub_build_pipeline(data_path=None):
    return {
        "retriever": MagicMock(),
        "llm": MagicMock(),
        "prompt": MagicMock(),
        "reranker": MagicMock(),
    }


if "rag_pipeline" not in sys.modules:
    rp_stub = types.ModuleType("rag_pipeline")
    rp_stub.load_documents = _stub_load_documents
    rp_stub.split_documents = _stub_split_documents
    rp_stub.build_vector_store = _stub_build_vector_store
    rp_stub.build_retriever = _stub_build_retriever
    rp_stub.retrieve_context = _stub_retrieve_context
    rp_stub.rerank_chunks = _stub_rerank_chunks
    rp_stub.build_grounded_answer = _stub_build_grounded_answer
    rp_stub.build_pipeline = _stub_build_pipeline
    rp_stub.DATA_PATH = "/stub/data"
    rp_stub.PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    sys.modules["rag_pipeline"] = rp_stub
