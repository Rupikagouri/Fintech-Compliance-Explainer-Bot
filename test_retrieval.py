import pytest
from langchain_core.documents import Document

from rag_pipeline import (
    build_retriever,
    load_documents,
    split_documents,
)


def test_load_documents_returns_documents(tmp_path):
    (tmp_path / "sample.txt").write_text("A sample knowledge document.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_load_documents_raises_file_not_found_with_path(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="missing"):
        load_documents(missing)


def test_load_documents_raises_value_error_when_empty(tmp_path):
    with pytest.raises(ValueError, match="No \.txt"):
        load_documents(tmp_path)


def test_split_documents_produces_multiple_chunks():
    chunks = split_documents([Document(page_content="x" * 600)])
    assert len(chunks) > 1


def test_split_chunks_do_not_exceed_500_characters():
    chunks = split_documents([Document(page_content="x" * 1200)])
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)


def test_split_assigns_sequential_chunk_ids():
    chunks = split_documents([Document(page_content="x" * 1200)])
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))


def test_build_retriever_returns_results_for_known_query():
    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            self.search_kwargs = search_kwargs
            return "retriever"

    store = FakeVectorStore()
    assert build_retriever(store) == "retriever"
    assert store.search_kwargs == {"k": 5}


def test_retriever_accepts_custom_k():
    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            return search_kwargs

    assert build_retriever(FakeVectorStore(), k=2) == {"k": 2}
