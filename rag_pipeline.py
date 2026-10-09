"""Document ingestion and RAG pipeline for the FinTech Compliance Explainer Bot.

Importing this module does not read documents or load / download machine-learning
models.  Call :func:`build_pipeline` when the application starts to load the
knowledge base and initialise all components.

LLM backend: Google Gemini Flash (``gemini-1.5-flash``).
Embeddings  : HuggingFace ``sentence-transformers/all-MiniLM-L6-v2`` (local, free).
Vector store: FAISS (in-memory).
Reranker    : Cross-encoder ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (local, free).

Environment variable required
------------------------------
``GEMINI_API_KEY`` – your Google AI Studio API key.
"""

from __future__ import annotations

import os
import hashlib
import re
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

DATA_PATH = Path(__file__).resolve().parent / "data"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base" / "india_fintech"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Gemini model name (Gemini 1.5 Flash is free-tier accessible)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are an Indian fintech systems explainer. Answer in clear English, adjusting depth to the question.

RULES:
- Use supplied sources as evidence. Explain stable technical concepts, but mark them as general practice when unsourced.
- For Indian legal, regulatory, limit, fee, or deadline claims, use sourced evidence and state its effective/verification date. If evidence is missing or may be stale, say so.
- Explain participants, sequence, exchanged data, message states, money movement, accounting, failures, recovery, and security as relevant.
- Synthesize multiple documents when needed. Distinguish mandatory regulation from illustrative architecture and industry practice.
- Cite supporting documents inline using supplied labels, e.g. [Source: knowledge_base/india_fintech/02_payment_rails.txt]. Never invent sources.
- If context does not support an important part of the answer, identify the gap instead of guessing.
- Stay within Indian fintech systems and workflows. Do not provide personalized financial, legal, tax, insurance, or investment advice.
- Never claim to inspect a user's account or process a real transaction. Never request OTPs, PINs, CVVs, passwords, or complete credentials.

Context:
{context}

Question:
{question}

Answer:"""

# ---------------------------------------------------------------------------
# Document loading and splitting
# ---------------------------------------------------------------------------


def load_documents(data_path: str | Path = DATA_PATH) -> list[Document]:
    """Load UTF-8 ``.txt`` documents from *data_path* recursively.

    Raises
    ------
    FileNotFoundError
        If *data_path* does not exist.
    ValueError
        If no ``.txt`` files are found under *data_path*.
    """
    directory = Path(data_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    roots = [directory]
    if directory.resolve() == DATA_PATH.resolve() and KNOWLEDGE_BASE_PATH.is_dir():
        roots.append(KNOWLEDGE_BASE_PATH)
    files = sorted({p.resolve() for root in roots if root.is_dir() for p in root.rglob("*.txt")})
    if not files:
        raise ValueError(f"No .txt files found in data directory: {directory}")

    project_root = Path(__file__).resolve().parent
    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        body = text
        if text.startswith("DOCUMENT_ID:") and "\n---\n" in text:
            header, body = text.split("\n---\n", 1)
            for line in header.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip().lower()] = value.strip()
        rel = path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.name
        metadata.update({"source": rel, "source_filename": path.name,
                         "document_id": metadata.get("document_id", path.stem),
                         "title": metadata.get("title", path.stem)})
        documents.append(Document(page_content=body.strip(), metadata=metadata))
    return documents


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into 500-character chunks with 50-character overlap.

    Each chunk receives a sequential ``chunk_id`` in its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True)
    chunks = splitter.split_documents(docs)
    headings_by_source: dict[str, list[tuple[int, str]]] = {}
    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        headings_by_source[source] = [(m.start(), m.group(1).strip()) for m in re.finditer(
            r"(?m)^([A-Z][A-Z0-9 /&(),’'’—–:-]{3,})\s*$", doc.page_content
        )]
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
        source = str(chunk.metadata.get("source", ""))
        start = int(chunk.metadata.get("start_index", 0))
        preceding = [title for position, title in headings_by_source.get(source, []) if position <= start]
        chunk.metadata["section"] = preceding[-1] if preceding else str(chunk.metadata.get("title", "Document"))
        chunk.metadata["chunk_uid"] = hashlib.sha256(
            f"{source}:{chunk_id}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()[:20]
    return chunks


# ---------------------------------------------------------------------------
# Vector store and retriever
# ---------------------------------------------------------------------------


def build_vector_store(split_docs: list[Document]) -> FAISS:
    """Embed *split_docs* and store them in an in-memory 384-dimensional FAISS index.

    Raises
    ------
    ValueError
        If *split_docs* is empty.
    """
    if not split_docs:
        raise ValueError("Cannot build a vector store without document chunks")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = faiss.IndexFlatL2(384)
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    store.add_documents(split_docs)
    return store


def build_retriever(vector_store: FAISS, k: int = 5) -> Any:
    """Return a vector-store retriever that fetches the *k* nearest chunks.

    Raises
    ------
    ValueError
        If *k* is less than 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    return vector_store.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# LLM: Gemini Flash
# ---------------------------------------------------------------------------


def _build_gemini_llm(model_name: str = GEMINI_MODEL) -> Any:
    """Construct and return a LangChain-wrapped Gemini LLM.

    Reads the API key from the ``GEMINI_API_KEY`` environment variable.

    Raises
    ------
    EnvironmentError
        If ``GEMINI_API_KEY`` is not set.
    ImportError
        If ``langchain-google-genai`` is not installed.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Obtain a key from https://aistudio.google.com/ and export it before "
            "running the application."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langchain-google-genai is not installed. "
            "Run: pip install langchain-google-genai"
        ) from exc

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Grounded answer generation
# ---------------------------------------------------------------------------


def retrieve_context(retriever: Any, query: str) -> list[Document]:
    """Invoke the retriever and return retrieved ``Document`` objects."""
    return retriever.invoke(query)


def rerank_chunks(
    reranker: CrossEncoder, query: str, chunks: list[Document], top_k: int = 3
) -> list[Document]:
    """Re-score *chunks* against *query* using the cross-encoder and return the
    top *top_k* chunks sorted by descending relevance score."""
    if not chunks:
        return []
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


def build_grounded_answer(
    llm: Any,
    prompt_template: ChatPromptTemplate,
    question: str,
    context_chunks: list[Document],
) -> str:
    """Generate an answer grounded in *context_chunks* using the LLM.

    Formats the retrieved context into the prompt and invokes Gemini.
    Returns the text content of the response.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk.metadata.get('source', 'unknown')} | Section: {chunk.metadata.get('section', 'document')} | Title: {chunk.metadata.get('title', 'Untitled')}]\n{chunk.page_content}"
        for chunk in context_chunks
    )
    prompt = prompt_template.format_messages(
        context=context_text, question=question
    )
    response = llm.invoke(prompt)
    # LangChain LLMs return AIMessage; extract the text content
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def build_pipeline(data_path: str | Path = DATA_PATH) -> dict[str, Any]:
    """Load the knowledge base and construct the full retrieval and generation stack.

    Returns a dictionary with keys:
    - ``retriever``: LangChain retriever backed by FAISS.
    - ``llm``: LangChain-wrapped Gemini Flash model.
    - ``prompt``: ``ChatPromptTemplate`` for grounded answers.
    - ``reranker``: Cross-encoder model for re-scoring retrieved chunks.
    """
    documents = load_documents(data_path)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks)
    return {
        "retriever": build_retriever(vector_store, k=10),
        "llm": _build_gemini_llm(),
        "prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE),
        "reranker": CrossEncoder(RERANKER_MODEL),
    }
