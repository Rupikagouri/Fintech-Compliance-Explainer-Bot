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
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

DATA_PATH = Path(__file__).resolve().parent / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Gemini model name (Gemini 1.5 Flash is free-tier accessible)
GEMINI_MODEL = "gemini-1.5-flash"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a helpful FinTech compliance and payment-process explainer.

RULES:
- Answer ONLY using the provided context below.
- If the answer is not found in the context, respond with exactly:
  "I don't have enough information in my knowledge base to answer that. \
Please consult your bank or a qualified professional."
- Do NOT give personalised financial advice or product recommendations.
- Do NOT process, initiate, or describe how to process real transactions.
- Be clear, factual, and easy to understand.

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

    files = sorted(directory.glob("**/*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in data directory: {directory}")

    loader = DirectoryLoader(
        str(directory),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into 500-character chunks with 50-character overlap.

    Each chunk receives a sequential ``chunk_id`` in its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
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
        chunk.page_content for chunk in context_chunks
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
        "retriever": build_retriever(vector_store, k=5),
        "llm": _build_gemini_llm(),
        "prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE),
        "reranker": CrossEncoder(RERANKER_MODEL),
    }
