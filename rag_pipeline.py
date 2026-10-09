"""Document ingestion and RAG pipeline construction.

Importing this module does not read documents or load/download machine-learning
models. Call :func:`build_pipeline` when the application starts.
"""

from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

DATA_PATH = Path(__file__).resolve().parent / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PROMPT_TEMPLATE = """You are a factual assistant.

Rules:
- Answer ONLY using the provided context
- If not found, say: Not found in the document

Context:
{context}

Question:
{question}

Answer:
"""


def load_documents(data_path: str | Path) -> list[Document]:
    """Load UTF-8 text documents from *data_path* recursively."""
    directory = Path(data_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    files = sorted(directory.glob("**/*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in data directory: {directory}")

    loader = DirectoryLoader(
        str(directory), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into 500-character chunks with 50-character overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
    return chunks


def build_vector_store(split_docs: list[Document]) -> FAISS:
    """Embed chunks and place them in an in-memory 384-dimensional FAISS index."""
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
    """Create a vector-store retriever returning the nearest *k* chunks."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return vector_store.as_retriever(search_kwargs={"k": k})


def build_pipeline(data_path: str | Path = DATA_PATH) -> dict[str, Any]:
    """Load the knowledge base and construct the retrieval and generation stack."""
    documents = load_documents(data_path)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks)
    return {
        "retriever": build_retriever(vector_store, k=5),
        "llm": ChatOllama(model="llama3.2"),
        "prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE),
        "reranker": CrossEncoder(RERANKER_MODEL),
    }
