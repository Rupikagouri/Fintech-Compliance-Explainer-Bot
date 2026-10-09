# Part 1 Documentation: Files and Work Completed

This document explains the contents of `implementation_part1` in plain language. It covers the handoff notes, the project overview page, the copied Python implementation, its data, dependencies, and tests.

## What Part 1 is about

The project role documented for Person 1 is **problem definition, product framing, and scope**. The project is intended to explain digital-payment processes and FinTech terms in accessible language. Part 1 sets out the intended users, the topics the assistant should explain, its goals, and the boundary that it must not process payments or give personalized financial advice.

The `files/` subfolder also contains a copied setup and retrieval implementation. That code is a separate technical handoff included in this folder. Its text data is an older set of general AI, machine-learning, and economy documents; it is not yet a curated knowledge base about payment flows, transaction verification, settlement, and compliance. The distinction matters when using these artifacts as the foundation for the FinTech project.

## Folder map

```text
implementation_part1/
├── README.md
├── PART1_DOCUMENTATION.md
├── index.html
└── files/
    ├── .gitignore
    ├── requirements.txt
    ├── rag_pipeline.py
    ├── data/
    │   ├── dl.txt
    │   ├── genai.txt
    │   ├── india.txt
    │   ├── lr.txt
    │   ├── ml.txt
    │   └── usa.txt
    └── tests/
        ├── __init__.py
        ├── test_project_setup.py
        └── test_retrieval.py
```

## Top-level files

### `README.md`

The short Part 1 handoff note. It records the product problem and framing, intended users, supported subject areas, objectives, and safety limits. It also distinguishes the current product role from the earlier retrieval code copied under `files/`.

### `PART1_DOCUMENTATION.md`

This file. It is the detailed guide to the folder, what each file is for, how the retrieval code is organized, and what its current limits are.

### `index.html`

A self-contained HTML project overview page. Its markup contains the page content and its CSS and JavaScript are embedded in the same file, so it does not need a separate frontend build system. The page includes a progress checklist whose selections are saved in browser storage, descriptions for the listed project files, and a control to copy the documented check commands. Its canvas contains the animated rupee-note and coin illustration; pointer movement shifts the notes, and clicking or using Enter/Space sends a coin burst. The animation observes the browser's reduced-motion preference.

This page is a handoff/overview artifact. It is not the Streamlit application, and it does not run document retrieval or answer questions.

## Files under `files/`

### `.gitignore`

Lists local or generated files Git should leave out of version control. It excludes Python virtual environments, Python bytecode and cache folders, pytest cache, editor settings, operating-system metadata files, and `.env` files that may contain local configuration or secrets. `.env.example` is explicitly allowed so a safe sample configuration can be committed.

### `requirements.txt`

Pins the Python package versions used by this implementation. Pinning versions makes it easier for team members to install the same dependency set. The packages cover:

- **LangChain** packages for document loading, prompts, splitting, vector-store integration, and the Ollama chat model adapter.
- **FAISS** for similarity search over document embeddings.
- **Hugging Face and sentence-transformers** for embedding and cross-encoder models.
- **Streamlit and FastAPI**, plus **Uvicorn**, for the project's application interfaces.
- **Pydantic** for data models/configuration and **pytest** for tests.

The package list alone does not install language or embedding models. The first use of the Hugging Face models may download model files, and the Ollama model must be available locally to use the configured chat model.

### `rag_pipeline.py`

Contains the document ingestion and retrieval setup. Importing the module defines functions and configuration; the expensive document/model work happens when the functions are called.

Key settings:

- `DATA_PATH` points to the `data/` directory next to this Python file.
- `EMBEDDING_MODEL` selects `sentence-transformers/all-MiniLM-L6-v2` to turn text into numerical vectors.
- `RERANKER_MODEL` selects a cross-encoder model intended to re-score retrieved text.
- `PROMPT_TEMPLATE` instructs a factual assistant to answer from the supplied context and to say “Not found in the document” when the answer is absent.

Functions, in order:

1. **`load_documents(data_path)`** checks that the given directory exists and contains `.txt` files, then reads those files as UTF-8 LangChain `Document` objects. It raises a clear `FileNotFoundError` for a missing directory and `ValueError` when no text files are present.
2. **`split_documents(docs)`** breaks long documents into chunks of at most 500 characters, with 50 characters of overlap so nearby context is retained. It adds a sequential `chunk_id` to each chunk's metadata.
3. **`build_vector_store(split_docs)`** rejects an empty chunk list, creates embeddings with the selected Hugging Face model, creates an in-memory FAISS index with 384 dimensions, and adds the chunks to it.
4. **`build_retriever(vector_store, k=5)`** asks the vector store for its nearest `k` chunks. It rejects values below one; by default it returns the five nearest chunks.
5. **`build_pipeline(data_path)`** runs loading, splitting, and vector-store creation, then returns a dictionary containing the retriever, an Ollama `llama3.2` model object, the prompt template, and a cross-encoder reranker object.

The function assembles these components, but this file does not implement a complete question-answering call: it does not connect retrieved documents to the prompt and LLM, invoke the LLM, or apply the reranker to a query. There is no `ask()` function here. Those are separate integration steps.

### Files under `data/`

These six UTF-8 text files are example source material loaded by `load_documents`. Together, they demonstrate document ingestion and retrieval, but they are not a complete or domain-aligned FinTech payment knowledge base.

- **`dl.txt`** gives a short introduction to deep learning, neural networks, and example tasks such as image recognition and language processing.
- **`genai.txt`** defines generative AI, gives examples of generated content, and mentions model families such as GANs and Transformers.
- **`india.txt`** summarizes India's economy with GDP, population, growth, currency conversion, and sector information.
- **`lr.txt`** introduces linear regression, its equation and variables, and a small scikit-learn example.
- **`ml.txt`** describes machine learning, its broad categories, and some applications.
- **`usa.txt`** discusses the U.S. economy, GDP, sectors, growth, the U.S. dollar, trade, and economic strengths.

Because the files cover mixed subjects, retrieved answers will only be as relevant and reliable as these source documents. For the planned FinTech explainer, this data set needs to be replaced or supplemented with reviewed material on payment initiation, verification, compliance, processing, settlement, and related concepts.

### Files under `tests/`

#### `__init__.py`

An empty package marker for the tests directory. It lets Python treat the directory as a package where needed.

#### `test_project_setup.py`

Contains six basic checks for expected project structure: a data directory with text files, a requirements file, a templates page, a `.gitignore` with environment exclusions, and the tests package marker. These checks verify that expected files exist; they do not test the RAG answer quality.

The setup check for `templates/index.html` expects a `templates/` directory at the project root. That template is not included in this `implementation_part1/files/` snapshot, so this particular check requires the full working project layout to pass.

#### `test_retrieval.py`

Contains eight tests for retrieval building blocks:

- Loading a text file returns a LangChain `Document`.
- A missing data directory raises `FileNotFoundError`.
- An empty data directory raises `ValueError`.
- Long documents are split into multiple chunks no longer than 500 characters.
- Chunks receive sequential IDs.
- The retriever is created with the default `k=5` or a caller-supplied value.

The retriever tests use a small fake vector store for the retriever configuration checks, so those checks do not download models or build a real FAISS index. There are no tests here for embedding generation, vector-store creation, reranking, LLM responses, safety behavior, or end-to-end question answering.

## What Part 1 produced

The product-framing work established the problem to solve, the audience, the educational purpose, the supported topics, the desired outcomes, and the actions the assistant must not perform. The folder also preserves an earlier technical retrieval slice: pinned dependencies, ignored local environment files, sample text data, document loading and chunking, FAISS/retriever setup, and 14 setup/retrieval tests.

The current code is a starting point rather than a finished FinTech assistant. The copied data does not yet match the product's payment-process subject matter, the answer pipeline is not connected end to end, and the tests do not cover model responses or safety boundaries. The setup/tests have not been run as part of preparing this documentation.
