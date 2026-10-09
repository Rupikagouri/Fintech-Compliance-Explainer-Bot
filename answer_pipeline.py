"""End-to-end question-answering pipeline for the FinTech Compliance Explainer Bot.

This module wires together:
- Input safety guardrails (safety.py)
- Document retrieval and cross-encoder re-ranking (rag_pipeline.py)
- Grounded answer generation via Gemini Flash (rag_pipeline.py)
- Output safety guardrails (safety.py)
- Grounding verification (safety.py)

Public API
----------
``ask(question, pipeline)``
    The only function external callers need.  Pass the user's question and the
    pipeline dict returned by :func:`rag_pipeline.build_pipeline`.

``AskResponse``
    Named-tuple returned by :func:`ask`.  Fields:

    - ``answer``   – text to show the user.
    - ``safe``     – ``True`` when the full pipeline passed all guards.
    - ``grounded`` – ``True`` when the answer overlaps meaningfully with the
                     retrieved context.  ``False`` may mean limited KB coverage.
    - ``guard_category`` – which guard fired, or empty string if none.
    - ``chunks_used``    – list of document chunk strings that grounded the answer.
"""

from __future__ import annotations

from typing import Any, NamedTuple

# Safety module: no heavy deps, always importable
from safety import (
    NOT_IN_KB_RESPONSE,
    GuardResult,
    is_grounded,
    run_input_guards,
    run_output_guards,
)


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------


class AskResponse(NamedTuple):
    """Structured response returned by :func:`ask`.

    Attributes
    ----------
    answer:
        The text to display to the user.
    safe:
        ``True`` when no safety guard was triggered.
    grounded:
        ``True`` when the answer is grounded in the retrieved context.
        Always ``True`` for guard-blocked responses (the canned message itself
        is safe by definition).
    guard_category:
        Short label of the guard that blocked the request (e.g.,
        ``"prompt_injection"``).  Empty string when ``safe is True``.
    chunks_used:
        The document chunk texts that were passed to the LLM.  Empty list for
        blocked requests.
    """

    answer: str
    safe: bool
    grounded: bool
    guard_category: str = ""
    chunks_used: list[str] = []


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def ask(question: str, pipeline: dict[str, Any]) -> AskResponse:
    """Ask a FinTech compliance question and return a structured response.

    The pipeline executes the following steps:

    1. **Input guard** – prompt-injection, out-of-scope, and topic checks.
       If any guard fires, a canned safe response is returned immediately.
    2. **Retrieve** – fetch the top-k document chunks from FAISS.
    3. **Rerank** – cross-encoder re-scores and selects the most relevant chunks.
    4. **Generate** – Gemini Flash generates an answer from the context.
    5. **Output guard** – financial-advice and product-recommendation check.
       If the guard fires, the canned safe response replaces the LLM output.
    6. **Grounding check** – heuristic word-overlap check between answer and context.
       If ungrounded, a caveat is appended to the answer.

    Parameters
    ----------
    question:
        The user's question string.
    pipeline:
        Dict returned by :func:`rag_pipeline.build_pipeline`, containing keys
        ``retriever``, ``llm``, ``prompt``, and ``reranker``.

    Returns
    -------
    AskResponse
        See class docstring for field descriptions.
    """
    # ------------------------------------------------------------------
    # Step 1: Input safety guards
    # ------------------------------------------------------------------
    input_guard: GuardResult = run_input_guards(question)
    if not input_guard.safe:
        return AskResponse(
            answer=input_guard.reason,
            safe=False,
            grounded=True,  # Canned responses are always safe/grounded
            guard_category=input_guard.category,
            chunks_used=[],
        )

    # ------------------------------------------------------------------
    # Step 2: Retrieve context from vector store
    # ------------------------------------------------------------------
    # Note: retrieve_context / rerank_chunks / build_grounded_answer are
    # inlined here to avoid a top-level `import rag_pipeline`, which would
    # trigger `import faiss` even when running tests without faiss installed.

    retriever = pipeline["retriever"]
    retrieved_docs = retriever.invoke(question)

    # ------------------------------------------------------------------
    # Step 3: Rerank retrieved chunks
    # ------------------------------------------------------------------
    reranker = pipeline["reranker"]
    if retrieved_docs:
        pairs = [(question, doc.page_content) for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        top_chunks = [doc for _, doc in ranked[:3]]
    else:
        top_chunks = []
    chunk_texts = [chunk.page_content for chunk in top_chunks]

    # ------------------------------------------------------------------
    # Step 4: Generate grounded answer via Gemini
    # ------------------------------------------------------------------
    llm = pipeline["llm"]
    prompt_template = pipeline["prompt"]
    context_text = "\n\n---\n\n".join(chunk.page_content for chunk in top_chunks)
    prompt_messages = prompt_template.format_messages(
        context=context_text, question=question
    )
    llm_response = llm.invoke(prompt_messages)
    raw_answer = (
        str(llm_response.content).strip()
        if hasattr(llm_response, "content")
        else str(llm_response).strip()
    )

    # ------------------------------------------------------------------
    # Step 5: Output safety guards
    # ------------------------------------------------------------------
    output_guard: GuardResult = run_output_guards(raw_answer)
    if not output_guard.safe:
        return AskResponse(
            answer=output_guard.reason,
            safe=False,
            grounded=True,
            guard_category=output_guard.category,
            chunks_used=chunk_texts,
        )

    # ------------------------------------------------------------------
    # Step 6: Grounding check
    # ------------------------------------------------------------------
    grounded = is_grounded(raw_answer, chunk_texts)
    final_answer = raw_answer
    if not grounded:
        final_answer = (
            raw_answer
            + "\n\n⚠️ "
            + NOT_IN_KB_RESPONSE
        )

    return AskResponse(
        answer=final_answer,
        safe=True,
        grounded=grounded,
        guard_category="",
        chunks_used=chunk_texts,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper (no pre-built pipeline needed — builds lazily)
# ---------------------------------------------------------------------------

_default_pipeline: dict[str, Any] | None = None


def ask_with_default_pipeline(question: str) -> AskResponse:
    """Like :func:`ask` but builds and caches the pipeline on first call.

    Useful for interactive scripts and demos.  The pipeline is built once and
    reused for subsequent calls in the same process.

    Raises
    ------
    EnvironmentError
        If ``GEMINI_API_KEY`` is not set (propagated from
        :func:`rag_pipeline.build_pipeline`).
    """
    global _default_pipeline
    if _default_pipeline is None:
        from rag_pipeline import build_pipeline  # local import to avoid circular

        _default_pipeline = build_pipeline()
    return ask(question, _default_pipeline)
