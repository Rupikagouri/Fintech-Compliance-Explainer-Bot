"""Streamlit chat interface for the FinTech Compliance Explainer Bot.

Connects to :func:`answer_pipeline.ask_with_default_pipeline`, which:
  - Runs input safety guards (prompt injection, out-of-scope, topic filter)
  - Retrieves relevant chunks from the FAISS knowledge base
  - Calls Gemini Flash to generate a grounded answer
  - Runs output guards (financial advice check)
  - Appends a caveat if the answer is not grounded in the KB

Usage
-----
    export GEMINI_API_KEY="your-key-here"
    streamlit run app.py

The pipeline is built once on first query and cached for the session.
No real payments are processed and no banking credentials are requested.
"""

from __future__ import annotations

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration – must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinTech Compliance Explainer",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Make the user bubble stand out */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1e3a5f;
    }
    /* Slightly softer assistant bubble */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1a1a2e;
    }
    /* Safety/refusal callout */
    .safety-box {
        background: #3b1c1c;
        border-left: 4px solid #e05c5c;
        padding: 0.75rem 1rem;
        border-radius: 0.4rem;
        margin-top: 0.5rem;
    }
    /* Grounding caveat */
    .caveat-box {
        background: #2b2b10;
        border-left: 4px solid #c8a020;
        padding: 0.6rem 0.9rem;
        border-radius: 0.4rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💸 FinTech Compliance\nExplainer Bot")
    st.caption("Powered by Gemini Flash · RAG · Safety guardrails")

    st.markdown("---")
    st.markdown("### 🔑 API Key")

    # Allow key entry in the sidebar so the app is usable without setting an
    # environment variable (e.g. during a live demo).
    sidebar_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AIza...",
        help="Obtain a free key at https://aistudio.google.com/",
    )
    if sidebar_key:
        os.environ["GEMINI_API_KEY"] = sidebar_key

    st.markdown("---")
    st.markdown("### 💬 Try asking")
    example_questions = [
        "What happens after I click Pay?",
        "What is a compliance check?",
        "Why is my payment pending?",
        "What is KYC and why is it required?",
        "Explain AML in simple terms",
        "What is RTGS vs NEFT?",
        "What is settlement in payments?",
        "What does 'Verifying…' mean?",
        "What is a chargeback?",
        "What is 3D Secure?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"eg_{q}"):
            st.session_state["pending_example"] = q

    st.markdown("---")
    st.markdown(
        "⚠️ **Informational only.** This bot explains how payment systems work. "
        "It does not process real transactions, access accounts, or give "
        "personalised financial advice.",
        unsafe_allow_html=False,
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("pending_example", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💸 FinTech Compliance Explainer")
st.caption(
    "Ask me anything about digital payment flows, KYC, AML, compliance checks, "
    "or settlement — in plain language."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Check API key before attempting any query
# ---------------------------------------------------------------------------

api_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())

if not api_key_present:
    st.warning(
        "**No Gemini API key found.**  "
        "Enter your key in the sidebar or set the `GEMINI_API_KEY` environment "
        "variable before asking a question.  "
        "Get a free key at https://aistudio.google.com/",
        icon="🔑",
    )

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state["messages"]:
    role = msg["role"]
    with st.chat_message(role):
        _safe = msg.get("safe", True)
        _grounded = msg.get("grounded", True)
        _category = msg.get("guard_category", "")

        if role == "assistant" and not _safe:
            # Safety refusal – render with a coloured callout
            icon = {
                "prompt_injection": "🚫",
                "out_of_scope": "🏦",
                "unsupported_topic": "🔍",
                "financial_advice": "⚖️",
            }.get(_category, "⚠️")
            st.markdown(
                f'<div class="safety-box">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Handle example question button clicks
# ---------------------------------------------------------------------------

pending = st.session_state.pop("pending_example", None)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_input: str | None = st.chat_input(
    "Ask about payments, KYC, AML, compliance, settlement…",
    disabled=not api_key_present,
)

# Prefer a typed question; fall back to a clicked example
question = user_input or pending

# ---------------------------------------------------------------------------
# Process question
# ---------------------------------------------------------------------------

if question:
    # Store and render the user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Import here so the heavy RAG deps load only when needed
    from answer_pipeline import ask_with_default_pipeline  # noqa: E402

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = ask_with_default_pipeline(question)
            except EnvironmentError as exc:
                # GEMINI_API_KEY missing or invalid
                error_text = (
                    f"**Configuration error:** {exc}  \n"
                    "Please add your Gemini API key in the sidebar."
                )
                st.error(error_text, icon="🔑")
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": error_text,
                        "safe": False,
                        "grounded": False,
                        "guard_category": "configuration_error",
                    }
                )
                st.stop()
            except Exception as exc:  # noqa: BLE001
                error_text = (
                    f"**Unexpected error:** {exc}  \n"
                    "Please try again or rephrase your question."
                )
                st.error(error_text, icon="❌")
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": error_text,
                        "safe": False,
                        "grounded": False,
                        "guard_category": "runtime_error",
                    }
                )
                st.stop()

        # -----------------------------------------------------------------
        # Render the response
        # -----------------------------------------------------------------
        if not response.safe:
            # Safety guard fired — show a coloured callout
            icon = {
                "prompt_injection": "🚫",
                "out_of_scope": "🏦",
                "unsupported_topic": "🔍",
                "financial_advice": "⚖️",
            }.get(response.guard_category, "⚠️")

            st.markdown(
                f'<div class="safety-box">{icon} {response.answer}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Normal answer
            # Split off the grounding caveat (appended by answer_pipeline)
            # so it can be styled differently from the main answer.
            caveat_marker = "\n\n⚠️ "
            if caveat_marker in response.answer:
                main_text, caveat = response.answer.split(caveat_marker, 1)
                st.markdown(main_text)
                st.markdown(
                    f'<div class="caveat-box">⚠️ {caveat}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(response.answer)

    # Store assistant message in history
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": response.answer,
            "safe": response.safe,
            "grounded": response.grounded,
            "guard_category": response.guard_category,
        }
    )
