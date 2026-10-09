"""FinTech Compliance & Transaction Flow Explainer Bot — Streamlit application.

This is the Person 3 application layer.  It:
  - Renders the Streamlit UI styled to match the project's index.html design.
  - Initialises the RAG pipeline once per session using st.session_state.
  - Routes user questions through the RAG pipeline (retrieve → rerank → generate).
  - Handles all application-level errors gracefully (missing API key, pipeline
    failures, empty input) so users never see a raw Python traceback.

Environment variable required
------------------------------
``GEMINI_API_KEY`` – your Google AI Studio API key.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinTech Explainer Bot",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system — injected CSS that mirrors the index.html colour palette
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Google Font import ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root palette (parchment / forest-green theme from index.html) ── */
:root {
    --paper:    #e8e2d4;
    --white:    #f8f5eb;
    --ink:      #24251f;
    --muted:    #77756c;
    --line:     #cfc8b8;
    --green:    #2c6a50;
    --green-dk: #244b38;
    --green-lt: #e2e4d8;
    --lime:     #d8f078;
    --amber:    #f4ead1;
    --shadow:   0 4px 18px rgba(46,39,25,.10);
}

/* ── Global resets ──────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
    background-color: var(--paper) !important;
    color: var(--ink) !important;
}

/* ── Streamlit app chrome ───────────────────────────────────────── */
.stApp { background-color: var(--paper) !important; }

/* Keep Streamlit's header visible so its built-in sidebar toggle works. */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
button[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #f1f0eb !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebarContent"] { padding-top: 1.8rem !important; }

/* ── Brand mark (sidebar top) ───────────────────────────────────── */
.brand-mark {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 28px 4px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 22px;
}
.brand-glyph {
    display: grid;
    place-items: center;
    width: 36px; height: 36px;
    border-radius: 11px;
    background: var(--green-dk);
    color: #eff8dc;
    font-size: 18px;
    font-weight: 700;
    flex-shrink: 0;
}
.brand-text { font-size: 13px; font-weight: 700; letter-spacing: -.02em; line-height: 1.2; }
.brand-text small { display: block; font-size: 9px; font-weight: 500;
                    letter-spacing: .08em; color: var(--muted); margin-top: 2px; }

/* ── Sidebar nav labels ─────────────────────────────────────────── */
.nav-label {
    font-size: 9px; font-weight: 750; letter-spacing: .14em;
    text-transform: uppercase; color: #9a9e98;
    padding: 0 4px; margin-bottom: 6px; margin-top: 20px;
}

/* ── Info tiles (sidebar) ───────────────────────────────────────── */
.info-tile {
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(248,245,235,.85);
    margin-bottom: 10px;
}
.info-tile-icon { font-size: 16px; }
.info-tile h4 { margin: 8px 0 4px; font-size: 11px; font-weight: 640; color: var(--ink); }
.info-tile p  { margin: 0; font-size: 9px; color: var(--muted); line-height: 1.6; }

/* ── Safety badge (sidebar) ─────────────────────────────────────── */
.safety-badge {
    padding: 10px 13px;
    border: 1px solid #d2d6c8;
    border-radius: 12px;
    background: #f3f2e9;
    margin-top: 14px;
    font-size: 9px; color: #52634c; line-height: 1.6;
}
.safety-badge strong { display: block; margin-bottom: 4px; font-size: 10px; }

/* ── Page header ────────────────────────────────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 18px 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 28px;
}
.page-wordmark {
    font-size: 11px; font-weight: 760; letter-spacing: .09em;
    display: flex; align-items: center; gap: 7px;
}
.wordmark-glyph { color: var(--green); font-size: 16px; }
.wordmark-light { color: var(--muted); font-size: 9px; font-weight: 600; }
.live-badge {
    display: flex; align-items: center; gap: 7px;
    padding: 6px 10px;
    border: 1px solid var(--line);
    border-radius: 20px;
    background: #f3efe4;
    color: #68675e;
    font-size: 10px;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #71816a;
    box-shadow: 0 0 0 3px #dfe4d8;
    display: inline-block;
}

/* ── Hero banner ────────────────────────────────────────────────── */
.hero-banner {
    position: relative;
    overflow: hidden;
    padding: 28px 34px;
    border: 2px solid #22231e;
    border-radius: 22px;
    background: var(--green-dk);
    color: #fff;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(36,31,20,.16);
}
.hero-banner::after {
    position: absolute; top: -90px; right: 80px;
    width: 260px; height: 260px;
    border: 1px solid rgba(237,255,213,.13);
    border-radius: 50%; content: "";
    box-shadow: 0 0 0 30px rgba(237,255,213,.025), 0 0 0 60px rgba(237,255,213,.015);
}
.hero-eyebrow {
    font-size: 9px; font-weight: 760; letter-spacing: .17em;
    text-transform: uppercase; color: var(--lime);
    margin-bottom: 10px;
}
.hero-title {
    font-size: clamp(24px, 3vw, 38px);
    font-weight: 510; letter-spacing: -.055em; line-height: .96;
    color: #fff; margin: 0 0 12px;
    font-family: Georgia, 'Times New Roman', serif;
}
.hero-sub { font-size: 11px; color: #c0d0c1; line-height: 1.65; max-width: 480px; margin: 0; }
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    margin-top: 16px; padding: 5px 9px;
    border-radius: 20px; border: 1px solid rgba(255,255,255,.15);
    background: rgba(255,255,255,.1);
    color: #e2eddb; font-size: 8px; font-weight: 650; letter-spacing: .05em;
}

/* ── Chat container ─────────────────────────────────────────────── */
.chat-wrap {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(248,245,235,.85);
    box-shadow: var(--shadow);
    overflow: hidden;
    margin-bottom: 20px;
}
.chat-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--line);
    background: #f3f0e7;
    font-size: 11px; font-weight: 640; color: var(--ink);
    display: flex; align-items: center; gap: 8px;
}

/* ── Message bubbles ────────────────────────────────────────────── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 18px 4px;
}
.msg-user .bubble {
    max-width: 72%;
    padding: 11px 15px;
    border-radius: 16px 16px 4px 16px;
    background: var(--green-dk);
    color: #fff;
    font-size: 12px; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(36,75,56,.25);
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 4px 18px 12px;
}
.msg-bot .avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--green-lt); color: var(--green-dk);
    font-size: 12px; font-weight: 700;
    display: grid; place-items: center;
    flex-shrink: 0; margin-right: 9px; margin-top: 2px;
}
.msg-bot .bubble {
    max-width: 78%;
    padding: 12px 15px;
    border-radius: 4px 16px 16px 16px;
    background: var(--white);
    border: 1px solid var(--line);
    font-size: 12px; line-height: 1.7;
    color: var(--ink);
    box-shadow: 0 2px 6px rgba(46,39,25,.06);
}

/* ── Error / warning message ────────────────────────────────────── */
.msg-error {
    margin: 6px 18px 14px;
    padding: 11px 14px;
    border-radius: 10px;
    border: 1px solid #e0c4a0;
    background: #fdf4e3;
    color: #7a5c2a;
    font-size: 11px; line-height: 1.6;
}
.msg-error strong { color: #6b4c1f; }

/* ── Empty state ────────────────────────────────────────────────── */
.empty-state {
    padding: 40px 20px;
    text-align: center;
    color: var(--muted);
}
.empty-state .empty-icon { font-size: 36px; margin-bottom: 12px; }
.empty-state p { font-size: 12px; line-height: 1.7; max-width: 340px; margin: 0 auto; }

/* ── Suggestion chips ───────────────────────────────────────────── */
.suggestions {
    padding: 12px 18px 16px;
    border-top: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 7px;
}
.suggestion-label {
    font-size: 9px; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted);
    width: 100%; margin-bottom: 4px;
}

/* ── Input area ─────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    background: var(--white) !important;
    color: var(--ink) !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px rgba(44,106,80,.12) !important;
}
.stTextInput > label { font-size: 11px !important; font-weight: 600 !important; color: var(--muted) !important; }

/* ── Send button ────────────────────────────────────────────────── */
.stButton > button {
    background: var(--green-dk) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 650 !important;
    letter-spacing: .03em !important;
    padding: 0.48rem 1.2rem !important;
    transition: background .18s ease !important;
}
.stButton > button:hover { background: var(--green) !important; }
.stButton > button:disabled { background: #c4bfb2 !important; cursor: not-allowed !important; }

/* ── Clear button (secondary) ───────────────────────────────────── */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--green) !important;
    border: 1px solid var(--line) !important;
}
.stButton > button[kind="secondary"]:hover { background: var(--green-lt) !important; }

/* ── Spinner ────────────────────────────────────────────────────── */
.stSpinner > div { color: var(--green) !important; }

/* ── Streamlit selectbox / dropdown overrides ───────────────────── */
.stSelectbox select {
    background: var(--white) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}

/* ── Metric cards ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(248,245,235,.85) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"] { font-size: 10px !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; color: var(--ink) !important; font-weight: 670 !important; }

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--paper); }
::-webkit-scrollbar-thumb { background: #c4bfb2; border-radius: 10px; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------


def _pipeline_error_message(exc: Exception) -> str:
    """Convert a pipeline exception into a user-friendly message."""
    msg = str(exc)
    if "GEMINI_API_KEY" in msg or "API key" in msg.lower():
        return (
            "⚠️ **API key not configured.** "
            "Please set the `GEMINI_API_KEY` environment variable and restart the app. "
            "You can get a free key from [Google AI Studio](https://aistudio.google.com/)."
        )
    if "No .txt files" in msg or "not found" in msg.lower():
        return (
            "⚠️ **Knowledge base not found.** "
            "The `data/` directory is missing or contains no `.txt` files. "
            "Please ensure the knowledge base files are in place."
        )
    if "langchain_google_genai" in msg or "ImportError" in msg.lower():
        return (
            "⚠️ **Missing dependency.** "
            "Run `pip install langchain-google-genai` to enable Gemini support."
        )
    return f"⚠️ **Something went wrong.** {msg}"


def _is_empty(text: str) -> bool:
    """Return True if *text* is blank or contains only whitespace."""
    return not text or not text.strip()


def _ask(pipeline: dict, question: str) -> str:
    """Route *question* through Person 2's answer_pipeline and return the answer.

    Uses ``answer_pipeline.ask()`` which handles:
    - Input safety guards (prompt injection, out-of-scope, unsupported topic)
    - Retrieval + cross-encoder re-ranking
    - Gemini Flash answer generation
    - Output safety guards (financial advice check)
    - Grounding verification

    Falls back to direct rag_pipeline calls if answer_pipeline is unavailable.

    Raises any exception from the pipeline so the caller can handle it.
    """
    try:
        from answer_pipeline import ask as pipeline_ask

        response = pipeline_ask(question, pipeline)
        return response.answer
    except ImportError:
        # Fallback: call rag_pipeline functions directly (Person 1's interface)
        from rag_pipeline import build_grounded_answer, rerank_chunks, retrieve_context

        chunks = retrieve_context(pipeline["retriever"], question)
        top_chunks = rerank_chunks(pipeline["reranker"], question, chunks, top_k=3)
        answer = build_grounded_answer(
            pipeline["llm"], pipeline["prompt"], question, top_chunks
        )
        return answer


def _init_pipeline() -> dict | None:
    """Initialise the RAG pipeline once and cache it in session_state.

    Returns the pipeline dict or None if initialisation failed (error is stored
    in st.session_state["pipeline_error"]).
    """
    if "pipeline" in st.session_state:
        return st.session_state["pipeline"]
    if "pipeline_error" in st.session_state:
        return None  # already failed — don't retry on every rerun

    try:
        from rag_pipeline import build_pipeline

        with st.spinner("Loading knowledge base… (first run may take a moment)"):
            pipeline = build_pipeline()
        st.session_state["pipeline"] = pipeline
        st.session_state.pop("pipeline_error", None)
        return pipeline
    except Exception as exc:  # noqa: BLE001
        st.session_state["pipeline_error"] = _pipeline_error_message(exc)
        return None


def _init_chat_history() -> None:
    """Ensure session_state["messages"] exists."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def _render_message(role: str, content: str) -> None:
    """Render a single chat message using the project's bubble style."""
    if role == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
    elif role == "assistant":
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="avatar">₹</div>'
            f'<div class="bubble">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif role == "error":
        st.markdown(
            f'<div class="msg-error">{content}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        # Brand mark
        st.markdown(
            """
            <div class="brand-mark">
              <div class="brand-glyph">₹</div>
              <div class="brand-text">
                FinTech Bot
                <small>COMPLIANCE EXPLAINER · RAG</small>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Nav labels
        st.markdown('<div class="nav-label">Learn about</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="info-tile">
              <span class="info-tile-icon">📂</span>
              <h4>Payment flow</h4>
              <p>Initiation → Verification → Compliance → Processing → Settlement</p>
            </div>
            <div class="info-tile">
              <span class="info-tile-icon">📖</span>
              <h4>FinTech terms</h4>
              <p>UPI, NEFT, RTGS, KYC, AML, SWIFT, settlement, and more.</p>
            </div>
            <div class="info-tile">
              <span class="info-tile-icon">🛡️</span>
              <h4>Compliance checks</h4>
              <p>What happens during AML, sanctions screening, and fraud detection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-label">Safety scope</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="safety-badge">
              <strong>✅ This bot will</strong>
              Explain payment processes and FinTech terms in plain language.
              <br><br>
              <strong>❌ This bot will not</strong>
              Process payments, access your account, give financial advice,
              or recommend investments.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clear conversation button at the bottom
        st.markdown("<br>", unsafe_allow_html=True)

        # Coin animation strip (mirrors index.html templates coin canvas)
        import streamlit.components.v1 as components
        components.html(
            """
<!doctype html>
<html>
<head><meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: transparent; }
  canvas { display: block; width: 100%; height: 56px; cursor: pointer; }
</style>
</head>
<body>
<canvas id="sideCoins" width="400" height="56"
        aria-label="Animated coin row. Click to burst."></canvas>
<script>
(function(){
  var c = document.getElementById('sideCoins');
  var ctx = c.getContext('2d');
  var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var coins = Array.from({length: 7}, function(_, i) {
    return { x: 28 + i * 52, phase: i * 0.45 };
  });
  function drawCoin(x, y, r) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2);
    ctx.fillStyle = '#b89755'; ctx.fill();
    ctx.strokeStyle = '#55503d'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.strokeStyle = 'rgba(255,245,202,.6)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x, y, r*.76, 0, Math.PI*2); ctx.stroke();
    ctx.fillStyle = '#f0dfad';
    ctx.font = 'bold ' + (r*.95) + 'px Georgia';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('\u20B9', x, y+1);
  }
  var burstCoins = [];
  function animate(t) {
    if (prefersReduced) return;
    ctx.clearRect(0, 0, c.width, c.height);
    coins.forEach(function(coin) {
      var y = 28 + Math.sin(t/800 + coin.phase) * 10;
      drawCoin(coin.x, y, 15);
    });
    for (var i = burstCoins.length - 1; i >= 0; i--) {
      var b = burstCoins[i];
      var age = (t - b.born) / 900;
      if (age >= 1) { burstCoins.splice(i, 1); continue; }
      ctx.globalAlpha = 1 - age;
      drawCoin(b.x + b.vx*age*60, b.y - b.vy*age*70, 15*(1 - age*.3));
      ctx.globalAlpha = 1;
    }
    requestAnimationFrame(animate);
  }
  c.addEventListener('click', function(e) {
    var box = c.getBoundingClientRect();
    var x = (e.clientX - box.left) * (c.width / box.width);
    var y = (e.clientY - box.top)  * (c.height / box.height);
    for (var i = 0; i < 6; i++) {
      burstCoins.push({
        x: x, y: y, born: performance.now(),
        vx: (Math.random()-.4)*2.2, vy: .8+Math.random()*1.4
      });
    }
  });
  if (!prefersReduced) requestAnimationFrame(animate);
  else coins.forEach(function(coin){ drawCoin(coin.x, 28, 15); });
})();
</script>
</body>
</html>
""",
            height=68,
            scrolling=False,
        )

        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def _render_header() -> None:
    st.markdown(
        """
        <div class="page-header">
          <div class="page-wordmark">
            <span class="wordmark-glyph">✳</span>
            FINTECH EXPLAINER
            <span class="wordmark-light">/ COMPLIANCE BOT</span>
          </div>
          <div class="live-badge">
            <span class="status-dot"></span> RAG · GEMINI FLASH
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    """Render the hero banner with the parchment theme and animated coin canvas.

    The canvas animation mirrors the one in index.html: floating banknotes with
    pointer-driven parallax and a coin burst on click/Enter, all drawn with the
    same palette as the parchment design system.
    """
    import streamlit.components.v1 as components  # local import — only needed here

    hero_html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root {
    --paper: #e9e4d5;
    --ink:   #24251f;
    --green-dk: #244b38;
    --lime:  #d8f078;
    --muted: #69685f;
    --line:  #cfc8b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }

  .hero {
    position: relative;
    display: flex;
    align-items: center;
    overflow: hidden;
    height: 300px;
    padding: 28px 34px;
    border: 2px solid #22231e;
    border-radius: 22px;
    background: var(--green-dk);
    color: #fff;
    box-shadow: 0 10px 30px rgba(36,31,20,.16);
  }
  /* inner engraved border */
  .hero::before {
    position: absolute;
    inset: 8px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 13px;
    content: "";
    pointer-events: none;
  }
  .hero-copy {
    position: relative;
    z-index: 2;
    width: 44%;
    flex-shrink: 0;
  }
  .hero-eyebrow {
    font-size: 9px;
    font-weight: 760;
    letter-spacing: .17em;
    text-transform: uppercase;
    color: var(--lime);
    margin-bottom: 10px;
  }
  .hero-title {
    font-size: clamp(26px, 4.5vw, 48px);
    font-weight: 510;
    letter-spacing: -.055em;
    line-height: .92;
    color: #fff;
    margin-bottom: 13px;
    font-family: Georgia, 'Times New Roman', serif;
    text-transform: uppercase;
  }
  .hero-sub {
    font-size: 11px;
    color: #c0d0c1;
    line-height: 1.65;
    max-width: 360px;
    margin-bottom: 14px;
  }
  .hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 9px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.15);
    background: rgba(255,255,255,.1);
    color: #e2eddb;
    font-size: 8px;
    font-weight: 650;
    letter-spacing: .05em;
  }

  canvas.money-art {
    position: absolute;
    top: 0; right: 0;
    width: 60%; height: 100%;
    cursor: grab;
    z-index: 1;
    touch-action: pan-y;
    outline: none;
  }
  canvas.money-art:active { cursor: grabbing; }

  .art-hint {
    position: absolute;
    z-index: 3;
    right: 18px; bottom: 14px;
    color: rgba(255,255,255,.38);
    font-size: 8px;
    letter-spacing: .06em;
    text-transform: uppercase;
  }
</style>
</head>
<body>
<div class="hero">
  <div class="hero-copy">
    <p class="hero-eyebrow">✳ FINTECH · KNOWLEDGE &amp; COMPLIANCE</p>
    <h1 class="hero-title">Make sense<br>of money<br>in motion.</h1>
    <p class="hero-sub">
      Ask anything about digital payment flows, compliance checks,
      settlement, or FinTech terminology. Get clear, plain-language
      answers grounded in a curated knowledge base.
    </p>
    <span class="hero-tag">✳ INFORMATIONAL ONLY · NOT FINANCIAL ADVICE</span>
  </div>
  <canvas class="money-art" id="moneyCanvas"
          role="button" tabindex="0"
          aria-label="Interactive animated illustration of money in a payment flow. Click or press Enter to send coins floating.">
  </canvas>
  <span class="art-hint">MOVE TO EXPLORE · CLICK TO SEND A COIN</span>
</div>

<script>
(function () {
  var canvas = document.getElementById('moneyCanvas');
  var ctx = canvas.getContext('2d');
  var notes = Array.from({length: 6}, function(_, i) {
    return {
      x: .46 + (i % 3) * .18 + (Math.random() - .5) * .07,
      y: .18 + Math.floor(i / 3) * .28 + (Math.random() - .5) * .07,
      scale: .5 + Math.random() * .32,
      angle: (Math.random() - .5) * .38,
      phase: Math.random() * Math.PI * 2,
      drift: .013 + Math.random() * .016
    };
  });
  var coins = [];
  var width = 0, height = 0, dpr = 1, pointerX = 0, pointerY = 0;
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function resize() {
    var box = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = box.width; height = box.height;
    canvas.width  = Math.round(width  * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (reducedMotion) draw(0);
  }

  function drawNote(x, y, size, angle, alpha) {
    alpha = alpha === undefined ? 1 : alpha;
    ctx.save();
    ctx.translate(x, y); ctx.rotate(angle); ctx.globalAlpha = alpha;
    var w = size, h = size * .61;
    ctx.fillStyle = '#d8c99d'; ctx.strokeStyle = '#555747';
    ctx.lineWidth = Math.max(1, size * .009);
    ctx.shadowColor = 'rgba(48,42,29,.18)'; ctx.shadowBlur = 13; ctx.shadowOffsetY = 7;
    ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(-w/2, -h/2, w, h, 5); }
    else { ctx.rect(-w/2, -h/2, w, h); }
    ctx.fill(); ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.strokeStyle = 'rgba(73,75,59,.58)';
    ctx.lineWidth = Math.max(.7, size * .004);
    ctx.strokeRect(-w*.45, -h*.38, w*.9, h*.76);
    ctx.beginPath(); ctx.ellipse(0, 0, h*.27, h*.36, 0, 0, Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(0, 0, h*.19, h*.27, 0, 0, Math.PI*2); ctx.stroke();
    ctx.fillStyle = '#484b3c';
    ctx.font = 'bold ' + Math.max(12, size*.2) + 'px Georgia';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('\u20B9', 0, 1);
    ctx.font = 'bold ' + Math.max(7, size*.055) + 'px Georgia';
    ctx.fillText('RESERVE BANK OF INDIA', 0, -h*.34);
    ctx.fillText('ONE RUPEE', 0, h*.36);
    [-1, 1].forEach(function(s) {
      ctx.fillText('\u20B9', s*w*.37, 0);
      for (var j = -2; j <= 2; j++) {
        ctx.beginPath(); ctx.moveTo(s*w*.29, j*h*.075); ctx.lineTo(s*w*.34, j*h*.075); ctx.stroke();
      }
    });
    ctx.restore();
  }

  function drawCoin(x, y, r, spin) {
    spin = spin || 0;
    ctx.save(); ctx.translate(x, y);
    ctx.scale(Math.max(.18, Math.abs(Math.cos(spin))), 1);
    ctx.fillStyle = '#b89755'; ctx.strokeStyle = '#55503d'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.ellipse(0, 0, r, r*.76, 0, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,245,202,.75)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.ellipse(0, 0, r*.78, r*.57, 0, 0, Math.PI*2); ctx.stroke();
    ctx.fillStyle = '#f0dfad';
    ctx.font = 'bold ' + (r*.95) + 'px Georgia';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('\u20B9', 0, 1);
    ctx.restore();
  }

  function draw(time) {
    ctx.clearRect(0, 0, width, height);
    // engraved orbit lines (decorative horizontal waves)
    ctx.strokeStyle = 'rgba(200,220,190,.18)'; ctx.lineWidth = 1;
    for (var row = 0; row < 4; row++) {
      ctx.beginPath();
      for (var x = -10; x <= width + 10; x += 8) {
        var y = height*(.68 + row*.057) + Math.sin(x*.012 + time*.00045 + row)*8 + pointerY*7;
        if (x === -10) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    // floating banknotes
    notes.forEach(function(note, i) {
      var bob = reducedMotion ? 0 : Math.sin(time*.0007 + note.phase) * 10;
      drawNote(
        width * (note.x - pointerX*.035),
        height * note.y + bob + pointerY * 9,
        Math.min(width, height) * note.scale,
        note.angle + pointerX*.11 + Math.sin(time*.0004 + note.phase)*.025,
        .96 - i * .04
      );
    });
    // static decorative coins
    [0, 1, 2].forEach(function(i) {
      drawCoin(
        width * (.63 + i*.12 - pointerX*.025),
        height * (.76 + Math.sin(time*.001 + i*2)*.025),
        Math.min(width, height) * (.043 - i * .006),
        time*.001 + i * 1.7
      );
    });
    // launched coins
    for (var ci = coins.length - 1; ci >= 0; ci--) {
      var c = coins[ci];
      var age = (time - c.born) / 1150;
      if (age >= 1) { coins.splice(ci, 1); continue; }
      drawCoin(
        c.x + c.vx * age * 90,
        c.y - c.vy * age * 100,
        Math.min(width, height) * .045 * (1 - age * .25),
        time * .004 + ci
      );
    }
    ctx.globalAlpha = 1;
    if (!reducedMotion) requestAnimationFrame(draw);
  }

  function toss(event) {
    var box = canvas.getBoundingClientRect();
    var cx = event && event.clientX ? event.clientX - box.left : width * .65;
    var cy = event && event.clientY ? event.clientY - box.top  : height * .72;
    for (var i = 0; i < 7; i++) {
      coins.push({
        x: cx, y: cy,
        born: performance.now(),
        vx: (Math.random() - .35) * 2,
        vy: .7 + Math.random() * 1.4
      });
    }
    if (reducedMotion) draw(performance.now());
  }

  canvas.addEventListener('pointermove', function(e) {
    var box = canvas.getBoundingClientRect();
    pointerX = (e.clientX - box.left) / box.width - .5;
    pointerY = (e.clientY - box.top)  / box.height - .5;
  });
  canvas.addEventListener('pointerleave', function() { pointerX = 0; pointerY = 0; });
  canvas.addEventListener('click', toss);
  canvas.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toss(); }
  });

  window.addEventListener('resize', resize);
  resize();
  if (!reducedMotion) requestAnimationFrame(draw);
})();
</script>
</body>
</html>
"""
    components.html(hero_html, height=320, scrolling=False)


def _render_metrics(message_count: int) -> None:
    col1, col2, col3 = st.columns(3)
    from rag_pipeline import DATA_PATH

    kb_path = Path(__file__).resolve().parent / "knowledge_base" / "india_fintech"
    knowledge_file_count = len(list(DATA_PATH.rglob("*.txt"))) + len(list(kb_path.rglob("*.txt")))
    col1.metric("Knowledge files", knowledge_file_count, help="FinTech workflow documents in the knowledge base")
    col2.metric("Questions asked", message_count, help="This session")
    col3.metric("LLM", "Gemini Flash", help="Google Gemini 1.5 Flash")


def _render_pipeline_error(error_msg: str) -> None:
    """Show a styled error banner when the pipeline could not be initialised."""
    st.markdown(
        f'<div class="msg-error">{error_msg}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _render_sidebar()
    _render_header()
    _render_hero()

    _init_chat_history()
    user_messages = [m for m in st.session_state["messages"] if m["role"] == "user"]
    _render_metrics(len(user_messages))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline initialisation ──────────────────────────────────────────
    pipeline = _init_pipeline()

    if pipeline is None:
        _render_pipeline_error(
            st.session_state.get(
                "pipeline_error",
                "⚠️ The knowledge base pipeline could not be loaded.",
            )
        )
        # Still render the chat history (read-only) so the user can see prior answers
        _render_chat_history()
        return  # Cannot accept new questions without a working pipeline

    # ── Chat window ──────────────────────────────────────────────────────
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-header">💬 Ask me about Indian FinTech systems</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state["messages"]:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">₹</div>
              <p>
                Ask a question like <em>"What happens after I click Pay?"</em>
                or <em>"What is a compliance check?"</em> — and get a clear,
                plain-language explanation.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state["messages"]:
            _render_message(msg["role"], msg["content"])

    # Suggestion chips
    st.markdown(
        """
        <div class="suggestions">
          <span class="suggestion-label">Try asking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Why is a UPI payment pending after debit?",
        "How does merchant settlement reconciliation work?",
        "What is an Account Aggregator consent artefact?",
        "What do LOS and LMS do in digital lending?",
        "How do fintech APIs prevent duplicate payments?",
    ]
    sug_cols = st.columns(len(suggestions))
    for col, sug in zip(sug_cols, suggestions):
        with col:
            if st.button(sug, key=f"sug_{sug}", use_container_width=True):
                _handle_question(pipeline, sug)

    st.markdown("</div>", unsafe_allow_html=True)  # close .chat-wrap

    # ── Input row ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Your question",
            placeholder="e.g. How does a fintech payment move from authorization to settlement?",
            key="user_input",
            label_visibility="collapsed",
        )
    with col_send:
        send_clicked = st.button("Ask →", use_container_width=True, key="send_btn")

    if send_clicked or (user_input and user_input != st.session_state.get("_last_input", "")):
        if send_clicked:
            _handle_question(pipeline, user_input)


def _render_chat_history() -> None:
    """Render stored messages without accepting new input."""
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-header">💬 Conversation history</div>',
        unsafe_allow_html=True,
    )
    for msg in st.session_state.get("messages", []):
        _render_message(msg["role"], msg["content"])
    st.markdown("</div>", unsafe_allow_html=True)


def _handle_question(pipeline: dict, question: str) -> None:
    """Process a user question: validate, run pipeline, store results, rerun."""
    if _is_empty(question):
        # Append a transient warning that empty input was submitted
        st.session_state["messages"].append(
            {
                "role": "error",
                "content": (
                    "⚠️ Please type a question before clicking Ask. "
                    "For example: <em>What is a compliance check?</em>"
                ),
            }
        )
        st.rerun()
        return

    question = question.strip()

    # Store user message
    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["_last_input"] = question

    # Get answer from pipeline
    with st.spinner("Thinking…"):
        try:
            answer = _ask(pipeline, question)
        except Exception as exc:  # noqa: BLE001
            answer = None
            error_msg = _pipeline_error_message(exc)

    if answer is not None:
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    else:
        st.session_state["messages"].append({"role": "error", "content": error_msg})

    st.rerun()


if __name__ == "__main__":
    main()
