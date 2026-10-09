"""Financial-safety guardrails, prompt-injection protection, and unsupported-question
handling for the FinTech Compliance Explainer Bot.

All public functions return a ``GuardResult`` named-tuple so callers always get a
structured response with both a boolean flag and a human-readable reason.
"""

from __future__ import annotations

import re
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class GuardResult(NamedTuple):
    """Outcome of a safety check.

    Attributes:
        safe: ``True`` when the input/output passes the guard; ``False`` when it
            was blocked.
        reason: Human-readable explanation of why the guard was triggered.  Empty
            string when ``safe is True``.
        category: Short label for the type of guard that fired (e.g.,
            ``"financial_advice"``).  Empty string when ``safe is True``.
    """

    safe: bool
    reason: str = ""
    category: str = ""


# ---------------------------------------------------------------------------
# Topic support list
# ---------------------------------------------------------------------------

# Keywords that indicate the question is within the bot's knowledge domain.
_SUPPORTED_TOPIC_PATTERNS: list[str] = [
    r"\bpayment\b",
    r"\btransaction\b",
    r"\bsettle(ment|d|s)?\b",
    r"\bcompl(iance|y|iant)\b",
    r"\bkyc\b",
    r"\baml\b",
    r"\bverif(ication|y|ied|ying)\b",
    r"\bfraud\b",
    r"\bauthentic(ation|ate|ated)?\b",
    r"\bauthori[sz](ation|e|ed)?\b",
    r"\bupi\b",
    r"\bneft\b",
    r"\brtgs\b",
    r"\bimps\b",
    r"\bswift\b",
    r"\biban\b",
    r"\bbeneficiar(y|ies)\b",
    r"\banccount\b",
    r"\bbank(ing)?\b",
    r"\bcard\b",
    r"\bwallet\b",
    r"\bdigital\s+payment\b",
    r"\bonline\s+payment\b",
    r"\bpayment\s+gateway\b",
    r"\bpayment\s+(flow|pipeline|process|stage|step)\b",
    r"\bsettl(e|ed|ing|ement)\b",
    r"\bclearing\b",
    r"\binterbank\b",
    r"\bfiat\b",
    r"\bcurrenc(y|ies)\b",
    r"\bfintec?h\b",
    r"\bregulat(ion|ory|or|ed)\b",
    r"\brbi\b",
    r"\bnpci\b",
    r"\bpmla\b",
    r"\bfatf\b",
    r"\bofac\b",
    r"\bsar\b",
    r"\bstr\b",
    r"\bctr\b",
    r"\bpep\b",
    r"\bsanction(s)?\b",
    r"\bmoney\s+laundering\b",
    r"\bterrorist\s+financ(ing|e)\b",
    r"\bdue\s+diligence\b",
    r"\bknow\s+your\s+customer\b",
    r"\banti.money\b",
    r"\brefund\b",
    r"\bchargeback\b",
    r"\bdeclin(e|ed|ing)\b",
    r"\bpending\b",
    r"\bprocessing\b",
    r"\binitiat(ion|e|ed|ing)\b",
    r"\bpin\b",
    r"\botp\b",
    r"\bbiometric\b",
    r"\b3d\s*secure\b",
    r"\bmerc?hant\b",
    r"\bacquirer\b",
    r"\bissuer\b",
    r"\binterchange\b",
    r"\bmdr\b",
    r"\bcorrespondent\s+bank\b",
    r"\bwhat.*(happen(s)?\s*(after|when|if)?)\b",
    r"\bwhy.*(payment|transaction|money)\b",
    r"\bhow.*(payment|transaction|settlement|compliance|kyc|aml|upi|neft|rtgs|imps)\b",
    r"\bwhat\s+is\s+(a\s+)?(payment|transaction|settlement|compliance|kyc|aml|upi|neft|rtgs|imps|swift|verification|fraud|authoris|authenti|sanction|beneficiar|chargeback|clearing|interbank|fintech|wallet|bank|card|otp|pin|biometric)\b",
    r"\bwhat\s+does\s+.{0,30}(payment|transaction|settlement|compliance|kyc|aml|upi|neft|rtgs|imps)\b",
    r"\bwhat\s+are\s+.{0,30}(payment|transaction|settlement|compliance|kyc|aml|check|rule|requir)\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\btell\s+me\s+about\b",
]

_SUPPORTED_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _SUPPORTED_TOPIC_PATTERNS
]

# ---------------------------------------------------------------------------
# Prompt-injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[str] = [
    # Role-swap attempts
    r"ignore\s+(all\s+)?(previous|prior|above|system|your)\s+(instruction|prompt|rule|guideline|constraint)s?",
    r"ignore\s+your\s+(system\s+)?(prompt|instruction|rule|guideline|constraint)s?",
    r"disregard\s+(all\s+)?(previous|prior|above|system|your)\s+(instruction|prompt|rule|guideline|constraint)s?",
    r"forget\s+(everything|all|your\s+(instruction|prompt|rule|guideline|constraint))s?",
    r"you\s+are\s+(now|no\s+longer)\s+a",
    r"pretend\s+(you\s+are|to\s+be)\s+a",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?\w+",
    r"new\s+(instruction|prompt|system\s+prompt|role)s?\s*:",
    r"system\s*:\s*you\s+(are|must|should)",
    r"<\s*system\s*>",
    r"\[system\]",
    r"\\u003c\s*system",
    # Jailbreak phrases
    r"jailbreak",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"developer\s+mode",
    r"unlock\s+(all|your|hidden)\s+(mode|capabilit|feature)",
    # Attempt to leak system prompt
    r"(print|repeat|output|tell\s+me|reveal|show)\s+(your\s+)?(system\s+)?(prompt|instruction|context|guideline)s?",
    r"what\s+(are\s+)?your\s+(system\s+)?(instruction|prompt|guideline|constraint)s?",
    # Data exfiltration via prompt
    r"translate\s+(the\s+)?(above|previous|system)\s+(to|into)",
    r"base64\s*(encode|decode)",
    # Indirect injection via content
    r"###\s*instruction",
    r"---\s*instruction",
    r"user:\s*ignore",
    r"assistant:\s*sure",
]

_INJECTION_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS
]

# ---------------------------------------------------------------------------
# Financial advice / prohibited output patterns
# ---------------------------------------------------------------------------

_FINANCIAL_ADVICE_PATTERNS: list[str] = [
    # Specific financial recommendations
    r"\b(you\s+should|i\s+(recommend|advise|suggest)(\s+you)?)\s+(invest|buy|sell|transfer|withdraw|deposit)",
    r"\b(best|top|highest.rated)\s+(bank|investment|fund|stock|crypto|wallet|account)\b",
    r"\bi\s+would\s+(invest|buy|sell|put\s+money)",
    r"(invest|put|move)\s+your\s+(money|savings|funds|capital)\s+in",
    r"\byou\s+should\s+(invest|buy|sell|use|choose|go\s+with|switch\s+to)\b",
    r"\bbetter\s+than\b.{0,40}\b(bank|fund|stock|investment|wallet|account)\b",
    # Specific rate / return predictions
    r"\b(will|is\s+going\s+to)\s+(increase|decrease|rise|fall|go\s+up|go\s+down)\b.{0,20}\b(stock|price|rate|return)\b",
    r"\bguaranteed?\s+(return|profit|yield|interest)\b",
    r"\byou\s+will\s+(earn|make|profit|gain)\b",
    # Account / transaction action
    r"\b(close|switch|move)\s+your\s+(account|money|funds)\b",
    r"\btransfer\s+your\s+(money|funds|savings)\s+to\b",
]

_FINANCIAL_ADVICE_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _FINANCIAL_ADVICE_PATTERNS
]

# ---------------------------------------------------------------------------
# Sensitive / out-of-scope query patterns
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_PATTERNS: list[str] = [
    r"\b(access|check|look\s+up|retrieve|get|view)\s+(my|your|their)\s+(account|balance|statement|transaction(s)?|funds)\b",
    r"\bprocess\s+(a|my|this|the)\s+(payment|transaction|transfer)\b",
    r"\b(make|execute|complete|do|perform|initiate|send)\s+(a|my|this|the)\s+(payment|transaction|transfer|wire)\b",
    r"\bsend\s+(money|funds|cash|₹|\$|rs\.?\s*\d)",
    r"\b(tell\s+me\s+)?(my|your|their)\s+(account\s+number|card\s+number|cvv|pin|password|otp|secret)\b",
    r"\bwhat\s+is\s+my\s+(balance|account|pin|password)\b",
    r"\b(hack|bypass|circumvent|exploit|steal)\b",
    r"\b(illegall?y?|fraudulentl?y?)\b",
    r"\b(launder|laundering)\s+money\b",  # asking HOW TO launder (education is fine; instructions are not)
    r"\bhow\s+to\s+(evade|avoid|bypass)\s+(kyc|aml|sanctions|compliance|detection|reporting)\b",
]

_OUT_OF_SCOPE_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _OUT_OF_SCOPE_PATTERNS
]

# ---------------------------------------------------------------------------
# Canned safe responses
# ---------------------------------------------------------------------------

UNSUPPORTED_TOPIC_RESPONSE = (
    "I can only answer questions about digital payment processes, transaction flows, "
    "compliance checks (KYC, AML, sanctions), verification, and settlement. "
    "Your question seems to be about something else. "
    "Please try rephrasing or ask a different question about FinTech payments."
)

PROMPT_INJECTION_RESPONSE = (
    "I noticed your message contains patterns that look like an attempt to change "
    "my operating instructions. I can only answer questions about digital payment "
    "processes and FinTech compliance. Please ask a genuine question and I'll do "
    "my best to help."
)

FINANCIAL_ADVICE_RESPONSE = (
    "I can explain how digital payments, compliance processes, and financial systems "
    "work in general terms, but I'm not able to give personalised financial advice, "
    "recommend specific products, or tell you where to invest your money. "
    "For personal financial decisions, please consult a qualified financial adviser."
)

OUT_OF_SCOPE_RESPONSE = (
    "I'm an informational assistant — I can explain how payment processes and "
    "compliance rules work, but I can't access accounts, process real transactions, "
    "or handle private financial data. For account-specific help, please contact "
    "your bank or payment provider directly."
)

NOT_IN_KB_RESPONSE = (
    "The knowledge base does not contain enough information to answer this question "
    "confidently. I've summarised what the retrieved context does say, but please "
    "verify with your bank or a qualified professional for anything important."
)

# ---------------------------------------------------------------------------
# Public guard functions
# ---------------------------------------------------------------------------


def check_prompt_injection(text: str) -> GuardResult:
    """Return a failed ``GuardResult`` if *text* contains prompt-injection patterns."""
    for pattern in _INJECTION_PATTERNS_COMPILED:
        if pattern.search(text):
            return GuardResult(
                safe=False,
                reason=PROMPT_INJECTION_RESPONSE,
                category="prompt_injection",
            )
    return GuardResult(safe=True)


def check_out_of_scope(text: str) -> GuardResult:
    """Return a failed ``GuardResult`` if *text* requests real account access or
    asks for instructions on illegal activity."""
    for pattern in _OUT_OF_SCOPE_PATTERNS_COMPILED:
        if pattern.search(text):
            return GuardResult(
                safe=False,
                reason=OUT_OF_SCOPE_RESPONSE,
                category="out_of_scope",
            )
    return GuardResult(safe=True)


def check_supported_topic(text: str) -> GuardResult:
    """Return a failed ``GuardResult`` if *text* does not appear to be about
    FinTech payments, compliance, or related topics."""
    for pattern in _SUPPORTED_PATTERNS_COMPILED:
        if pattern.search(text):
            return GuardResult(safe=True)
    return GuardResult(
        safe=False,
        reason=UNSUPPORTED_TOPIC_RESPONSE,
        category="unsupported_topic",
    )


def check_financial_advice(text: str) -> GuardResult:
    """Return a failed ``GuardResult`` if *text* (typically an LLM output) contains
    financial advice or specific product recommendations."""
    for pattern in _FINANCIAL_ADVICE_COMPILED:
        if pattern.search(text):
            return GuardResult(
                safe=False,
                reason=FINANCIAL_ADVICE_RESPONSE,
                category="financial_advice",
            )
    return GuardResult(safe=True)


def run_input_guards(user_query: str) -> GuardResult:
    """Run all input-side guards in priority order.

    Guards are applied in the order:
    1. Prompt injection (highest priority – always block immediately).
    2. Out-of-scope / harmful requests.
    3. Unsupported topic.

    Returns the first failing ``GuardResult``, or a passing one if all guards
    pass.
    """
    for guard in (check_prompt_injection, check_out_of_scope, check_supported_topic):
        result = guard(user_query)
        if not result.safe:
            return result
    return GuardResult(safe=True)


def run_output_guards(llm_output: str) -> GuardResult:
    """Run all output-side guards on the LLM-generated response.

    Guards applied:
    1. Financial advice / product recommendation check.

    Returns the first failing ``GuardResult``, or a passing one if all pass.
    """
    for guard in (check_financial_advice,):
        result = guard(llm_output)
        if not result.safe:
            return result
    return GuardResult(safe=True)


def is_grounded(answer: str, context_chunks: list[str], min_overlap: int = 3) -> bool:
    """Heuristic check: does *answer* contain at least *min_overlap* content words
    that also appear in *context_chunks*?

    This is a lightweight grounding check.  A ``False`` result means the answer
    may be hallucinated; it does NOT mean the answer is definitely wrong.

    Parameters
    ----------
    answer:
        The LLM-generated answer text.
    context_chunks:
        List of document chunk strings retrieved from the vector store.
    min_overlap:
        Minimum number of unique content words that must appear in both the
        answer and the context.  Defaults to 3.
    """
    # Tokenise simply: lowercase alphabetic words longer than 3 chars
    _stop = {
        "that", "this", "with", "from", "have", "will", "more", "been",
        "they", "them", "their", "what", "when", "where", "which", "would",
        "could", "should", "about", "into", "than", "then", "also", "some",
        "your", "just", "such", "each", "over", "after", "these", "those",
        "very", "only", "both", "most", "much", "many", "even", "well",
    }

    def _words(text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _stop
        }

    answer_words = _words(answer)
    context_text = " ".join(context_chunks)
    context_words = _words(context_text)
    overlap = answer_words & context_words
    return len(overlap) >= min_overlap
