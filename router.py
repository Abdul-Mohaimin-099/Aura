"""
router.py — Intent classification for incoming queries.
Uses Groq (fast 8B model) to classify into RAG / WEB_SEARCH / GENERAL / OCR.
"""

from __future__ import annotations

from enum import Enum

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

import config


class Intent(str, Enum):
    RAG = "RAG"
    WEB_SEARCH = "WEB_SEARCH"
    GENERAL = "GENERAL"
    OCR = "OCR"


_SYSTEM_PROMPT: str = (
    "You are a query router. Given the user's message and what resources "
    "are available, reply with EXACTLY one word:\n"
    "  • RAG        — if asking about uploaded documents / PDFs.\n"
    "  • OCR        — if asking about an uploaded image or its visual content.\n"
    "  • WEB_SEARCH — if the question needs real-time or current information.\n"
    "  • GENERAL    — for all other queries (creative writing, math, coding, etc.).\n"
    "Reply with ONLY the single word."
)


def classify(query: str, has_vectorstore: bool, has_ocr: bool = False) -> Intent:
    """Classify user intent via a fast Groq LLM call."""
    parts: list[str] = []
    if has_vectorstore:
        parts.append("A document knowledge base IS available.")
    else:
        parts.append("No document knowledge base is available.")
    if has_ocr:
        parts.append("Image OCR text IS available.")
    else:
        parts.append("No image has been uploaded.")
    context = " ".join(parts)

    try:
        llm = ChatGroq(
            model=config.GROQ_ROUTER_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
        )
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"[Context: {context}]\n\nUser query: {query}"),
            ],
            config={"callbacks": config.get_tracer(), "tags": ["router"]},
        )
        label = response.content.strip().upper()
        if label in Intent.__members__:
            return Intent(label)
    except Exception:
        pass

    # Fallback: prioritise available resources
    if has_vectorstore:
        return Intent.RAG
    if has_ocr:
        return Intent.OCR
    return Intent.GENERAL


def validate_intent(
    intent: Intent,
    has_vectorstore: bool,
    has_ocr: bool,
) -> Intent:
    """Correct impossible intents (e.g. RAG with no vectorstore)."""
    if intent == Intent.RAG and not has_vectorstore:
        return Intent.OCR if has_ocr else Intent.GENERAL
    if intent == Intent.OCR and not has_ocr:
        return Intent.RAG if has_vectorstore else Intent.GENERAL
    return intent
