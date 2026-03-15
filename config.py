"""
config.py — Central configuration for Aura.
Loads .env, exposes typed constants for models, keys, and tracing.
"""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# ── API keys ──────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Model identifiers ────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK: str = "gemini-2.5-flash-lite"
GROQ_MODEL: str = "llama-3.3-70b-versatile"  # confirmed live on Groq
GROQ_ROUTER_MODEL: str = "llama-3.1-8b-instant"  # confirmed live on Groq — fast router
EMBEDDING_MODEL: str = (
    "all-MiniLM-L6-v2"  # Local HuggingFace — no API quota, fast, reliable
)
VISION_MODEL: str = "gemini-2.5-flash"
VISION_MODEL_FALLBACK: str = "gemini-2.5-flash-lite"
# Search agent — Gemini 3.1 Flash Lite Preview (primary) with Google Search grounding
# Falls back to 2.5 Flash Lite (stable) when preview quota is exhausted
SEARCH_MODEL: str = "gemini-3.1-flash-lite-preview"
SEARCH_MODEL_FALLBACK: str = "gemini-2.5-flash-lite"

# ── Chunking parameters ──────────────────────────────────────
# Sized for markdown content (tables/headings add overhead vs plain text)
PARENT_CHUNK_SIZE: int = 1500
PARENT_CHUNK_OVERLAP: int = 200
CHILD_CHUNK_SIZE: int = 500
CHILD_CHUNK_OVERLAP: int = 80

# ── Token / context limits ──────────────────────────────────
MAX_CONTEXT_CHARS: int = 6000  # hard cap on total RAG context string
MAX_HISTORY_TURNS: int = 2  # number of user/assistant pairs to keep in context
MAX_OUTPUT_TOKENS: int = 1024  # cap LLM output tokens

# ── LangSmith ────────────────────────────────────────────────
LANGSMITH_TRACING: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Aura")


def get_tracer() -> list[BaseCallbackHandler]:
    """Return a LangChainTracer callback list when tracing is enabled."""
    if not LANGSMITH_TRACING:
        return []
    try:
        from langchain_core.tracers import LangChainTracer

        return [LangChainTracer(project_name=LANGSMITH_PROJECT)]
    except Exception:
        return []


def get_trace_config(
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent LangSmith trace config for all invoke/stream calls."""
    trace_tags = ["aura", *(tags or [])]
    trace_metadata: dict[str, Any] = {"project": LANGSMITH_PROJECT}
    if metadata:
        trace_metadata.update(metadata)

    return {
        "callbacks": get_tracer(),
        "run_name": run_name,
        "tags": trace_tags,
        "metadata": trace_metadata,
    }
