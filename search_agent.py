"""
search_agent.py — Search & Chat Agent.
Primary:  Gemini 3.1 Flash Lite Preview  (gemini-3.1-flash-lite-preview)
Fallback: Gemini 2.5 Flash Lite          (gemini-2.5-flash-lite)

Preview models have a low free-tier QPM quota. When a 429 RESOURCE_EXHAUSTED
is returned, this module retries up to 3 times with exponential back-off, then
automatically falls back to the stable 2.5 Flash Lite model so the user never
sees an error page.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types

import config

# ── Lazy Gemini client ────────────────────────────────────────
_gemini_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return (and cache) a Google GenAI client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _gemini_client


_SYSTEM: str = (
    "You are **Aura**, a smart AI assistant with real-time web access via Google Search. "
    "For questions about current events, news, prices, weather, sports, or anything time-sensitive, use Google Search before answering. "
    "For general chat and knowledge questions, answer directly and helpfully. "
    "Use markdown formatting. Be accurate, concise, and conversational."
)


def _call(model: str, contents: list[types.Content]) -> tuple[str, list[str]]:
    """Call generate_content and return answer text plus structured sources."""
    response = _get_client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.5,
            max_output_tokens=1024,
        ),
    )

    answer = (response.text or "").strip() or "⚠️ No response. Try rephrasing."

    sources: list[str] = []
    try:
        grounding = response.candidates[0].grounding_metadata
        for chunk in grounding.grounding_chunks or []:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None) and web.uri not in sources:
                sources.append(web.uri)
    except Exception:
        pass

    return answer, sources


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def handle(query: str, chat_history: list[dict]) -> tuple[str, list[str]]:
    """
    Process a query using Gemini with Google Search grounding.

    Retry logic:
      - Attempt primary model (3.1 Flash Lite Preview) up to 3 times.
      - Back-off: 1 s → 2 s between retries.
      - If all retries fail with 429, fall back to Gemini 2.5 Flash Lite.
      - Non-quota errors are returned immediately without retrying.
    """
    # Build typed Content list — keep last N messages from configured turns
    contents: list[types.Content] = []
    window = config.MAX_HISTORY_TURNS * 2
    for m in chat_history[-window:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["text"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=query)]))

    # ── Primary model with retries ────────────────────────────
    for attempt in range(3):
        try:
            return _call(config.SEARCH_MODEL, contents)
        except Exception as exc:
            if _is_quota_error(exc):
                if attempt < 2:
                    time.sleep(2**attempt)  # 1 s, 2 s
                    continue
                # All retries on preview model exhausted → fall back
                break
            return f"❌ Error: {exc}", []

    # ── Fallback: stable Gemini 2.5 Flash Lite ─────────────────
    try:
        return _call(config.SEARCH_MODEL_FALLBACK, contents)
    except Exception as exc:
        return f"❌ Error (quota exhausted on both models): {exc}", []
