"""
vision_service.py — Gemini Vision OCR / image reasoning.
Uses the google.genai SDK to analyse uploaded images.
"""

from __future__ import annotations

from typing import Optional

from google import genai
from google.genai import types

import config

# ── Lazy client ───────────────────────────────────────────────
_client: genai.Client | None = None

_DEFAULT_PROMPT: str = (
    "You are an Academic Document Digitizer. "
    "You receive photos of university noticeboards, circulars, and handwritten routines. "
    "Follow these rules strictly:\n\n"
    "1) Transcribe all visible text accurately, preserving original wording, numbers, punctuation, "
    "line breaks, and headings as closely as possible.\n"
    "2) Identify and extract key academic entities, especially:\n"
    "   - dates (exam dates, submission dates, event dates),\n"
    "   - deadlines (application, fee, assignment, registration, etc.),\n"
    "   - room numbers / venue references.\n"
    "3) Output the final extracted data as a clean Markdown table. Use columns:\n"
    "   | Source Text | Date | Deadline | Room/Venue | Notes |\n"
    "   Include one row per distinct item/announcement/schedule entry.\n"
    "4) If any text is unclear or illegible, do NOT guess. Mark uncertain values as `[illegible]` "
    "and add a short 'Missing / Unclear Parts' section listing exactly what could not be read.\n"
    "5) Do not invent details. If an item is absent, leave the table cell as `-`."
)


def _get_client() -> genai.Client:
    """Return (and cache) a google.genai Client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def analyze_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    prompt: Optional[str] = None,
) -> str:
    """
    Send an image to Gemini Vision for OCR / reasoning.
    Retries with VISION_MODEL_FALLBACK (gemini-2.5-flash-lite) on failure.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        mime_type:   MIME type (e.g. "image/png").
        prompt:      Optional targeted prompt.

    Returns:
        Model's textual analysis.

    Raises:
        RuntimeError: On API failure of both primary and fallback.
    """
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    gen_cfg = types.GenerateContentConfig(temperature=0.2, max_output_tokens=4096)
    contents = [prompt or _DEFAULT_PROMPT, image_part]

    for model, label in [
        (config.VISION_MODEL, "primary"),
        (config.VISION_MODEL_FALLBACK, "fallback"),
    ]:
        try:
            response = _get_client().models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
            if response and response.text:
                return response.text
            return "⚠️ The model returned an empty response for this image."
        except Exception as exc:
            last_exc = exc
            if label == "primary":
                continue  # try fallback
            raise RuntimeError(
                f"Gemini Vision failed on both models. Last error: {last_exc}"
            ) from last_exc
    # unreachable, but satisfies type checkers
    raise RuntimeError("Gemini Vision: unexpected exit")
