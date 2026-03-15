"""
app.py — Streamlit entry point for Aura.
Unified agent: automatically routes queries to RAG, OCR, Web Search, or General Chat.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import graph as aura_graph
import ingestion
import vision_service
import ui_helpers
import sidebar
import config


_URL_PREFIXES = ("http://", "https://")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Aura – AI Assistant",
    page_icon="Aura_Icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_helpers.inject_css()

# ── Session state defaults ────────────────────────────────────
_DEFAULTS: dict = {
    "messages": [],
    "chat_sessions": [],  # list of {title, messages}
    "vectorstore": None,
    "retriever": None,
    "ocr_text": None,
    "pending_pdf": None,
    "pending_img": None,
}


def _init_session_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# One-time migration from dual-agent to unified
def _migrate_legacy_state() -> None:
    if "active_agent" not in st.session_state:
        return

    old_msgs = st.session_state.get("search_messages", []) + st.session_state.get(
        "rag_messages", []
    )
    if old_msgs and not st.session_state.messages:
        st.session_state.messages = old_msgs

    for old_key in ("active_agent", "search_messages", "rag_messages", "chip_query"):
        st.session_state.pop(old_key, None)


def _lc_history(messages: list[dict]) -> list:
    """Convert stored messages to LangChain format, pruned to context window."""
    window = config.MAX_HISTORY_TURNS * 2
    out = []
    for m in messages[-window:]:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


def _search_history(messages: list[dict]) -> list[dict]:
    """Convert stored messages to Gemini format, pruned to context window."""
    window = config.MAX_HISTORY_TURNS * 2
    return [
        {"role": "user" if m["role"] == "user" else "model", "text": m["content"]}
        for m in messages[-window:]
    ]


def _is_url_source(source: str) -> bool:
    return isinstance(source, str) and source.lower().startswith(_URL_PREFIXES)


def _non_url_sources(sources: list[str]) -> list[str]:
    return [s for s in sources if not _is_url_source(s)]


def _render_assistant_message(
    content: str, intent: str | None, sources: list[str]
) -> None:
    if intent:
        ui_helpers.render_intent_badge(intent)

    display_content = ui_helpers.append_inline_source_icons(content, sources)
    st.markdown(display_content)

    chip_sources = _non_url_sources(sources)
    if chip_sources:
        ui_helpers.render_sources(chip_sources)


def _render_message_history() -> None:
    for msg in st.session_state.messages:
        avatar = "Aura_Icon.png" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                _render_assistant_message(
                    content=msg["content"],
                    intent=msg.get("intent"),
                    sources=msg.get("sources") or [],
                )
            else:
                st.markdown(msg["content"])


def _collect_chat_input() -> tuple[str | None, list, object | None]:
    chat_pdf_files: list = []
    chat_img_file = None

    try:
        chat_payload = st.chat_input(
            "Ask Aura anything…",
            accept_file="multiple",
            file_type=["pdf", "png", "jpg", "jpeg", "webp"],
        )
    except TypeError:
        chat_payload = st.chat_input("Ask Aura anything…")

    query: str | None = None
    if isinstance(chat_payload, str):
        query = chat_payload
    elif chat_payload:
        query = getattr(chat_payload, "text", None)
        files = getattr(chat_payload, "files", None)

        if files is None and isinstance(chat_payload, dict):
            query = chat_payload.get("text", query)
            files = chat_payload.get("files")

        for file in files or []:
            file_type = (getattr(file, "type", "") or "").lower()
            file_name = (getattr(file, "name", "") or "").lower()
            if file_type == "application/pdf" or file_name.endswith(".pdf"):
                chat_pdf_files.append(file)
            elif file_type.startswith("image/") or file_name.endswith(_IMAGE_SUFFIXES):
                if chat_img_file is None:
                    chat_img_file = file

    return query, chat_pdf_files, chat_img_file


def _pdf_sig(files: list) -> list[tuple]:
    return [(f.name, f.size) for f in files] if files else []


def _process_pdf_uploads(pdf_files: list) -> None:
    if not pdf_files:
        return
    if _pdf_sig(pdf_files) == _pdf_sig(st.session_state.pending_pdf or []):
        return

    st.session_state.pending_pdf = list(pdf_files)
    all_docs = []
    for file in pdf_files:
        with st.spinner(f"Loading {file.name}…"):
            try:
                all_docs.extend(ingestion.load_pdf(file))
            except RuntimeError as err:
                st.error(str(err))

    if not all_docs:
        return

    with st.spinner("Building vector index…"):
        try:
            vectorstore = ingestion.build_vectorstore(all_docs)
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = ingestion.get_retriever(vectorstore)
            st.toast(f"✅ Indexed {len(all_docs)} pages!", icon="📚")
        except RuntimeError as err:
            st.error(str(err))


def _process_image_upload(img_file: object | None) -> None:
    if not img_file or img_file == st.session_state.pending_img:
        return

    st.session_state.pending_img = img_file
    with st.spinner("Analyzing image with Gemini Vision…"):
        try:
            st.session_state.ocr_text = vision_service.analyze_image(
                img_file.read(),
                img_file.type or "image/png",
            )
            st.toast("✅ Image analyzed!", icon="🖼️")
        except RuntimeError as err:
            st.error(str(err))


def _run_query(query: str) -> None:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    lc_hist = _lc_history(st.session_state.messages)
    search_hist = _search_history(st.session_state.messages)

    with st.spinner("Thinking…"):
        intent, content, sources = aura_graph.stream_response(
            query=query,
            chat_history=lc_hist,
            search_history=search_hist,
            retriever=st.session_state.retriever,
            ocr_text=st.session_state.ocr_text or "",
        )

    with st.chat_message("assistant", avatar="Aura_Icon.png"):
        srcs = sources or []
        if isinstance(content, str):
            _render_assistant_message(content=content, intent=intent, sources=srcs)
            answer = content
        else:
            if intent:
                ui_helpers.render_intent_badge(intent)
            answer = st.write_stream(content)

            chip_sources = _non_url_sources(srcs)
            if chip_sources:
                ui_helpers.render_sources(chip_sources)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "intent": intent,
            "sources": sources,
        }
    )


_init_session_state()
_migrate_legacy_state()


# ── Sidebar ───────────────────────────────────────────────────
sidebar.render()

# ── Display past messages ─────────────────────────────────────
_render_message_history()

# ── Welcome screen (only when no messages) ────────────────────
chip_suggestion: str | None = None
if not st.session_state.messages:
    chip_suggestion = ui_helpers.render_welcome()

# ── Chat input ────────────────────────────────────────────────
query, chat_pdf_files, chat_img_file = _collect_chat_input()

query = query or chip_suggestion

# ── Attachment toolbar (typewriter JS / quick actions) ────────────────────────
ui_helpers.render_attach_bar()
sidebar_pdf_files = st.session_state.get("_sidebar_pdf") or []
sidebar_img_file = st.session_state.get("_sidebar_img")

pdf_file = [*sidebar_pdf_files, *chat_pdf_files]
img_file = chat_img_file or sidebar_img_file

_process_pdf_uploads(pdf_file)
_process_image_upload(img_file)

# ── Handle query ──────────────────────────────────────────────
if query:
    _run_query(query)
