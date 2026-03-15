"""
sidebar.py — Premium dark sidebar for Aura.
Pull-in / pull-out via Streamlit's built-in collapse button.
"""

from __future__ import annotations

import streamlit as st


def _save_current_session() -> None:
    """Push the active conversation into chat_sessions (max 20 kept)."""
    msgs = st.session_state.messages
    user_msgs = [m for m in msgs if m["role"] == "user"]
    if not user_msgs:
        return
    title = user_msgs[0]["content"][:48] + (
        "\u2026" if len(user_msgs[0]["content"]) > 48 else ""
    )
    sessions: list = st.session_state.get("chat_sessions", [])
    sessions.insert(0, {"title": title, "messages": list(msgs)})
    st.session_state.chat_sessions = sessions[:20]


def _reset_runtime_state() -> None:
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.ocr_text = None
    st.session_state.pending_pdf = None
    st.session_state.pending_img = None


def _render_brand_header() -> None:
    st.markdown('<div class="sidebar-brand">', unsafe_allow_html=True)
    ic_col, name_col = st.columns([1, 3.5])
    with ic_col:
        st.image("Aura_Icon.png", width=30)
    with name_col:
        st.markdown(
            '<div style="display:flex;align-items:center;height:100%;padding-top:5px;">'
            '<span class="sidebar-app-name">Aura</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_new_chat_button() -> None:
    st.markdown('<div style="padding:2px 10px 10px;">', unsafe_allow_html=True)
    if st.button("+ New chat", use_container_width=True, key="new_chat"):
        _save_current_session()
        _reset_runtime_state()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _sync_sidebar_attachment_state() -> None:
    st.session_state["_sidebar_pdf"] = []
    st.session_state["_sidebar_img"] = None


def _render_context_status() -> None:
    has_kb = bool(st.session_state.get("vectorstore"))
    has_img = bool(st.session_state.get("ocr_text"))

    if has_kb:
        st.markdown(
            '<div class="kb-active">📚 Knowledge base active</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear document", use_container_width=True, key="clear_kb"):
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.pending_pdf = None
            st.rerun()

    if has_img:
        st.markdown(
            '<div class="kb-image">🖼️ Image loaded</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear image", use_container_width=True, key="clear_img"):
            st.session_state.ocr_text = None
            st.session_state.pending_img = None
            st.rerun()

    if has_kb or has_img:
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


def _restore_session(session: dict) -> None:
    st.session_state.messages = session["messages"]
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.ocr_text = None
    st.session_state.pending_pdf = None
    st.session_state.pending_img = None


def _render_chat_history() -> None:
    sessions: list = st.session_state.get("chat_sessions", [])
    live_msgs = st.session_state.messages
    live_user = [m for m in live_msgs if m["role"] == "user"]

    st.markdown(
        '<div class="sidebar-section-header">Chat history</div>',
        unsafe_allow_html=True,
    )

    if not live_user and not sessions:
        st.markdown(
            '<div style="padding:4px 16px 8px;font-size:0.8rem;color:#3b4055;">'
            "No conversations yet</div>",
            unsafe_allow_html=True,
        )
        return

    if live_user:
        live_title = live_user[0]["content"][:44] + (
            "\u2026" if len(live_user[0]["content"]) > 44 else ""
        )
        st.markdown(
            f'<div class="sidebar-history-item sidebar-history-active">'
            f"{live_title}</div>",
            unsafe_allow_html=True,
        )

    for idx, session in enumerate(sessions[:10]):
        if st.button(
            session["title"],
            key=f"session_{idx}",
            use_container_width=True,
        ):
            _restore_session(session)
            st.rerun()


def render() -> None:
    """Render the sidebar (collapsible via built-in Streamlit toggle)."""
    with st.sidebar:
        _render_brand_header()
        _render_new_chat_button()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        _sync_sidebar_attachment_state()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        _render_context_status()
        _render_chat_history()
