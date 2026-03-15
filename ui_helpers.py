"""
ui_helpers.py — Premium dark UI helpers for Aura.
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as _components

# ── CSS loader ────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "styles.css"
_CSS_CACHE: str | None = None


def inject_css() -> None:
    global _CSS_CACHE
    if _CSS_CACHE is None:
        _CSS_CACHE = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{_CSS_CACHE}</style>", unsafe_allow_html=True)


# ── Intent maps ───────────────────────────────────────────────
_CSS_MAP = {
    "RAG": "intent-rag",
    "WEB_SEARCH": "intent-web",
    "GENERAL": "intent-gen",
    "OCR": "intent-ocr",
}
_LABEL_MAP = {
    "RAG": "Document RAG",
    "WEB_SEARCH": "Web Search",
    "GENERAL": "General Chat",
    "OCR": "Vision Analysis",
}

# ── Suggestion chips ─────────────────────────────────────────
_CHIPS = [
    ("📄", "Summarize a document"),
    ("🌍", "Latest news"),
    ("💡", "Explain a concept"),
    ("✍️", "Help me write"),
]


def render_welcome() -> str | None:
    """
    Render the premium welcome screen with Aura icon, gradient heading,
    subtitle, and glass suggestion cards.
    Returns the chip label clicked, or None.
    """
    st.markdown('<div style="height:4rem;"></div>', unsafe_allow_html=True)

    # ── Aura brand row (icon + name) ─────────────────────────
    left_pad, brand_col, right_pad = st.columns([2.2, 1.6, 2.2])
    with brand_col:
        icon_col, name_col = st.columns([0.9, 1.1], vertical_alignment="center")
        with icon_col:
            st.image("Aura_Icon.png", width=74)
        with name_col:
            st.markdown(
                '<span class="welcome-brand-name">Aura</span>',
                unsafe_allow_html=True,
            )

    # ── Heading ───────────────────────────────────────────────
    st.markdown(
        '<p class="welcome-heading">'
        '<span class="welcome-heading-accent">What can I help with?</span>'
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="welcome-sub">'
        "Upload documents, search the web, analyze images, or just chat."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Chip pills — centered row ────────────────────────────
    pad = 0.5
    cols = st.columns([pad] + [1] * len(_CHIPS) + [pad])
    for col, (icon, label) in zip(cols[1:-1], _CHIPS):
        with col:
            if st.button(
                f"{icon}  {label}",
                key=f"chip_{label}",
                use_container_width=True,
            ):
                return label

    return None


def render_intent_badge(intent: str) -> None:
    """Render a frosted glass pill indicating which agent handled the response."""
    css_cls = _CSS_MAP.get(intent, "intent-gen")
    label = _LABEL_MAP.get(intent, intent)
    st.markdown(
        f'<span class="intent-badge {css_cls}">'
        f'<span class="intent-badge-dot"></span>'
        f"{label}"
        f"</span>",
        unsafe_allow_html=True,
    )


def render_sources(sources: list[str]) -> None:
    """Render RAG source document chips below an assistant response."""
    chips_html = " ".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
    st.markdown(
        f'<div class="source-row">{chips_html}</div>',
        unsafe_allow_html=True,
    )


def append_inline_source_icons(
    content: str, sources: list[str], icon: str = "🔗"
) -> str:
    """Append icon-only markdown links for URL sources at the end of content."""
    url_sources = [
        s
        for s in sources
        if isinstance(s, str) and s.lower().startswith(("http://", "https://"))
    ]
    if not url_sources:
        return content

    icon_links = " ".join(f"[{icon}]({url})" for url in url_sources)
    return f"{content.rstrip()} {icon_links}".strip()


def _inject_input_js() -> None:
    """Typewriter placeholder animation + expand_more fix via parent-frame JS."""
    phrases = [
        "Ask Aura anything\u2026",
        "What\u2019s happening in the world today?",
        "Help me analyze a document\u2026",
        "Explain a concept simply\u2026",
        "Summarize my uploaded PDF\u2026",
        "Help me draft an email\u2026",
        "Latest AI research and news\u2026",
        "What does this image show?",
    ]
    ph_json = json.dumps(phrases)
    js = f"""
    (function() {{
        var p = window.parent;
        if (p._auraTyper) {{ p._auraTyper.stop(); }}

        // ── Fix: hide expand_more text icon inside any popover trigger ──
        function hideExpandMore() {{
            // Target stIcon spans whose own text content is exactly expand_more
            var icons = p.document.querySelectorAll(
                '[data-testid="stPopover"] > button [data-testid="stIcon"]'
            );
            icons.forEach(function(el) {{
                el.style.display = 'none';
            }});
            // Also catch any stIcon that renders as literal text 'expand_more'
            var allIcons = p.document.querySelectorAll('[data-testid="stIcon"]');
            allIcons.forEach(function(el) {{
                // Use childNodes to avoid picking up SVG child text
                var direct = '';
                el.childNodes.forEach(function(n) {{
                    if (n.nodeType === 3) direct += n.nodeValue;
                }});
                if (direct.trim() === 'expand_more') {{
                    el.style.visibility = 'hidden';
                    el.style.fontSize = '0';
                    el.style.lineHeight = '0';
                    el.style.width = '0';
                    el.style.overflow = 'hidden';
                }}
            }});
        }}
        hideExpandMore();
        var obsIcon = new MutationObserver(hideExpandMore);
        obsIcon.observe(p.document.body, {{ childList: true, subtree: true }});

        // ── Typewriter placeholder animation ──
        var phrases = {ph_json};
        var idx = 0, ch = 0, del = false, t = null;

        function el() {{ return p.document.querySelector('[data-testid="stChatInputTextArea"]'); }}

        function tick() {{
            var e = el();
            if (!e) {{ t = setTimeout(tick, 400); return; }}
            if (p.document.activeElement === e && e.value.length > 0) {{
                t = setTimeout(tick, 400); return;
            }}
            var txt = phrases[idx];
            if (!del) {{
                ch++;
                e.placeholder = txt.slice(0, ch);
                if (ch >= txt.length) {{ t = setTimeout(function(){{ del=true; tick(); }}, 2600); }}
                else {{ t = setTimeout(tick, 48); }}
            }} else {{
                ch--;
                e.placeholder = txt.slice(0, ch);
                if (ch <= 0) {{
                    del = false;
                    idx = (idx + 1) % phrases.length;
                    t = setTimeout(tick, 500);
                }} else {{ t = setTimeout(tick, 22); }}
            }}
        }}

        p._auraTyper = {{ stop: function() {{ if(t) clearTimeout(t); obsIcon.disconnect(); }} }};
        t = setTimeout(tick, 1200);
    }})();
    """
    _components.html(f"<script>{js}</script>", height=0, scrolling=False)


def _inject_sidebar_toggle() -> None:
    """
    Inject a floating pill button at the right edge of the sidebar so the user
    can collapse / expand the sidebar without hunting for the tiny header button.
    The button delegates clicks to the native stSidebarCollapseButton.
    """
    js = """
    (function() {
        var p = window.parent;
        if (p._auraSidebarToggle) return;   // already injected

        // Build the pill button
        var btn = p.document.createElement('div');
        btn.id = 'aura-sidebar-toggle';
        // Chevron-left SVG: collapse = point left, expand = rotated right via CSS class
        btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>';
        p.document.body.appendChild(btn);

        function isSidebarOpen() {
            var colBtn = p.document.querySelector(
                '[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button'
            );
            return !!colBtn;
        }

        function syncState() {
            var open = isSidebarOpen();
            var sb = p.document.querySelector('[data-testid="stSidebar"]');
            var sbWidth = sb ? sb.getBoundingClientRect().width : 0;
            if (open && sbWidth > 0) {
                btn.classList.remove('sidebar-collapsed');
                btn.style.left = Math.max(sbWidth - 12, 0) + 'px';
            } else {
                btn.classList.add('sidebar-collapsed');
                btn.style.left = '0px';
            }
        }

        btn.addEventListener('click', function() {
            var collapseBtn = p.document.querySelector(
                '[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button'
            );
            var expandBtn = p.document.querySelector(
                '[data-testid="collapsedControl"] button'
            );
            if (collapseBtn) {
                collapseBtn.click();
            } else if (expandBtn) {
                expandBtn.click();
            }
            setTimeout(syncState, 380);
        });

        // Watch for sidebar open/close DOM changes
        var mo = new MutationObserver(function() { syncState(); });
        mo.observe(p.document.body, { childList: true, subtree: true, attributes: true });
        syncState();

        p._auraSidebarToggle = true;
    })();
    """
    _components.html(f"<script>{js}</script>", height=0, scrolling=False)


def render_attach_bar() -> None:
    """
    Run the typewriter placeholder animation on the chat input and inject
    the sidebar toggle pill button.
    File uploaders are rendered in the sidebar by sidebar.py.
    """
    _inject_input_js()
    _inject_sidebar_toggle()
