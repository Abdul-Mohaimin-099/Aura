"""
graph.py — LangGraph orchestration for Aura.
Replaces the manual if/else dispatch in app.py with a compiled StateGraph.
"""

from __future__ import annotations

import re
from typing import Any, Generator, Union
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

import router
import chat_chain
import rag_chain
import search_agent
import config


# ── State ────────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """State carried through the Aura orchestration graph."""

    # Inputs (set by caller)
    query: str
    chat_history: list  # LangChain HumanMessage/AIMessage history
    search_history: list[dict]  # Gemini format: {"role", "text"}
    retriever: Any  # BaseRetriever (not serializable)
    ocr_text: str
    has_retriever: bool
    has_ocr: bool
    code: str
    student_grade: str

    # Outputs (set by nodes)
    intent: str  # "RAG" | "WEB_SEARCH" | "GENERAL" | "OCR"
    answer: str
    sources: list[str]


_AURA_CODE_MENTOR_SYSTEM: str = (
    "You are Aura Code Mentor, a pedagogical coding teacher for Classes 1-12. "
    "Analyze student code using Socratic questioning. "
    "Do not immediately provide corrected full code. "
    "If a syntax issue exists, explicitly mention it and include the line number when possible. "
    "Use encouraging language and guide the student to reason about the fix."
)


def _extract_syntax_warning(review_text: str) -> str | None:
    """Extract and format a Streamlit warning for syntax errors with line number."""
    text = review_text.lower()
    if "syntax" not in text and "indentationerror" not in text:
        return None

    line_match = re.search(r"\bline\s*(\d+)\b", review_text, flags=re.IGNORECASE)
    if not line_match:
        line_match = re.search(r"\bL(\d+)\b", review_text)

    if line_match:
        line_no = line_match.group(1)
        return f"⚠️ Warning: Possible syntax error around line {line_no}."

    return (
        "⚠️ Warning: Possible syntax error detected, but line number was not identified."
    )


def code_review_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node for code review with token streaming and syntax warning extraction.

    Inputs from state:
      - code: str
      - student_grade: str

    Returns:
      - answer_stream: Generator[str] for st.write_stream(...)
      - warning_box: dict populated after stream completes, e.g. warning_box["warning"]

    Streamlit usage example:
      out = code_review_node(state)
      st.write_stream(out["answer_stream"])
      if out["warning_box"]["warning"]:
          st.warning(out["warning_box"]["warning"])
    """
    code = state.get("code", "")
    student_grade = str(state.get("student_grade", "Unknown"))

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        streaming=True,
    )

    messages = [
        SystemMessage(content=_AURA_CODE_MENTOR_SYSTEM),
        HumanMessage(
            content=(
                f"Student grade level: {student_grade}\n"
                "Task: Review the student's code. Use a supportive, teaching-first style.\n"
                "If you detect a syntax error, state the likely line number clearly.\n\n"
                "Student code:\n"
                f"```\n{code}\n```"
            )
        ),
    ]

    warning_box: dict[str, str | None] = {"warning": None, "full_text": None}

    def _stream() -> Generator[str, None, None]:
        chunks: list[str] = []
        for chunk in llm.stream(
            messages,
            config=config.get_trace_config(
                run_name="graph.code_review.stream",
                tags=["graph", "code-review", "stream"],
                metadata={"module": "graph", "node": "code_review_node"},
            ),
        ):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not token:
                continue
            chunks.append(token)
            yield token

        full_text = "".join(chunks)
        warning_box["full_text"] = full_text
        warning_box["warning"] = _extract_syntax_warning(full_text)

    return {
        "answer_stream": _stream(),
        "warning_box": warning_box,
    }


# ── Nodes ────────────────────────────────────────────────────


def router_node(state: GraphState) -> dict:
    """Classify user intent and validate against available resources."""
    intent = router.classify(
        state["query"],
        has_vectorstore=state["has_retriever"],
        has_ocr=state["has_ocr"],
    )
    intent = router.validate_intent(
        intent,
        has_vectorstore=state["has_retriever"],
        has_ocr=state["has_ocr"],
    )
    return {"intent": intent.value}


def general_chat_node(state: GraphState) -> dict:
    """Handle general conversation via Groq Llama 3.3."""
    answer = chat_chain.handle(state["query"], state["chat_history"])
    return {"answer": answer, "sources": []}


def rag_node(state: GraphState) -> dict:
    """Retrieve documents and generate answer via RAG pipeline."""
    answer, sources = rag_chain.handle(
        state["query"],
        state["retriever"],
        state["chat_history"],
    )
    return {"answer": answer, "sources": sources}


def web_search_node(state: GraphState) -> dict:
    """Search the web via Gemini with Google Search grounding."""
    answer, sources = search_agent.handle(state["query"], state["search_history"])
    return {"answer": answer, "sources": sources}


def ocr_node(state: GraphState) -> dict:
    """Answer questions about analyzed image content."""
    augmented = f"Image analysis:\n{state['ocr_text']}\n\nQuestion: {state['query']}"
    answer = chat_chain.handle(augmented, state["chat_history"])
    return {"answer": answer, "sources": []}


# ── Routing ──────────────────────────────────────────────────


def _route_by_intent(state: GraphState) -> str:
    """Conditional edge: return the node name matching the classified intent."""
    return state["intent"]


# ── Graph construction ───────────────────────────────────────


def build_graph():
    """Construct and compile the Aura orchestration graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("GENERAL", general_chat_node)
    workflow.add_node("RAG", rag_node)
    workflow.add_node("WEB_SEARCH", web_search_node)
    workflow.add_node("OCR", ocr_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        _route_by_intent,
        {"GENERAL": "GENERAL", "RAG": "RAG", "WEB_SEARCH": "WEB_SEARCH", "OCR": "OCR"},
    )
    for node in ("GENERAL", "RAG", "WEB_SEARCH", "OCR"):
        workflow.add_edge(node, END)

    return workflow.compile()


# Module-level compiled graph (available for invoke / tracing / testing)
graph = build_graph()


# ── Streaming helper for Streamlit ───────────────────────────


def stream_response(
    query: str,
    chat_history: list,
    search_history: list[dict],
    retriever: Any,
    ocr_text: str,
) -> tuple[str, Union[Generator[str, None, None], str], list[str]]:
    """
    Route the query and return streaming-ready output for Streamlit.

    Returns (intent, content, sources) where content is a Generator[str]
    for streaming intents or a plain str for web_search.
    """
    has_retriever = retriever is not None
    has_ocr = bool(ocr_text)

    intent = router.classify(query, has_vectorstore=has_retriever, has_ocr=has_ocr)
    intent = router.validate_intent(
        intent, has_vectorstore=has_retriever, has_ocr=has_ocr
    )

    if intent == router.Intent.GENERAL:
        return intent.value, chat_chain.stream(query, chat_history), []

    elif intent == router.Intent.RAG:
        gen, sources = rag_chain.stream(query, retriever, chat_history)
        return intent.value, gen, sources

    elif intent == router.Intent.OCR:
        augmented = f"Image analysis:\n{ocr_text}\n\nQuestion: {query}"
        return intent.value, chat_chain.stream(augmented, chat_history), []

    else:  # WEB_SEARCH
        answer, sources = search_agent.handle(query, search_history)
        return intent.value, answer, sources
