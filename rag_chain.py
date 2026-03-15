"""
rag_chain.py — RAG query handler.
Retrieves chunks, rebuilds parent context, generates answer via Llama 3.3 70B.
Context is markdown-formatted (extracted via PyMuPDF4LLM).
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

import config

_RAG_SYSTEM: str = (
    "You are **Aura**, a knowledgeable AI assistant. Answer using ONLY "
    "the retrieved context below. If the context is insufficient, say so "
    "honestly. Cite document sections in your answer.\n\n"
    "The context is extracted as markdown and may contain tables, headings, "
    "and structured formatting — use this structure to give precise answers.\n\n"
    "## Retrieved Context\n{context}"
)


def handle(
    query: str,
    retriever: BaseRetriever,
    chat_history: list,
) -> tuple[str, list[str]]:
    """
    Run the RAG pipeline: retrieve → generate.
    Context and history are hard-capped to avoid 413 errors.
    """
    try:
        docs = retriever.invoke(
            query,
            config={"callbacks": config.get_tracer(), "tags": ["rag-retrieval"]},
        )

        # De-duplicate using parent content, cap total context size
        seen: set[str] = set()
        context_parts: list[str] = []
        sources: list[str] = []
        total_chars = 0

        for doc in docs:
            parent = doc.metadata.get("parent_content", doc.page_content)
            if parent not in seen:
                if total_chars + len(parent) > config.MAX_CONTEXT_CHARS:
                    # Try the child chunk instead — smaller
                    snippet = doc.page_content[: config.MAX_CONTEXT_CHARS - total_chars]
                    if snippet.strip():
                        context_parts.append(snippet)
                        total_chars += len(snippet)
                    break
                seen.add(parent)
                context_parts.append(parent)
                total_chars += len(parent)
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            ref = f"{src} (p.{page})" if page != "" else src
            if ref not in sources:
                sources.append(ref)

        context_text = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "No relevant context found."
        )

        # Trim history to last N turns to stay within token limits
        trimmed_history = chat_history[-(config.MAX_HISTORY_TURNS * 2) :]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM),
                MessagesPlaceholder("chat_history"),
                ("human", "{query}"),
            ]
        )
        llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.4,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": context_text,
                "chat_history": trimmed_history,
                "query": query,
            },
            config={"callbacks": config.get_tracer(), "tags": ["rag-generation"]},
        )
        return answer, sources

    except Exception as exc:
        return f"❌ RAG pipeline error: {exc}", []


def stream(
    query: str,
    retriever: BaseRetriever,
    chat_history: list,
) -> tuple:
    """
    Run the RAG pipeline with streaming generation.
    Returns (generator, sources_list).
    The generator yields text chunks; sources are resolved before streaming starts.
    """
    try:
        docs = retriever.invoke(query)

        seen: set[str] = set()
        context_parts: list[str] = []
        sources: list[str] = []
        total_chars = 0

        for doc in docs:
            parent = doc.metadata.get("parent_content", doc.page_content)
            if parent not in seen:
                if total_chars + len(parent) > config.MAX_CONTEXT_CHARS:
                    snippet = doc.page_content[: config.MAX_CONTEXT_CHARS - total_chars]
                    if snippet.strip():
                        context_parts.append(snippet)
                        total_chars += len(snippet)
                    break
                seen.add(parent)
                context_parts.append(parent)
                total_chars += len(parent)
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            ref = f"{src} (p.{page})" if page != "" else src
            if ref not in sources:
                sources.append(ref)

        context_text = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "No relevant context found."
        )

        trimmed_history = chat_history[-(config.MAX_HISTORY_TURNS * 2) :]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM),
                MessagesPlaceholder("chat_history"),
                ("human", "{query}"),
            ]
        )
        llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.4,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            streaming=True,
        )
        chain = prompt | llm | StrOutputParser()

        def _gen():
            yield from chain.stream(
                {
                    "context": context_text,
                    "chat_history": trimmed_history,
                    "query": query,
                }
            )

        return _gen(), sources

    except Exception as exc:

        def _err():
            yield f"❌ RAG pipeline error: {exc}"

        return _err(), []
