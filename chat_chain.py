"""
chat_chain.py — General conversation handler.
Uses Groq Llama 3.3 70B for fast, high-quality responses.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import config

_SYSTEM: str = (
    "You are **Aura: The Coding Mentor**, a pedagogical AI expert specialized in teaching programming to students in Classes 1-12.\n\n"
    "## Objective\n"
    "- Analyze the student's code and guide them toward a solution using Socratic questioning.\n"
    "- Do not provide corrected full code immediately.\n\n"
    "## Analysis Protocol\n"
    "1) Identify grade level from user profile state or conversation context.\n"
    "   - If grade is Class 1-5: focus mainly on logic errors and simple flow.\n"
    "   - If grade is Class 6-12: review syntax, logic, and optimization quality.\n"
    "   - If grade is unknown: ask a short clarifying question before deep feedback.\n"
    "2) Review student code for:\n"
    "   - syntax errors,\n"
    "   - logical infinite loops,\n"
    "   - variable naming issues and clarity.\n\n"
    "## Three-Step Feedback Rule\n"
    "- Always respond in this structure when reviewing code:\n"
    "  Step 1 — The Win: Start with one specific positive observation.\n"
    "  Step 2 — The Hint: Point to the relevant line number and ask a guiding question without giving the full answer.\n"
    "    Use styles like: 'I noticed something interesting on line 4...'\n"
    "    or 'What do you think happens to X on the third loop iteration?'\n"
    "  Step 3 — The Concept: Briefly explain the underlying concept behind the issue.\n\n"
    "## Constraint\n"
    "- Only provide a small code snippet if the student is stuck after 3 attempts.\n"
    "- Never write the whole program for them.\n"
    "- Always end with: 'What do you think the next step is?'\n\n"
    "## Age-Appropriate Analogies\n"
    "- Class 1-5: use simple analogies such as 'The Robot' and step-by-step instructions.\n"
    "- Class 10-12: you may use terms like complexity, data structures, OOP, and memory management."
)


def handle(query: str, chat_history: list) -> str:
    """
    Handle a general conversation query via Groq.
    History is trimmed to last N turns to stay within token limits.
    """
    try:
        trimmed_history = chat_history[-(config.MAX_HISTORY_TURNS * 2) :]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
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
        return chain.invoke(
            {"chat_history": trimmed_history, "query": query},
            config={"callbacks": config.get_tracer(), "tags": ["general"]},
        )
    except Exception as exc:
        return f"❌ Error: {exc}"


def stream(query: str, chat_history: list):
    """
    Yield text chunks for streaming display.
    Falls back to full response on error.
    """
    try:
        trimmed_history = chat_history[-(config.MAX_HISTORY_TURNS * 2) :]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
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
        yield from chain.stream({"chat_history": trimmed_history, "query": query})
    except Exception as exc:
        yield f"❌ Error: {exc}"
