"""
ingestion.py — PDF / text loading, parent-child chunking, FAISS indexing,
and MMR retriever construction.
Uses PyMuPDF4LLM for high-quality markdown extraction from PDFs.
"""

from __future__ import annotations

import os
import tempfile
from typing import BinaryIO

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

# ── Lazy embeddings (loaded once, reused) ──────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and cache) the local HuggingFace embeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Loaders ───────────────────────────────────────────────────


def load_pdf(uploaded_file: BinaryIO) -> list[Document]:
    """Extract pages from an uploaded PDF as LLM-optimised markdown via PyMuPDF4LLM."""
    import pymupdf4llm

    # Use mkstemp so the file is closed before pymupdf opens it (Windows compat)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded_file.read())

        md_pages = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)

        source = getattr(uploaded_file, "name", "uploaded.pdf")
        docs: list[Document] = []
        for page in md_pages:
            text = page["text"].strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "page": page["metadata"]["page_number"],
                    },
                )
            )
        return docs
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF: {exc}") from exc
    finally:
        os.unlink(tmp_path)


def load_text(raw_text: str, source: str = "pasted_text") -> list[Document]:
    """Wrap raw text into LangChain Documents."""
    return [Document(page_content=raw_text, metadata={"source": source})]


# ── Splitting (parent → child) ───────────────────────────────


def split_documents(docs: list[Document]) -> list[Document]:
    """Split into large parents then smaller children for retrieval.

    Uses markdown-aware separators so headings, tables, and code blocks
    are kept intact whenever possible.
    """
    _MD_SEPARATORS = [
        "\n## ",      # h2
        "\n### ",     # h3
        "\n#### ",    # h4
        "\n\n",       # paragraph break
        "\n",         # line break
        " ",
        "",
    ]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=_MD_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=_MD_SEPARATORS,
    )
    children: list[Document] = []
    for idx, parent in enumerate(parent_splitter.split_documents(docs)):
        for child in child_splitter.split_documents([parent]):
            child.metadata["parent_index"] = idx
            child.metadata["parent_content"] = parent.page_content
            children.append(child)
    return children


# ── FAISS vector store ────────────────────────────────────────


def build_vectorstore(documents: list[Document]) -> FAISS:
    """Embed child chunks and build a FAISS index."""
    if not documents:
        raise RuntimeError("No documents provided — cannot build index.")
    try:
        chunks = split_documents(documents)
        if not chunks:
            raise RuntimeError(
                "Document splitting produced no chunks. Check PDF content."
            )
        return FAISS.from_documents(chunks, _get_embeddings())
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"FAISS build failed: {exc}") from exc


# ── MMR retriever (no external API calls) ─────────────────────


def get_retriever(vectorstore: FAISS, k: int = 6) -> BaseRetriever:
    """
    Return a Max Marginal Relevance retriever.
    MMR balances relevance with diversity — better results than plain similarity,
    no external API call needed.
    k and fetch_k are capped to the actual number of indexed vectors to prevent
    FAISS crashes when the document set is smaller than the requested k.
    """
    total = vectorstore.index.ntotal
    safe_k = min(k, max(total, 1))
    safe_fetch = min(safe_k * 3, max(total, 1))
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": safe_k, "fetch_k": safe_fetch, "lambda_mult": 0.6},
    )
