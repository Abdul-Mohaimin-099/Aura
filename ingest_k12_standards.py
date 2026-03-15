"""
Ingest Code.org K-12 standards + NCERT Class 12 CS into FAISS with grade-band metadata,
and retrieve only from matching difficulty level at query time.

What this script does:
1) Loads content from:
   - Code.org standards file(s): PDF/TXT/MD/JSON
   - NCERT Class 12 CS PDF
2) Splits text into chunks.
3) Tags each chunk with difficulty_level: Primary | Middle | High School.
4) Embeds chunks with all-MiniLM-L6-v2 and saves a local FAISS index.
5) Supports grade-filtered retrieval for user questions.

Examples
--------
Ingest:
python ingest_k12_standards.py ingest \
  --codeorg ./data/codeorg_standards.pdf \
  --ncert12 ./data/ncert_class12_cs.pdf \
  --index-dir ./faiss_k12

Query (explicit level):
python ingest_k12_standards.py query \
  --index-dir ./faiss_k12 \
  --question "Explain loops with an example" \
  --difficulty "Middle"

Query (infer from question text like "for class 4" / "grade 10"):
python ingest_k12_standards.py query \
  --index-dir ./faiss_k12 \
  --question "For class 10, what is an array?"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

PRIMARY = "Primary"
MIDDLE = "Middle"
HIGH_SCHOOL = "High School"
VALID_LEVELS = {PRIMARY, MIDDLE, HIGH_SCHOOL}


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    return path.read_text(encoding="utf-8")


def load_pdf_as_docs(pdf_path: Path, source_name: str) -> list[Document]:
    md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    docs: list[Document] = []
    for page in md_pages:
        text = page.get("text", "").strip()
        if not text:
            continue
        page_num = page.get("metadata", {}).get("page_number")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "source_path": str(pdf_path.resolve()),
                    "page": page_num,
                },
            )
        )
    return docs


def load_text_as_doc(path: Path, source_name: str) -> list[Document]:
    text = _read_text_file(path).strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": source_name,
                "source_path": str(path.resolve()),
            },
        )
    ]


def infer_level_from_chunk_text(text: str) -> Optional[str]:
    content = text.lower()

    primary_patterns = [
        r"\b(k[- ]?5|k[- ]?2|grade\s*(k|1|2|3|4|5)|class\s*(1|2|3|4|5))\b",
        r"\belementary\b",
    ]
    middle_patterns = [
        r"\b(grade\s*(6|7|8)|class\s*(6|7|8)|middle school)\b",
    ]
    high_patterns = [
        r"\b(grade\s*(9|10|11|12)|class\s*(9|10|11|12)|high school|senior secondary)\b",
    ]

    if any(re.search(p, content) for p in primary_patterns):
        return PRIMARY
    if any(re.search(p, content) for p in middle_patterns):
        return MIDDLE
    if any(re.search(p, content) for p in high_patterns):
        return HIGH_SCHOOL
    return None


def tag_difficulty_for_chunk(chunk: Document) -> str:
    source = str(chunk.metadata.get("source", "")).lower()

    if "ncert" in source and "12" in source:
        return HIGH_SCHOOL

    inferred = infer_level_from_chunk_text(chunk.page_content)
    if inferred:
        return inferred

    if "code.org" in source or "codeorg" in source:
        return MIDDLE

    return HIGH_SCHOOL


def chunk_and_tag(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["difficulty_level"] = tag_difficulty_for_chunk(chunk)
    return chunks


def ingest(codeorg_paths: Iterable[Path], ncert12_pdf: Path, index_dir: Path) -> None:
    docs: list[Document] = []

    for path in codeorg_paths:
        name = f"Code.org::{path.name}"
        if path.suffix.lower() == ".pdf":
            docs.extend(load_pdf_as_docs(path, source_name=name))
        else:
            docs.extend(load_text_as_doc(path, source_name=name))

    docs.extend(
        load_pdf_as_docs(ncert12_pdf, source_name=f"NCERT::Class12::{ncert12_pdf.name}")
    )

    if not docs:
        raise RuntimeError("No ingestible content found.")

    chunks = chunk_and_tag(docs)

    vs = FAISS.from_documents(chunks, get_embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))

    counts = {PRIMARY: 0, MIDDLE: 0, HIGH_SCHOOL: 0}
    for chunk in chunks:
        lvl = chunk.metadata.get("difficulty_level", HIGH_SCHOOL)
        if lvl in counts:
            counts[lvl] += 1

    print("✅ Ingestion complete")
    print(f"Total documents: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Difficulty distribution: {counts}")
    print(f"Saved FAISS index: {index_dir.resolve()}")


def infer_level_from_question(question: str) -> Optional[str]:
    q = question.lower()

    m_class = re.search(r"\bclass\s*(\d{1,2})\b", q)
    if m_class:
        cls = int(m_class.group(1))
        if 1 <= cls <= 5:
            return PRIMARY
        if 6 <= cls <= 9:
            return MIDDLE
        if 10 <= cls <= 12:
            return HIGH_SCHOOL

    m_grade = re.search(r"\bgrade\s*(\d{1,2})\b", q)
    if m_grade:
        grade = int(m_grade.group(1))
        if 1 <= grade <= 5:
            return PRIMARY
        if 6 <= grade <= 9:
            return MIDDLE
        if 10 <= grade <= 12:
            return HIGH_SCHOOL

    return None


def retrieve_by_difficulty(
    index_dir: Path,
    question: str,
    difficulty_level: Optional[str] = None,
    k: int = 6,
    fetch_k: int = 30,
) -> list[Document]:
    vs = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    level = difficulty_level or infer_level_from_question(question)
    if level and level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid difficulty level '{level}'. Use one of: {sorted(VALID_LEVELS)}"
        )

    candidates = vs.similarity_search(question, k=fetch_k)

    if level:
        filtered = [
            d for d in candidates if d.metadata.get("difficulty_level") == level
        ]
    else:
        filtered = candidates

    return filtered[:k]


def cmd_ingest(args: argparse.Namespace) -> None:
    codeorg_paths = [Path(p) for p in args.codeorg]
    for p in codeorg_paths:
        if not p.exists():
            raise FileNotFoundError(f"Code.org source not found: {p}")

    ncert12_pdf = Path(args.ncert12)
    if not ncert12_pdf.exists() or ncert12_pdf.suffix.lower() != ".pdf":
        raise FileNotFoundError(
            f"NCERT Class 12 PDF not found or invalid: {ncert12_pdf}"
        )

    ingest(codeorg_paths, ncert12_pdf, Path(args.index_dir))


def cmd_query(args: argparse.Namespace) -> None:
    level = args.difficulty
    if level and level not in VALID_LEVELS:
        raise ValueError(f"--difficulty must be one of: {sorted(VALID_LEVELS)}")

    results = retrieve_by_difficulty(
        index_dir=Path(args.index_dir),
        question=args.question,
        difficulty_level=level,
        k=args.k,
        fetch_k=max(args.fetch_k, args.k),
    )

    inferred = level or infer_level_from_question(args.question)
    print(f"Question: {args.question}")
    print(f"Applied difficulty filter: {inferred or 'None (all levels)'}")
    print(f"Retrieved chunks: {len(results)}")

    for i, doc in enumerate(results, start=1):
        src = doc.metadata.get("source", "unknown")
        lvl = doc.metadata.get("difficulty_level", "unknown")
        page = doc.metadata.get("page")
        preview = doc.page_content.replace("\n", " ")[:220]
        page_info = f" page={page}" if page is not None else ""
        print(f"\n[{i}] source={src} level={lvl}{page_info}")
        print(f"    {preview}...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest K-12 standards + NCERT Class 12 CS and query with grade-level filtering."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build and save FAISS index")
    p_ingest.add_argument(
        "--codeorg",
        nargs="+",
        required=True,
        help="One or more Code.org standards files (pdf/txt/md/json).",
    )
    p_ingest.add_argument(
        "--ncert12",
        required=True,
        help="NCERT Class 12 CS PDF path.",
    )
    p_ingest.add_argument(
        "--index-dir",
        required=True,
        help="Output directory for local FAISS index.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Run grade-filtered retrieval")
    p_query.add_argument(
        "--index-dir", required=True, help="Path to saved FAISS index."
    )
    p_query.add_argument("--question", required=True, help="User question text.")
    p_query.add_argument(
        "--difficulty",
        choices=sorted(VALID_LEVELS),
        default=None,
        help="Optional explicit filter: Primary | Middle | High School",
    )
    p_query.add_argument(
        "--k", type=int, default=6, help="Final number of chunks to return."
    )
    p_query.add_argument(
        "--fetch-k",
        type=int,
        default=30,
        help="Initial candidate pool before metadata filtering.",
    )
    p_query.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
