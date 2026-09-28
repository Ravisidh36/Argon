"""
Two-stage semantic chunking with markdown table preservation.

Stage 1: MarkdownHeaderTextSplitter — splits on H1/H2/H3 to keep
         semantically related paragraphs together and attaches header
         metadata to each chunk for citation.

Stage 2: Table-aware recursive splitting — identifies markdown pipe
         table blocks and keeps entire rows together (never splits the
         event name from its date columns).  Non-table text is split
         with RecursiveCharacterTextSplitter as before.

This is critical for academic calendars and similar structured documents
where an answer may span multiple columns in the same table row.
"""

import re
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# Slightly larger than before so that a typical 2–3 column table row (which
# can be ~200–300 chars) stays inside a single chunk with its context.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# A line that is part of a markdown pipe table starts with '|'
_TABLE_LINE_RE = re.compile(r"^\s*\|")
# A markdown table separator row e.g. |---|---|---|
_TABLE_SEP_RE = re.compile(r"^\s*\|[-| :]+\|\s*$")

_SegmentKind = Literal["text", "table"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_segments(text: str) -> list[tuple[_SegmentKind, str]]:
    """
    Partition *text* into alternating ('table', ...) and ('text', ...) runs.
    A "table" run is any contiguous block of lines that start with '|'.
    """
    segments: list[tuple[_SegmentKind, str]] = []
    current_lines: list[str] = []
    in_table = False

    for line in text.split("\n"):
        is_pipe = bool(_TABLE_LINE_RE.match(line))

        if is_pipe and not in_table:
            # Transition text → table
            if current_lines:
                segments.append(("text", "\n".join(current_lines)))
                current_lines = []
            in_table = True
            current_lines.append(line)

        elif not is_pipe and in_table:
            # Transition table → text
            segments.append(("table", "\n".join(current_lines)))
            current_lines = [line]
            in_table = False

        else:
            current_lines.append(line)

    if current_lines:
        segments.append(("table" if in_table else "text", "\n".join(current_lines)))

    return segments


def _split_table(table_text: str) -> list[str]:
    """
    If a markdown table fits in CHUNK_SIZE characters, return it as one chunk.
    Otherwise, split it into row-groups, repeating the header + separator in
    every group so each chunk is self-contained.
    """
    if len(table_text) <= CHUNK_SIZE:
        return [table_text]

    lines = table_text.split("\n")
    # Detect separator row (row 1 matches |---|---|)
    has_sep = len(lines) > 1 and bool(_TABLE_SEP_RE.match(lines[1]))
    header_lines = lines[: 2 if has_sep else 1]
    data_rows = lines[2 if has_sep else 1 :]

    chunks: list[str] = []
    current_rows: list[str] = []

    for row in data_rows:
        candidate = "\n".join(header_lines + current_rows + [row])
        if len(candidate) > CHUNK_SIZE and current_rows:
            chunks.append("\n".join(header_lines + current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append("\n".join(header_lines + current_rows))

    return chunks or [table_text]


def _split_preserving_tables(
    text: str,
    size_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """
    Split *text* with *size_splitter* while keeping markdown table blocks
    atomic.  Tables that exceed CHUNK_SIZE are split by row-group (with
    headers repeated) rather than mid-cell.
    """
    results: list[str] = []
    for kind, content in _extract_segments(text):
        if not content.strip():
            continue
        if kind == "table":
            results.extend(_split_table(content))
        else:
            results.extend(size_splitter.split_text(content))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_documents(
    markdown: str,
    source: str = "unknown",
    document_id: str = "unknown",
) -> list[Document]:
    """
    Convert a Markdown string into a list of LangChain Documents suitable
    for embedding and FAISS indexing.

    Each document carries metadata:
        source      — original filename
        document_id — SHA-256 hex digest of the source file
        chunk_id    — "{source}::{sequential_index}" for golden-set eval
        H1 / H2 / H3 — section header hierarchy from MarkdownHeaderSplitter
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Note: "\n" is intentionally NOT in this list.
        # Table rows are separated by "\n", so including it would break table
        # cells across chunks. Table splitting is handled by _split_table().
        separators=["\n\n", ". ", " ", ""],
    )

    header_chunks = header_splitter.split_text(markdown)

    documents: list[Document] = []
    chunk_idx = 0

    for header_chunk in header_chunks:
        sub_texts = _split_preserving_tables(header_chunk.page_content, size_splitter)

        for sub_text in sub_texts:
            if not sub_text.strip():
                continue
            documents.append(
                Document(
                    page_content=sub_text,
                    metadata={
                        **header_chunk.metadata,
                        "source": source,
                        "document_id": document_id,
                        "chunk_id": f"{source}::{chunk_idx}",
                    },
                )
            )
            chunk_idx += 1

    return documents