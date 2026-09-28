from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# Chunk size / overlap are in characters (cheap proxy for tokens — roughly
# 4 chars/token for English). ~500 tokens keeps a chunk focused on one idea
# without starving the LLM of context once it's retrieved.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def create_documents(markdown: str, source: str = "unknown", document_id: str = "unknown"):
    """
    Two-stage chunking:
      1. Split on markdown headers -> preserves semantic/structural
         boundaries and gives us H1/H2/H3 metadata for citation.
      2. Re-split any oversized section with a recursive character
         splitter -> bounds chunk size so no single embedding has to
         represent a wall of unrelated content, and adds overlap so we
         don't sever a sentence/idea exactly at a chunk boundary.
    """

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Tries these separators in order — paragraph, then line, then
        # sentence-ish, then word — before ever hard-cutting mid-word.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    header_chunks = header_splitter.split_text(markdown)

    documents = []
    chunk_idx = 0

    for header_chunk in header_chunks:
        sub_texts = size_splitter.split_text(header_chunk.page_content)

        for sub_text in sub_texts:
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