"""
Document reader using Docling for structured document parsing.

Supports PDF, DOCX, PPTX, HTML, XLSX, images, and other formats that Docling
handles natively. Exports to Markdown so the downstream chunking pipeline can
continue to use MarkdownHeaderTextSplitter without modification.

Tables are preserved as GitHub-flavoured Markdown pipe tables so that related
columns (e.g. "Event | From | To") stay together in the same chunk.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Lazily initialise the Docling converter (models load once, reused after)."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def doc_reader(path: str) -> str:
    """
    Parse the file at *path* with Docling and return it as a structured Markdown string.

    Raises FileNotFoundError if the file does not exist.
    Raises RuntimeError if Docling conversion fails.
    Raises ValueError if Docling produces empty output.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    converter = _get_converter()
    try:
        result = converter.convert(str(file_path))
    except Exception as exc:
        raise RuntimeError(
            f"Docling could not convert '{path}': {exc}"
        ) from exc

    markdown = result.document.export_to_markdown()

    if not markdown or not markdown.strip():
        raise ValueError(
            f"Docling produced empty output for '{path}'. "
            "The file may be empty, image-only without OCR, or in an unsupported format."
        )

    return markdown
