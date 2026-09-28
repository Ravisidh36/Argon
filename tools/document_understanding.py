from pathlib import Path

from docling.document_converter import DocumentConverter


# Docling's converter is relatively expensive to initialize, so keep one
# process-local instance and reuse it for all documents.
_converter = DocumentConverter()


def doc_reader(path: str) -> str:
    """Convert a supported document into structured Markdown for RAG.

    Docling preserves document structure, including headings and tables,
    which is important for RAG over structured documents such as academic
    calendars.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    try:
        result = _converter.convert(str(file_path))
        return result.document.export_to_markdown()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse document '{file_path.name}' with Docling: {exc}"
        ) from exc
