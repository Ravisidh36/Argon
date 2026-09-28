import os

from tools.document_understanding import doc_reader
from retrieval.cleaner import clean_markdown
from retrieval.chunking import create_documents
from retrieval.vector_store import create_vector_store
from retrieval.document_registry import hash_file, is_indexed, register_document


def index_document(path):
    # Automatically resolve if file is inside data/documents/ or relative path
    if not os.path.exists(path):
        doc_dir_path = os.path.join("data", "documents", path)
        if os.path.exists(doc_dir_path):
            path = doc_dir_path
        else:
            return f"Error: File '{path}' does not exist. Checked '{path}' and '{doc_dir_path}'."

    file_hash = hash_file(path)

    if is_indexed(file_hash):
        message = "Document already indexed. Skipping."
        print(message)
        return message

    filename = os.path.basename(path)
    print(f"Indexing {filename}...")

    try:
        markdown = doc_reader(path)
        markdown = clean_markdown(markdown)
        documents = create_documents(
            markdown,
            source=filename,
            document_id=file_hash,
        )

        # Only register the document AFTER the vector store save succeeds.
        create_vector_store(documents)
    except Exception as e:
        return f"Indexing failed, no vectors were saved: {e}"

    register_document(
        file_hash=file_hash,
        filename=filename,
        path=path,
        num_chunks=len(documents),
    )

    message = f"Indexed {len(documents)} chunks successfully."
    print(message)
    return message


index_document_tool = {
    "name": "index_document",
    "description": (
        "Parse and index a document for retrieval. Supports PDF, DOCX, PPTX, XLSX, HTML, and more. "
        "Call this when the user asks to 'index', 'load', or 'add' a file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the document file (e.g. '2.pdf', 'report.docx')",
            }
        },
        "required": ["path"],
    },
}
