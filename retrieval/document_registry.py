"""
Persistent, content-based document registry for indexing deduplication.

Documents are identified by a SHA-256 hash of their raw file bytes, not
by filename or path. This means:
  - Indexing the exact same file twice (even under a different path) is
    detected and skipped.
  - Renaming a file does NOT cause re-indexing (same bytes -> same hash).
  - Modifying a file's content, even slightly, DOES cause re-indexing,
    since even one changed byte produces a completely different hash.

The registry itself is a small JSON file stored alongside the FAISS
index, so it survives application restarts (it is not an in-memory
list/set/dict that resets every run).
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

from retrieval import vector_store


def hash_file(path: str) -> str:
    """
    Streams the file in chunks rather than reading it all into memory at
    once (matters for large PDFs), and returns its SHA-256 hex digest.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _registry_path():
    # Read vector_store.VECTOR_STORE_PATH at call time (not at import
    # time) so anything that points the vector store elsewhere - tests,
    # future config changes - is automatically respected here too,
    # keeping the registry and the FAISS index physically colocated.
    return os.path.join(vector_store.VECTOR_STORE_PATH, "document_registry.json")


def load_registry() -> dict:
    """Loads the registry fresh from disk. No in-memory state is assumed
    or cached, so this always reflects reality even after a restart."""
    path = _registry_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    path = _registry_path()
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)

    # Write to a temp file first, then atomically replace the real file.
    # Without this, a crash or interruption mid-write could leave a
    # half-written, corrupted JSON file behind, silently breaking
    # dedup on the next run.
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def is_indexed(file_hash: str, registry: dict | None = None) -> bool:
    if registry is None:
        registry = load_registry()
    return file_hash in registry


def register_document(file_hash: str, filename: str, path: str, num_chunks: int) -> None:
    """
    Records that a document has been successfully indexed.

    IMPORTANT: callers must only call this AFTER the FAISS vector store
    has been saved successfully. If this were called first (or if it
    were called regardless of success), a failed embed/save could leave
    the registry claiming a document is indexed when its vectors were
    never actually persisted - a silent, hard-to-debug desync between
    "what we think we have" and "what's actually in FAISS".
    """
    registry = load_registry()
    registry[file_hash] = {
        "document_id": file_hash,
        "filename": filename,
        "path": path,
        "num_chunks": num_chunks,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_registry(registry)
