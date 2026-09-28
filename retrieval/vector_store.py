"""
FAISS vector store with model-specific storage and embedding compatibility guard.

A new embedding model produces vectors in a different vector space (different
dimension and meaning). Keeping model-specific directories isolates indexes so
switching models never accidentally mixes incompatible vectors. In addition,
an embedding metadata file (embedding_meta.json) is persisted alongside the FAISS
index to guard against model mismatch.
"""

import json
import os
import re

from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL
from retrieval.embeddings import embeddings


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")


VECTOR_STORE_PATH = os.path.join("vector_store", _model_slug(EMBEDDING_MODEL))
_META_FILENAME = "embedding_meta.json"


def _meta_path() -> str:
    return os.path.join(VECTOR_STORE_PATH, _META_FILENAME)


def _read_stored_model() -> str | None:
    path = _meta_path()
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("model")


def _write_model_meta() -> None:
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump({"model": EMBEDDING_MODEL}, f, indent=2)


def _check_model_compat() -> None:
    """
    Raises RuntimeError if the stored index was built with a different model.
    Skips the check when no index exists yet (fresh start).
    """
    stored = _read_stored_model()
    if stored is None:
        return  # no index yet — nothing to guard against
    if stored != EMBEDDING_MODEL:
        raise RuntimeError(
            f"\nEmbedding model mismatch — cannot use this FAISS index!\n"
            f"  Index was built with : {stored}\n"
            f"  Current model is     : {EMBEDDING_MODEL}\n\n"
            "Vectors from different models live in incompatible spaces.\n"
            "Re-index your documents with the current embedding model.\n"
        )


def create_vector_store(documents):
    """
    Add documents to the model-specific FAISS index, creating it from scratch if needed.
    Writes/updates embedding_meta.json on every successful save.
    """
    _check_model_compat()
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(index_file):
        db = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        db.add_documents(documents)
    else:
        db = FAISS.from_documents(documents, embeddings)

    db.save_local(VECTOR_STORE_PATH)
    _write_model_meta()


def load_vector_store() -> FAISS:
    """
    Load the FAISS index for the currently configured embedding model.
    Raises FileNotFoundError if no index has been created yet.
    """
    _check_model_compat()
    index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"No FAISS index exists for embedding model '{EMBEDDING_MODEL}'. "
            "Re-index your documents with the current embedding model first."
        )

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
