"""
Tests for document-level deduplication in the indexing pipeline.

These tests run fully offline: the real Gemini embeddings model and the
real PDF-to-markdown converter are both replaced with fakes, so no
network call or API key is needed. What's under test is the pipeline
logic itself - hashing, the persistent registry, and FAISS staying
incremental - not the quality of embeddings or PDF parsing.

Run with:
    pytest tests/test_deduplication.py -v
"""

import hashlib
import pytest
from langchain_core.embeddings import Embeddings

from retrieval import index_document as index_document_module
from retrieval import vector_store
from retrieval import document_registry


class FakeEmbeddings(Embeddings):
    """
    Deterministic, offline stand-in for the real embeddings model.
    Same text always maps to the same vector (via a hash), which is all
    FAISS needs to build/search an index - we aren't testing embedding
    *quality* here, just that the indexing/dedup pipeline behaves
    correctly around whatever embeddings it's given.
    """

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """
    Points the vector store (and therefore the registry, which derives
    its path from vector_store.VECTOR_STORE_PATH) at a fresh temp
    directory per test, and swaps in the fake embeddings model.
    """
    store_path = str(tmp_path / "vector_store")
    monkeypatch.setattr(vector_store, "VECTOR_STORE_PATH", store_path)
    monkeypatch.setattr(vector_store, "embeddings", FakeEmbeddings())
    return store_path


@pytest.fixture(autouse=True)
def fake_doc_reader(monkeypatch):
    """
    index_document.py does `from tools.document_understanding import
    doc_reader`, which binds the function into index_document's own
    namespace at import time - so we must patch it there
    (index_document_module.doc_reader), not on the original module,
    or the patch silently won't take effect.
    """

    def _reader(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    monkeypatch.setattr(index_document_module, "doc_reader", _reader)


def make_fake_pdf(tmp_path, name, content):
    """A plain text file standing in for a PDF - fine here since
    fake_doc_reader reads it as raw text instead of running it through
    real PDF parsing."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_index_new_pdf_adds_vectors(isolated_store, tmp_path):
    pdf_path = make_fake_pdf(tmp_path, "a.pdf", "Hello world, this is document A.")

    result = index_document_module.index_document(pdf_path)

    assert "Indexed" in result
    db = vector_store.load_vector_store()
    assert len(db.docstore._dict) >= 1


def test_index_same_pdf_again_is_skipped(isolated_store, tmp_path):
    pdf_path = make_fake_pdf(tmp_path, "a.pdf", "Hello world, this is document A.")

    index_document_module.index_document(pdf_path)
    count_before = len(vector_store.load_vector_store().docstore._dict)

    result = index_document_module.index_document(pdf_path)

    assert result == "Document already indexed. Skipping."
    count_after = len(vector_store.load_vector_store().docstore._dict)
    assert count_after == count_before


def test_two_different_pdfs_both_remain_searchable(isolated_store, tmp_path):
    pdf_a = make_fake_pdf(tmp_path, "a.pdf", "Content about cats and kittens.")
    pdf_b = make_fake_pdf(tmp_path, "b.pdf", "Content about dogs and puppies.")

    index_document_module.index_document(pdf_a)
    index_document_module.index_document(pdf_b)

    db = vector_store.load_vector_store()
    sources = {doc.metadata.get("source") for doc in db.docstore._dict.values()}
    assert "a.pdf" in sources
    assert "b.pdf" in sources


def test_modified_pdf_is_treated_as_new_document(isolated_store, tmp_path):
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_text("Original content version one.", encoding="utf-8")

    result_1 = index_document_module.index_document(str(pdf_path))
    assert "Indexed" in result_1

    # Same filename, different bytes -> different hash -> must NOT be
    # skipped just because the filename matches.
    pdf_path.write_text("Completely different content, version two.", encoding="utf-8")

    result_2 = index_document_module.index_document(str(pdf_path))
    assert "Indexed" in result_2
    assert result_2 != "Document already indexed. Skipping."


def test_dedup_persists_across_restart(isolated_store, tmp_path):
    pdf_path = make_fake_pdf(tmp_path, "a.pdf", "Persistent dedup test content.")

    index_document_module.index_document(pdf_path)

    # Simulate a restart: load the registry fresh from disk with no
    # in-memory state carried over from the call above.
    file_hash = document_registry.hash_file(pdf_path)
    fresh_registry = document_registry.load_registry()
    assert document_registry.is_indexed(file_hash, fresh_registry)

    result = index_document_module.index_document(pdf_path)
    assert result == "Document already indexed. Skipping."
