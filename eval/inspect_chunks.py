"""
Run this after indexing a document to see every chunk with its chunk_id.
Use this output to hand-build eval/golden_set.json: pick a handful of
questions you'd realistically ask, and note which chunk_id(s) actually
contain the answer.

Usage:
    python -m eval.inspect_chunks
"""

from retrieval.vector_store import load_vector_store


def inspect():
    db = load_vector_store()

    # FAISS's in-memory docstore keyed by internal id -> Document.
    # This is the only way to see every indexed chunk, not just top-k
    # results of a similarity search.
    all_docs = list(db.docstore._dict.values())

    print(f"Total chunks indexed: {len(all_docs)}\n")

    for doc in all_docs:
        chunk_id = doc.metadata.get("chunk_id", "MISSING")
        headers = {k: v for k, v in doc.metadata.items() if k.startswith("H")}

        print(f"[{chunk_id}] {headers}")
        print(doc.page_content)
        print("-" * 60)
        print()


if __name__ == "__main__":
    inspect()
