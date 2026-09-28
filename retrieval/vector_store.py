import os
from langchain_community.vectorstores import FAISS
from retrieval.embeddings import embeddings

VECTOR_STORE_PATH = "vector_store"


def create_vector_store(documents):
    """
    Adds documents to the existing index if one exists, instead of
    overwriting it. Without this, indexing a second document would
    silently erase every chunk from the first one.
    """

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


def load_vector_store():

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )