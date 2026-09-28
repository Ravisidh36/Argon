import os

from langchain_huggingface import HuggingFaceEmbeddings


# Local embedding model: runs 100% locally on CPU/GPU without any API keys.
# BGE-M3 provides a stronger multilingual/semantic embedding model for RAG
# than the previous all-MiniLM-L6-v2 model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
