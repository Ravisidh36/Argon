from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

# Normalized embeddings: converting raw vectors to unit length means FAISS
# L2 search gives the same ranking as cosine similarity — which is what
# BGE-M3 is trained/evaluated with.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
