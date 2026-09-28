from langchain_huggingface import HuggingFaceEmbeddings

# Local embedding model: runs 100% locally on CPU/GPU without any API keys
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)