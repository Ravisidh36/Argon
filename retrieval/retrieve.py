from retrieval.vector_store import load_vector_store
from retrieval.rerank import rerank_with_scores

# How many candidates the fast bi-encoder search pulls before the
# slower, more accurate cross-encoder reranks them down to k.
CANDIDATE_POOL_SIZE = 10


def retrieve_document(query: str, k: int = 3) -> str:
    """
    Retrieve the top-k most relevant chunks for *query*.

    Returns a formatted string with chunk number, cross-encoder score,
    metadata, and content — ready to be injected into an LLM prompt.
    Scores are included for observability (higher = more relevant).
    """
    db = load_vector_store()
    candidates = db.similarity_search(query, k=CANDIDATE_POOL_SIZE)
    scored_docs = rerank_with_scores(query, candidates, top_k=k)

    context = ""
    for i, (score, doc) in enumerate(scored_docs, 1):
        context += (
            f"Chunk {i} [relevance={score:.4f}]\n"
            f"Metadata: {doc.metadata}\n"
            f"Content:\n{doc.page_content}\n\n"
        )

    return context


retrieve_document_tool = {
    "name": "retrieve_document",
    "description": (
        "Retrieve relevant information from indexed documents, resumes, personal records, and local PDFs "
        "to answer questions about individuals, projects, education, grades, CGPA, or content in files. "
        "ALWAYS check retrieve_document first before using web_search when asked about a person's resume or background."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question or search query."
            }
        },
        "required": ["query"]
    }
}