from retrieval.vector_store import load_vector_store
from retrieval.rerank import rerank_with_scores

# How many candidates the fast bi-encoder search pulls before the
# slower, more accurate cross-encoder reranks them down to k.
CANDIDATE_POOL_SIZE = 10


def retrieve_document(
    query: str,
    document: str | None = None,
    k: int = 3,
) -> str:
    """
    Retrieve the top-k most relevant chunks.

    If a document filename is provided, retrieval is restricted to
    chunks belonging to that document before reranking.
    """

    db = load_vector_store()

    # Document-specific retrieval
    if document:
        candidates = db.similarity_search(
            query,
            k=CANDIDATE_POOL_SIZE,
            filter={"source": document},
        )
    else:
        # Search across all indexed documents
        candidates = db.similarity_search(
            query,
            k=CANDIDATE_POOL_SIZE,
        )

    if not candidates:
        if document:
            return (
                f"No relevant information was found in the specified "
                f"document '{document}'."
            )

        return "No relevant information was found in the indexed documents."

    # Rerank only the candidates that survived document filtering.
    scored_docs = rerank_with_scores(
        query,
        candidates,
        top_k=k,
    )

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
        "Retrieve relevant information from indexed documents. "
        "Use this for questions about local PDFs, resumes, academic "
        "calendars, timetables, schedules, holidays, grades, projects, "
        "or any other content stored in indexed documents. "
        "If the user specifies a document filename, ALWAYS provide "
        "that filename in the 'document' argument so retrieval is "
        "restricted to that document."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The semantic information to search for. "
                    "Do not include the filename here."
                ),
            },
            "document": {
                "type": "string",
                "description": (
                    "Optional filename to restrict retrieval to, "
                    "for example '3.pdf'."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Number of final chunks to return after reranking.",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}