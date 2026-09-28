from retrieval.vector_store import load_vector_store
from retrieval.rerank import rerank

# How many candidates the fast bi-encoder search pulls before the
# slower, more accurate cross-encoder reranks them down to k.
CANDIDATE_POOL_SIZE = 10


def retrieve_document(query, k=3):

    db = load_vector_store()
    candidates = db.similarity_search(query, k=CANDIDATE_POOL_SIZE)
    docs = rerank(query, candidates, top_k=k)

    context = ""

    for i, doc in enumerate(docs, 1):
        context += (
            f"Chunk {i}\n"
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