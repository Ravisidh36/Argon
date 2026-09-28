from sentence_transformers import CrossEncoder

_model = None


def get_reranker():
    """
    Lazily loads the cross-encoder once and reuses it (loading the model
    is slow; scoring with it is fast).
    """
    global _model
    if _model is None:
        # Small, CPU-friendly cross-encoder trained on MS MARCO (a real
        # search-relevance dataset). Runs locally, no API call needed.
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query, documents, top_k=3):
    """
    Re-scores a shortlist of candidate documents by how relevant they
    actually are to this specific query (query + chunk seen together),
    instead of relying only on embedding distance. Returns the top_k,
    reordered by that score, most relevant first.
    """
    if not documents:
        return []

    model = get_reranker()

    # Cross-encoder input format: (query, candidate_text) pairs.
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda pair: pair[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]
