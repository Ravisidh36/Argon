"""
Measures retrieval quality against a hand-labeled golden set, and
compares two strategies side-by-side:
  - baseline:  pure bi-encoder cosine similarity search
  - reranked:  bi-encoder candidate pool -> cross-encoder reranking

Metrics:
  - Recall@k: of the chunks actually relevant to a query, what fraction
    appeared in the top-k retrieved results?
  - MRR (Mean Reciprocal Rank): 1 / (rank of the first relevant chunk),
    averaged across queries. Rewards getting the right answer near the top,
    not just "somewhere in the list."

Usage:
    python -m eval.evaluate_retrieval
"""

import json
from retrieval.vector_store import load_vector_store
from retrieval.rerank import rerank

GOLDEN_SET_PATH = "eval/golden_set.json"
K_VALUES = [1, 3, 5]
MAX_K = max(K_VALUES)
CANDIDATE_POOL_SIZE = 10


def load_golden_set(path=GOLDEN_SET_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return None
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def get_chunk_ids(docs):
    return [doc.metadata.get("chunk_id") for doc in docs]


def evaluate_strategy(db, golden_set, use_rerank, verbose=False):
    recall_scores = {k: [] for k in K_VALUES}
    mrr_scores = []

    for case in golden_set:
        query = case["query"]
        relevant_ids = case["relevant_chunk_ids"]

        if use_rerank:
            candidates = db.similarity_search(query, k=CANDIDATE_POOL_SIZE)
            results = rerank(query, candidates, top_k=MAX_K)
        else:
            results = db.similarity_search(query, k=MAX_K)

        retrieved_ids = get_chunk_ids(results)

        mrr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
        for k in K_VALUES:
            score = recall_at_k(retrieved_ids, relevant_ids, k)
            if score is not None:
                recall_scores[k].append(score)

        if verbose:
            label = "RERANKED" if use_rerank else "BASELINE"
            print(f"  [{label}] {query}")
            print(f"      expected:  {relevant_ids}")
            print(f"      retrieved: {retrieved_ids}")

    summary = {
        f"Recall@{k}": sum(v) / len(v) if v else 0.0
        for k, v in recall_scores.items()
    }
    summary["MRR"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    return summary


def evaluate():
    db = load_vector_store()
    golden_set = load_golden_set()

    print(f"Evaluating {len(golden_set)} queries...\n")

    baseline = evaluate_strategy(db, golden_set, use_rerank=False, verbose=True)
    print()
    reranked = evaluate_strategy(db, golden_set, use_rerank=True, verbose=True)
    print()

    print("=" * 50)
    print(f"{'Metric':<12}{'Baseline':>12}{'Reranked':>12}{'Delta':>12}")
    print("=" * 50)
    for metric in baseline:
        b = baseline[metric]
        r = reranked[metric]
        delta = r - b
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<12}{b:>12.3f}{r:>12.3f}{sign}{delta:>11.3f}")


if __name__ == "__main__":
    evaluate()
