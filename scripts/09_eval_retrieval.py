"""
Phase 4 - retrieval eval.

Compares BM25-only, dense-only, and hybrid RRF retrieval against a
hand-labeled query set. Run from repo root:

    uv run scripts/09_eval_retrieval.py

INPUT you need to fill in first:
    data/eval/labeled_queries.jsonl
    Each line: {"query": str, "relevant_doc_ids": [str, ...]}
    Use scripts/eval_helpers/inspect_candidates.py to build these.

Loads bm25/faiss/model/docs ONCE up front, not per query -- retrieve()
itself reloads everything on every call, which is fine for one-off CLI
use but too slow for looping over a labeled set.
"""

import json
from pathlib import Path
from collections import defaultdict
import importlib.util

ROOT = Path(__file__).parent.parent

spec = importlib.util.spec_from_file_location(
    "hybrid_retrieve", ROOT / "scripts" / "07_hybrid_retrieve.py"
)
hr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr)

K = 5


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for d in retrieved_k if d in relevant)
    return hits / len(retrieved_k)


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    retrieved_k = set(retrieved[:k])
    hits = sum(1 for d in relevant if d in retrieved_k)
    return hits / len(relevant)


def mrr(retrieved, relevant):
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(retrieved, relevant, k):
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def load_queries(path):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "FILL_IN" in row["relevant_doc_ids"]:
                print(f"[skip] not yet labeled: {row['query']!r}")
                continue
            queries.append(row)
    return queries


def run_eval(queries, search_fn, label, k=K):
    scores = defaultdict(list)
    for row in queries:
        relevant = set(row["relevant_doc_ids"])
        retrieved = search_fn(row["query"])  # list of doc_ids, best-first
        scores["precision"].append(precision_at_k(retrieved, relevant, k))
        r = recall_at_k(retrieved, relevant, k)
        if r is not None:
            scores["recall"].append(r)
        scores["mrr"].append(mrr(retrieved, relevant))
        scores["hit_rate"].append(hit_rate_at_k(retrieved, relevant, k))

    print(f"\n=== {label} (k={k}) ===")
    for metric, vals in scores.items():
        avg = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {metric:>10}: {avg:.3f}  (n={len(vals)})")
    return scores


def main():
    queries = load_queries(ROOT / "data" / "eval" / "labeled_queries.jsonl")
    if not queries:
        print("No labeled queries found. Fill in data/eval/labeled_queries.jsonl first.")
        return
    print(f"Loaded {len(queries)} labeled queries.")

    print("Loading resources (bm25, faiss, model, docs) once...")
    bm25, doc_ids = hr.load_bm25()
    index, faiss_idx_to_doc_id = hr.load_faiss()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hr.MODEL_NAME, device="cpu")

    def bm25_only(query):
        results = hr.bm25_search(query, bm25, doc_ids, hr.BM25_TOP_K)
        return [d for d, _ in results]

    def dense_only(query):
        results = hr.dense_search(query, index, faiss_idx_to_doc_id, model, hr.DENSE_TOP_K)
        return [d for d, _ in results]

    def hybrid(query):
        bm25_results = hr.bm25_search(query, bm25, doc_ids, hr.BM25_TOP_K)
        dense_results = hr.dense_search(query, index, faiss_idx_to_doc_id, model, hr.DENSE_TOP_K)
        fused = hr.reciprocal_rank_fusion(bm25_results, dense_results)
        return [d for d, _ in fused]

    run_eval(queries, bm25_only, "BM25-only")
    run_eval(queries, dense_only, "Dense-only")
    run_eval(queries, hybrid, "Hybrid RRF")


if __name__ == "__main__":
    main()