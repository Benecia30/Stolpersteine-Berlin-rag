"""
Phase 4 - labeling helper.

Prints a wide candidate pool (BM25 top-20 UNION dense top-20, before RRF
fusion narrows it to 5) alongside each doc's facts, so you can eyeball
which doc_ids are actually correct for a query and paste them into
data/eval/labeled_queries.jsonl by hand.

Use the wider pool, not just retrieve()'s fused top-5 -- you want to catch
correct docs that hybrid ranked low (false negatives for your eval) too,
not just confirm what it already surfaced.

Usage:
    uv run scripts/eval_helpers/inspect_candidates.py "Kinder die ermordet wurden"
"""

import sys
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

spec = importlib.util.spec_from_file_location(
    "hybrid_retrieve", ROOT / "scripts" / "07_hybrid_retrieve.py"
)
hr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr)


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run scripts/eval_helpers/inspect_candidates.py "your query"')
        return

    query = sys.argv[1]

    print("Loading resources (bm25, faiss, model, docs)...")
    docs = hr.load_docs()
    bm25, doc_ids = hr.load_bm25()
    index, faiss_idx_to_doc_id = hr.load_faiss()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hr.MODEL_NAME, device="cpu")

    bm25_results = hr.bm25_search(query, bm25, doc_ids, hr.BM25_TOP_K)
    dense_results = hr.dense_search(query, index, faiss_idx_to_doc_id, model, hr.DENSE_TOP_K)

    bm25_ids = [d for d, _ in bm25_results]
    dense_ids = [d for d, _ in dense_results]
    candidates = list(dict.fromkeys(bm25_ids + dense_ids))  # union, dedup, order-preserving

    print(f'\nQuery: "{query}"')
    print(f"Candidates: {len(candidates)} (BM25 {len(bm25_ids)} + dense {len(dense_ids)}, deduped)\n")

    for doc_id in candidates:
        row = docs.get(doc_id)
        if row is None:
            print(f"[{doc_id}] NOT FOUND in documents.jsonl -- check for stale index")
            continue
        in_bm25 = "bm25" if doc_id in bm25_ids else "    "
        in_dense = "dense" if doc_id in dense_ids else "     "
        print(f"[{doc_id}]  ({in_bm25} {in_dense})")
        print(f"  {row['text_bm25'][:300]}")
        print()

    print("---")
    print("Copy the correct doc_ids into data/eval/labeled_queries.jsonl, e.g.:")
    print(json.dumps({"query": query, "relevant_doc_ids": ["<paste ids here>"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
