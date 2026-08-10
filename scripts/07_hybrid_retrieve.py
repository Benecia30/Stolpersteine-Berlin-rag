"""
Hybrid retrieval: BM25 (sparse) + e5-small (dense) -> Reciprocal Rank Fusion (RRF)
Returns top-n facts-only (text_bm25) payloads for the LLM. Prose never leaves this step.

Usage:
    uv run python scripts/07_hybrid_retrieve.py "Wer wurde nach Theresienstadt deportiert?"
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Must match scripts/05_build_bm25_index.py EXACTLY -- same tokenizer used at
# index-build time and query time, or BM25 scores are silently wrong.
TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


DOCS_PATH = Path("data/processed/documents.jsonl")
BM25_PATH = Path("data/processed/bm25_index.pkl")
FAISS_PATH = Path("data/processed/embeddings.faiss")
META_PATH = Path("data/processed/embeddings_meta.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"

BM25_TOP_K = 20
DENSE_TOP_K = 20
RRF_K = 60          # standard RRF damping constant
FINAL_TOP_N = 5      # how many facts-only payloads to hand the LLM


def load_docs():
    docs = {}
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d
    return docs


def load_bm25():
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"]


def load_faiss():
    index = faiss.read_index(str(FAISS_PATH))
    meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]
    faiss_idx_to_doc_id = {m["faiss_idx"]: m["doc_id"] for m in meta}
    return index, faiss_idx_to_doc_id


def bm25_search(query, bm25, doc_ids, top_k):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def dense_search(query, index, faiss_idx_to_doc_id, model, top_k):
    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_vec, top_k)
    return [(faiss_idx_to_doc_id[i], float(score)) for i, score in zip(I[0], D[0]) if i != -1]


def reciprocal_rank_fusion(bm25_results, dense_results, k=RRF_K):
    """
    bm25_results / dense_results: list of (doc_id, score) already sorted best-first.
    Returns list of (doc_id, fused_score) sorted best-first.
    """
    fused = {}
    for rank, (doc_id, _) in enumerate(bm25_results):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(dense_results):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def retrieve(query, top_n=FINAL_TOP_N):
    docs = load_docs()
    bm25, doc_ids = load_bm25()
    index, faiss_idx_to_doc_id = load_faiss()
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    bm25_results = bm25_search(query, bm25, doc_ids, BM25_TOP_K)
    dense_results = dense_search(query, index, faiss_idx_to_doc_id, model, DENSE_TOP_K)
    fused = reciprocal_rank_fusion(bm25_results, dense_results)[:top_n]

    payloads = []
    for doc_id, score in fused:
        d = docs[doc_id]
        payloads.append({
            "doc_id": doc_id,
            "rrf_score": round(score, 5),
            "text": d["text_bm25"],       # facts-only, never text_embed/prose
            "source_url": d["source_url"],
        })
    return payloads


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Wer wurde nach Theresienstadt deportiert?"
    print(f"Query: {query}\n")
    results = retrieve(query)
    for r in results:
        print(f"[{r['rrf_score']}] {r['doc_id']}")
        print(f"  {r['text'][:150]}...")
        print(f"  {r['source_url']}\n")