"""
Re-ranking on top of hybrid retrieval.

Flow:
    1. Hybrid retrieve a wider candidate pool (e.g. top 25) via 07_hybrid_retrieve.retrieve()
    2. Score each (query, candidate) pair with a multilingual cross-encoder
    3. Return the top N re-ranked results

Uses a MULTILINGUAL cross-encoder (not the common English-only ms-marco-MiniLM)
because this dataset is German-language biographical text.

Install (already likely have sentence-transformers via your embedding step):
    uv add sentence-transformers
"""

from sentence_transformers import CrossEncoder

# Multilingual, trained on mMARCO (covers German) -- unlike the common
# English-only ms-marco-MiniLM-L-6-v2, which would likely hurt quality here.
RERANK_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder once (mirrors your st.cache_resource pattern in app.py)."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_n: int = 5, text_field: str = "text_bm25") -> list[dict]:
    """
    Re-score a candidate pool with a cross-encoder and return the top_n.

    Args:
        query: the user's question
        candidates: list of doc dicts from hybrid retrieve(), e.g. [{"stolperstein_id": ..., "text_bm25": ..., ...}, ...]
        top_n: how many to keep after re-ranking
        text_field: which field on each candidate dict to score against (use the
                    facts-only field, same one used for LLM prompts, not the longer prose field)

    Returns:
        candidates re-ordered by cross-encoder score, truncated to top_n,
        each with a "rerank_score" key added for transparency/logging.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c[text_field]) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]


def retrieve_with_rerank(query: str, retrieve_fn, candidate_pool: int = 25, top_n: int = 5) -> list[dict]:
    """
    Convenience wrapper: widen hybrid retrieval, then re-rank down to top_n.

    Args:
        query: user question
        retrieve_fn: your existing retrieve() from 07_hybrid_retrieve.py
        candidate_pool: how many candidates to pull from hybrid search before re-ranking
                        (wider than final top_n so the cross-encoder has real choices to make)
        top_n: final number of documents to return

    Example:
        from scripts.07_hybrid_retrieve import retrieve
        results = retrieve_with_rerank("Wer wurde nach Theresienstadt deportiert?", retrieve, candidate_pool=25, top_n=5)
    """
    candidates = retrieve_fn(query, top_n=candidate_pool)
    return rerank(query, candidates, top_n=top_n)
