"""
Build a BM25 keyword index over the structured-facts-only text_bm25 field.

This index is derived entirely from public, committable data (no prose),
so both the index artifact and the code are safe to commit.

Output:
  data/processed/bm25_index.pkl  — pickled (bm25, doc_ids, tokenized_corpus)
"""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
OUTPUT_PATH = Path("data/processed/bm25_index.pkl")

# German-aware tokenizer: lowercase, keep umlauts/ß (they carry meaning,
# e.g. "Straße" vs "Strasse", "für" vs "fur"), split on non-word chars.
TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def main() -> None:
    doc_ids = []
    tokenized_corpus = []

    with DOCUMENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_ids.append(doc["doc_id"])
            tokenized_corpus.append(tokenize(doc["text_bm25"]))

    print(f"Documents loaded: {len(doc_ids)}")

    bm25 = BM25Okapi(tokenized_corpus)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)

    print(f"BM25 index saved: {OUTPUT_PATH}")

    # Quick sanity check
    test_query = "Theresienstadt Mitte"
    scores = bm25.get_scores(tokenize(test_query))
    top_idx = scores.argsort()[::-1][:3]
    print(f"\nSanity check — top 3 results for '{test_query}':")
    for idx in top_idx:
        print(f"  {doc_ids[idx]}  (score: {scores[idx]:.2f})")


if __name__ == "__main__":
    main()