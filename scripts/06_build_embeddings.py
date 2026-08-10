"""
Build FAISS dense index from text_embed field using multilingual-e5-small.
Input:  data/processed/documents.jsonl
Output: data/processed/embeddings.faiss
        data/processed/embeddings_meta.jsonl  (row order == FAISS index order)
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCS_PATH = Path("data/processed/documents.jsonl")
INDEX_PATH = Path("data/processed/embeddings.faiss")
META_PATH = Path("data/processed/embeddings_meta.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64

def load_docs():
    docs = []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def main():
    docs = load_docs()
    print(f"Loaded {len(docs)} documents")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    dim = model.get_embedding_dimension()
    print(f"Model loaded, embedding dim = {dim}")

    texts = [f"passage: {d['text_embed']}" for d in docs]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for cosine sim via inner product
    ).astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index written: {INDEX_PATH} ({index.ntotal} vectors)")

    with open(META_PATH, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({
                "faiss_idx": i,
                "doc_id": d["doc_id"],
            }, ensure_ascii=False) + "\n")
    print(f"Meta written: {META_PATH}")

    # sanity check
    q = model.encode(["query: Wer wurde nach Theresienstadt deportiert?"], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, 3)
    print("\nSanity check — top 3 for 'Wer wurde nach Theresienstadt deportiert?':")
    for score, idx in zip(D[0], I[0]):
        print(f"  {docs[idx]['doc_id']} (score: {score:.4f})")

if __name__ == "__main__":
    main()