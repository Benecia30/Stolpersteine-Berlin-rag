"""
Phase 5 - Streamlit interface.

Wraps 08_generate_answer.py with a simple Q&A UI. Reuses the retrieval and
generation logic from 07/08 directly (no duplicated pipeline logic) but
caches the heavy resources (BM25, FAISS, embedding model) once per session
instead of reloading them on every question -- retrieve() as written in 07
reloads everything from disk on every call, fine for CLI/eval use, bad for
an interactive UI where someone asks several questions in a row.

Run:
    uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501

In GitHub Codespaces, forward port 8501 (Codespaces usually prompts you
automatically) and open the forwarded URL.
"""

import importlib.util
import os
from pathlib import Path

import streamlit as st
from groq import Groq

ROOT = Path(__file__).parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hr = load_module("hybrid_retrieve", ROOT / "scripts" / "07_hybrid_retrieve.py")
ga = load_module("generate_answer_mod", ROOT / "scripts" / "08_generate_answer.py")

st.set_page_config(page_title="Stolpersteine Berlin Q&A", page_icon="🕯️", layout="centered")


@st.cache_resource(show_spinner="Loading search index (first run only)...")
def load_resources():
    docs = hr.load_docs()
    bm25, doc_ids = hr.load_bm25()
    index, faiss_idx_to_doc_id = hr.load_faiss()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hr.MODEL_NAME, device="cpu")
    return docs, bm25, doc_ids, index, faiss_idx_to_doc_id, model


docs, bm25, doc_ids, index, faiss_idx_to_doc_id, model = load_resources()


def cached_retrieve(question: str, top_n: int = 5):
    """Same hybrid RRF logic as 07_hybrid_retrieve.retrieve(), but using the
    cached resources above instead of reloading them from disk."""
    bm25_results = hr.bm25_search(question, bm25, doc_ids, hr.BM25_TOP_K)
    dense_results = hr.dense_search(question, index, faiss_idx_to_doc_id, model, hr.DENSE_TOP_K)
    fused = hr.reciprocal_rank_fusion(bm25_results, dense_results)[:top_n]

    payloads = []
    for doc_id, score in fused:
        d = docs[doc_id]
        payloads.append({
            "doc_id": doc_id,
            "rrf_score": round(score, 5),
            "text": d["text_bm25"],
            "source_url": d["source_url"],
        })
    return payloads


def cached_generate_answer(question: str, top_n: int = 5):
    """Same logic as 08_generate_answer.generate_answer(), but calling
    cached_retrieve() instead of hr.retrieve() to avoid a full resource
    reload on every question."""
    payloads = cached_retrieve(question, top_n)
    if not payloads:
        return {
            "answer": "No relevant Stolpersteine records were found for this question.",
            "sources": [],
            "citation_check": None,
            "payloads": [],
        }

    context = ga.build_context(payloads)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=ga.MODEL_NAME,
        messages=[
            {"role": "system", "content": ga.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    answer = completion.choices[0].message.content
    citation_check = ga.verify_citations(answer, payloads)

    return {
        "answer": answer,
        "sources": [{"doc_id": p["doc_id"], "url": p["source_url"]} for p in payloads],
        "citation_check": citation_check,
        "payloads": payloads,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🕯️ Berlin Stolpersteine Q&A")
st.caption(
    "Ask about victims commemorated by Stolpersteine (memorial stones) in Mitte, "
    "Charlottenburg-Wilmersdorf, and Tempelhof-Schöneberg. Answers are grounded "
    "only in the official Stolpersteine-Berlin.de dataset — the system will say "
    "'I don't know' rather than guess."
)

EXAMPLE_QUERIES = [
    "Wer wurde nach Theresienstadt deportiert?",
    "Wer lebte in der Ackerstraße?",
    "Wer hat überlebt?",
]

with st.sidebar:
    st.subheader("Example questions")
    for eq in EXAMPLE_QUERIES:
        if st.button(eq, use_container_width=True):
            st.session_state["question_input"] = eq
    st.divider()
    st.caption(
        "Known limitations: age/child-status isn't a separate indexed field, so "
        "'children who were murdered' relies on the model inferring age from birth "
        "and deportation dates. Exact house-number queries (e.g. a specific street "
        "number) may miss the right record among several on the same street."
    )

question = st.text_input(
    "Ask a question (German or English)",
    key="question_input",
    placeholder="Wer wurde nach Theresienstadt deportiert?",
)

ask = st.button("Ask", type="primary")

if ask and question:
    with st.spinner("Searching and generating answer..."):
        result = cached_generate_answer(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    check = result["citation_check"]
    if check is not None:
        if check["invalid"]:
            st.warning(f"⚠️ Citation check flagged possibly incorrect IDs: {sorted(check['invalid'])}")
        else:
            st.success("✅ All citations verified against retrieved sources.")

    if result["sources"]:
        with st.expander(f"📖 Sources ({len(result['sources'])})"):
            for s in result["sources"]:
                st.markdown(f"- [`{s['doc_id']}`]({s['url']})")

    if result["payloads"]:
        with st.expander("🔍 Retrieved context (what the model actually saw)"):
            for p in result["payloads"]:
                st.markdown(f"**`{p['doc_id']}`** — RRF score {p['rrf_score']}")
                st.text(p["text"])
elif ask and not question:
    st.info("Type a question first.")
