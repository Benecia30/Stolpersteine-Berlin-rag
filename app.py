"""
Phase 5 - Streamlit interface + lightweight monitoring.

The application:
- caches BM25, FAISS and the embedding model once per Streamlit session
- performs hybrid BM25 + dense retrieval with RRF fusion
- generates strictly grounded answers with Groq
- verifies citations against retrieved document IDs
- logs query-level monitoring metrics
- collects explicit thumbs-up / thumbs-down feedback

Run:
    uv run streamlit run app.py \
        --server.address 0.0.0.0 \
        --server.port 8501
"""

import importlib.util
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from utils.logger import log_feedback, log_query


ROOT = Path(__file__).parent

load_dotenv(ROOT / ".env")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hr = load_module(
    "hybrid_retrieve",
    ROOT / "scripts" / "07_hybrid_retrieve.py",
)

ga = load_module(
    "generate_answer_mod",
    ROOT / "scripts" / "08_generate_answer.py",
)


st.set_page_config(
    page_title="Stolpersteine Berlin Q&A",
    page_icon="🕯️",
    layout="centered",
)


@st.cache_resource(
    show_spinner="Loading search index (first run only)..."
)
def load_resources():
    docs = hr.load_docs()
    bm25, doc_ids = hr.load_bm25()
    index, faiss_idx_to_doc_id = hr.load_faiss()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        hr.MODEL_NAME,
        device="cpu",
    )

    return (
        docs,
        bm25,
        doc_ids,
        index,
        faiss_idx_to_doc_id,
        model,
    )


(
    docs,
    bm25,
    doc_ids,
    index,
    faiss_idx_to_doc_id,
    model,
) = load_resources()


def cached_retrieve(
    question: str,
    top_n: int = 5,
):
    """
    Same hybrid RRF retrieval logic as
    scripts/07_hybrid_retrieve.py, but uses cached resources.
    """

    bm25_results = hr.bm25_search(
        question,
        bm25,
        doc_ids,
        hr.BM25_TOP_K,
    )

    dense_results = hr.dense_search(
        question,
        index,
        faiss_idx_to_doc_id,
        model,
        hr.DENSE_TOP_K,
    )

    fused = hr.reciprocal_rank_fusion(
        bm25_results,
        dense_results,
    )[:top_n]

    payloads = []

    for doc_id, score in fused:
        document = docs[doc_id]

        payloads.append(
            {
                "doc_id": doc_id,
                "rrf_score": round(score, 5),
                "text": document["text_bm25"],
                "source_url": document["source_url"],
            }
        )

    return payloads


def cached_generate_answer(
    question: str,
    top_n: int = 5,
):
    """
    Same generation logic as 08_generate_answer.py,
    with additional timing information for monitoring.
    """

    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # Retrieval timing
    # ---------------------------------------------------------

    retrieval_start = time.perf_counter()

    payloads = cached_retrieve(
        question,
        top_n,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if not payloads:
        total_time = (
            time.perf_counter()
            - total_start
        )

        return {
            "answer": (
                "No relevant Stolpersteine records were "
                "found for this question."
            ),
            "sources": [],
            "citation_check": None,
            "payloads": [],
            "metrics": {
                "retrieval_time_seconds": retrieval_time,
                "generation_time_seconds": 0.0,
                "total_time_seconds": total_time,
            },
        }

    context = ga.build_context(payloads)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    # ---------------------------------------------------------
    # Generation timing
    # ---------------------------------------------------------

    generation_start = time.perf_counter()

    client = Groq(
        api_key=os.environ["GROQ_API_KEY"]
    )

    completion = client.chat.completions.create(
        model=ga.MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": ga.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0.0,
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    answer = completion.choices[0].message.content

    citation_check = ga.verify_citations(
        answer,
        payloads,
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    return {
        "answer": answer,
        "sources": [
            {
                "doc_id": p["doc_id"],
                "url": p["source_url"],
            }
            for p in payloads
        ],
        "citation_check": citation_check,
        "payloads": payloads,
        "metrics": {
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": generation_time,
            "total_time_seconds": total_time,
        },
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None

if "latest_question" not in st.session_state:
    st.session_state["latest_question"] = None

if "latest_query_id" not in st.session_state:
    st.session_state["latest_query_id"] = None

if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🕯️ Berlin Stolpersteine Q&A")

st.caption(
    "Ask about victims commemorated by Stolpersteine "
    "(memorial stones) across seven Berlin districts: "
    "Charlottenburg-Wilmersdorf, Mitte, "
    "Tempelhof-Schöneberg, Friedrichshain-Kreuzberg, "
    "Pankow, Steglitz-Zehlendorf, and Neukölln. "
    "Answers are grounded only in the project's "
    "structured Stolpersteine data."
)


EXAMPLE_QUERIES = [
    "Wer wurde nach Theresienstadt deportiert?",
    "Wer lebte in der Ackerstraße?",
    "Wer wurde nach Auschwitz deportiert und lebte in Pankow?",
]


with st.sidebar:
    st.subheader("Example questions")

    for eq in EXAMPLE_QUERIES:
        if st.button(
            eq,
            use_container_width=True,
        ):
            st.session_state[
                "question_input"
            ] = eq

    st.divider()

    st.caption(
        "Known limitations: age/child-status is not a "
        "separate indexed field. Exact house-number "
        "queries can also be difficult because neither "
        "BM25 nor the dense retriever explicitly models "
        "street numbers."
    )


question = st.text_input(
    "Ask a question (German or English)",
    key="question_input",
    placeholder=(
        "Wer wurde nach Theresienstadt deportiert?"
    ),
)

ask = st.button(
    "Ask",
    type="primary",
)


# ---------------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------------

if ask and question:
    with st.spinner(
        "Searching and generating answer..."
    ):
        result = cached_generate_answer(
            question
        )

    metrics = result["metrics"]

    query_id = log_query(
        question=question,
        answer=result["answer"],
        sources=result["sources"],
        citation_check=result["citation_check"],
        retrieval_time_seconds=metrics[
            "retrieval_time_seconds"
        ],
        generation_time_seconds=metrics[
            "generation_time_seconds"
        ],
        total_time_seconds=metrics[
            "total_time_seconds"
        ],
    )

    # Persist result because clicking feedback causes
    # Streamlit to rerun the entire script.
    st.session_state[
        "latest_result"
    ] = result

    st.session_state[
        "latest_question"
    ] = question

    st.session_state[
        "latest_query_id"
    ] = query_id

    st.session_state[
        "feedback_given"
    ] = None


elif ask and not question:
    st.info("Type a question first.")


# ---------------------------------------------------------------------------
# Display latest result
# ---------------------------------------------------------------------------

result = st.session_state[
    "latest_result"
]

if result is not None:

    st.markdown("### Answer")

    st.write(
        result["answer"]
    )

    check = result[
        "citation_check"
    ]

    if check is not None:
        if check["invalid"]:
            st.warning(
                "⚠️ Citation check flagged possibly "
                f"incorrect IDs: "
                f"{sorted(check['invalid'])}"
            )
        else:
            st.success(
                "✅ All citations verified against "
                "retrieved sources."
            )

    # -----------------------------------------------------------------------
    # Feedback
    # -----------------------------------------------------------------------

    st.markdown("#### Was this answer helpful?")

    if (
        st.session_state[
            "feedback_given"
        ]
        is None
    ):
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "👍 Helpful",
                use_container_width=True,
                key=(
                    "helpful_"
                    + st.session_state[
                        "latest_query_id"
                    ]
                ),
            ):
                log_feedback(
                    query_id=st.session_state[
                        "latest_query_id"
                    ],
                    question=st.session_state[
                        "latest_question"
                    ],
                    feedback="helpful",
                )

                st.session_state[
                    "feedback_given"
                ] = "helpful"

                st.rerun()

        with col2:
            if st.button(
                "👎 Not helpful",
                use_container_width=True,
                key=(
                    "not_helpful_"
                    + st.session_state[
                        "latest_query_id"
                    ]
                ),
            ):
                log_feedback(
                    query_id=st.session_state[
                        "latest_query_id"
                    ],
                    question=st.session_state[
                        "latest_question"
                    ],
                    feedback="not_helpful",
                )

                st.session_state[
                    "feedback_given"
                ] = "not_helpful"

                st.rerun()

    else:
        if (
            st.session_state[
                "feedback_given"
            ]
            == "helpful"
        ):
            st.success(
                "Thanks for the feedback! 👍"
            )
        else:
            st.info(
                "Thanks for the feedback! 👎"
            )

    # -----------------------------------------------------------------------
    # Sources
    # -----------------------------------------------------------------------

    if result["sources"]:
        with st.expander(
            f"📖 Sources "
            f"({len(result['sources'])})"
        ):
            for source in result[
                "sources"
            ]:
                st.markdown(
                    f"- [`{source['doc_id']}`]"
                    f"({source['url']})"
                )

    # -----------------------------------------------------------------------
    # Retrieved context
    # -----------------------------------------------------------------------

    if result["payloads"]:
        with st.expander(
            "🔍 Retrieved context "
            "(what the model actually saw)"
        ):
            for payload in result[
                "payloads"
            ]:
                st.markdown(
                    f"**`{payload['doc_id']}`** — "
                    f"RRF score "
                    f"{payload['rrf_score']}"
                )

                st.text(
                    payload["text"]
                )

    # -----------------------------------------------------------------------
    # Timing information
    # -----------------------------------------------------------------------

    metrics = result["metrics"]

    with st.expander(
        "⚙️ Response metrics"
    ):
        st.write(
            f"Retrieval: "
            f"{metrics['retrieval_time_seconds']:.3f} s"
        )

        st.write(
            f"LLM generation: "
            f"{metrics['generation_time_seconds']:.3f} s"
        )

        st.write(
            f"Total: "
            f"{metrics['total_time_seconds']:.3f} s"
        )
