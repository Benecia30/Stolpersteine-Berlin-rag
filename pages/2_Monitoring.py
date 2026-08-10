"""
Monitoring dashboard for the Stolpersteine Berlin RAG application.

Uses:
- logs/queries.jsonl
- logs/feedback.jsonl

Shows:
- query volume
- feedback distribution
- citation-validation rate
- latency statistics
- latency over time
- recent queries
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent

QUERY_LOG = ROOT / "logs" / "queries.jsonl"
FEEDBACK_LOG = ROOT / "logs" / "feedback.jsonl"


st.set_page_config(
    page_title="RAG Monitoring",
    page_icon="📊",
    layout="wide",
)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows)


queries = load_jsonl(QUERY_LOG)
feedback = load_jsonl(FEEDBACK_LOG)


st.title("📊 RAG Monitoring Dashboard")

st.caption(
    "Runtime monitoring for the Berlin Stolpersteine Q&A system. "
    "Query logs contain latency, source and citation-validation metrics; "
    "feedback logs contain explicit user ratings."
)


if queries.empty:
    st.info(
        "No query monitoring data is available yet. "
        "Ask some questions in the main application first."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

queries["timestamp"] = pd.to_datetime(
    queries["timestamp"],
    errors="coerce",
)

if not feedback.empty:
    feedback["timestamp"] = pd.to_datetime(
        feedback["timestamp"],
        errors="coerce",
    )


# Join feedback onto query-level monitoring data
if not feedback.empty:
    feedback_small = feedback[
        [
            "query_id",
            "feedback",
        ]
    ].drop_duplicates(
        subset=["query_id"],
        keep="last",
    )

    combined = queries.merge(
        feedback_small,
        on="query_id",
        how="left",
    )
else:
    combined = queries.copy()
    combined["feedback"] = None


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

total_queries = len(queries)

feedback_count = combined["feedback"].notna().sum()

helpful_count = (
    combined["feedback"] == "helpful"
).sum()

not_helpful_count = (
    combined["feedback"] == "not_helpful"
).sum()

if feedback_count:
    helpful_rate = (
        helpful_count / feedback_count * 100
    )
else:
    helpful_rate = 0.0


citation_pass_count = (
    queries["citation_valid"] == True
).sum()

citation_pass_rate = (
    citation_pass_count / total_queries * 100
    if total_queries
    else 0.0
)

avg_total_latency = (
    queries["total_time_seconds"].mean()
)

median_total_latency = (
    queries["total_time_seconds"].median()
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total queries",
    total_queries,
)

col2.metric(
    "Helpful rate",
    f"{helpful_rate:.1f}%",
    help=(
        f"{helpful_count} helpful / "
        f"{feedback_count} rated queries"
    ),
)

col3.metric(
    "Citation pass rate",
    f"{citation_pass_rate:.1f}%",
    help=(
        f"{citation_pass_count} of "
        f"{total_queries} generated answers passed "
        "the citation-ID validation check."
    ),
)

col4.metric(
    "Median response time",
    f"{median_total_latency:.2f} s",
)


st.divider()


# ---------------------------------------------------------------------------
# Feedback + citations
# ---------------------------------------------------------------------------

left, right = st.columns(2)


with left:
    st.subheader("👍 User feedback")

    if feedback_count:
        feedback_counts = (
            combined["feedback"]
            .dropna()
            .value_counts()
            .rename(
                index={
                    "helpful": "Helpful",
                    "not_helpful": "Not helpful",
                }
            )
        )

        st.bar_chart(
            feedback_counts
        )

        st.write(
            f"**{helpful_count} helpful** and "
            f"**{not_helpful_count} not helpful** "
            f"out of {feedback_count} rated answers."
        )

    else:
        st.info(
            "No explicit feedback has been recorded yet."
        )


with right:
    st.subheader("✅ Citation validation")

    citation_counts = (
        queries["citation_valid"]
        .map(
            {
                True: "Passed",
                False: "Failed",
            }
        )
        .value_counts()
    )

    st.bar_chart(
        citation_counts
    )

    st.write(
        f"**{citation_pass_rate:.1f}%** of generated "
        "answers passed programmatic citation validation."
    )


st.divider()


# ---------------------------------------------------------------------------
# Latency monitoring
# ---------------------------------------------------------------------------

st.subheader("⏱️ Response latency")

latency_cols = [
    "retrieval_time_seconds",
    "generation_time_seconds",
    "total_time_seconds",
]

latency_summary = pd.DataFrame(
    {
        "Metric": [
            "Retrieval",
            "Generation",
            "Total",
        ],
        "Mean (s)": [
            queries[
                "retrieval_time_seconds"
            ].mean(),
            queries[
                "generation_time_seconds"
            ].mean(),
            queries[
                "total_time_seconds"
            ].mean(),
        ],
        "Median (s)": [
            queries[
                "retrieval_time_seconds"
            ].median(),
            queries[
                "generation_time_seconds"
            ].median(),
            queries[
                "total_time_seconds"
            ].median(),
        ],
        "Max (s)": [
            queries[
                "retrieval_time_seconds"
            ].max(),
            queries[
                "generation_time_seconds"
            ].max(),
            queries[
                "total_time_seconds"
            ].max(),
        ],
    }
)


st.dataframe(
    latency_summary.style.format(
        {
            "Mean (s)": "{:.3f}",
            "Median (s)": "{:.3f}",
            "Max (s)": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


latency_chart = (
    queries[
        [
            "timestamp",
            "retrieval_time_seconds",
            "generation_time_seconds",
            "total_time_seconds",
        ]
    ]
    .dropna()
    .sort_values("timestamp")
    .set_index("timestamp")
)

st.line_chart(
    latency_chart
)


st.caption(
    "The first request may include model/index warm-up effects. "
    "Later requests better represent steady-state interactive latency."
)


st.divider()


# ---------------------------------------------------------------------------
# Answer / retrieval monitoring
# ---------------------------------------------------------------------------

left, right = st.columns(2)


with left:
    st.subheader("📚 Sources per answer")

    source_counts = (
        queries["num_sources"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        source_counts
    )


with right:
    st.subheader("📝 Answer length")

    answer_lengths = (
        queries[
            [
                "timestamp",
                "answer_length_chars",
            ]
        ]
        .dropna()
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    st.line_chart(
        answer_lengths
    )


st.divider()


# ---------------------------------------------------------------------------
# Feedback-linked failures
# ---------------------------------------------------------------------------

st.subheader("🔎 Queries receiving negative feedback")

negative = combined[
    combined["feedback"] == "not_helpful"
].copy()


if negative.empty:
    st.success(
        "No negatively rated queries have been recorded."
    )

else:
    columns_to_show = [
        "question",
        "num_sources",
        "citation_valid",
        "retrieval_time_seconds",
        "generation_time_seconds",
        "total_time_seconds",
    ]

    available_columns = [
        col
        for col in columns_to_show
        if col in negative.columns
    ]

    st.dataframe(
        negative[
            available_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Negative feedback is useful for identifying failure "
        "patterns that citation validation alone cannot detect. "
        "A response can be perfectly cited but still fail to "
        "answer the user's intended question."
    )


st.divider()


# ---------------------------------------------------------------------------
# Recent query log
# ---------------------------------------------------------------------------

st.subheader("🕘 Recent queries")

display_columns = [
    "timestamp",
    "question",
    "feedback",
    "citation_valid",
    "num_sources",
    "total_time_seconds",
]

available_display_columns = [
    column
    for column in display_columns
    if column in combined.columns
]

recent = (
    combined
    .sort_values(
        "timestamp",
        ascending=False,
    )
    [available_display_columns]
)


st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------------
# Raw monitoring data
# ---------------------------------------------------------------------------

with st.expander(
    "🧾 Raw monitoring data"
):
    st.write(
        "Query log"
    )

    st.dataframe(
        queries,
        use_container_width=True,
    )

    st.write(
        "Feedback log"
    )

    if feedback.empty:
        st.write(
            "No feedback records."
        )
    else:
        st.dataframe(
            feedback,
            use_container_width=True,
        )
