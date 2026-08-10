"""
Lightweight monitoring utilities for the Streamlit application.

Logs:
- every generated answer to logs/queries.jsonl
- explicit user feedback to logs/feedback.jsonl

JSONL keeps the monitoring layer simple and reproducible without requiring
a separate database service.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

QUERY_LOG = LOG_DIR / "queries.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: dict):
    _ensure_log_dir()

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_query(
    question: str,
    answer: str,
    sources: list,
    citation_check,
    retrieval_time_seconds: float,
    generation_time_seconds: float,
    total_time_seconds: float,
    retrieval_method: str = "hybrid_rrf",
) -> str:
    """
    Log one completed user query.

    Returns a unique query_id so later feedback can be linked
    to the exact generated answer.
    """

    query_id = str(uuid.uuid4())

    invalid_citations = []

    if citation_check is not None:
        invalid_citations = sorted(
            list(citation_check.get("invalid", []))
        )

    citation_valid = len(invalid_citations) == 0

    record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer_length_chars": len(answer),
        "retrieval_method": retrieval_method,
        "num_sources": len(sources),
        "source_ids": [
            source.get("doc_id")
            for source in sources
        ],
        "citation_valid": citation_valid,
        "invalid_citations": invalid_citations,
        "retrieval_time_seconds": round(
            retrieval_time_seconds, 4
        ),
        "generation_time_seconds": round(
            generation_time_seconds, 4
        ),
        "total_time_seconds": round(
            total_time_seconds, 4
        ),
    }

    _append_jsonl(QUERY_LOG, record)

    return query_id


def log_feedback(
    query_id: str,
    question: str,
    feedback: str,
):
    """
    Log explicit user feedback for a previously generated answer.

    feedback should normally be:
        helpful
        not_helpful
    """

    record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "feedback": feedback,
    }

    _append_jsonl(FEEDBACK_LOG, record)
