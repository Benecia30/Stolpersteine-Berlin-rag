"""
Phase 4 - judge calibration.

The generation eval (10_eval_generation.py) scored 8/9 real queries at a perfect
5/5/5. That's either a genuinely strong system, or a lenient judge rubber-stamping
everything. This script feeds the SAME judge a set of deliberately broken answers
to find out which.

Uses real context from a real query (Hans Frost, single clean doc) so the only
variable is the answer text itself.

Run from repo root:
    uv run scripts/eval_helpers/calibrate_judge.py

A properly working judge should score every case below LOW on the dimension it's
designed to break, ideally 1-2, and should NOT need extra hints to catch it --
these are the same kind of errors real generation could plausibly produce.
"""

import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

spec = importlib.util.spec_from_file_location(
    "eval_gen", ROOT / "scripts" / "10_eval_generation.py"
)
eval_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_gen)

spec2 = importlib.util.spec_from_file_location(
    "generate_answer_mod", ROOT / "scripts" / "08_generate_answer.py"
)
ga = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ga)


# Real context for a real, clean query -- gives the judge something real to compare against.
QUERY = "Hans Frost"
REAL_CONTEXT = (
    "[stolperstein_03639] Hans Frost. lived at Witzlebenstr. 20 in Charlottenburg-Wilmersdorf. "
    "born 1931. Born: 23. April 1931 in Berlin. Deportation: am 17. November 1941 nach Kowno, "
    "Fort IX. Fate: Ermordet. Stolperstein laid: 23. April 2013. (Source: https://www.stolpersteine-berlin.de/...)"
)

TEST_CASES = [
    {
        "label": "hallucinated_citation",
        "targets": "groundedness / citation_accuracy",
        "answer": (
            "Hans Frost lived at Witzlebenstr. 20 in Charlottenburg-Wilmersdorf and was "
            "deported on 17 November 1941 to Kowno, Fort IX, where he was murdered "
            "[stolperstein_09999]."
        ),
        "invalid_citations": {"stolperstein_09999"},
    },
    {
        "label": "fabricated_fact_valid_citation",
        "targets": "groundedness",
        "answer": (
            "Hans Frost lived at Witzlebenstr. 20 and was deported to Auschwitz in 1943, "
            "where he was murdered [stolperstein_03639]."
        ),
        "invalid_citations": set(),  # citation ID is valid, the FACT attached to it is wrong
    },
    {
        "label": "misattributed_citation",
        "targets": "citation_accuracy",
        "answer": (
            "Hans Frost was born in Berlin in 1931 [stolperstein_03639]. He lived at "
            "Alexanderplatz his whole life and worked as a teacher [stolperstein_03639]."
        ),
        "invalid_citations": set(),  # ID is valid but doesn't support the address/job claims
    },
    {
        "label": "overconfident_no_hedge",
        "targets": "hedging_appropriate",
        "answer": (
            "Hans Frost was married with three children and was active in the local "
            "synagogue before his deportation."
        ),
        "invalid_citations": set(),  # no citation at all, pure invention, context has none of this
    },
    {
        "label": "needless_overhedge",
        "targets": "hedging_appropriate",
        "answer": (
            "I cannot be completely certain, but the context might possibly suggest that "
            "someone named Hans Frost may have lived somewhere in Berlin at some point, "
            "though I cannot confirm any further details."
        ),
        "invalid_citations": set(),  # context clearly supports a confident answer, this underclaims
    },
]


def main():
    print(f"Calibrating judge against query: {QUERY!r}\n")

    results = []
    for case in TEST_CASES:
        print(f"--- {case['label']} (targets: {case['targets']}) ---")
        grade = eval_gen.judge(
            QUERY,
            REAL_CONTEXT,
            case["answer"],
            invalid_citations=case["invalid_citations"],
        )
        for k in ("groundedness", "citation_accuracy", "hedging_appropriate"):
            print(f"  {k}: {grade[k]}  ({grade[k + '_reason']})")
        results.append({"label": case["label"], "targets": case["targets"], **grade})
        print()

    out_path = ROOT / "data" / "eval" / "judge_calibration_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=== Summary ===")
    print("Each case should score LOW (1-2) on its targeted dimension.")
    print("If any score HIGH (4-5) on its targeted dimension, the judge is too lenient")
    print("and the 5/5/5 scores on real queries should not be fully trusted.\n")
    for r in results:
        flags = []
        for dim in ("groundedness", "citation_accuracy", "hedging_appropriate"):
            if dim.replace("_", " ") in r["targets"].replace("_", " ") or dim in r["targets"]:
                if r[dim] >= 4:
                    flags.append(f"{dim}={r[dim]} (SHOULD BE LOW)")
        status = "⚠️  JUDGE TOO LENIENT" if flags else "OK - judge caught it"
        print(f"  {r['label']}: {status}  {flags}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
