"""
Phase 4 - end-to-end eval (LLM-as-judge).

For each labeled query, runs your existing 08_generate_answer pipeline,
then has a separate Groq call grade the answer on:
    - groundedness   (every claim traceable to a retrieved [doc_id])
    - citation_accuracy  (citations point to the correct facts, not just valid IDs)
    - hedging_appropriate  ("I don't know" used correctly, not over/under-hedged)

Run from repo root:
    uv run scripts/10_eval_generation.py

Uses data/eval/labeled_queries.jsonl (same file as 09). "relevant_doc_ids" isn't used
here directly (that's for retrieval eval) -- generation eval judges the answer against
whatever context 08 actually retrieved, not against ground truth.

generate_answer() returns {"answer", "sources", "citation_check"} -- no raw context
string. Context text for the judge is reconstructed from documents.jsonl using the
doc_ids in "sources", loaded once up front, so this doesn't call retrieve() a second
time per query (retrieve() reloads the model/index on every call, expensive in a loop).
"""

import json
import importlib.util
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

spec = importlib.util.spec_from_file_location(
    "generate_answer_mod", ROOT / "scripts" / "08_generate_answer.py"
)
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)

client = Groq(api_key=os.environ["GROQ_API_KEY"])

JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_SYSTEM_PROMPT = """You are grading answers from a RAG system about Berlin Stolpersteine \
(Holocaust memorial stones). You will be given a question, the retrieved context \
(facts-only, from data/processed documents), and the generated answer.

You will also be told whether an automated citation checker found any invalid/hallucinated \
[doc_id] citations in the answer -- treat any invalid citation as an automatic groundedness \
score of 1, regardless of how plausible the surrounding text reads.

Score each dimension 1-5 and give a one-sentence reason:

1. groundedness: Is every factual claim in the answer traceable to the provided context? \
   5 = fully grounded, no invented facts. 1 = significant fabrication or invalid citations.
2. citation_accuracy: Do the [doc_id] citations point to the correct facts they support \
   (not just valid IDs, but the RIGHT ID for that specific claim)? \
   5 = all citations correct. 1 = citations wrong or misattributed.
3. hedging_appropriate: Does the answer say "I don't know" when the context doesn't \
   support an answer, and NOT hedge when the context does clearly support one? \
   5 = perfectly calibrated. 1 = confidently wrong or needlessly evasive.

Respond ONLY with JSON, no markdown fences, no preamble:
{"groundedness": int, "groundedness_reason": str, "citation_accuracy": int, \
"citation_accuracy_reason": str, "hedging_appropriate": int, "hedging_appropriate_reason": str}
"""


def build_context_from_sources(sources, docs):
    blocks = []
    for s in sources:
        doc = docs.get(s["doc_id"])
        if doc is None:
            continue
        blocks.append(f"[{s['doc_id']}] {doc['text_bm25']} (Source: {s['url']})")
    return "\n\n".join(blocks)


def judge(query, context, answer, invalid_citations, notes=""):
    user_msg = f"QUESTION: {query}\n\nRETRIEVED CONTEXT:\n{context}\n\nGENERATED ANSWER:\n{answer}"
    if invalid_citations:
        user_msg += f"\n\nAUTOMATED CHECK FOUND INVALID CITATIONS: {sorted(invalid_citations)}"
    if notes:
        user_msg += f"\n\nNOTES FOR GRADING: {notes}"

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def load_queries(path):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def main():
    queries = load_queries(ROOT / "data" / "eval" / "labeled_queries.jsonl")
    if not queries:
        print("No labeled queries found.")
        return

    print("Loading documents.jsonl once...")
    docs = ga.retrieve_mod.load_docs()

    results = []
    for row in queries:
        print(f"\nGenerating + judging: {row['query']!r}")
        gen = ga.generate_answer(row["query"])

        if gen["citation_check"] is None:
            print("  (no context retrieved, skipping judge)")
            continue

        context = build_context_from_sources(gen["sources"], docs)
        invalid = gen["citation_check"]["invalid"]

        grade = judge(
            row["query"],
            context,
            gen["answer"],
            invalid_citations=invalid,
            notes=row.get("note", ""),
        )
        results.append({"query": row["query"], "invalid_citations": sorted(invalid), **grade})

        for k in ("groundedness", "citation_accuracy", "hedging_appropriate"):
            print(f"  {k}: {grade[k]}  ({grade[k + '_reason']})")
        if invalid:
            print(f"  ⚠️  invalid citations flagged by 08's own checker: {sorted(invalid)}")

    out_path = ROOT / "data" / "eval" / "generation_eval_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if results:
        for metric in ("groundedness", "citation_accuracy", "hedging_appropriate"):
            avg = sum(r[metric] for r in results) / len(results)
            print(f"\nAVG {metric}: {avg:.2f} / 5")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()