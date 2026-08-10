"""
Phase 3: Groq LLM integration with strict grounding.

Takes a user question -> hybrid retrieve (07) -> build facts-only context ->
Groq chat completion with a grounding-only system prompt -> cited answer.

Usage:
    uv run python scripts/08_generate_answer.py "Wer wurde nach Theresienstadt deportiert?"
"""
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
retrieve_mod = import_module("07_hybrid_retrieve")

MODEL_NAME = "llama-3.3-70b-versatile"  # confirm current free-tier model name on Groq's console before relying on this
TOP_N_CONTEXT = 5

SYSTEM_PROMPT = """You are a research assistant answering questions about the Stolpersteine \
(memorial stones for victims of Nazi persecution) in Berlin, using ONLY the facts provided \
in the context below.

STRICT RULES:
1. Answer ONLY using information explicitly present in the provided context. Do not use \
outside knowledge about the Holocaust, Berlin, or history in general, even if you believe \
it to be true.
2. Every factual claim in your answer must be traceable to a specific doc_id in the context. \
Cite the doc_id in square brackets after each claim, e.g. "...deported in 1942 [stolperstein_00001]."
3. If the context does not contain enough information to answer the question, say so \
explicitly. Do not fill gaps with plausible-sounding inference or general historical knowledge.
4. Never speculate about a person's fate, age, family situation, or any detail not stated \
in the context, even if it seems like a reasonable guess.
5. If multiple people match the question, mention each one with their own citation rather \
than merging them into a single vague statement.
6. Write in the same language as the user's question (German or English).
7. The context you receive is always exactly 5 documents, retrieved out of 7,205 total \
records. If the question does not name or clearly identify a specific person, address, or \
event (for example, it only names a data field like "Verlegedatum"/laying date, or asks a \
general category question with no distinguishing detail), the 5 documents you were given \
are an arbitrary, non-representative sample -- NOT the answer to a general query about all \
7,205 records. In that case, do not list the 5 people as if they answer the question. \
Instead, say the question needs to specify a person, address, or other identifying detail, \
since it does not currently point to a specific answerable topic in the data."""



def build_context(payloads):
    blocks = []
    for p in payloads:
        blocks.append(f"[{p['doc_id']}] {p['text']} (Source: {p['source_url']})")
    return "\n\n".join(blocks)


def verify_citations(answer: str, payloads: list) -> dict:
    """Check that every [doc_id] cited in the answer actually came from retrieved context.
    Catches hallucinated IDs and typos (e.g. model writing 'stolperstone' instead of
    'stolperstein') that would otherwise silently pass as valid-looking citations."""
    valid_ids = {p["doc_id"] for p in payloads}
    cited_ids = set(re.findall(r"\[(stolperstein_\d+)\]", answer))
    return {
        "cited": cited_ids,
        "valid": cited_ids & valid_ids,
        "invalid": cited_ids - valid_ids,       # hallucinated or malformed IDs
        "unused_context": valid_ids - cited_ids,  # retrieved but never cited
    }


def generate_answer(question: str, top_n: int = TOP_N_CONTEXT) -> dict:
    payloads = retrieve_mod.retrieve(question, top_n=top_n)
    if not payloads:
        return {
            "answer": "No relevant Stolpersteine records were found for this question.",
            "sources": [],
            "citation_check": None,
        }

    context = build_context(payloads)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # deterministic, minimizes fabrication risk
    )

    answer = completion.choices[0].message.content
    citation_check = verify_citations(answer, payloads)

    return {
        "answer": answer,
        "sources": [{"doc_id": p["doc_id"], "url": p["source_url"]} for p in payloads],
        "citation_check": citation_check,
    }


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Wer wurde nach Theresienstadt deportiert?"
    result = generate_answer(question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources retrieved:")
    for s in result["sources"]:
        print(f"  [{s['doc_id']}] {s['url']}")

    check = result["citation_check"]
    if check:
        print("\nCitation check:")
        print(f"  Valid citations:   {sorted(check['valid'])}")
        if check["invalid"]:
            print(f"  ⚠️  INVALID/HALLUCINATED citations: {sorted(check['invalid'])}")
        if check["unused_context"]:
            print(f"  Unused context (retrieved, not cited): {sorted(check['unused_context'])}")