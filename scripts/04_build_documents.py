"""
Merge the district CSV (names, coordinates, address) with the scraped
structured facts and private prose into one unified documents.jsonl.

Produces, per stolperstein_id:
  - text_bm25  : structured-facts-only sentence (safe, committable, used
                 both for BM25 indexing AND as the payload fed to the LLM
                 at generation time)
  - text_embed : structured facts + private prose appended (used only to
                 build embeddings; never surfaced to the LLM or user)

Output: data/processed/documents.jsonl (gitignored — derived in part from
private prose via text_embed; keep the committed artifact list to code +
the structured JSONL, not this merged file, unless you strip text_embed
before committing).
"""

import json
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/raw/stolpersteine_7_districts.csv")
STRUCTURED_PATH = Path("data/raw/biographies_structured.jsonl")
PROSE_PATH = Path("data/cache/biographies_prose.jsonl")
OUTPUT_PATH = Path("data/processed/documents.jsonl")


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load a JSONL file into a dict keyed by stolperstein_id."""
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["stolperstein_id"]] = record
    return records


def build_text_bm25(row: pd.Series, structured: dict | None) -> str:
    """Structured-facts-only sentence. No prose. This is what both the
    BM25 index and the LLM prompt payload are built from."""
    parts = []

    name = f"{row.get('Vorname', '')} {row.get('Nachname', '')}".strip()
    if name:
        parts.append(name)

    address = row.get("Straße + Hausnummer") or row.get("Straße")
    bezirk = row.get("Bezirk")
    if address and bezirk:
        parts.append(f"lived at {address} in {bezirk}")
    elif bezirk:
        parts.append(f"lived in {bezirk}")

    geburtsjahr = row.get("Geburtsjahr")
    if pd.notna(geburtsjahr):
        parts.append(f"born {geburtsjahr}")

    if structured:
        geboren = structured.get("geboren")
        if geboren:
            parts.append(f"Born: {geboren}")

        deportation = structured.get("deportation")
        if deportation:
            parts.append(f"Deportation: {deportation}")

        fate = structured.get("fate")
        if fate:
            parts.append(f"Fate: {fate}")

        verlegedatum = structured.get("verlegedatum")
        if verlegedatum:
            parts.append(f"Stolperstein laid: {verlegedatum}")

    return ". ".join(parts) + "."


def build_text_embed(text_bm25: str, prose: dict | None) -> str:
    """Facts + private prose, for embeddings only. Never shown to the LLM
    or the user, never committed as raw text."""
    if prose and prose.get("bio_text_raw"):
        return f"{text_bm25} {prose['bio_text_raw']}"
    return text_bm25


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    structured = load_jsonl(STRUCTURED_PATH)
    prose = load_jsonl(PROSE_PATH)

    print(f"CSV rows: {len(df)}")
    print(f"Structured records available: {len(structured)}")
    print(f"Prose records available: {len(prose)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_structured = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            sid = row["stolperstein_id"]
            struct_record = structured.get(sid)
            prose_record = prose.get(sid)

            if struct_record is None:
                missing_structured += 1
                continue  # skip rows not yet scraped (partial-run safe)

            text_bm25 = build_text_bm25(row, struct_record)
            text_embed = build_text_embed(text_bm25, prose_record)

            doc = {
                "doc_id": sid,
                "name": f"{row.get('Vorname', '')} {row.get('Nachname', '')}".strip(),
                "verlegeort": struct_record.get("verlegeort"),
                "bezirk": struct_record.get("bezirk_ortsteil") or row.get("Bezirk"),
                "geboren": struct_record.get("geboren"),
                "deportation": struct_record.get("deportation"),
                "fate": struct_record.get("fate"),
                "verlegedatum": struct_record.get("verlegedatum"),
                "source_url": struct_record.get("source_url"),
                "source_refs": struct_record.get("source_refs", []),
                "latitude": row.get("Breitengrad"),
                "longitude": row.get("Längengrad"),
                "text_bm25": text_bm25,
                "text_embed": text_embed,
            }

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    print(f"Documents written: {written}")
    print(f"Skipped (not yet scraped): {missing_structured}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()