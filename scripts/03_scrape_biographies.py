import json
import time
import random
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

INPUT_CSV = "data/raw/stolpersteine_7_districts.csv"

# PUBLIC, committable output: structured facts only
OUTPUT_STRUCTURED = "data/raw/biographies_structured.jsonl"

# PRIVATE, gitignored output: copyrighted narrative text, used only as
# local retrieval context, never redistributed
OUTPUT_PROSE_CACHE = "data/cache/biographies_prose.jsonl"

FAILED_LOG = "data/raw/scrape_failures.jsonl"
CHECKPOINT_EVERY = 50
MIN_DELAY = 1.0
MAX_DELAY = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (academic research project; LLM Zoomcamp coursework; contact: your_email@example.com)"
}



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def fetch(url: str) -> tuple[str, str]:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=15,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text, response.url


def normalize_text(value: str | None) -> str | None:
    """Collapse repeated spaces, tabs, and line breaks."""
    if not value:
        return None
    return " ".join(value.split())


def text_or_none(node):
    return normalize_text(node.get_text(" ", strip=True)) if node else None


def parse_structured_fields(html: str, source_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    dl = soup.find("dl", class_="st-table")
    fields = {}
    if dl:
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            label = dt.get_text(strip=True)
            value = normalize_text(dd.get_text(" ", strip=True))
            fields[label] = value

    # fate is sometimes in a dt's inner div rather than plain text
    fate_div = dl.find("div", class_="field-fate") if dl else None
    fate = text_or_none(fate_div)

    # source reference codes, e.g. [BArch], [BLHA]
    source_spans = soup.select("#block-stolpersteinsourcesblock span[title]")
    source_refs = [s.get_text(strip=True) for s in source_spans]

    return {
        "source_url": source_url,
        "verlegeort": fields.get("Verlegeort"),
        "bezirk_ortsteil": fields.get("Bezirk/Ortsteil"),
        "verlegedatum": fields.get("Verlegedatum"),
        "geboren": fields.get("Geboren"),
        "deportation": fields.get("Deportation"),
        "fate": fate,
        "source_refs": source_refs,
    }


def parse_prose_text(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    bio_container = soup.select_one(".field_bio .field__item")
    return (
        normalize_text(bio_container.get_text(" ", strip=True))
        if bio_container
        else None
    )


def already_done_ids(path: str) -> set:
    if not Path(path).exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["stolperstein_id"])
            except Exception:
                continue
    return ids


def main():
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()

    done_ids = already_done_ids(OUTPUT_STRUCTURED)
    print(f"Already scraped: {len(done_ids)}")

    todo = df[~df["stolperstein_id"].isin(done_ids)]
    print(f"Remaining: {len(todo)}")

    struct_f = open(OUTPUT_STRUCTURED, "a", encoding="utf-8")
    prose_f = open(OUTPUT_PROSE_CACHE, "a", encoding="utf-8")
    fail_f = open(FAILED_LOG, "a", encoding="utf-8")

    count = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        stub_url = row["Url zu Biografie"]
        stolperstein_id = row["stolperstein_id"]
        try:
            real_html, real_url = fetch(stub_url)

            structured = parse_structured_fields(real_html, real_url)
            structured["stolperstein_id"] = stolperstein_id
            struct_f.write(json.dumps(structured, ensure_ascii=False) + "\n")

            prose = parse_prose_text(real_html)
            prose_f.write(json.dumps({
                "stolperstein_id": stolperstein_id,
                "source_url": real_url,
                "bio_text_raw": prose,
            }, ensure_ascii=False) + "\n")

        except Exception as e:
            fail_f.write(json.dumps({
                "stolperstein_id": stolperstein_id,
                "url": stub_url,
                "error": str(e),
            }, ensure_ascii=False) + "\n")

        count += 1
        if count % CHECKPOINT_EVERY == 0:
            struct_f.flush()
            prose_f.flush()
            fail_f.flush()
            print(f"Checkpoint: {count} processed")

        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    struct_f.close()
    prose_f.close()
    fail_f.close()
    print("Done.")


if __name__ == "__main__":
    main()
