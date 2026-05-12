#!/usr/bin/env python3
"""
News Atom Lite — Batch Extractor
---------------------------------
Process multiple articles in one run. Accepts a folder of .txt files
or a CSV with url/file/title/date/journalist/org columns.

IDs are globally unique across the entire batch — counter state is
persisted in {output-dir}/.newsatom_counters.json.

Output is appended to events.jsonl and atoms.jsonl in the output directory.
Use --overwrite to replace existing files at the start of each batch run.

Usage:
  # Process a folder of .txt files
  python batch_extract.py --input-dir ./articles --backend openai --org "Scroll.in" --org-prefix SCR

  # Process a CSV file
  python batch_extract.py --csv articles.csv --backend anthropic --org-prefix SCR

  # With Gemma fine-tuned model
  python batch_extract.py --input-dir ./articles --backend huggingface \
      --model sannuta/newsatom-gemma-2b --gemma-format --org-prefix SCR

CSV format (columns, header required):
  file          — path to .txt file (use 'url' column for URLs instead)
  url           — article URL
  title         — article headline (optional)
  date          — publication date YYYY-MM-DD (optional)
  journalist    — byline (optional)
  org           — publisher name (optional)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from newsatom.fetcher import fetch_url, read_file
from newsatom.splitter import split_sentences
from newsatom.prompt import build_prompt
from newsatom.parser import parse_output
from newsatom.wrapper import stamp_records, make_prefix
from newsatom.counter import get_next_counters
from newsatom.backends import get_backend

SENTENCE_WARNING_THRESHOLD = 50


def process_article(
    text, url, title, date, journalist, org,
    backend, model_id, gemma_format,
    output_dir, org_prefix, events_path, atoms_path,
):
    sentences = split_sentences(text)
    n = len(sentences)

    if not sentences:
        print("     Skipped — no sentences found.")
        return 0, 0

    if n > SENTENCE_WARNING_THRESHOLD:
        print(f"     ⚠  {n} sentences — may exceed context limit on smaller models.")

    prompt = build_prompt(
        sentences=sentences, pub_date=date, url=url, title=title,
        journalist=journalist, org=org, gemma_format=gemma_format,
    )

    raw_output = backend.generate(prompt)
    events, atoms = parse_output(raw_output)

    if not events and not atoms:
        print("     Warning: model returned no output.")
        return 0, 0

    event_start, atom_start = get_next_counters(
        output_dir=output_dir, org_prefix=org_prefix,
        n_events=len(events), n_atoms=len(atoms),
    )

    origin = {
        "url": url, "title": title, "journalist": journalist,
        "organization": org, "created_at": date,
    }

    events, atoms = stamp_records(
        events=events, atoms=atoms, origin=origin, model_id=model_id,
        org_prefix=org_prefix, event_counter_start=event_start, atom_counter_start=atom_start,
    )

    with open(events_path, "a") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    with open(atoms_path, "a") as f:
        for atom in atoms:
            f.write(json.dumps(atom, ensure_ascii=False) + "\n")

    return len(events), len(atoms)


def load_articles_from_dir(input_dir):
    return [
        {"file": str(p), "title": p.stem.replace("_", " ").replace("-", " "),
         "date": "", "journalist": "", "org": ""}
        for p in sorted(Path(input_dir).glob("*.txt"))
    ]


def load_articles_from_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {"file": row.get("file", ""), "url": row.get("url", ""),
             "title": row.get("title", ""), "date": row.get("date", ""),
             "journalist": row.get("journalist", ""), "org": row.get("org", "")}
            for row in csv.DictReader(f)
        ]


def main():
    parser = argparse.ArgumentParser(description="News Atom Lite — batch extractor.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Directory of .txt article files")
    source.add_argument("--csv", help="CSV file with article metadata")

    parser.add_argument("--backend", required=True,
                        choices=["openai", "anthropic", "huggingface", "ollama", "local"])
    parser.add_argument("--model")
    parser.add_argument("--model-path")
    parser.add_argument("--gemma-format", action="store_true")
    parser.add_argument("--org", default="",
                        help="Publisher name — used to derive ID prefix if --org-prefix not set")
    parser.add_argument("--org-prefix", default="",
                        help="3-letter ID prefix e.g. SCR → SCR-EVT0001")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files at start of batch run.")

    args = parser.parse_args()

    if args.backend == "local" and not args.model_path:
        parser.error("--model-path is required for --backend local")

    articles = load_articles_from_dir(args.input_dir) if args.input_dir else load_articles_from_csv(args.csv)

    if not articles:
        print("No articles found.", file=sys.stderr)
        sys.exit(1)

    org_prefix = args.org_prefix if args.org_prefix else make_prefix(args.org or "UNK")

    if not args.org and not args.org_prefix:
        print("⚠  Warning: no --org or --org-prefix provided. IDs will use prefix 'UNK'.\n")

    print(f"News Atom Lite — batch mode")
    print(f"  Articles: {len(articles)}")
    print(f"  Backend:  {args.backend}")
    print(f"  Prefix:   {org_prefix}")
    print(f"  Output:   {args.output_dir}\n")

    backend = get_backend(name=args.backend, model=args.model, model_path=args.model_path)
    model_id = args.model or args.model_path or args.backend

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    events_path = Path(args.output_dir) / "events.jsonl"
    atoms_path = Path(args.output_dir) / "atoms.jsonl"

    if args.overwrite:
        events_path.write_text("")
        atoms_path.write_text("")

    total_events = total_atoms = failed = 0

    for i, article in enumerate(articles, 1):
        source_label = article.get("file") or article.get("url", "")
        print(f"[{i}/{len(articles)}] {Path(source_label).name if article.get('file') else source_label[:60]}")

        try:
            if article.get("url"):
                text, meta = fetch_url(article["url"])
                title = article.get("title") or meta.get("title", "")
                date = article.get("date") or meta.get("date", "")
                journalist = article.get("journalist") or meta.get("journalist", "")
                org = article.get("org") or args.org or meta.get("org", "")
                url = article["url"]
            else:
                text = read_file(article["file"])
                title, date, journalist = article.get("title", ""), article.get("date", ""), article.get("journalist", "")
                org = article.get("org") or args.org
                url = article["file"]

            n_events, n_atoms = process_article(
                text=text, url=url, title=title, date=date, journalist=journalist, org=org,
                backend=backend, model_id=model_id, gemma_format=args.gemma_format,
                output_dir=args.output_dir, org_prefix=org_prefix,
                events_path=events_path, atoms_path=atoms_path,
            )
            print(f"     {n_events} events, {n_atoms} atoms")
            total_events += n_events
            total_atoms += n_atoms

        except Exception as e:
            print(f"     ✗ Failed: {e}")
            failed += 1

        if i < len(articles) and args.delay > 0:
            time.sleep(args.delay)

    print(f"\n✓ Batch complete.")
    print(f"  Processed: {len(articles) - failed}/{len(articles)}")
    print(f"  Total events: {total_events}")
    print(f"  Total atoms:  {total_atoms}")
    if failed:
        print(f"  Failed:       {failed}")
    print(f"  Events → {events_path}")
    print(f"  Atoms  → {atoms_path}")


if __name__ == "__main__":
    main()
