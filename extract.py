#!/usr/bin/env python3
"""
News Atom Lite — Extractor
--------------------------
Extract structured Events and Atoms from a news article.
Works with any language model — OpenAI, Anthropic, HuggingFace,
Ollama, or any local model via transformers.

Usage:
  python extract.py --url https://... --backend openai
  python extract.py --file article.txt --backend anthropic
  python extract.py --file article.txt --backend ollama --model llama3.2
  python extract.py --file article.txt --backend huggingface --model sannuta/newsatom-gemma-2b --gemma-format
  python extract.py --file article.txt --backend local --model-path ./my-model --gemma-format
"""

import argparse
import json
import sys
from pathlib import Path

from newsatom.fetcher import fetch_url, read_file
from newsatom.splitter import split_sentences
from newsatom.prompt import build_prompt
from newsatom.parser import parse_output
from newsatom.wrapper import stamp_records
from newsatom.counter import get_next_counters
from newsatom.backends import get_backend

SENTENCE_WARNING_THRESHOLD = 200


def main():
    parser = argparse.ArgumentParser(
        description="News Atom Lite — extract Events and Atoms from a news article.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
backends:
  openai        OpenAI API (requires OPENAI_API_KEY env var)
  anthropic     Anthropic API (requires ANTHROPIC_API_KEY env var)
  huggingface   HuggingFace Inference API (requires HF_TOKEN env var)
  ollama        Ollama local server (no API key needed)
  local         Local model via transformers (requires --model-path)

examples:
  python extract.py --url https://example.com/article --backend openai
  python extract.py --file article.txt --backend anthropic
  python extract.py --file article.txt --backend ollama --model llama3.2
  python extract.py --file article.txt --backend huggingface --model sannuta/newsatom-gemma-2b --gemma-format
        """
    )

    # Input
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="URL of the article to fetch")
    source.add_argument("--file", help="Path to a plain text file containing the article body")

    # Backend
    parser.add_argument(
        "--backend", required=True,
        choices=["openai", "anthropic", "huggingface", "ollama", "local"],
        help="Which model backend to use"
    )
    parser.add_argument("--model", help="Model name or ID (backend-specific)")
    parser.add_argument("--model-path", help="Path to local model directory (--backend local only)")
    parser.add_argument(
        "--gemma-format", action="store_true",
        help="Use Gemma 2 IT chat format (<start_of_turn> markers). "
             "Required for fine-tuned models trained on Gemma 2 IT format."
    )

    # Article metadata
    parser.add_argument("--title", default="", help="Article headline")
    parser.add_argument("--date", default="", help="Publication date (YYYY-MM-DD)")
    parser.add_argument("--journalist", default="", help="Byline")
    parser.add_argument("--org", default="", help="Publisher name")

    # Output
    parser.add_argument("--output-dir", default=".", help="Output directory (default: current)")
    parser.add_argument("--prefix", default="", help="Filename prefix e.g. 'adani_'")
    parser.add_argument("--org-prefix", default="",
                        help="3-letter ID prefix e.g. 'SCR' → SCR-EVT0001. "
                             "Auto-derived from --org if not set.")

    args = parser.parse_args()

    if args.backend == "local" and not args.model_path:
        parser.error("--model-path is required when using --backend local")

    # ── Step 1: Get article text ───────────────────────────────────────────────
    print("→ Loading article...")
    if args.url:
        text, meta = fetch_url(args.url)
        url = args.url
        title = args.title or meta.get("title", "")
        date = args.date or meta.get("date", "")
        journalist = args.journalist or meta.get("journalist", "")
        org = args.org or meta.get("org", "")
    else:
        text = read_file(args.file)
        url = args.file
        title, date, journalist, org = args.title, args.date, args.journalist, args.org

    if not text.strip():
        print("Error: article text is empty.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Split sentences ────────────────────────────────────────────────
    print("→ Splitting sentences...")
    sentences = split_sentences(text)
    n = len(sentences)
    print(f"   {n} sentences")

    if not sentences:
        print("Error: no sentences found.", file=sys.stderr)
        sys.exit(1)

    if n > SENTENCE_WARNING_THRESHOLD:
        print(
            f"\n⚠  Warning: {n} sentences exceeds the recommended limit of "
            f"{SENTENCE_WARNING_THRESHOLD}. Smaller models may truncate this "
            f"article. Consider splitting it before extraction.\n"
        )

    # ── Step 3: Build prompt ──────────────────────────────────────────────────
    prompt = build_prompt(
        sentences=sentences,
        pub_date=date,
        url=url,
        title=title,
        journalist=journalist,
        org=org,
        gemma_format=args.gemma_format,
    )

    # ── Step 4: Call model ────────────────────────────────────────────────────
    print(f"→ Calling {args.backend} backend...")
    backend = get_backend(
        name=args.backend,
        model=args.model,
        model_path=args.model_path,
    )
    raw_output = backend.generate(prompt)

    # ── Step 5: Parse output ──────────────────────────────────────────────────
    print("→ Parsing output...")
    events, atoms = parse_output(raw_output)
    print(f"   {len(events)} events, {len(atoms)} atoms")

    # ── Step 6: Reserve IDs and stamp wrapper fields ───────────────────────────
    print("→ Stamping wrapper fields...")
    output_dir = args.output_dir
    org_prefix = args.org_prefix

    event_start, atom_start = get_next_counters(
        output_dir=output_dir,
        org_prefix=org_prefix,
        n_events=len(events),
        n_atoms=len(atoms),
    )

    model_id = args.model or args.model_path or args.backend
    origin = {
        "url": url,
        "title": title,
        "journalist": journalist,
        "organization": org,
        "created_at": date,
    }

    events, atoms = stamp_records(
        events=events,
        atoms=atoms,
        origin=origin,
        model_id=model_id,
        org_prefix=org_prefix,
        event_counter_start=event_start,
        atom_counter_start=atom_start,
    )

    # ── Step 7: Write output ──────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    events_path = Path(output_dir) / f"{args.prefix}events.jsonl"
    atoms_path = Path(output_dir) / f"{args.prefix}atoms.jsonl"

    with open(events_path, "a") as f:  # append — supports batch runs
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    with open(atoms_path, "a") as f:
        for atom in atoms:
            f.write(json.dumps(atom, ensure_ascii=False) + "\n")

    print(f"\n✓ Done.")
    print(f"  Events → {events_path}")
    print(f"  Atoms  → {atoms_path}")


if __name__ == "__main__":
    main()
