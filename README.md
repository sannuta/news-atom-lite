# News Atom Lite

Extract structured **Events** and **Atoms** from news articles using any language model.

---
# News Atom Lite

> A structured knowledge extraction tool for journalism.

Journalism produces vast amounts of verified, attributed, timestamped knowledge — and then buries it in prose. **News Atom Lite** extracts that knowledge into structured records: discrete events and the sentence-level envelopes that document them.

This is a simplified implementation of the [News Atom](https://newsatom.xyz) schema — a metadata blueprint for building a machine-readable system of primary record for journalism. News Atom Lite formalises two schemas: **Events v2.0** and **News Atom Lite** (sentence-level knowledge units), and provides a model-agnostic CLI to extract them from any article using any language model.

The extraction rules are drawn from the [News Atom](https://newsatom.xyz) standard, developed by [Sannuta Raghu](https://newsatom.xyz), and shared here as part of a journalism commons — a foundation others can build on.

---

An EVENT is a real-world happening that exists independently of how any article describes it. The same event occurred once, on a specific date, with specific actors, regardless of how many articles report it. Events are described using structured fields: agent, participants, action, object, location, date.

An ATOM is a sentence-level record of how a specific event was reported. Where the event captures what happened, the atom captures how journalism documented it — the exact words used, who was quoted, what was attributed to whom, and whether the sentence reports the event as a fact or interprets its meaning. Every atom is anchored to one or more events via event_label.

One event accumulates atoms from many articles. This creates a machine-readable system of primary record for journalism.

---

## Installation

```bash
git clone https://github.com/sannuta/news-atom-lite
cd news-atom-lite
pip install -r requirements.txt
```

---

## Usage

### From a URL

```bash
python extract.py \
  --url https://example.com/article \
  --backend openai \
  --date 2025-04-22 \
  --title "Article headline" \
  --journalist "Reporter Name" \
  --org "Publication Name"
```

### From a plain text file

```bash
python extract.py \
  --file article.txt \
  --backend anthropic
```

### Specify output location

```bash
python extract.py \
  --file article.txt \
  --backend ollama \
  --model llama3.2 \
  --output-dir ./output \
  --prefix my_article_
```

This writes:
- `./output/my_article_events.jsonl`
- `./output/my_article_atoms.jsonl`

---

## Backends

Choose any backend with `--backend`. Each requires its own dependency and credentials.

| Backend | `--backend` | Default model | Requires |
|---------|-------------|---------------|---------|
| OpenAI | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` env var |
| Anthropic | `anthropic` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` env var |
| HuggingFace | `huggingface` | `mistralai/Mistral-7B-Instruct-v0.3` | `HF_TOKEN` env var |
| Ollama | `ollama` | `llama3.2` | Ollama running locally |
| Local | `local` | — | `--model-path` |

Override the default model with `--model`:

```bash
python extract.py --file article.txt --backend openai --model gpt-4o
python extract.py --file article.txt --backend anthropic --model claude-sonnet-4-20250514
python extract.py --file article.txt --backend ollama --model mistral
```

### Setting credentials

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace
export HF_TOKEN=hf_...
```

### Ollama

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
python extract.py --file article.txt --backend ollama --model llama3.2
```

### Local model

```bash
pip install torch transformers

python extract.py \
  --file article.txt \
  --backend local \
  --model-path ./path/to/model
```

---

## Output format

Two JSONL files per article.

### events.jsonl — one event per line

```json
{
  "type": "event",
  "event_label": "2024-11-21 Adani Group cancelled $600-million bond offering",
  "date": "2024-11-21",
  "agent": "Adani Group",
  "participants": ["Adani Group", "Adani Green Energy"],
  "action": "cancelled",
  "object": "$600-million bond offering",
  "related_event": "2024 US prosecutors indicted Gautam Adani for bribery",
  "relation_phrase": "in the wake of"
}
```

### atoms.jsonl — one atom per line

```json
{
  "type": "atom",
  "event_label": "2024-11-21 Adani Group cancelled $600-million bond offering",
  "information_type": "observed_fact",
  "direct_quote": false,
  "subject": "the Adani Group",
  "predicate": "cancelled",
  "object": "its bond offering of $600 million",
  "date": "2024-11-21",
  "original_text": "The Adani Group cancelled its bond offering of $600 million following the US indictment.",
  "entities": [
    {"name": "Adani Group", "type": "organization"},
    {"name": "$600 million", "type": "quantity"}
  ],
  "language": "en"
}
```

See `examples/` for a full worked example.

---

## Schema reference

See [schema.md](schema.md) for the complete field definitions, rules, and examples.

**Event fields:** `type`, `event_label`, `date`, `agent`, `participants`, `action`, `object`, `location`, `related_event`, `relation_phrase`

**Atom fields:** `type`, `event_label`, `information_type`, `direct_quote`, `source`, `subject`, `predicate`, `object`, `date`, `location`, `original_text`, `entities`, `language`

---

## Key concepts

**`information_type`**
`observed_fact` — the journalist is a witness or conduit of a witness.
`sensemaking` — the journalist is an explainer: definitions, background, analysis, uncertainty.

**`direct_quote`**
`true` if and only if typographic double quotation marks are present in the sentence. Deterministic — no judgment required.

**`event_label`**
`YYYY-MM-DD [AGENT] [ACTION] [OBJECT] [@LOCATION]` — max 12 words. The same event keeps the same label across all atoms that reference it.

**`agent` vs `subject`**
`agent` (event) = the causal agent. `subject` (atom) = the grammatical subject. They differ in passive constructions — and that difference is intentional.

---

## Adding a new backend

1. Create `newsatom/backends/mymodel_backend.py`
2. Subclass `ModelBackend` and implement `generate(prompt) -> str`
3. Register it in `newsatom/backends/__init__.py`

```python
from .base import ModelBackend

class MyModelBackend(ModelBackend):
    def generate(self, prompt: str) -> str:
        # call your model here
        return response_text
```

---

## Notes

- Pass the article **body only** — strip navigation, subscription prompts, photo captions, and section headers.  URL fetching does this automatically; `--file` input assumes clean text.
- Wrapper fields (`atom_id`, `event_id`, `origin`, `review_process`) are not generated by the model. Stamp them in your own pipeline.
- Very long articles (200+ sentences) may hit context limits on smaller models. Split them before extraction.
- Model quality varies. Larger, instruction-tuned models produce more consistent schema adherence. See [schema.md](schema.md) for the rules the model needs to follow.

---

## Licence

MIT

---

## Batch processing

Process a folder of articles or a CSV in one run. IDs are globally unique across the entire batch.

```bash
# Process a folder of .txt files
python batch_extract.py \
  --input-dir ./articles \
  --backend openai \
  --org "Scroll.in" \
  --org-prefix SCR \
  --output-dir ./output

# Process a CSV
python batch_extract.py \
  --csv articles.csv \
  --backend anthropic \
  --org-prefix SCR
```

CSV format (header required):

```
file,title,date,journalist,org
articles/adani.txt,Adani Group stocks fall,2024-11-21,Scroll Staff,Scroll.in
articles/waqf.txt,Congress challenges Waqf Bill,2025-04-04,Scroll Staff,Scroll.in
```

Use `--url` column instead of `--file` for URL-based fetching.

---

## ID persistence

IDs are globally unique across runs. A counter file is stored at `{output-dir}/.newsatom_counters.json` and updated after every run.

Run 1 on 5 articles → events `SCR-EVT0001` to `SCR-EVT0047`  
Run 2 on 3 more articles → events `SCR-EVT0048` onwards

The counter file is created automatically. Add it to `.gitignore` if you don't want to commit it.

---

## Using the fine-tuned newsatom model

When `sannuta/newsatom-gemma-2b` is available on Hugging Face, use the `--gemma-format` flag:

```bash
python extract.py \
  --file article.txt \
  --backend huggingface \
  --model sannuta/newsatom-gemma-2b \
  --gemma-format \
  --org "My Publication" \
  --org-prefix PUB
```

The `--gemma-format` flag switches the prompt to the Gemma 2 IT chat format (`<start_of_turn>` markers) that the fine-tuned model was trained on. Use it only with Gemma-based fine-tuned models — general-purpose models don't need it.
