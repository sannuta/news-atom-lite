"""
Builds the extraction prompt.

For general-purpose models (OpenAI, Anthropic, Ollama):
  The prompt is sent as a user message, with the system instruction
  inline or as a system prompt via the backend.

For fine-tuned models trained on Gemma 2 IT format:
  The prompt uses <start_of_turn> markers.
  Use build_prompt(gemma_format=True) for these models.
"""

from datetime import datetime


INSTRUCTION = """Extract all Events and Atoms from the news article below.

An EVENT is a discrete real-world happening:
  type, event_label, date, agent, participants, action, object,
  location (optional), related_event (optional), relation_phrase (optional)

An ATOM is a sentence-level record of how an event was reported:
  type, event_label, information_type, direct_quote, source (optional),
  subject, predicate, object, date (optional), location (optional),
  original_text, entities, language

Rules:
- One atom per sentence. Terminal punctuation = sentence boundary.
- event_label format: "YYYY-MM-DD [AGENT] [ACTION] [OBJECT] [@LOCATION]"
- information_type: "observed_fact" (journalist as witness) or "sensemaking" (journalist as explainer)
- direct_quote: true only if typographic double quote marks are present in the sentence
- agent: the causal agent, NOT the grammatical subject of passive constructions
- original_text: exact verbatim sentence, never modified

Output format — produce two sections:

--- EVENTS ---
One standalone JSON object per event, separated by a blank line.

--- ATOMS ---
One standalone JSON object per atom, separated by a blank line.

No array brackets. No preamble. No explanation."""


def build_prompt(
    sentences: list[str],
    pub_date: str = "",
    url: str = "",
    title: str = "",
    journalist: str = "",
    org: str = "",
    gemma_format: bool = False,
) -> str:
    """
    Build the extraction prompt.

    Args:
        sentences:    List of article sentences (already split and cleaned).
        pub_date:     Publication date (YYYY-MM-DD).
        url:          Article URL.
        title:        Article headline.
        journalist:   Byline.
        org:          Publisher name.
        gemma_format: If True, wraps in Gemma 2 IT chat tokens.
                      Use for fine-tuned models trained on Gemma 2 IT format.

    Returns:
        Full prompt string.
    """
    n = len(sentences)

    day_of_week = ""
    if pub_date:
        try:
            dt = datetime.strptime(pub_date[:10], "%Y-%m-%d")
            day_of_week = dt.strftime("%A")
        except ValueError:
            pass

    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    meta_lines = []
    if pub_date:
        meta_lines.append(f"Publication date: {pub_date}" + (f" ({day_of_week})" if day_of_week else ""))
    if url:
        meta_lines.append(f"URL: {url}")
    if title:
        meta_lines.append(f"Title: {title}")
    if org:
        meta_lines.append(f"Organization: {org}")
    if journalist:
        meta_lines.append(f"Journalist: {journalist}")

    meta_block = "\n".join(meta_lines)
    article_block = (
        f"{meta_block}\n\n"
        f"Article sentences — {n} sentences, produce exactly {n} atoms:\n\n"
        f"{numbered}"
    )

    if gemma_format:
        return (
            f"<start_of_turn>user\n"
            f"{INSTRUCTION}\n\n"
            f"{article_block}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    else:
        return f"{INSTRUCTION}\n\n{article_block}"
