"""
Builds the extraction prompt for News Atom Lite.

For general-purpose models (OpenAI, Anthropic, Ollama):
  The prompt is sent as a user message.

For fine-tuned models trained on Gemma 2 IT format:
  Use build_prompt(gemma_format=True) to wrap in <start_of_turn> markers.
"""

from datetime import datetime


INSTRUCTION = """You are generating structured extractions from news articles.
Your job is to parse the article below with absolute precision.
Accuracy is not optional. Every field must be correct.
Do not guess. Do not infer. Do not approximate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFINITIONS:

An EVENT is a real-world happening that exists independently of how any article
describes it. The same event occurred once, on a specific date, with specific
actors, regardless of how many articles report it. Events are described using
structured fields: agent, participants, action, object, location, date.

An ATOM is a sentence-level record of how a specific event was reported.
Where the event captures what happened, the atom captures how journalism
documented it — the exact words used, who was quoted, what was attributed to
whom, and whether the sentence reports the event as a fact or interprets its
meaning. Every atom is anchored to one or more events via event_label.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION ORDER:

Step 1 — Read the full article. Identify all discrete events.
Step 2 — For each sentence, determine which event(s) it touches.
Step 3 — Output all events first, then all atoms.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SKIP:
"Also read" sections, navigation elements, editor's notes, related article
links, embedded tweets, embedded social media posts, embedded videos, and
any text that is not original journalistic content.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FORMAT:

Produce two outputs, clearly marked:

--- EVENTS ---
One standalone JSON object per event, separated by a blank line.

--- ATOMS ---
One standalone JSON object per atom, separated by a blank line.

No array brackets. No preamble. No postamble. No explanation.
The "type" field on every object identifies whether it is an event or an atom.
Do not generate atom_id or event_id values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVENT RULES:

RULE E1 — ONE EVENT PER DISTINCT HAPPENING:
Every distinct action by a distinct actor on a distinct object is a separate
event. Extract as many events as the article contains. When in doubt, extract
more events rather than fewer.

Always separate:
- The primary incident (collapse, verdict, announcement, arrest)
- Each official statement or reaction to the incident
- Each government or authority response
- Each legal action (FIR, arrest, charge, verdict)
- Each visit by a public figure
- Historical background events (with their historical date)
- Natural events that caused or contributed to the primary incident

RULE E2 — AGENT IS THE CAUSAL AGENT, NOT THE GRAMMATICAL SUBJECT:
Category 1 — Causal agent explicit:
  "The wall collapsed during heavy rain" → agent: "heavy rain"
Category 2 — Agent stated after "by":
  "The bill was passed by Parliament" → agent: "Parliament"
Category 3 — Institutional action:
  "The police registered an FIR" → agent: "police"
Category 4 — Agent genuinely unstated:
  "Seven persons were killed" → agent: "unknown"

NEVER use the thing that was acted upon as the agent:
  "The wall collapsed" → agent is NOT "wall"
  "Seven persons died" → agent is NOT "seven persons"
  "The petition was dismissed" → agent is NOT "petition"

RULE E3 — ACTION IS THE ROOT VERB:
Extract the root verb from passive constructions.
"was renamed" → "renamed"
"have been killed" → "killed"
"has been filed" → "filed"

RULE E4 — EVENT_LABEL FORMAT:
Format: "YYYY-MM-DD [AGENT] [ACTION] [OBJECT] [@LOCATION]" — max 12 words.
When agent is unknown, use primary participant instead.
CORRECT: "2025-04-01 Dhami renamed 17 places @Uttarakhand"
CORRECT: "2026-04-29 wall collapsed @Bowring Hospital Bengaluru"
WRONG:   "2026-04-29 Wall collapse at Bowring and Lady Curzon Hospital kills seven"

RULE E5 — RELATED EVENTS (optional):
related_event  — the event_label of the referenced event, matching exactly
                 an event_label in this output.
relation_phrase — the exact connecting phrase verbatim from the text.
                 Examples: "following the incident", "in response to the
                 verdict", "amid the protests"

Both fields are omitted if no explicit connecting phrase exists.
NEVER infer a relationship. Never populate based on temporal proximity alone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ATOM RULES:

RULE A1 — ONE SENTENCE = ONE ATOM:
A sentence ends at its terminal punctuation mark (. ? !)
One terminal punctuation mark = one atom. Always. No exceptions.

Exception: a direct quote block with a single attribution is one atom
regardless of internal punctuation.
"X. Y. Z," Speaker said, qualifying context. → ONE atom.
"X." Speaker said. "Y." → TWO atoms.

RULE A2 — EVENT_LABEL links atom to event:
Must exactly match an event_label from the events output above.
Single event touched → string.
Multiple events touched → array of strings.

RULE A3 — INFORMATION_TYPE:
"observed_fact" — journalist as witness or conduit of a witness.
  The sentence reports an action, decision, statement, or event.
"sensemaking" — journalist as explainer.
  Definitions, processes, mechanisms, analysis, background, uncertainty.

Uncertainty sentences are ALWAYS sensemaking:
  "It was not immediately clear whether..." → sensemaking
  "It remains to be seen..." → sensemaking

RULE A4 — DIRECT_QUOTE:
true if and only if typographic double quotation marks (" ") are present
in the sentence. Deterministic. No judgment required.
"The minister said the bill was passed." → direct_quote: false
"The minister said \"the bill was passed.\"" → direct_quote: true

Indirect attribution WITHOUT quotation marks is NOT a direct quote:
"the newspaper quoted X as saying" → direct_quote: false
"X told reporters that Y happened" → direct_quote: false

RULE A5 — SOURCE:
Only populated when direct_quote is true.
The exact attribution phrase as it appears in the sentence.
Omit entirely if direct_quote is false.
Omit entirely if direct_quote is true but no attribution phrase exists.
NEVER use "article", "report", or "source" as the source value.

RULE A6 — SUBJECT: Grammatical subject of the sentence.

RULE A7 — PREDICATE: Main verb or verb phrase. 1–5 words maximum.
CORRECT: "collapsed" / "visited" / "was not clear"
WRONG: "was not immediately clear whether the structure was in a dilapidated condition"

RULE A8 — OBJECT: What the action is directed at. Concise.
CORRECT: "compound wall"
WRONG: "compound wall at the hospital that collapsed during heavy rain on Wednesday"

RULE A9 — DATE: Resolved ISO 8601. Omit if no temporal reference in sentence.

RULE A10 — LOCATION: Where action occurs. Stated in sentence. Omit otherwise.

RULE A11 — ORIGINAL_TEXT: Exact sentence verbatim. Never modified.

RULE A12 — ENTITIES: Array of {name, type}.
Types: person | organization | location | event | concept | quantity

Extract ALL of the following:
- Proper nouns: people, places, named organisations
- Institutional nouns when referring to specific bodies: "police", "court",
  "hospital", "government", "army", "ministry"
- Publications cited: "The Hindu", "Reuters" → organization
- Titles with names: "Chief Minister Siddaramaiah" → name: "Siddaramaiah", type: "person"
- Quantities central to the event: "seven persons", "$265 million"

RULE A13 — LANGUAGE: Always "en" for English-language articles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATE RESOLUTION RULES:

Use publication date as the anchor for all temporal references.

Unambiguous — resolve:
  Named days: "on Monday" in article published Tuesday 2025-04-01 → "2025-03-31"
  "yesterday" → publication date minus one day
  "earlier today" → publication date
  "last month" → previous calendar month
  Month references: "in January" → "2025-01"

Ambiguous — OMIT date field entirely:
  "last week", "recently", "earlier this month", "in recent weeks"

Historical: use the date of the actual occurrence.
  "in the 1980s" → "1980s" (never reduce a decade to a single year)

No temporal reference → omit date field entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONSERVATIVE DEFAULTS:

When in doubt:
- Uncertain about agent → use "unknown" rather than guessing
- Uncertain about event boundary → extract as separate event
- Uncertain about date → omit rather than guess
- Uncertain about entities → include rather than exclude
- Uncertain about information_type → observed_fact if something happened,
  sensemaking if explaining"""


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
        meta_lines.append(
            f"Publication date: {pub_date}"
            + (f" ({day_of_week})" if day_of_week else "")
        )
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
