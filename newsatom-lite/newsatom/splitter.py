"""
Splits article body text into clean sentences using pysbd.
"""

import re

SKIP_PATTERNS = [
    re.compile(r"^also read", re.IGNORECASE),
    re.compile(r"^follow us on", re.IGNORECASE),
    re.compile(r"^subscribe", re.IGNORECASE),
    re.compile(r"^get the best of", re.IGNORECASE),
    re.compile(r"^before you scroll", re.IGNORECASE),
    re.compile(r"^read more", re.IGNORECASE),
    re.compile(r"^share$", re.IGNORECASE),
    re.compile(r"@[A-Za-z0-9_]{2,}"),
    re.compile(r"^https?://\S+$"),
]

MIN_WORD_COUNT = 5


def split_sentences(text: str) -> list[str]:
    """Split article text into sentences using pysbd. Requires: pip install pysbd"""
    try:
        import pysbd
    except ImportError:
        raise ImportError("Sentence splitting requires: pip install pysbd")

    segmenter = pysbd.Segmenter(language="en", clean=True)
    raw_sentences = segmenter.segment(text)

    clean = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < MIN_WORD_COUNT:
            continue
        if any(p.search(sentence) for p in SKIP_PATTERNS):
            continue
        if sentence.isupper() and len(sentence) < 60:
            continue
        clean.append(sentence)

    return clean
