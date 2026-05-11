"""
Parses raw model output into lists of event dicts and atom dicts.
"""

import json
import re


def parse_output(raw_output: str) -> tuple[list[dict], list[dict]]:
    """Parse raw model output into (events, atoms)."""
    events_section, atoms_section = _split_sections(raw_output)
    events = _parse_records(events_section, "event")
    atoms = _parse_records(atoms_section, "atom")
    return events, atoms


def _split_sections(text: str) -> tuple[str, str]:
    events_match = re.search(r"---\s*EVENTS\s*---", text, re.IGNORECASE)
    atoms_match = re.search(r"---\s*ATOMS\s*---", text, re.IGNORECASE)

    if events_match and atoms_match:
        return text[events_match.end():atoms_match.start()], text[atoms_match.end():]
    elif atoms_match:
        return "", text[atoms_match.end():]
    elif events_match:
        return text[events_match.end():], ""
    else:
        return "", text


def _parse_records(text: str, expected_type: str) -> list[dict]:
    records = []
    for json_str in _extract_json_blocks(text):
        try:
            record = json.loads(json_str)
            if isinstance(record, dict):
                records.append(record)
        except json.JSONDecodeError as e:
            fixed = _attempt_fix(json_str)
            if fixed:
                records.append(fixed)
            else:
                print(f"   Warning: skipped malformed {expected_type}: {e}")
    return records


def _extract_json_blocks(text: str) -> list[str]:
    blocks = []
    depth = 0
    start = None
    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start:i+1])
                start = None
    return blocks


def _attempt_fix(json_str: str) -> dict | None:
    fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None
