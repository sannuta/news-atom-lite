"""
Persistent ID counter for News Atom Lite.

Stores the last-used event and atom counter per org prefix in a
JSON file alongside the output, so IDs are globally unique across
multiple extraction runs.

Counter file: {output_dir}/.newsatom_counters.json
"""

import json
from pathlib import Path

COUNTER_FILENAME = ".newsatom_counters.json"


def load_counters(output_dir: str) -> dict:
    path = Path(output_dir) / COUNTER_FILENAME
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_counters(output_dir: str, counters: dict) -> None:
    path = Path(output_dir) / COUNTER_FILENAME
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(counters, f, indent=2)


def get_next_counters(
    output_dir: str,
    org_prefix: str,
    n_events: int,
    n_atoms: int,
) -> tuple[int, int]:
    """Reserve a block of IDs. Returns (event_start, atom_start)."""
    counters = load_counters(output_dir)
    prefix_key = org_prefix.upper()
    prefix_counters = counters.get(prefix_key, {"last_event": 0, "last_atom": 0})

    event_start = prefix_counters["last_event"] + 1
    atom_start = prefix_counters["last_atom"] + 1

    prefix_counters["last_event"] += n_events
    prefix_counters["last_atom"] += n_atoms
    counters[prefix_key] = prefix_counters

    save_counters(output_dir, counters)
    return event_start, atom_start


def reset_counters(output_dir: str, org_prefix: str = None) -> None:
    """Reset counters. Pass org_prefix to reset only that prefix."""
    if org_prefix:
        counters = load_counters(output_dir)
        counters.pop(org_prefix.upper(), None)
        save_counters(output_dir, counters)
    else:
        path = Path(output_dir) / COUNTER_FILENAME
        if path.exists():
            path.unlink()
