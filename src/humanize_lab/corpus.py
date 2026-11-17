"""Shared corpora and prompt templates.

The 12-document, entity-annotated corpus from Experiment 03 is the canonical
benchmark for every downstream experiment: same sources, same entity lists,
so numbers are directly comparable across experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import function_word_distribution

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "experiments" / "03_sweet_spot_tuning_and_comparison" / "data" / "dataset.json"

SYSTEM_NEUTRAL = (
    "You are an editor. Rewrite the user's text in your own words. "
    "Keep every name, date and number. Output only the rewritten text."
)
SYSTEM_HUMANIZE = (
    "You are a human writer. Rewrite the text so it reads like a person wrote it: "
    "vary sentence length, use plain wording. Keep facts. Output only the rewrite."
)
SYSTEM_SUMMARY = (
    "You are an editor. Summarize the user's text in one short paragraph. "
    "Keep the key names, dates and numbers. Output only the summary."
)

USER_TEMPLATES = {
    "neutral": "Rewrite this text in your own words:\n\n{text}\n",
    "humanize": "Rewrite this text so it sounds human, not machine-written:\n\n{text}\n",
    "summarize": "Summarize this text:\n\n{text}\n",
    "continue": "{text}\n",
}


def load_docs() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def load_source(doc_id: str) -> dict:
    for d in load_docs():
        if d["id"] == doc_id:
            return d
    raise KeyError(doc_id)


def human_reference_dist() -> dict:
    """Function-word distribution of the human-written benchmark sources."""
    return function_word_distribution([d["text"] for d in load_docs()])


def build_prompts(docs: list[dict], mode: str = "neutral", system: str | None = None) -> tuple[list[str], list[str]]:
    """Return (user_prompts, system_prompt). Chat rendering happens later so
    each experiment can pick its own template."""
    tmpl = USER_TEMPLATES[mode]
    sysmsg = system or (SYSTEM_HUMANIZE if mode == "humanize" else SYSTEM_NEUTRAL)
    return [tmpl.format(text=d["text"]) for d in docs], sysmsg
