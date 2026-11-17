"""The workhorse: run a matrix of decoding conditions over the benchmark corpus.

Every experiment in this repo reduces to "load a model, run N conditions x S seeds
over the 12-document corpus, score with metrics.summarize + perplexity". This
module implements exactly that so experiments stay short and comparable.
"""

from __future__ import annotations

import time
from collections import Counter

import torch

from . import metrics
from .corpus import human_reference_dist, load_docs
from .models import BASE_MODEL, DEFAULT_MODEL, get_tokenizer, load, render_chat
from .sampling import generate


def build_prompt(tokenizer, doc: dict, mode: str = "neutral", system: str | None = None) -> str:
    from .corpus import SYSTEM_HUMANIZE, SYSTEM_NEUTRAL, USER_TEMPLATES

    if system is None:
        system = SYSTEM_HUMANIZE if mode == "humanize" else SYSTEM_NEUTRAL
    return render_chat(tokenizer, system, USER_TEMPLATES[mode].format(text=doc["text"]))


def eval_conditions(
    conditions: dict,
    *,
    docs: list[dict] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    max_new_tokens: int = 100,
    mode: str = "neutral",
    system: str | None = None,
    model_id: str = DEFAULT_MODEL,
    model=None,
    tokenizer=None,
    scorer_model=None,
    scorer_tokenizer=None,
    score_ppl: bool = True,
    human_dist: Counter | None = None,
    verbose: bool = True,
    capture_texts: bool = True,
) -> tuple[list[dict], dict]:
    """Run every condition and return (metric rows, {condition: [texts]}).

    ``conditions[name]`` may be:
      * a list of processors  -> a fresh list is used per seed
      * a dict with keys ``factory`` (callable -> processors), ``mask_fn``,
        ``position_fn``
    """
    docs = docs or load_docs()
    if tokenizer is None or model is None:
        tokenizer, model = load(model_id)
    human_dist = human_dist or human_reference_dist()
    device = model.device

    if score_ppl and scorer_model is None:
        scorer_tokenizer, scorer_model = load(BASE_MODEL)

    prompts = [build_prompt(tokenizer, d, mode, system) for d in docs]
    rows: list[dict] = []
    texts_by_cond: dict[str, list[str]] = {}

    for name, cond in conditions.items():
        for seed in seeds:
            if isinstance(cond, dict):
                procs = cond.get("factory", lambda: [])()
                mask_fn = cond.get("mask_fn")
                pos_fn = cond.get("position_fn")
                extra = cond.get("extra", {})
                state = cond.get("state")
            else:
                procs, mask_fn, pos_fn, extra, state = cond, None, None, {}, None
            if procs is not None and callable(procs) and not isinstance(procs, list):
                procs = procs()

            t0 = time.time()
            out = generate(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                processors=procs,
                mask_fn=mask_fn,
                position_fn=pos_fn,
                seed=seed,
                capture_trace=True,
                **extra,
            )
            dt = time.time() - t0
            ppls = metrics.perplexity(scorer_model, scorer_tokenizer, out.texts) if score_ppl else [float("nan")] * len(out.texts)
            gltr = metrics.gltr_buckets(out.trace)

            for d, text, ppl in zip(docs, out.texts, ppls):
                m = metrics.summarize(text, d["text"], d["entities"], human_dist)
                row = {
                    "condition": name,
                    "seed": seed,
                    "doc_id": d["id"],
                    "domain": d["domain"],
                    **m,
                    "ppl": ppl,
                    **gltr,
                    "gen_seconds": dt / len(docs),
                }
                if state is not None:
                    row.update(state() if callable(state) else state)
                rows.append(row)
            texts_by_cond.setdefault(name, []).extend(out.texts)
        if verbose:
            sub = [r for r in rows if r["condition"] == name]
            print(
                f"  {name:28s} words={_mean(sub,'n_words'):5.1f} copy={_mean(sub,'copy_4gram'):.2f} "
                f"ent={_mean(sub,'entity_recall'):.2f} d2={_mean(sub,'distinct_2'):.2f} "
                f"burst={_mean(sub,'burstiness'):.2f} boil={_mean(sub,'boilerplate'):.1f} "
                f"ppl={_mean(sub,'ppl'):6.1f}",
                flush=True,
            )
    return rows, texts_by_cond


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


SUMMARY_KEYS = [
    "n_words",
    "distinct_1",
    "distinct_2",
    "rep_4",
    "rep_5",
    "copy_4gram",
    "entity_recall",
    "boilerplate",
    "hedge",
    "burstiness",
    "sent_len_mean",
    "style_jsd",
    "compression",
    "ppl",
    "gltr_top10",
    "gltr_tail",
]


def summarize_conditions(rows: list[dict], keys: list[str] | None = None) -> list[dict]:
    from .tracking import group_mean

    keys = keys or [k for k in SUMMARY_KEYS if any(k in r for r in rows)]
    return group_mean(rows, "condition", keys)
