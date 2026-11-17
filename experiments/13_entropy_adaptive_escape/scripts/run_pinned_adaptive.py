"""Experiment 13b — Entity pinning x adaptive heat (E11 x E13 combination).

E11 (entity pinning): copy anchors greedily -> recall recovers, head stays at 0.99.
E13 (adaptive heat):   heat only confident steps -> reaches the human GLTR
                       profile (0.849 @ PPL 35), but recall collapses to 0.08.

Neither is sufficient alone, and they act on disjoint token classes, so the
combination is the natural test: pin the facts, heat everything else.

Conditions
  baseline                  T=0.7 + top-p 0.9
  adapt_c0.50               E13 alone
  pin_only                  E11 alone (greedy anchors, base temperature elsewhere)
  pin+adapt_c0.50/0.60/0.70 combined
  pin+adapt_c0.70_h1.15     combined, gentler heat

Run: uv run python experiments/13_entropy_adaptive_escape/scripts/run_pinned_adaptive.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from humanize_lab.corpus import human_reference_dist, load_docs
from humanize_lab.evaluate import eval_conditions, summarize_conditions
from humanize_lab.models import load
from humanize_lab.processors import AdaptiveTemperature, Temperature, TopP
from humanize_lab.routers import PinnedAdaptive, fact_token_ids
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "13_entropy_adaptive_escape" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
MAX_NEW = 100

tok, _ = load()
FACTS = fact_token_ids(tok)


def pinned(base=0.7, high=1.3, conf=0.6, pin=True):
    def make():
        return [PinnedAdaptive(FACTS, base=base, high=high, conf=conf, pin=pin), TopP(0.9)]

    return make


def adapted(base=0.7, high=1.3, conf=0.6):
    def make():
        return [AdaptiveTemperature(base=base, high=high, conf=conf), TopP(0.9)]

    return make


CONDITIONS = {
    "baseline_T0.7": {"factory": lambda: [Temperature(0.7), TopP(0.9)]},
    "adapt_c0.50": {"factory": adapted(0.7, 1.3, 0.50)},
    "pin_only": {"factory": pinned(base=0.7, high=0.7, conf=0.0, pin=True)},
    "pin+adapt_c0.50": {"factory": pinned(0.7, 1.3, 0.50)},
    "pin+adapt_c0.60": {"factory": pinned(0.7, 1.3, 0.60)},
    "pin+adapt_c0.70": {"factory": pinned(0.7, 1.3, 0.70)},
    "pin+adapt_c0.70_h1.15": {"factory": pinned(0.7, 1.15, 0.70)},
}

if __name__ == "__main__":
    print(env_fingerprint())
    print(f"fact-token ids: {len(FACTS)} / {len(tok)} ({100*len(FACTS)/len(tok):.1f}%)")
    docs = load_docs()
    t0 = time.time()
    rows, texts = eval_conditions(
        CONDITIONS, docs=docs, seeds=SEEDS, max_new_tokens=MAX_NEW,
        human_dist=human_reference_dist(),
    )
    print(f"total {time.time()-t0:.1f}s")

    pd.DataFrame(rows).to_csv(OUT / "pinned_adaptive_raw.csv", index=False)
    sdf = pd.DataFrame(summarize_conditions(rows))
    sdf["recall_per_copy"] = sdf["entity_recall"] / sdf["copy_4gram"].clip(lower=1e-3)
    sdf.to_csv(OUT / "pinned_adaptive_summary.csv", index=False)

    cols = ["condition", "n_words", "gltr_top10", "copy_4gram", "entity_recall", "recall_per_copy",
            "distinct_2", "burstiness", "boilerplate", "ppl"]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "pinned_adaptive_summary.md").write_text(
        "# Entity pinning x adaptive heat — summary\n\nHuman GLTR top-10 baseline: 0.797\n\n" + table + "\n"
    )
    print("\n" + table)

    save_json(texts, OUT / "pinned_adaptive_samples.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in sdf.iterrows():
        ax.scatter(r["ppl"], r["gltr_top10"], s=70)
        ax.annotate(r["condition"], (r["ppl"], r["gltr_top10"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.797, color="#E45756", ls=":", label="human GLTR 0.797")
    ax.set_xscale("log")
    ax.set_xlabel("perplexity (base scorer, log)")
    ax.set_ylabel("GLTR top-10")
    ax.set_title("Entity pinning x adaptive heat")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pinned_adaptive.png", dpi=130)
    print(f"saved -> {OUT}")
