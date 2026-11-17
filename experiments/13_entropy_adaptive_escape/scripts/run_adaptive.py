"""Experiment 13 — Entropy-adaptive head escape.

E12 (head-conformity floor) showed two things at once:

  * the model puts 92-99 % of its mass on ten tokens at safe temperatures, so a
    sampler cannot reach the human 0.797 GLTR profile;
  * the only way to reach it is T >= 1.3, where perplexity explodes to 5 700+.

E12's proposed fix was to apply the heat *selectively*: copying is decided at
confident steps, so raise temperature only when the model is confident and keep
it low everywhere else. This experiment tests that directly, plus the inverted
control (heat the uncertain steps) to show the direction matters.

Conditions
  baseline        T=0.7 + top-p 0.9
  fixed_hot       T=1.3 + top-p 0.9             (E12's soup point, reference)
  adapt_c0.50/0.60/0.70   T=1.3 if top1 > c, else T=0.7
  adapt_inverse   T=1.3 if top1 <= 0.60, else T=0.7   (control)

Run: uv run python experiments/13_entropy_adaptive_escape/scripts/run_adaptive.py
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
from humanize_lab.processors import AdaptiveTemperature, Temperature, TopP
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "13_entropy_adaptive_escape" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
MAX_NEW = 100


def adapt(base: float, high: float, conf: float, invert: bool = False):
    def make():
        return [AdaptiveTemperature(base=base, high=high, conf=conf, invert=invert), TopP(0.9)]

    return make


CONDITIONS = {
    "baseline_T0.7": {"factory": lambda: [Temperature(0.7), TopP(0.9)]},
    "fixed_hot_T1.3": {"factory": lambda: [Temperature(1.3), TopP(0.9)]},
    "adapt_c0.30": {"factory": adapt(0.7, 1.3, 0.30)},
    "adapt_c0.40": {"factory": adapt(0.7, 1.3, 0.40)},
    "adapt_c0.50": {"factory": adapt(0.7, 1.3, 0.50)},
    "adapt_c0.60": {"factory": adapt(0.7, 1.3, 0.60)},
    "adapt_c0.70": {"factory": adapt(0.7, 1.3, 0.70)},
    "adapt_c0.50_h1.15": {"factory": adapt(0.7, 1.15, 0.50)},
    "adapt_c0.60_h1.15": {"factory": adapt(0.7, 1.15, 0.60)},
    "adapt_c0.70_h1.15": {"factory": adapt(0.7, 1.15, 0.70)},
    "adapt_inverse_c0.60": {"factory": adapt(0.7, 1.3, 0.60, invert=True)},
}

if __name__ == "__main__":
    print(env_fingerprint())
    docs = load_docs()
    t0 = time.time()
    rows, texts = eval_conditions(
        CONDITIONS, docs=docs, seeds=SEEDS, max_new_tokens=MAX_NEW,
        human_dist=human_reference_dist(),
    )
    print(f"total {time.time()-t0:.1f}s")

    pd.DataFrame(rows).to_csv(OUT / "adaptive_raw.csv", index=False)
    sdf = pd.DataFrame(summarize_conditions(rows))
    sdf.to_csv(OUT / "adaptive_summary.csv", index=False)

    cols = ["condition", "n_words", "gltr_top10", "copy_4gram", "entity_recall", "distinct_2",
            "rep_4", "burstiness", "boilerplate", "ppl"]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "adaptive_summary.md").write_text(
        "# Entropy-adaptive head escape — summary\n\nHuman GLTR top-10 baseline: 0.797\n\n" + table + "\n"
    )
    print("\n" + table)

    save_json(texts, OUT / "samples.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["gltr_top10", "copy_4gram", "ppl"]):
        ax.bar(sdf["condition"], sdf[m], color="#F28E2B")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(0.797, color="#E45756", ls=":", label="human")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(OUT / "adaptive.png", dpi=130)
    print(f"saved -> {OUT}")
