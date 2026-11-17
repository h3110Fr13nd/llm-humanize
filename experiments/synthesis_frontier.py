"""Cross-experiment frontier plot: human-likeness vs the accuracy cost of buying it.

Every experiment in the repo measures the same 12-document corpus with the same
metrics, so their summary CSVs are directly comparable. This script reads them
and draws the two axes that actually trade against each other:

  x = GLTR top-10 conformity on the base model (human text = 0.797)
  y = entity recall / copy_4gram  (>1 means the model says more new facts than
                                   it copies phrases; the baseline is ~0.55)

Right panel: the same points against log10 PPL, to show *how much fluency* each
step along the axis costs. A point that reaches the human GLTR band without a
PPL explosion is the goal; E13's pinned-adaptive sampler is the only one that
does.

Run: uv run python experiments/synthesis_frontier.py
"""

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SERIES = {
    "E10 decoders": ("experiments/10_decoder_zoo_shootout/outputs/decoder_shootout_summary.csv", "condition", "o", "#4E79A7"),
    "E11 pinning": ("experiments/11_entity_pinned_head_escape/outputs/entity_pinned_summary.csv", "condition", "s", "#F28E2B"),
    "E12 temperature": ("experiments/12_head_conformity_floor/outputs/head_floor_summary.csv", "condition", "^", "#E15759"),
    "E13 adaptive": ("experiments/13_entropy_adaptive_escape/outputs/adaptive_summary.csv", "condition", "D", "#59A14F"),
    "E13b pinned+adaptive": ("experiments/13_entropy_adaptive_escape/outputs/pinned_adaptive_summary.csv", "condition", "P", "#B07AA1"),
    "E05 window attn": ("experiments/05_speculative_window_attention/outputs/window_attention_summary.csv", "condition", "v", "#76B7B2"),
    "E06 steering": ("experiments/06_cross_text_tone_style_transfer/outputs/steering_summary.csv", "condition", "X", "#9C755F"),
    "E09a DPO": ("experiments/09_rl_post_training_humanization/09a_dpo_anti_boilerplate/outputs/dpo_eval_summary.csv", "stage", "h", "#EDC948"),
}
HUMAN_GLTR = 0.797


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    rows = []

    for label, (path, keycol, marker, colour) in SERIES.items():
        p = ROOT / path
        if not p.exists():
            print(f"  skip {label}: {path} missing")
            continue
        df = pd.read_csv(p)
        df = df[np.isfinite(pd.to_numeric(df["ppl"], errors="coerce")) | (df["ppl"] == np.inf)]
        df["ratio"] = df["entity_recall"] / df["copy_4gram"].replace(0, np.nan)
        df["ppl_plot"] = pd.to_numeric(df["ppl"], errors="coerce").clip(lower=1.0)
        ax1.scatter(df["gltr_top10"], df["ratio"], marker=marker, s=55, color=colour,
                    alpha=0.85, edgecolor="white", linewidth=0.6, label=label)
        ax2.scatter(df["gltr_top10"], np.log10(df["ppl_plot"]), marker=marker, s=55,
                    color=colour, alpha=0.85, edgecolor="white", linewidth=0.6, label=label)
        for _, r in df.iterrows():
            rows.append({"series": label, "condition": r[keycol], "gltr_top10": r["gltr_top10"],
                         "recall_over_copy": r["ratio"], "ppl": r["ppl"],
                         "copy_4gram": r["copy_4gram"], "entity_recall": r["entity_recall"]})

    for ax in (ax1, ax2):
        ax.axvline(HUMAN_GLTR, color="k", ls="--", lw=1.2)
        ax.set_xlabel("GLTR top-10 conformity (human = 0.797)")
        ax.grid(alpha=0.3)

    ax1.set_ylabel("entity recall / 4-gram copy  (higher = more new facts per copied phrase)")
    ax1.set_title("What human-likeness buys: information per copied phrase")
    ax1.axhline(1.0, color="grey", ls=":", lw=1)

    ax2.set_ylabel("log10 perplexity under SmolLM-135M base")
    ax2.set_title("Cost of the same move: fluency collapse")
    ax2.annotate("human text", (HUMAN_GLTR, ax2.get_ylim()[0]), rotation=90,
                 fontsize=8, va="bottom", ha="right")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(OUT / "synthesis_frontier.png", dpi=140)

    pd.DataFrame(rows).to_csv(OUT / "synthesis_frontier.csv", index=False)

    df = pd.DataFrame(rows)
    df["dist_to_human"] = (df["gltr_top10"] - HUMAN_GLTR).abs()
    print("\nclosest points to the human GLTR profile that keep recall/copy > 1:")
    cand = df[(df.recall_over_copy > 1.0) & (df.ppl < 200)].sort_values("dist_to_human").head(6)
    print(cand.round(3).to_string(index=False))
    print(f"\nsaved -> {OUT / 'synthesis_frontier.png'} and synthesis_frontier.csv "
          f"({len(df)} points from {df.series.nunique()} experiments)")


if __name__ == "__main__":
    main()
