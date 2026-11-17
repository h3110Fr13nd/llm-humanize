"""Experiment 05 — Speculative / asymmetric window attention.

Hypothesis: the copy-and-echo behaviour of a small decoder comes from generated
tokens dominating their own attention. Restrict *self*-attention to a sliding
window while keeping the whole source prefix visible (asymmetric attention).

Ablations that isolate the mechanism:
  * window_full_src_W  : generated tokens see all of the source + last W generated
  * window_all_W       : uniform sliding window, source included
  * no_src             : source is masked out during generation (control:
                         proves the source attention is what preserves facts)

Run: uv run python experiments/05_speculative_window_attention/scripts/run_window_attention.py
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
from humanize_lab.processors import Temperature, TopP, window_mask_fn
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "05_speculative_window_attention" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
MAX_NEW = 100
DECODER = lambda: [Temperature(0.8), TopP(0.9)]  # noqa: E731

CONDITIONS = {
    "full_attention": {"factory": DECODER, "mask_fn": None},
    "win4_src_full": {"factory": DECODER, "mask_fn": window_mask_fn(4)},
    "win8_src_full": {"factory": DECODER, "mask_fn": window_mask_fn(8)},
    "win16_src_full": {"factory": DECODER, "mask_fn": window_mask_fn(16)},
    "win32_src_full": {"factory": DECODER, "mask_fn": window_mask_fn(32)},
    "win64_src_full": {"factory": DECODER, "mask_fn": window_mask_fn(64)},
    "win16_all": {"factory": DECODER, "mask_fn": window_mask_fn(16, source_full=False)},
    "win16_no_src": {"factory": DECODER, "mask_fn": window_mask_fn(16, mask_source=True)},
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

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "window_attention_raw.csv", index=False)
    sdf = pd.DataFrame(summarize_conditions(rows))
    sdf.to_csv(OUT / "window_attention_summary.csv", index=False)

    cols = ["condition", "n_words", "copy_4gram", "rep_5", "entity_recall", "distinct_2",
            "burstiness", "ppl", "gltr_top10", "style_jsd"]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "window_attention_summary.md").write_text("# Window attention — summary\n\n" + table + "\n")
    print("\n" + table)

    save_json({k: v for k, v in texts.items()}, OUT / "samples_all_conditions.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, m in zip(axes, ["copy_4gram", "rep_5", "entity_recall", "ppl"]):
        ax.bar(sdf["condition"], sdf[m], color="#59A14F")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "window_attention.png", dpi=130)
    print(f"saved -> {OUT}")
