"""Experiment 10 — Decoder zoo shootout on a 135M model.

Runs 10 decoding strategies over the 12-document / 3-seed benchmark and scores
every rewrite on plagiarism, factual recall, diversity, burstiness, boilerplate
and model perplexity (plus GLTR head conformity against a human baseline).

Run: uv run python experiments/10_decoder_zoo_shootout/scripts/run_decoder_shootout.py
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
from humanize_lab.metrics import gltr_from_texts
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, load
from humanize_lab.processors import (
    DRY,
    Eta,
    Greedy,
    HybridPostTopP,
    MinP,
    MirostatV2,
    RepetitionPenalty,
    Temperature,
    TopH,
    TopP,
    Typical,
)
from humanize_lab.tracking import env_fingerprint, md_table, save_json, save_rows

OUT = ROOT / "experiments" / "10_decoder_zoo_shootout" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2)
MAX_NEW = 100

CONDITIONS = {
    "greedy": {"factory": lambda: [Greedy()]},
    "top_p_0.9": {"factory": lambda: [Temperature(0.8), TopP(0.9)]},
    "top_p_0.9+rep1.3": {"factory": lambda: [Temperature(0.8), RepetitionPenalty(1.3), TopP(0.9)]},
    "min_p_0.05": {"factory": lambda: [MinP(0.05, temperature=0.8)]},
    "typical_p_0.9": {"factory": lambda: [Temperature(0.8), Typical(0.9)]},
    "eta_1e-4": {"factory": lambda: [Temperature(0.8), Eta(1e-4)]},
    "top_h_2.0": {"factory": lambda: [Temperature(0.8), TopH(2.0)]},
    "mirostat_v2_tau3": {"factory": lambda: [Temperature(0.8), MirostatV2(3.0)]},
    "dry_0.8+top_p_0.9": {"factory": lambda: [Temperature(0.8), DRY(0.8, 1.75, 2, 4), TopP(0.9)]},
    "hybrid_tau0.5_x0.4": {"factory": lambda: [HybridPostTopP(0.5, 0.4)]},
}

if __name__ == "__main__":
    env = env_fingerprint()
    print(f"env: {env}")
    docs = load_docs()
    human = human_reference_dist()
    print(f"{len(docs)} docs, {len(CONDITIONS)} conditions, seeds={SEEDS}")

    t0 = time.time()
    rows, texts = eval_conditions(
        CONDITIONS,
        docs=docs,
        seeds=SEEDS,
        max_new_tokens=MAX_NEW,
        human_dist=human,
        model_id=DEFAULT_MODEL,
    )
    print(f"total {time.time()-t0:.1f}s")

    # --- human GLTR baseline (real next-tokens of the source corpus) ---
    tok_b, base = load(BASE_MODEL)
    human_gltr = gltr_from_texts(base, tok_b, [d["text"] for d in docs])
    print("\nhuman GLTR baseline:", {k: round(v, 3) for k, v in human_gltr.items()})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "decoder_shootout_raw.csv", index=False)

    summary = summarize_conditions(rows)
    sdf = pd.DataFrame(summary)

    def z(col):
        v = sdf[col]
        return (v - v.mean()) / (v.std(ddof=0) + 1e-9)

    sdf["human_score"] = (
        1.2 * z("distinct_2")
        + 0.8 * z("burstiness")
        + 0.8 * z("entity_recall")
        - 1.2 * z("copy_4gram")
        - 0.8 * z("boilerplate")
        - 0.6 * z("ppl")
    )
    sdf["head_gap_vs_human"] = sdf["gltr_top10"] - human_gltr.get("gltr_top10", 0.0)
    sdf = sdf.sort_values("human_score", ascending=False)
    sdf.to_csv(OUT / "decoder_shootout_summary.csv", index=False)

    cols = [
        "condition", "n_words", "copy_4gram", "entity_recall", "distinct_2", "rep_4",
        "burstiness", "boilerplate", "style_jsd", "ppl", "gltr_top10", "head_gap_vs_human",
        "human_score",
    ]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "decoder_shootout_summary.md").write_text(
        f"# Decoder zoo shootout — summary (mean over {len(docs)} docs x {len(SEEDS)} seeds)\n\n"
        f"Human GLTR baseline on the same corpus: `{ {k: round(v,3) for k,v in human_gltr.items()} }`\n\n"
        f"{table}\n"
    )
    print("\n" + table)

    save_json(env, OUT / "env.json")
    save_json({k: v for k, v in texts.items()}, OUT / "samples_all_conditions.json")
    save_json(human_gltr, OUT / "human_gltr_baseline.json")

    metrics_to_plot = ["copy_4gram", "entity_recall", "distinct_2", "burstiness", "boilerplate", "ppl"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, m in zip(axes.flatten(), metrics_to_plot):
        s = sdf.sort_values(m)
        ax.barh(s["condition"], s[m], color="#4C78A8")
        ax.set_title(m)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "decoder_shootout.png", dpi=130)
    print(f"\nsaved -> {OUT}")
