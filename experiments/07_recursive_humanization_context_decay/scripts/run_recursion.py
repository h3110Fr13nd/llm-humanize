"""Experiment 07 — Recursive humanization and context decay.

Feed each rewrite back in as the next round's source, T0 -> T1 -> ... -> T6,
and measure how facts, length, diversity and fluency decay. This is the
behavioural analogue of E06's steering drift and the practical failure mode of
"run the humanizer again until it sounds human".

Run: uv run python experiments/07_recursive_humanization_context_decay/scripts/run_recursion.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import difflib

import matplotlib.pyplot as plt
import pandas as pd

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_NEUTRAL, USER_TEMPLATES, human_reference_dist, load_docs
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, load, render_chat
from humanize_lab.processors import Greedy, HybridPostTopP, Temperature, TopP
from humanize_lab.sampling import generate
from humanize_lab.tracking import env_fingerprint, group_mean, md_table, save_json

OUT = ROOT / "experiments" / "07_recursive_humanization_context_decay" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ROUNDS = 6
MAX_NEW = 100
SEEDS = (0, 1, 2)
DECODERS = {
    "greedy": lambda: [Greedy()],
    "top_p_0.9": lambda: [Temperature(0.8), TopP(0.9)],
    "hybrid_tau0.5_x0.4": lambda: [HybridPostTopP(0.5, 0.4)],
}

if __name__ == "__main__":
    print(env_fingerprint())
    docs = load_docs()
    tok, model = load(DEFAULT_MODEL)
    stok, scorer = load(BASE_MODEL)
    human = human_reference_dist()

    rows, trajectories = [], {}
    t0 = time.time()
    for dec_name, factory in DECODERS.items():
        for seed in SEEDS:
            current = [d["text"] for d in docs]
            for rnd in range(1, ROUNDS + 1):
                previous = list(current)
                prompts = [render_chat(tok, SYSTEM_NEUTRAL, USER_TEMPLATES["neutral"].format(text=t)) for t in current]
                out = generate(model, tok, prompts, max_new_tokens=MAX_NEW, processors=factory(), seed=seed)
                current = out.texts
                ppls = metrics.perplexity(scorer, stok, current)
                for d, text, prev, ppl in zip(docs, current, previous, ppls):
                    m = metrics.summarize(text, prev, d["entities"], human)  # copy vs previous round
                    m_orig = metrics.summarize(text, d["text"], d["entities"], human)  # copy vs original
                    rows.append({
                        "decoder": dec_name, "seed": seed, "round": rnd, "doc_id": d["id"],
                        "ppl": ppl,
                        "entity_recall_vs_original": m_orig["entity_recall"],
                        "copy_vs_previous": m["copy_4gram"],
                        "copy_vs_original": m_orig["copy_4gram"],
                        "churn_vs_previous": 1.0 - difflib.SequenceMatcher(None, prev, text).ratio(),
                        "similarity_to_original": difflib.SequenceMatcher(None, d["text"], text).ratio(),
                        **{k: m[k] for k in ("n_words", "distinct_2", "burstiness", "boilerplate", "style_jsd", "compression")},
                    })
                    trajectories.setdefault(f"{dec_name}|s{seed}|{d['id']}", []).append(
                        round(m_orig["entity_recall"], 3))
            print(f"  {dec_name} seed {seed} done (round {ROUNDS})", flush=True)
    print(f"total {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recursion_raw.csv", index=False)

    agg = df.groupby(["decoder", "round"]).mean(numeric_only=True).reset_index()
    agg.to_csv(OUT / "recursion_by_round.csv", index=False)
    cols = ["decoder", "round", "n_words", "entity_recall_vs_original", "churn_vs_previous",
            "copy_vs_previous", "copy_vs_original", "similarity_to_original", "distinct_2",
            "burstiness", "boilerplate", "ppl"]
    table = md_table(agg[cols].to_dict("records"), cols)
    (OUT / "recursion_summary.md").write_text("# Recursive humanization — by round\n\n" + table + "\n")
    print("\n" + table)

    # entity half-life: first round where recall <= 50 % of round-1 recall
    half_life = {}
    for dec in DECODERS:
        sub = agg[agg["decoder"] == dec].sort_values("round")
        base = float(sub.iloc[0]["entity_recall_vs_original"])
        hl = None
        for _, r in sub.iterrows():
            if r["entity_recall_vs_original"] <= base / 2:
                hl = int(r["round"])
                break
        half_life[dec] = {"round1_recall": base, "half_life_round": hl}
    save_json(half_life, OUT / "entity_half_life.json")
    print("\nentity half-life:", half_life)

    save_json(trajectories, OUT / "trajectories.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for dec in DECODERS:
        sub = agg[agg["decoder"] == dec]
        axes[0].plot(sub["round"], sub["entity_recall_vs_original"], "o-", label=dec)
        axes[1].plot(sub["round"], sub["ppl"], "o-", label=dec)
        axes[2].plot(sub["round"], sub["copy_vs_previous"], "o-", label=dec)
    for ax, t in zip(axes, ["entity recall vs original", "perplexity", "copy vs previous round"]):
        ax.set_title(t); ax.set_xlabel("round"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[1].set_yscale("log")
    plt.tight_layout(); plt.savefig(OUT / "recursion.png", dpi=130)
    print(f"saved -> {OUT}")
