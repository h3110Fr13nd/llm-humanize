"""Experiment 11 — entity-pinned head escape (see src/humanize_lab/routers.py).

Spawned by E10: standard truncation is head-locked (GLTR top-10 ~0.99 vs 0.80
human) and the only head-escaping decoder destroys entity recall. This routes
on token *class* instead of confidence:

  * capitalised / numeric tokens that appear in the source -> greedy copy (anchors)
  * everything else -> mixture of nucleus (1-q) and the post-nucleus boundary token (q)

Run: uv run python experiments/11_entity_pinned_head_escape/scripts/run_entity_pinned.py
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
from humanize_lab.processors import Temperature, TopP
from humanize_lab.routers import EntityPinnedRouter, fact_token_ids, preamble_ban
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "11_entity_pinned_head_escape" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
MAX_NEW = 100

tok, _ = load()


def pinned(**kw):
    facts = fact_token_ids(tok)

    def make():
        return [EntityPinnedRouter(facts, **kw)]

    return make


def pinned_ban(**kw):
    facts = fact_token_ids(tok)

    def make():
        b1, bp = preamble_ban(tok)
        return [b1, bp, EntityPinnedRouter(facts, **kw)]

    return make


CONDITIONS = {
    "top_p_0.9": {"factory": lambda: [Temperature(0.8), TopP(0.9)]},
    "ban_only": {"factory": lambda: [*preamble_ban(tok), Temperature(0.8), TopP(0.9)]},
    "pin_only": {"factory": pinned(post_prob=0.0)},
    "pin_post_q0.25": {"factory": pinned(post_prob=0.25)},
    "pin_post_q0.50": {"factory": pinned(post_prob=0.5)},
    "pin_post_q1.0": {"factory": pinned(post_prob=1.0)},
    "pin_post_q0.5+ban": {"factory": pinned_ban(post_prob=0.5)},
    "pin_post_q1.0+ban": {"factory": pinned_ban(post_prob=1.0)},
}

if __name__ == "__main__":
    print(env_fingerprint())
    facts = fact_token_ids(tok)
    print(f"fact-token ids: {len(facts)} / {len(tok)} ({100*len(facts)/len(tok):.1f}% of vocab)")

    t0 = time.time()
    rows, texts = eval_conditions(
        CONDITIONS, docs=load_docs(), seeds=SEEDS, max_new_tokens=MAX_NEW,
        human_dist=human_reference_dist(),
    )
    print(f"total {time.time()-t0:.1f}s")

    pd.DataFrame(rows).to_csv(OUT / "entity_pinned_raw.csv", index=False)
    sdf = pd.DataFrame(summarize_conditions(rows))
    sdf.to_csv(OUT / "entity_pinned_summary.csv", index=False)

    cols = ["condition", "n_words", "copy_4gram", "entity_recall", "distinct_2",
            "burstiness", "boilerplate", "ppl", "gltr_top10", "style_jsd"]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "entity_pinned_summary.md").write_text("# Entity-pinned head escape — summary\n\n" + table + "\n")
    print("\n" + table)

    save_json(texts, OUT / "samples_all_conditions.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, m in zip(axes, ["gltr_top10", "entity_recall", "copy_4gram", "boilerplate"]):
        ax.bar(sdf["condition"], sdf[m], color="#E15759")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "entity_pinned.png", dpi=130)
    print(f"saved -> {OUT}")
