"""Experiment 06 — Cross-text tone/style transfer via activation steering.

Build a difference-in-means style vector (human sources minus model rewrites of
the *same* documents, so content is controlled) and inject it into the residual
stream during generation at a relative strength (fraction of the residual norm).

Run: uv run python experiments/06_cross_text_tone_style_transfer/scripts/run_steering.py
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
import torch

from humanize_lab.corpus import human_reference_dist, load_docs
from humanize_lab.evaluate import build_prompt, eval_conditions, summarize_conditions
from humanize_lab.models import load
from humanize_lab.processors import Temperature, TopP
from humanize_lab.sampling import generate
from humanize_lab.steering import projection, residual_norms, steer, style_vector
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "06_cross_text_tone_style_transfer" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2)
MAX_NEW = 100
LAYERS = (6, 12, 24)
ALPHAS = (0.10, 0.25, 0.50)

tok, model = load()
docs = load_docs()

if __name__ == "__main__":
    print(env_fingerprint())

    # 1. model-voice corpus: top-p rewrites of the same documents
    prompts = [build_prompt(tok, d) for d in docs]
    t0 = time.time()
    base = generate(model, tok, prompts, max_new_tokens=MAX_NEW, processors=[Temperature(0.8), TopP(0.9)], seed=99)
    print(f"model-voice corpus generated in {time.time()-t0:.1f}s")

    human_texts = [d["text"] for d in docs]
    human = human_reference_dist()

    # 2. per-layer style vectors, projection separation, residual scale
    rn = residual_norms(model, tok, human_texts + base.texts, list(LAYERS))
    vectors, proj_rows = {}, []
    for layer in LAYERS:
        v = style_vector(model, tok, human_texts, base.texts, layer)
        vectors[layer] = v.detach().cpu()
        ph = projection(model, tok, human_texts, v, layer)
        pm = projection(model, tok, base.texts, v, layer)
        sep = float((ph.mean() - pm.mean()) / (torch.cat([ph, pm]).std() + 1e-9))
        proj_rows.append({"layer": layer, "human_proj": float(ph.mean()), "model_proj": float(pm.mean()),
                          "separation_sigma": sep, "vector_norm": float(v.norm()),
                          "residual_norm": rn[layer], "vector_as_frac_of_residual": float(v.norm() / rn[layer])})
        print(f"  layer {layer:2d}: separation {sep:5.2f} sigma | vector {v.norm():6.1f} "
              f"= {100*v.norm()/rn[layer]:4.1f}% of residual norm {rn[layer]:.0f}")
    pd.DataFrame(proj_rows).to_csv(OUT / "style_vector_projection.csv", index=False)

    # 3. steering during generation
    import humanize_lab.evaluate as ev

    original_generate = ev.generate

    def patched_generate(*args, **kwargs):
        cfg = kwargs.pop("_steer", None)
        if cfg is None:
            return original_generate(*args, **kwargs)
        layer, vec, alpha = cfg
        with steer(model, layer, vec.to(model.device), alpha, residual_norm=rn[layer]):
            return original_generate(*args, **kwargs)

    ev.generate = patched_generate

    conditions = {"no_steering": {"factory": lambda: [Temperature(0.8), TopP(0.9)]}}
    for layer in LAYERS:
        for alpha in ALPHAS:
            conditions[f"L{layer}_a{alpha}"] = {
                "factory": lambda: [Temperature(0.8), TopP(0.9)],
                "extra": {"_steer": (layer, vectors[layer], alpha)},
            }
    g = torch.Generator().manual_seed(0)
    rand = torch.randn_like(vectors[12])
    conditions["L12_random_a0.25"] = {
        "factory": lambda: [Temperature(0.8), TopP(0.9)],
        "extra": {"_steer": (12, rand, 0.25)},
    }

    t0 = time.time()
    rows, texts = eval_conditions(conditions, docs=docs, seeds=SEEDS, max_new_tokens=MAX_NEW, human_dist=human)
    print(f"steered generation total {time.time()-t0:.1f}s")
    ev.generate = original_generate

    pd.DataFrame(rows).to_csv(OUT / "steering_raw.csv", index=False)
    sdf = pd.DataFrame(summarize_conditions(rows))
    sdf.to_csv(OUT / "steering_summary.csv", index=False)
    cols = ["condition", "n_words", "style_jsd", "burstiness", "copy_4gram", "entity_recall",
            "distinct_2", "ppl", "gltr_top10", "boilerplate"]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "steering_summary.md").write_text("# Activation steering — summary\n\n" + table + "\n")
    print("\n" + table)

    save_json(texts, OUT / "samples.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, m in zip(axes, ["style_jsd", "burstiness", "ppl"]):
        ax.bar(sdf["condition"], sdf[m], color="#B07AA1")
        ax.set_title(m); ax.tick_params(axis="x", rotation=70, labelsize=7); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "steering.png", dpi=130)
    print(f"saved -> {OUT}")
