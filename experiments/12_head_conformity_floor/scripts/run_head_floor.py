"""Experiment 12 — Is head escape even possible by decoding alone?

E10 measured GLTR top-10 = 0.98–1.00 for every truncation family vs 0.797 for
human text. The truncation families all looked identical, which suggests the
*model's own distribution* may be the binding constraint, not the sampler.

This experiment measures the floor analytically and empirically:

  predicted(T) = mean probability mass that the model puts on its own top-10
                 tokens at temperature T along a real generation trajectory
  measured(T)  = GLTR top-10 fraction of tokens actually sampled at temperature T
                 with NO truncation at all

If measured ≈ predicted at every T, then no logits-level trick can reach the
human 0.80 profile, and the intervention must be training (E09) or architecture.

Run: uv run python experiments/12_head_conformity_floor/scripts/run_head_floor.py
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
import torch.nn.functional as F

from humanize_lab.corpus import human_reference_dist, load_docs
from humanize_lab.evaluate import build_prompt, eval_conditions, summarize_conditions
from humanize_lab.models import get_tokenizer, load, encode_prompts
from humanize_lab.processors import Greedy, Temperature
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "12_head_conformity_floor" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
TEMPS = (0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0)
SEEDS = (0, 1, 2)
MAX_NEW = 100

CONDITIONS = {f"T={t}": {"factory": (lambda t=t: [Temperature(t)])} for t in TEMPS}


@torch.no_grad()
def predicted_head_mass(model, tokenizer, docs, draft_tokens: int = 60) -> dict[float, float]:
    """Fraction of model mass on its own top-10 tokens, along a real trajectory."""
    prompts = [build_prompt(tokenizer, d) for d in docs]
    ids, am, _ = encode_prompts(tokenizer, prompts, model.device)
    L = ids.shape[1]
    # deterministic greedy draft to get a realistic trajectory
    cache = None
    out_ids = ids
    for step in range(draft_tokens):
        inp = out_ids if step == 0 else out_ids[:, -1:]
        pos = (am.cumsum(-1) - 1).clamp(min=0) if step == 0 else (am.sum(-1) + step).view(-1, 1)
        attn = torch.cat([am, torch.ones((ids.shape[0], step), dtype=am.dtype, device=ids.device)], 1)
        o = model(input_ids=inp, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        out_ids = torch.cat([out_ids, o.logits[:, -1, :].argmax(-1, keepdim=True)], 1)

    logits = model(input_ids=out_ids, attention_mask=torch.ones_like(out_ids)).logits[:, L - 1 : -1, :].float()
    res = {}
    for t in TEMPS:
        p = F.softmax(logits / t, dim=-1)
        top10 = p.topk(10, dim=-1).values.sum(-1)
        res[t] = float(top10.mean())
    return res


if __name__ == "__main__":
    print(env_fingerprint())
    docs = load_docs()
    tok, model = load()

    t0 = time.time()
    predicted = predicted_head_mass(model, tok, docs)
    print("predicted top-10 mass:", {k: round(v, 3) for k, v in predicted.items()})

    rows, texts = eval_conditions(
        CONDITIONS, docs=docs, seeds=SEEDS, max_new_tokens=MAX_NEW,
        human_dist=human_reference_dist(), tokenizer=tok, model=model,
    )
    print(f"total {time.time()-t0:.1f}s")

    pd.DataFrame(rows).to_csv(OUT / "head_floor_raw.csv", index=False)
    sdf = pd.DataFrame(summarize_conditions(rows))
    sdf["temperature"] = [float(c.split("=")[1]) for c in sdf["condition"]]
    sdf["predicted_top10_mass"] = sdf["temperature"].map(predicted)
    sdf["gap_measured_minus_predicted"] = sdf["gltr_top10"] - sdf["predicted_top10_mass"]
    sdf = sdf.sort_values("temperature")
    sdf.to_csv(OUT / "head_floor_summary.csv", index=False)

    cols = ["condition", "temperature", "predicted_top10_mass", "gltr_top10",
            "gap_measured_minus_predicted", "copy_4gram", "entity_recall", "distinct_2", "ppl", "n_words"]
    table = md_table(sdf[cols].to_dict("records"), cols)
    (OUT / "head_floor_summary.md").write_text("# Head-conformity floor — summary\n\n" + table + "\n")
    print("\n" + table)

    save_json({str(k): v for k, v in predicted.items()}, OUT / "predicted_mass.json")
    save_json(texts, OUT / "samples.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(sdf["temperature"], sdf["gltr_top10"], "o-", label="measured GLTR top-10", color="#4C78A8")
    ax1.plot(sdf["temperature"], sdf["predicted_top10_mass"], "s--", label="predicted top-10 mass", color="#F58518")
    ax1.axhline(0.797, color="#E45756", ls=":", label="human baseline 0.797")
    ax1.set_xlabel("temperature"); ax1.set_ylabel("top-10 fraction"); ax1.grid(alpha=0.3); ax1.legend()
    plt.tight_layout(); plt.savefig(OUT / "head_floor.png", dpi=130)
    print(f"saved -> {OUT}")
