"""Experiment 09a — DPO distillation: can preference training remove the
assistant preamble and move the head-conformity floor?

E10/E12 established that no decoding rule can leave the model's top-10 head
(GLTR 0.98-1.00 vs 0.797 for human text) without destroying the text, and E11
showed the cheapest available win is *banning* the "Here's a rewritten
version:" opener (copy 0.505 -> 0.191, PPL 12.9 -> 9.8). A ban is a runtime
crutch. DPO asks whether the same suppression can be moved into the weights,
and — the E12 target — whether it can shift the output distribution itself.

Pairs (same prompt, no reward model):
  chosen   = top-p(0.9) output with the preamble banned   (E11 "ban_only")
  rejected = greedy output                                 (E10 baseline)

Trained with a hand-rolled LoRA + DPO (src/humanize_lab/tuning.py): LoRA is a
separate module, so the DPO reference policy is the same model with the
adapter disabled — no second copy of the weights in 4 GB of VRAM.

Run: uv run python experiments/09_rl_post_training_humanization/09a_dpo_anti_boilerplate/scripts/run_dpo.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_HUMANIZE, SYSTEM_NEUTRAL, USER_TEMPLATES, human_reference_dist, load_docs
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, load, render_chat
from humanize_lab.processors import Greedy, Temperature, TopP
from humanize_lab.routers import preamble_ban
from humanize_lab.sampling import generate
from humanize_lab.tracking import env_fingerprint, md_table, save_json
from humanize_lab.tuning import DPOData, LoRAConfig, apply_lora, train_dpo

OUT = ROOT / "experiments" / "09_rl_post_training_humanization" / "09a_dpo_anti_boilerplate" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MAX_NEW = 100
EVAL_DOCS = 12


def passages(text: str, target: int = 60) -> list[str]:
    """Split a document into ~target-word passages on sentence boundaries."""
    sents = metrics.sentences(text)
    out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > target * 1.4:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return [p for p in out if len(p.split()) >= 25]


def main() -> None:
    print(env_fingerprint())
    docs = load_docs()
    human = human_reference_dist()
    tok, model = load(DEFAULT_MODEL)
    stok, scorer = load(BASE_MODEL)
    ban1, banp = preamble_ban(tok)

    # ---------------------------------------------------------------- 1. data
    pieces: list[tuple[str, str]] = []  # (system, passage)
    for d in docs:
        for p in passages(d["text"]):
            pieces.append((SYSTEM_NEUTRAL, p))
            pieces.append((SYSTEM_HUMANIZE, p))
    print(f"{len(pieces)} training prompts from {len(docs)} docs "
          f"({len(pieces)//2} passages x 2 instructions)")

    prompts = [render_chat(tok, s, USER_TEMPLATES[mode].format(text=p))
               for (s, p), mode in zip(pieces, ["neutral", "humanize"] * (len(pieces) // 2))]
    # rebuild explicitly so mode always matches the system prompt
    prompts = []
    for i, (s, p) in enumerate(pieces):
        mode = "neutral" if i % 2 == 0 else "humanize"
        prompts.append(render_chat(tok, s, USER_TEMPLATES[mode].format(text=p)))

    t0 = time.time()
    chosen_out = generate(model, tok, prompts, max_new_tokens=MAX_NEW, seed=7,
                          processors=[ban1, banp, Temperature(0.8), TopP(0.9)])
    rejected_out = generate(model, tok, prompts, max_new_tokens=MAX_NEW, seed=7, processors=[Greedy()])
    print(f"pairs generated in {time.time()-t0:.1f}s")

    triples = []
    pair_rows = []
    for (s, p), pr, ch, rj in zip(pieces, prompts, chosen_out.texts, rejected_out.texts):
        if not ch.strip() or not rj.strip() or ch.strip() == rj.strip():
            continue
        triples.append((pr, ch, rj))
        for label, txt in (("chosen", ch), ("rejected", rj)):
            m = metrics.summarize(txt, p, [], human)
            pair_rows.append({"label": label, "words": m["n_words"], "copy_4gram": m["copy_4gram"],
                              "boilerplate": m["boilerplate"], "burstiness": m["burstiness"],
                              "distinct_2": m["distinct_2"]})
    pd.DataFrame(pair_rows).to_csv(OUT / "pair_quality.csv", index=False)
    pq = pd.DataFrame(pair_rows).groupby("label").mean(numeric_only=True)
    print("\npair quality (mean):")
    print(pq.round(3).to_string())

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(triples), generator=g).tolist()
    n_test = max(8, len(triples) // 6)
    test = DPOData([triples[i] for i in idx[:n_test]])
    train = DPOData([triples[i] for i in idx[n_test:]])
    print(f"\ntrain {len(train)} pairs, held-out {len(test)} pairs")
    save_json({"n_train": len(train), "n_test": len(test),
               "sample": [{"prompt": t[0][-260:], "chosen": t[1], "rejected": t[2]} for t in triples[:3]]},
              OUT / "pairs_preview.json")

    # ------------------------------------------------------------- 2. baseline
    eval_prompts = [render_chat(tok, SYSTEM_NEUTRAL, USER_TEMPLATES["neutral"].format(text=d["text"])) for d in docs]

    def evaluate(tag: str) -> list[dict]:
        out = generate(model, tok, eval_prompts, max_new_tokens=MAX_NEW, processors=[Greedy()],
                       capture_trace=True)
        ppls = metrics.perplexity(scorer, stok, out.texts)
        gltr = metrics.gltr_buckets(out.trace)
        rows = []
        for d, text, ppl in zip(docs, out.texts, ppls):
            m = metrics.summarize(text, d["text"], d["entities"], human)
            rows.append({"stage": tag, "doc_id": d["id"], "ppl": ppl, **m, **gltr})
        means = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
        print(f"  {tag:8s} copy={means['copy_4gram']:.3f} ent={means['entity_recall']:.3f} "
              f"boil={means['boilerplate']:.2f} burst={means['burstiness']:.3f} "
              f"ppl={means['ppl']:.1f} gltr={means['gltr_top10']:.3f}")
        return rows

    print("\nevaluation at GREEDY decoding:")
    before = evaluate("before")

    # 4 GB of VRAM: the scorer and the policy cannot both sit on the GPU while
    # DPO holds 4 forward passes of full-vocabulary logits.
    scorer.to("cpu")
    torch.cuda.empty_cache()
    print(f"  VRAM after moving scorer to CPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ------------------------------------------------------------- 3. train
    handle = apply_lora(model, LoRAConfig(r=16, alpha=32.0, targets=("q_proj", "k_proj", "v_proj", "o_proj")))
    print(f"\nLoRA: {len(handle.modules)} modules, {handle.n_params():,} trainable params "
          f"({100*handle.n_params()/sum(p.numel() for p in model.parameters()):.2f}% of the model)")

    t0 = time.time()
    history = train_dpo(
        model, handle, tok, train, test,
        beta=0.1, lr=1e-3, epochs=5, batch_size=1, grad_accum=8, max_len=320, log_every=5,
    )
    print(f"training took {time.time()-t0:.1f}s")
    pd.DataFrame(history).to_csv(OUT / "dpo_history.csv", index=False)

    # ------------------------------------------------------------- 4. evaluate
    scorer.to(model.device)
    handle.set_enabled(True)
    after = evaluate("after")

    # save the adapter so the run is reproducible without retraining
    torch.save(
        {"config": {"r": 16, "alpha": 32.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
         "adapters": [{"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for m in handle.modules],
         "history": history},
        OUT / "lora_adapter.pt",
    )

    # ------------------------------------------------------------- 5. report
    rows = before + after
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dpo_eval_raw.csv", index=False)
    agg = df.groupby("stage").mean(numeric_only=True).reset_index()
    agg.to_csv(OUT / "dpo_eval_summary.csv", index=False)
    cols = ["stage", "n_words", "copy_4gram", "entity_recall", "boilerplate", "burstiness",
            "distinct_2", "rep_4", "style_jsd", "ppl", "gltr_top10", "gltr_10_100", "gltr_tail"]
    table = md_table(agg[cols].to_dict("records"), cols)
    (OUT / "dpo_eval_summary.md").write_text("# DPO anti-boilerplate — greedy evaluation\n\n" + table + "\n")
    print("\n" + table)

    gltr = pd.DataFrame(history)[["step", "dpo_loss", "reward_margin", "pref_acc"]].dropna(subset=["dpo_loss"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(gltr["step"], gltr["dpo_loss"], "o-")
    axes[0].set_title("DPO loss")
    axes[1].plot(gltr["step"], gltr["reward_margin"], "o-")
    axes[1].set_title("reward margin (chosen - rejected)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[2].bar(["before", "after"], [agg.loc[agg.stage == "before", "boilerplate"].iloc[0],
                                      agg.loc[agg.stage == "after", "boilerplate"].iloc[0]], color=["#999", "#E15759"])
    axes[2].set_title("boilerplate / 100 words (greedy)")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "dpo.png", dpi=130)
    save_json(env_fingerprint(), OUT / "env.json")
    save_json({"before": before, "after": after}, OUT / "eval_rows.json")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
