"""Experiment 04 — Global non-autoregressive rewrite (real diffusion LM vs AR).

Arm A: SmolLM-135M-Instruct, left-to-right, top-p / greedy (the baseline the
       whole repo uses).
Arm B: `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` — a real masked-diffusion LM
       (MDLM, 0.6 B, Qwen3 base). With block_size == max_new_tokens every token
       of the response is decided jointly, i.e. a genuine global canvas;
       block_size=32 makes it semi-autoregressive as an ablation.

Same 12 documents, same rewrite instruction, same metrics. The question is
whether bidirectionality fixes copying/boilerplate/drift that left-to-right
decoding cannot (E10/E12).

Run: uv run python experiments/04_text_diffusion_global_rewrite/scripts/run_diffusion_vs_ar.py
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

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_NEUTRAL, USER_TEMPLATES, human_reference_dist, load_docs
from humanize_lab.diffusion import generate as diff_generate
from humanize_lab.evaluate import build_prompt
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, load, render_chat
from humanize_lab.processors import Greedy, Temperature, TopP
from humanize_lab.sampling import generate as ar_generate
from humanize_lab.textfix import verbalise_numerals
from humanize_lab.tracking import env_fingerprint, group_mean, md_table, save_json

OUT = ROOT / "experiments" / "04_text_diffusion_global_rewrite" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DIFF_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
SEEDS = (0, 1, 2)
MAX_NEW = 64  # both arms; diffusion steps cost ~1 s each
METRIC_KEYS = ["n_words", "copy_4gram", "entity_recall", "distinct_2", "burstiness",
               "boilerplate", "style_jsd"]


def score_rows(name, seed, docs, texts, human, scorer, stok, extra=None):
    ppls = metrics.perplexity(scorer, stok, texts)
    rows = []
    for d, text, ppl in zip(docs, texts, ppls):
        m = metrics.summarize(text, d["text"], d["entities"], human)
        rows.append({"condition": name, "seed": seed, "doc_id": d["id"], "domain": d["domain"],
                     **{k: m[k] for k in METRIC_KEYS}, "ppl": ppl, **(extra or {})})
    return rows


if __name__ == "__main__":
    print(env_fingerprint())
    docs = load_docs()
    human = human_reference_dist()
    stok, scorer = load(BASE_MODEL)
    tok, ar_model = load(DEFAULT_MODEL)
    rows, samples = [], {}

    # ---------- Arm A: autoregressive ----------
    prompts = [build_prompt(tok, d) for d in docs]
    for name, procs, seeds in [
        ("ar_smol135m_top_p", lambda: [Temperature(0.8), TopP(0.9)], SEEDS),
        ("ar_smol135m_greedy", lambda: [Greedy()], (0,)),
    ]:
        for seed in seeds:
            t0 = time.time()
            out = ar_generate(ar_model, tok, prompts, max_new_tokens=MAX_NEW, processors=procs(), seed=seed)
            dt = time.time() - t0
            rows += score_rows(name, seed, docs, out.texts, human, scorer, stok, {"seconds": dt})
            samples.setdefault(name, []).extend(out.texts)
            print(f"  {name:26s} seed {seed} {dt:5.1f}s", flush=True)

    # ---------- Arm B: diffusion ----------
    # 4 GB VRAM: give the 0.6 B diffusion model the whole GPU. The AR arms and
    # the scorer are already done, so they are parked on CPU.
    ar_model.to("cpu")
    scorer.to("cpu")
    torch.cuda.empty_cache()
    print(f"VRAM after parking AR models on CPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    try:
        from humanize_lab.diffusion import load_diffusion_model

        dtok, dmodel = load_diffusion_model(DIFF_MODEL)
        print(f"diffusion model loaded: {DIFF_MODEL} "
              f"({torch.cuda.memory_allocated()/1e9:.2f} GB VRAM allocated)")
        dprompts = [
            dtok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_NEUTRAL},
                 {"role": "user", "content": USER_TEMPLATES["neutral"].format(text=d["text"])}],
                add_generation_prompt=True, tokenize=True, enable_thinking=False,
            )["input_ids"]
            for d in docs
        ]
        # Numerals in the conditioning text seed a comma+digit attractor that
        # swallows the whole canvas (E04c, outputs/numeral_probe.json). Spelling
        # them out removes it, so the same conditions are run a second time on
        # numeral-free prompts. This is the preprocessing fix, measured.
        dprompts_verbal = [
            dtok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_NEUTRAL},
                 {"role": "user", "content": USER_TEMPLATES["neutral"].format(
                     text=verbalise_numerals(d["text"]))}],
                add_generation_prompt=True, tokenize=True, enable_thinking=False,
            )["input_ids"]
            for d in docs
        ]
        pad_id = dtok.pad_token_id or dtok.eos_token_id
        # 4 GB card: the 0.6 B model (1.2 GB) plus two logit-sized temporaries is
        # the whole budget, so batch 1 it is. The peak-memory print below records
        # what each configuration actually costs.
        bs = 1

        def run_diff(name, seeds, prompts=None, **kw):
            prompts = dprompts if prompts is None else prompts
            for seed in seeds:
                torch.cuda.empty_cache()
                t0 = time.time()
                texts, confs = [], []
                for i in range(0, len(prompts), bs):
                    chunk = prompts[i : i + bs]
                    L = max(len(c) for c in chunk)
                    pt = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                    for j, c in enumerate(chunk):
                        pt[j, : len(c)] = torch.tensor(c, dtype=torch.long)
                    pl = torch.tensor([len(c) for c in chunk], dtype=torch.long)
                    torch.manual_seed(seed * 1000 + i)
                    ids, conf = diff_generate(
                        dmodel, dtok, pt.to(dmodel.device), pl.to(dmodel.device),
                        max_new_tokens=MAX_NEW, capture_confidence=True, **kw,
                    )
                    texts.extend(dtok.batch_decode(ids, skip_special_tokens=True))
                    confs.append(conf or 0.0)
                texts = [t.strip() for t in texts]
                dt = time.time() - t0
                rows.extend(score_rows(name, seed, docs, texts, human, scorer, stok,
                                       {"seconds": dt, "unmask_confidence": sum(confs) / len(confs)}))
                samples.setdefault(name, []).extend(texts)
                print(f"  {name:26s} seed {seed} {dt:5.1f}s conf={sum(confs)/len(confs):.3f} "
                      f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
                torch.cuda.reset_peak_memory_stats()

        run_diff("diff_global_T0", (0,), block_size=MAX_NEW, steps=MAX_NEW, temperature=0.0)
        run_diff("diff_global_T1", (0, 1), block_size=MAX_NEW, steps=MAX_NEW, temperature=1.0)
        run_diff("diff_block16_T1", (0,), block_size=16, steps=MAX_NEW, temperature=1.0)
        # same two samplers, numeral-free conditioning (the E04c fix)
        run_diff("diff_global_T1_verb", (0, 1), prompts=dprompts_verbal,
                 block_size=MAX_NEW, steps=MAX_NEW, temperature=1.0)
        run_diff("diff_block16_T1_verb", (0,), prompts=dprompts_verbal,
                 block_size=16, steps=MAX_NEW, temperature=1.0)
    except Exception as e:  # pragma: no cover - offline fallback
        print(f"!! diffusion arm skipped: {type(e).__name__}: {e}")
        save_json({"error": f"{type(e).__name__}: {e}"}, OUT / "diffusion_arm_error.json")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diffusion_vs_ar_raw.csv", index=False)
    agg = pd.DataFrame(group_mean(rows, "condition", METRIC_KEYS + ["ppl", "seconds"]))
    agg.to_csv(OUT / "diffusion_vs_ar_summary.csv", index=False)
    cols = ["condition", "n_words", "copy_4gram", "entity_recall", "distinct_2",
            "burstiness", "boilerplate", "style_jsd", "ppl", "seconds"]
    table = md_table(agg[cols].to_dict("records"), cols)
    (OUT / "diffusion_vs_ar_summary.md").write_text("# Diffusion vs AR — summary\n\n" + table + "\n")
    print("\n" + table)

    save_json(samples, OUT / "samples.json")
    save_json(env_fingerprint(), OUT / "env.json")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, m in zip(axes, ["copy_4gram", "entity_recall", "boilerplate", "ppl"]):
        ax.bar(agg["condition"], agg[m], color="#76B7B2")
        ax.set_title(m); ax.tick_params(axis="x", rotation=70, labelsize=7); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "diffusion_vs_ar.png", dpi=130)
    print(f"saved -> {OUT}")
