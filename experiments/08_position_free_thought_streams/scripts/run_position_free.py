"""Experiment 08a + 08c — position-free thought streams (feasible POC).

08a: how much does SmolLM-135M depend on *absolute position*?
     Teacher-forced perplexity of the same coherent text under four position
     regimes: normal RoPE, shuffled positions, all-zero positions, reversed.

08c: can the model unscramble a sentence it is given as a bag of words?
     Shuffle 15 % / 30 % / 50 % of a sentence, ask for the original order, and
     measure recovery (difflib similarity + Kendall tau of the recovered words).
     This is the "diffusion lexical unscrambler" idea reduced to its testable
     core with an AR model: a position-free syntax prior should recover order.

08b (temporal JEPA concept streams) is not attempted: it needs a trained
predictor, which is out of scope for a 4 GB laptop POC — see README gaps.

Run: uv run python experiments/08_position_free_thought_streams/scripts/run_position_free.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import difflib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_NEUTRAL, human_reference_dist, load_docs
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, encode_prompts, load, render_chat
from humanize_lab.processors import Temperature, TopP
from humanize_lab.sampling import generate
from humanize_lab.tracking import env_fingerprint, md_table, save_json

OUT = ROOT / "experiments" / "08_position_free_thought_streams" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def ppl_under_positions(model, tok, texts, position_mode: str, seed: int = 0) -> list[float]:
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    ids = enc.input_ids.to(model.device)
    am = enc.attention_mask.to(model.device)
    base_pos = (am.cumsum(-1) - 1).clamp(min=0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if position_mode == "normal":
        pos = base_pos
    elif position_mode == "zero":
        pos = torch.zeros_like(base_pos)
    elif position_mode == "reversed":
        pos = (am.sum(-1, keepdim=True) - 1 - base_pos).clamp(min=0)
    elif position_mode == "shuffled":
        pos = torch.stack([base_pos[i][torch.randperm(base_pos.shape[1], generator=g)] for i in range(ids.shape[0])]).to(model.device)
    else:
        raise ValueError(position_mode)
    logits = model(input_ids=ids, attention_mask=am, position_ids=pos).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    mask = am[:, 1:].bool()
    lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    out = []
    for b in range(ids.shape[0]):
        v = mask[b]
        out.append(float(torch.exp(-lp[b][v].mean())) if v.any() else float("inf"))
    return out


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Rank correlation between recovered order and the original order."""
    pos = {w: i for i, w in enumerate(a)}
    ranks = [pos[w] for w in b if w in pos]
    n = len(ranks)
    if n < 2:
        return 0.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                conc += 1
            elif ranks[i] > ranks[j]:
                disc += 1
    return (conc - disc) / (conc + disc + 1e-9)


def scramble(sentence: str, frac: float, rng: torch.Generator) -> str:
    w = sentence.split()
    if len(w) < 6:
        return sentence
    n_swap = max(1, int(len(w) * frac))
    for _ in range(n_swap):
        i = int(torch.randint(0, len(w) - 1, (1,), generator=rng))
        j = int(torch.randint(0, len(w) - 1, (1,), generator=rng))
        w[i], w[j] = w[j], w[i]
    return " ".join(w)


def bag_recall(original: str, output: str) -> tuple[float, float]:
    """(share of original content words that survived, share of output words
    that came from the original). Distinguishes "reordered the words" from
    "ignored the words and wrote something else"."""
    stop = set("a an the of to in and or for with as by is are was were be been it its this that from on at".split())
    o = [w.lower().strip(".,;:()") for w in original.split()]
    g = [w.lower().strip(".,;:()") for w in output.split()]
    o = [w for w in o if w and w not in stop]
    g = [w for w in g if w and w not in stop]
    if not o:
        return 0.0, 0.0
    oset = set(o)
    hit = sum(1 for w in g if w in oset)
    return (hit / len(oset), hit / max(len(g), 1))


if __name__ == "__main__":
    print(env_fingerprint())
    docs = load_docs()
    tok, model = load(DEFAULT_MODEL)
    human = human_reference_dist()

    # ---------- 08a: position ablation (teacher-forced PPL) ----------
    rows = []
    for mode in ("normal", "shuffled", "zero", "reversed"):
        for seed in (0, 1):
            ppls = ppl_under_positions(model, tok, [d["text"] for d in docs], mode, seed)
            for d, p in zip(docs, ppls):
                rows.append({"part": "08a_position_ppl", "condition": mode, "seed": seed,
                             "doc_id": d["id"], "ppl": p})
        print(f"  08a {mode:9s} mean ppl {sum(r['ppl'] for r in rows if r['condition']==mode)/len(docs):8.2f}")

    # ---------- 08c: lexical unscrambling ----------
    rng = torch.Generator().manual_seed(0)
    sents = []
    for d in docs:
        s = [x for x in metrics.sentences(d["text"]) if 8 <= len(x.split()) <= 25]
        if s:
            sents.append(s[len(s) // 2])
    print(f"  08c using {len(sents)} sentences")
    samples = {}
    for frac in (0.0, 0.15, 0.30, 0.50):
        scr = [scramble(s, frac, rng) for s in sents]
        prompts = [
            render_chat(tok, "You restore word order.",
                        "Reorder these words into the original grammatical sentence. "
                        "Output only the sentence.\n\nWords: " + s)
            for s in scr
        ]
        out = generate(model, tok, prompts, max_new_tokens=60, processors=[Temperature(0.7), TopP(0.9)], seed=0)
        samples[f"shuffle_{int(frac*100)}pct"] = [
            {"intact": s0, "input": s1, "output": t} for s0, s1, t in zip(sents, scr, out.texts)
        ]
        for s0, s1, text in zip(sents, scr, out.texts):
            recall, precision = bag_recall(s0, text)
            rows.append({
                "part": "08c_unscramble", "condition": f"shuffle_{int(frac*100)}pct", "seed": 0,
                "doc_id": "", "ppl": float("nan"),
                "similarity": difflib.SequenceMatcher(None, s0, text).ratio(),
                "kendall_tau": kendall_tau(s0.split(), text.split()),
                "exact": float(text.strip().rstrip(".") == s0.strip().rstrip(".")),
                "scrambled_similarity": difflib.SequenceMatcher(None, s0, s1).ratio(),
                "word_recall": recall,
                "word_precision": precision,
            })
        sub = [r for r in rows if r["condition"] == f"shuffle_{int(frac*100)}pct"]
        k = len(sub)
        print(f"  08c shuffle {int(frac*100):2d}%: sim {sum(r['similarity'] for r in sub)/k:.3f} "
              f"tau {sum(r['kendall_tau'] for r in sub)/k:+.3f} "
              f"word_recall {sum(r['word_recall'] for r in sub)/k:.3f} "
              f"(input sim {sum(r['scrambled_similarity'] for r in sub)/k:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "position_free_raw.csv", index=False)
    a = df[df["part"] == "08a_position_ppl"].groupby("condition")["ppl"].mean().reset_index()
    c = df[df["part"] == "08c_unscramble"].groupby("condition")[["similarity", "kendall_tau", "exact", "scrambled_similarity"]].mean().reset_index()
    a.to_csv(OUT / "position_ppl_summary.csv", index=False)
    c.to_csv(OUT / "unscramble_summary.csv", index=False)
    (OUT / "position_free_summary.md").write_text(
        "# 08a position ablation (teacher-forced PPL)\n\n" + md_table(a.to_dict("records")) +
        "\n\n# 08c lexical unscrambling\n\n" + md_table(c.to_dict("records")) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(a["condition"], a["ppl"]); axes[0].set_title("08a PPL by position regime"); axes[0].set_yscale("log")
    axes[1].bar(c["condition"], c["kendall_tau"]); axes[1].set_title("08c Kendall tau (recovered order)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "position_free.png", dpi=130)
    print(a.to_string(index=False)); print(c.to_string(index=False))
    save_json(samples, OUT / "samples.json")
    save_json(env_fingerprint(), OUT / "env.json")
    print(f"saved -> {OUT}")
