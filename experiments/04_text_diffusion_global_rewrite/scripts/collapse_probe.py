"""E04b — why does the MDLM canvas collapse on some documents?

E04 found that `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` produces fluent text
for 11/12 documents but a pure comma run for `01_space_telescopes`, and that
the two documents which ever collapsed (01, 12) are also the two with the
longest prompts (319 / 293 tokens vs 217-260 for the rest).

Two competing explanations:

  H1  *length*: the failure is a function of prompt length, so it is a
      conditioning/positional limit of this checkpoint.
  H2  *canvas geometry*: it is a function of how the masked block is laid out,
      so a smaller block (fewer positions denoised jointly) fixes it.

The probe separates them:

  A. block sweep (8 / 16 / 32 / 64) at temperature 1.0 for docs 01 (collapses),
     12 (collapsed at T=0), 03 (never collapses, short prompt, control).
  B. prompt-truncation sweep for doc 01: keep only the last k prompt tokens
     (k = 100, 150, 200, 250) with a fixed block of 64. If truncation fixes it,
     H1 wins and the failure is a long-context conditioning limit.

Run: uv run python experiments/04_text_diffusion_global_rewrite/scripts/collapse_probe.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_NEUTRAL, USER_TEMPLATES, load_docs
from humanize_lab.diffusion import generate as diff_generate
from humanize_lab.diffusion import load_diffusion_model
from humanize_lab.models import BASE_MODEL, load
from humanize_lab.tracking import save_json

OUT = ROOT / "experiments" / "04_text_diffusion_global_rewrite" / "outputs"
DIFF_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
MAX_NEW = 64
FOCUS = ["01_space_telescopes", "12_calvin_cycle_photosynthesis", "03_quantum_computing"]
CUTS = [100, 150, 200, 250]


def prompt_ids(tok, text):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_NEUTRAL},
         {"role": "user", "content": USER_TEMPLATES["neutral"].format(text=text)}],
        add_generation_prompt=True, tokenize=True, enable_thinking=False,
    )["input_ids"]


def run(dtok, dmodel, ids, **kw):
    pt = torch.tensor([ids], dtype=torch.long, device=dmodel.device)
    pl = torch.tensor([len(ids)], dtype=torch.long, device=dmodel.device)
    torch.cuda.empty_cache()
    out, conf = diff_generate(dmodel, dtok, pt, pl, max_new_tokens=MAX_NEW,
                              capture_confidence=True, **kw)
    text = dtok.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text, conf


def main() -> None:
    docs = {d["id"]: d for d in load_docs()}
    dtok, dmodel = load_diffusion_model(DIFF_MODEL)
    print(f"loaded {DIFF_MODEL} ({torch.cuda.memory_allocated()/1e9:.2f} GB)")

    rows = []

    # ---------- A. block sweep ----------
    for doc_id in FOCUS:
        ids = prompt_ids(dtok, docs[doc_id]["text"])
        for block in (8, 16, 32, 64):
            if MAX_NEW % block:
                continue
            t0 = time.time()
            text, conf = run(dtok, dmodel, ids, block_size=block, steps=MAX_NEW,
                             temperature=1.0, remasking="low_confidence")
            rows.append({"probe": "block_sweep", "doc_id": doc_id, "prompt_tokens": len(ids),
                         "block": block, "cut": None, "confidence": conf,
                         "text": text, "seconds": round(time.time() - t0, 1)})
            print(f"  block {block:2d} {doc_id:34s} conf={conf:.3f} "
                  f"commas={text.count(',')} {text[:70]!r}", flush=True)

    # ---------- B. prompt truncation ----------
    full = prompt_ids(dtok, docs["01_space_telescopes"]["text"])
    for k in CUTS:
        ids = full[-k:]
        text, conf = run(dtok, dmodel, ids, block_size=64, steps=MAX_NEW,
                         temperature=1.0, remasking="low_confidence")
        rows.append({"probe": "truncation", "doc_id": "01_space_telescopes",
                     "prompt_tokens": len(ids), "block": 64, "cut": k,
                     "confidence": conf, "text": text, "seconds": None})
        print(f"  cut {k:3d} tokens conf={conf:.3f} commas={text.count(',')} {text[:70]!r}",
              flush=True)

    # Free the 0.6 B model before scoring: 4 GB cannot hold it plus the scorer.
    del dmodel
    torch.cuda.empty_cache()
    stok, scorer = load(BASE_MODEL)
    for r in rows:
        r["ppl"] = metrics.perplexity(scorer, stok, [r["text"]])[0]
        r["comma_run"] = r["text"].count(",") > 20
        r["n_words"] = len(r["text"].split())
    del scorer
    torch.cuda.empty_cache()

    save_json(rows, OUT / "collapse_probe.json")
    print("\n| probe | doc | prompt tok | block | cut | conf | commas | words | ppl |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['probe']} | {r['doc_id'][:20]} | {r['prompt_tokens']} | {r['block']} "
              f"| {r['cut'] or '-'} | {r['confidence']:.3f} | {r['text'].count(',')} "
              f"| {r['n_words']} | {r['ppl']:.1f} |")
    print(f"\nsaved -> {OUT / 'collapse_probe.json'}")


if __name__ == "__main__":
    main()
