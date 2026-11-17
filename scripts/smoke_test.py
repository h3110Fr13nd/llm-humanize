"""Harness smoke test: load, batch-generate with several decoders, print metrics.

Run: uv run python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humanize_lab import generate, render_chat
from humanize_lab.corpus import human_reference_dist, load_docs
from humanize_lab.metrics import summarize
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, get_tokenizer, load, pick_device
from humanize_lab.processors import Greedy, MinP, Processor, RepetitionPenalty, TopP, window_mask_fn

docs = load_docs()[:3]
human = human_reference_dist()
device = pick_device()
print(f"device={device}")
t0 = time.time()
tok, model = load(DEFAULT_MODEL)
print(f"loaded {DEFAULT_MODEL} in {time.time()-t0:.1f}s")

prompts = [render_chat(tok, "You are an editor. Rewrite the text in your own words.", d["text"]) for d in docs]

conditions = {
    "greedy": [Greedy()],
    "top_p_0.9": [TopP(0.9)],
    "repetition_penalty": [RepetitionPenalty(1.3), TopP(0.9)],
    "hybrid_tau0.5_x0.4": [__import__("humanize_lab.processors", fromlist=["HybridPostTopP"]).HybridPostTopP(0.5, 0.4)],
}

for name, procs in conditions.items():
    t0 = time.time()
    out = generate(model, tok, prompts, max_new_tokens=80, processors=procs, seed=0, capture_trace=True)
    dt = time.time() - t0
    print(f"\n=== {name}  ({dt:.1f}s, {80*len(prompts)/dt:.0f} tok/s) ===")
    for d, text in zip(docs, out.texts):
        m = summarize(text, d["text"], d["entities"], human)
        print(f"  [{d['id'][:2]}] {text[:160]!r}")
        print(
            f"     words={m['n_words']} d1={m['distinct_1']:.2f} copy={m['copy_4gram']:.2f} "
            f"ent={m['entity_recall']:.2f} boil={m['boilerplate']:.1f} burst={m['burstiness']:.2f} "
            f"jsd={m['style_jsd']:.3f}"
        )

# window mask sanity check: output must differ from full attention
w = window_mask_fn(16)
out_win = generate(model, tok, prompts[:1], max_new_tokens=40, processors=[TopP(0.9)], mask_fn=w, seed=1)
out_full = generate(model, tok, prompts[:1], max_new_tokens=40, processors=[TopP(0.9)], seed=1)
print("\nwindow-16 == full?", out_win.texts[0] == out_full.texts[0])
print("window :", out_win.texts[0][:120])
print("full   :", out_full.texts[0][:120])

# position ablation sanity check
out_pos = generate(model, tok, prompts[:1], max_new_tokens=40, processors=[TopP(0.9)], seed=1,
                   position_fn=lambda step, real, dev: (real * 0 + 7).to(dev))
print("\nfrozen-position output:", out_pos.texts[0][:120])
print("\nOK")
