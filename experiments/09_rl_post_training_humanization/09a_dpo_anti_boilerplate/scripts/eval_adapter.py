"""09a — re-evaluate the committed LoRA adapter and save the generated text.

The DPO run saved metrics but not the text, which makes its headline claim
("the adapter replaces the preamble with generic assistant prose") impossible to
check from the repository. This script loads `outputs/lora_adapter.pt`, applies
it to the same 12 documents at greedy decoding, and writes
`outputs/eval_samples.json` — so the adapter artifact is both verified and
inspectable.

It also re-derives the summary row for the adapted model, which is a cheap
regression check: if it does not match `dpo_eval_summary.md` (modulo sampling,
greedy is deterministic), the committed artifact is not the one that was
measured.

Run: uv run python experiments/09_rl_post_training_humanization/09a_dpo_anti_boilerplate/scripts/eval_adapter.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_NEUTRAL, USER_TEMPLATES, human_reference_dist, load_docs
from humanize_lab.models import BASE_MODEL, DEFAULT_MODEL, load, render_chat
from humanize_lab.processors import Greedy
from humanize_lab.sampling import generate
from humanize_lab.tracking import save_json
from humanize_lab.tuning import LoRAConfig, apply_lora

OUT = ROOT / "experiments" / "09_rl_post_training_humanization" / "09a_dpo_anti_boilerplate" / "outputs"
MAX_NEW = 100


def main() -> None:
    docs = load_docs()
    human = human_reference_dist()
    tok, model = load(DEFAULT_MODEL)
    stok, scorer = load(BASE_MODEL)

    ckpt = torch.load(OUT / "lora_adapter.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    handle = apply_lora(model, LoRAConfig(r=cfg["r"], alpha=cfg["alpha"], targets=tuple(cfg["targets"])))
    n_loaded = 0
    for module, saved in zip(handle.modules, ckpt["adapters"]):
        module.A.data.copy_(saved["A"].to(module.A.device, module.A.dtype))
        module.B.data.copy_(saved["B"].to(module.B.device, module.B.dtype))
        n_loaded += 1
    print(f"loaded {n_loaded} LoRA modules ({handle.n_params():,} params, "
          f"{100 * handle.n_params() / sum(p.numel() for p in model.parameters()):.2f}% of the model)")

    prompts = [render_chat(tok, SYSTEM_NEUTRAL, USER_TEMPLATES["neutral"].format(text=d["text"])) for d in docs]
    rows = {}
    for stage, enabled in (("before", False), ("after", True)):
        handle.set_enabled(enabled)
        out = generate(model, tok, prompts, max_new_tokens=MAX_NEW, processors=[Greedy()], capture_trace=True)
        ppls = metrics.perplexity(scorer, stok, out.texts)
        rows[stage] = []
        for d, text, ppl in zip(docs, out.texts, ppls):
            m = metrics.summarize(text, d["text"], d["entities"], human)
            rows[stage].append({"doc_id": d["id"], "ppl": ppl, "text": text, **m})
        df = pd.DataFrame(rows[stage])
        print(f"  {stage:7s} copy={df.copy_4gram.mean():.3f} ent={df.entity_recall.mean():.3f} "
              f"boil={df.boilerplate.mean():.2f} ppl={df.ppl.mean():.2f} words={df.n_words.mean():.1f}")

    save_json(rows, OUT / "eval_samples.json")

    print("\nfirst sentence of each 'after' sample (the claim to check):")
    for r in rows["after"][:6]:
        print(f"  {r['doc_id'][:26]:26s} {r['text'][:88]!r}")
    print(f"\nsaved -> {OUT / 'eval_samples.json'}")


if __name__ == "__main__":
    main()
