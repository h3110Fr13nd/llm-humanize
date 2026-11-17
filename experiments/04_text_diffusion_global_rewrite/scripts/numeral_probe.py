"""E04c — is the MDLM canvas collapse a *numeral* attractor?

E04b ruled out the two obvious explanations. Block size does not matter
(doc 01 collapses at block 8 as well as 64) and prompt truncation does not
matter (doc 01 still collapses at a 100-token prompt). What the probe output
does show is revealing: the degenerating canvas emits *digits*

    1,2,3,4,5,6,7,8,9,1,1,2,3,4,5,6,7,8,9,1,1,2,...

and the two documents that ever collapsed (`01_space_telescopes`,
`12_calvin_cycle_photosynthesis`) are the number-dense ones: "2.4-meter",
"2.2 micrometers", "540 kilometers", "April 24, 1990", "December 1993",
"STS-61", "3,000 ancient galaxies" / "CO2", "ATP", "NADPH".

Hypothesis H3: the checkpoint never learned to denoise a comma-separated
digit list, so a *few* numeral tokens in the conditioning act as a seed for a
comma+digit attractor that overwrites the whole canvas.

Test: rewrite doc 01's numerals as words, keep everything else, and generate.
If it recovers, the mitigation for the pipeline is "verbalise numerals before
diffusion, restore digits afterwards". A reverse control injects numerals into
doc 03 (which never collapses) to check the causal direction.

Run: uv run python experiments/04_text_diffusion_global_rewrite/scripts/numeral_probe.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch

from humanize_lab import metrics
from humanize_lab.corpus import SYSTEM_NEUTRAL, USER_TEMPLATES, load_docs
from humanize_lab.diffusion import generate as diff_generate
from humanize_lab.diffusion import load_diffusion_model
from humanize_lab.models import BASE_MODEL, load
from humanize_lab.textfix import verbalise_numerals
from humanize_lab.tracking import save_json

OUT = ROOT / "experiments" / "04_text_diffusion_global_rewrite" / "outputs"
DIFF_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
MAX_NEW = 64

WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def verbalise(text: str) -> str:
    """Kept for the probe's own transcripts; the harness version lives in textfix."""
    return verbalise_numerals(text)


def prompt_ids(tok, text):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_NEUTRAL},
         {"role": "user", "content": USER_TEMPLATES["neutral"].format(text=text)}],
        add_generation_prompt=True, tokenize=True, enable_thinking=False,
    )["input_ids"]


def main() -> None:
    docs = {d["id"]: d for d in load_docs()}
    d01 = docs["01_space_telescopes"]["text"]
    d03 = docs["03_quantum_computing"]["text"]

    cases = [
        ("01 control (raw numerals)", d01),
        ("01 numerals verbalised", verbalise(d01)),
        ("01 numerals deleted", re.sub(r"\d+(?:[.,]\d+)*", "", d01)),
        ("03 control (few numerals)", d03),
        ("03 numerals injected", d03 + " Values: 1,2,3,4,5,6,7,8,9,10 and 2.4-meter, 540 kilometers, 1990."),
    ]
    print(f"doc 01 raw digit count: {len(re.findall(r'[0-9]', d01))}, "
          f"doc 03: {len(re.findall(r'[0-9]', d03))}")

    dtok, dmodel = load_diffusion_model(DIFF_MODEL)
    rows = []
    for name, text in cases:
        ids = prompt_ids(dtok, text)
        pt = torch.tensor([ids], dtype=torch.long, device=dmodel.device)
        pl = torch.tensor([len(ids)], dtype=torch.long, device=dmodel.device)
        torch.cuda.empty_cache()
        out, conf = diff_generate(dmodel, dtok, pt, pl, max_new_tokens=MAX_NEW,
                                  block_size=MAX_NEW, steps=MAX_NEW, temperature=1.0,
                                  capture_confidence=True)
        gen = dtok.batch_decode(out, skip_special_tokens=True)[0].strip()
        rows.append({"case": name, "prompt_tokens": len(ids), "confidence": conf,
                     "text": gen, "commas": gen.count(","),
                     "digits": len(re.findall(r"\d", gen))})
        print(f"  {name:32s} conf={conf:.3f} commas={gen.count(','):2d} "
              f"digits={len(re.findall(r'[0-9]', gen)):2d} {gen[:80]!r}", flush=True)

    del dmodel
    torch.cuda.empty_cache()
    stok, scorer = load(BASE_MODEL)
    for r in rows:
        r["ppl"] = metrics.perplexity(scorer, stok, [r["text"]])[0]
        r["n_words"] = len(r["text"].split())
        r["comma_run"] = r["commas"] > 20
    del scorer

    save_json(rows, OUT / "numeral_probe.json")
    print("\n| case | prompt tok | conf | commas | digits | words | ppl |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | {r['prompt_tokens']} | {r['confidence']:.3f} | {r['commas']} "
              f"| {r['digits']} | {r['n_words']} | {r['ppl']:.1f} |")
    print(f"\nsaved -> {OUT / 'numeral_probe.json'}")


if __name__ == "__main__":
    main()
