"""Decisive E04 probe: is our stock-class path equivalent to the checkpoint's own code?

Compares the masked-token distribution at the first <|mask|> position under:
  A. AutoModelForMaskedLM + trust_remote_code (the README path, bidirectional by design)
  B. stock Qwen3ForCausalLM + 4-D bool all-ones mask
  C. stock Qwen3ForCausalLM + dict mask {"full_attention": bool all-ones}
  D. stock Qwen3ForCausalLM + additive float 0.0 mask
  E. stock Qwen3ForCausalLM + no mask (causal)
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

MID = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(snap)
M = tok.mask_token_id

pre = tok("The capital of France is", add_special_tokens=False)["input_ids"]
x = torch.tensor([pre + [M] * 6], device="cuda")
L = x.shape[1]
ones_bool = torch.ones((1, 1, L, L), dtype=torch.bool, device="cuda")
zeros_float = torch.zeros((1, 1, L, L), dtype=torch.bfloat16, device="cuda")
pos = len(pre)


def show(tag, logits):
    l = logits[0, pos].float()
    top = l.topk(5)
    txt = ", ".join(f"{tok.decode([int(i)])!r}({float(v):.1f})" for v, i in zip(top.values, top.indices))
    print(f"{tag:32s} {txt}")


# ---- A: the checkpoint's own code ----
try:
    mref = AutoModelForMaskedLM.from_pretrained(snap, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    with torch.no_grad():
        la = mref(input_ids=x).logits
    show("A remote_code (no mask)", la)
    with torch.no_grad():
        la2 = mref(input_ids=x, attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda")).logits
    print("   A vs A(2-D mask) maxdiff", float((la - la2).abs().max()))
    del mref
    torch.cuda.empty_cache()
except Exception as e:  # noqa: BLE001
    print("A failed:", type(e).__name__, str(e)[:300])
    la = None

# ---- B..E: stock class ----
raw = json.load(open(f"{snap}/config.json"))
cfg = Qwen3Config(**{k: v for k, v in raw.items() if k in set(Qwen3Config().to_dict())})
cfg._attn_implementation = "sdpa"
m = Qwen3ForCausalLM(cfg)
m.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)
m = m.to(torch.bfloat16).to("cuda").eval()

with torch.no_grad():
    lb = m(input_ids=x, attention_mask=ones_bool).logits
    lc = m(input_ids=x, attention_mask={"full_attention": ones_bool}).logits
    ld = m(input_ids=x, attention_mask=zeros_float).logits
    le = m(input_ids=x).logits

show("B 4-D bool ones", lb)
show("C dict full_attention bool", lc)
show("D additive float zeros", ld)
show("E no mask (causal)", le)
if la is not None:
    for tag, t in (("B", lb), ("C", lc), ("D", ld), ("E", le)):
        print(f"  maxdiff A vs {tag}: {float((la - t).abs().max()):.4f}")
print("\nprompt tokens:", tok.decode(pre), "| masks at", pos, "..", L - 1)
