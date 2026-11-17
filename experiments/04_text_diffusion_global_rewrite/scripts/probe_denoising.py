"""E04 probe 2 — is the MDLM checkpoint broken, or just cold-starting badly?

probe_mask_semantics.py showed that at a fully masked 6-token suffix the top-5
are generic tokens (' a', ',', ' and') under every 4-D mask format, while causal
attention predicts 'Paris'. Two competing explanations:

  H1 alignment   the model knows the content but places it at a different offset
                 inside the masked span;
  H2 cold start  a fully masked suffix is out of distribution for this
                 checkpoint, so the first denoising step is noise.

Tests
  T1  mask a single known token *mid-sequence* (well-posed for diffusion):
      if bidirectional recovers it here, the mask plumbing is correct.
  T2  per-position top-1 across the whole masked suffix: is 'Paris' anywhere?
  T3  a 6-step greedy unmask loop: does a single unlucky step self-correct?
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(snap)
M = tok.mask_token_id

raw = json.load(open(f"{snap}/config.json"))
cfg = Qwen3Config(**{k: v for k, v in raw.items() if k in set(Qwen3Config().to_dict())})
model = Qwen3ForCausalLM(cfg)
missing, unexpected = model.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)
assert not missing and not unexpected, (missing[:3], unexpected[:3])
model = model.to(torch.bfloat16).to("cuda").eval()


def fwd(ids: list[int], bidirectional: bool = True) -> torch.Tensor:
    x = torch.tensor([ids], device="cuda")
    L = x.shape[1]
    kw = {}
    if bidirectional:
        kw["attention_mask"] = torch.ones((1, 1, L, L), dtype=torch.bool, device="cuda")
    with torch.no_grad():
        return model(input_ids=x, **kw).logits[0].float()


def top5(logits: torch.Tensor) -> str:
    v, i = logits.topk(5)
    return ", ".join(f"{tok.decode([int(j)])!r}({float(s):.1f})" for s, j in zip(v, i))


# ------------------------------------------------------------------ T1
text = "The Eiffel Tower is located in the city of Paris and it was completed in 1889."
ids = tok(text, add_special_tokens=False)["input_ids"]
pos = next(i for i, t in enumerate(ids) if "Paris" in tok.decode([t]))
masked = list(ids)
masked[pos] = M
print(f"T1 mid-sequence mask at index {pos} of {len(ids)} ({tok.decode([ids[pos]])!r})")
print("   bidirectional:", top5(fwd(masked, True)[pos]))
print("   causal       :", top5(fwd(masked, False)[pos]))

# ------------------------------------------------------------------ T2
pre = tok("The capital of France is", add_special_tokens=False)["input_ids"]
suffix = list(pre) + [M] * 6
lo = fwd(suffix, True)
print("\nT2 per-position top-1 across a fully masked 6-token suffix")
for p in range(len(pre), len(suffix)):
    v, i = lo[p].topk(1)
    print(f"   pos {p}: {tok.decode([int(i)])!r}  p={float(F.softmax(lo[p], -1).max()):.3f}")

# ------------------------------------------------------------------ T3
print("\nT3 greedy unmask loop (1 position per step, bidirectional)")
cur = list(suffix)
order = []
for step in range(6):
    lg = fwd(cur, True)
    cand = torch.full((len(cur),), -1e9, device="cuda")
    for p in range(len(pre), len(cur)):
        if cur[p] == M:
            cand[p] = float(F.softmax(lg[p], -1).max())
    p = int(cand.argmax())
    t = int(lg[p].argmax())
    cur[p] = t
    order.append((p, tok.decode([t])))
    print(f"   step {step}: pos {p} <- {tok.decode([t])!r}")
print("   final:", repr(tok.decode([t for _, t in [(0, x) for x in []]] or cur[len(pre):], skip_special_tokens=True)))
