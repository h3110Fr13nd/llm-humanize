"""Masked-diffusion LM sampling (adapted from the dLLM model card).

Used by Experiment 04 to compare a *real* discrete diffusion LM
(`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`, MDLM, 0.6 B) against the AR
SmolLM-135M on the same rewrite task.

``block_size`` controls how much of the response is denoised jointly:
block_size == max_new_tokens is a single global canvas; smaller blocks make the
process semi-autoregressive.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling, in float32.

    The upstream reference casts to float64 "for stability". On this machine a
    float64 copy of the logits at (batch 2, 224 tokens, 151 k vocab) costs
    545 MB *per tensor* and there are two of them, i.e. 1.09 GB of transient
    VRAM — which is exactly the `Tried to allocate 888.00 MiB` OOM the
    diffusion arm hit on the 4 GB card. float32 has plenty of mantissa for
    exp()/log() over logits in [-30, 30] and we argmax immediately, so the
    precision argument does not apply here.
    """
    if temperature == 0:
        return logits
    logits = logits.float()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, : int(remainder[i].item())] += 1
    return num


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_size: int = 128,
    steps: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    capture_confidence: bool = False,
    bidirectional: bool = True,
):
    """Return (token ids for the response region, mean unmask confidence).

    ``bidirectional=True`` passes a 4-D all-ones mask so every position can
    attend to every other position. This is **required** for masked diffusion:
    a stock causal mask means a masked position cannot see the future it is
    supposed to be denoising alongside, and the output degenerates (observed:
    10-word outputs, boilerplate 1067/100w, PPL 6.8e7).
    """
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch_size = prompt.size(0)
    total_length = int(prompt_lens.max().item() + max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=model.device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt[i, :length]
        x[i, length : length + max_new_tokens] = mask_id

    prompt_index = torch.arange(total_length, device=x.device).unsqueeze(0) < prompt_lens.unsqueeze(1)
    positions = torch.arange(total_length, device=x.device)

    def full_mask(n_rows: int) -> torch.Tensor:
        ones = torch.ones((n_rows, 1, total_length, total_length), dtype=torch.bool, device=x.device)
        return ones

    assert max_new_tokens % block_size == 0, "max_new_tokens must be a multiple of block_size"
    num_blocks = max_new_tokens // block_size
    steps_per_block = max(steps // num_blocks, 1)

    confidences: list[torch.Tensor] = []
    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size
        init_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer_tokens = get_num_transfer_tokens(init_mask, steps_per_block)

        for i in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                both = torch.cat([x, un_x], dim=0)
                kw = {"attention_mask": full_mask(both.shape[0])} if bidirectional else {}
                out = model(both, **kw).logits
                logits, un_logits = torch.chunk(out, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                kw = {"attention_mask": full_mask(x.shape[0])} if bidirectional else {}
                logits = model(x, **kw).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                # Probability of the chosen token without materialising the full
                # (batch, seq, 151k) softmax: p = exp(z_k - logsumexp(z)).
                # At batch 2 / 224 tokens the softmax tensor alone is 272 MB,
                # which the 4 GB card cannot spare.
                z = logits.float()
                x0_lp = torch.gather(z, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p = (x0_lp - torch.logsumexp(z, dim=-1)).exp()
                del z
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise NotImplementedError(remasking)

            confidence = torch.where(block_mask, x0_p, torch.full_like(x0_p, -np.inf))
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k == 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            if capture_confidence:
                confidences.append(confidence.masked_fill(~transfer_index, float("nan")))
            x[transfer_index] = x0[transfer_index]

    out = []
    for i in range(batch_size):
        n = int(prompt_lens[i].item())
        out.append(x[i, n : n + max_new_tokens].tolist())
    conf = None
    if capture_confidence and confidences:
        stacked = torch.stack(confidences)
        conf = float(stacked[~torch.isnan(stacked)].mean().item()) if (~torch.isnan(stacked)).any() else 0.0
    return out, conf


def load_diffusion_model(model_id: str = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", dtype=torch.bfloat16):
    """Load the MDLM checkpoint without the ``dllm`` package.

    The published checkpoint ships custom modelling code that ``import dllm``,
    which is not on PyPI. But ``A2DQwen3Model`` merely subclasses
    ``transformers.Qwen3Model`` (MDLM keeps the transformer and changes only
    the *training objective*), and the state dict keys are stock Qwen3 keys —
    verified here: 0 missing / 0 unexpected tensors. So the weights load into
    the ordinary ``Qwen3ForCausalLM`` and masked-diffusion sampling works on
    top of it. No remote code, no extra dependency.
    """
    import glob
    import json
    from pathlib import Path

    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    snap = glob.glob(
        str(Path.home() / ".cache/huggingface/hub/models--" / model_id.replace("/", "--") / "snapshots/*")
    )
    if not snap:
        from huggingface_hub import snapshot_download

        snap = [snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])]
    snap = snap[0]

    raw = json.load(open(f"{snap}/config.json"))
    valid = set(Qwen3Config().to_dict().keys())
    cfg = Qwen3Config(**{k: v for k, v in raw.items() if k in valid})
    model = Qwen3ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.to(dtype).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model
