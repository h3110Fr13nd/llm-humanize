"""Batched manual generation loop.

Why not ``model.generate``? Because every experiment here needs at least one
of: custom 4-D attention masks, custom position ids, per-step distribution
traces, or sampling from a hand-built distribution. This loop gives all four
while keeping the KV cache (so it is as fast as ``generate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .models import encode_prompts, new_cache, pick_device, stop_token_ids


@dataclass
class GenOut:
    texts: list[str]
    token_ids: list[list[int]]
    prompt_len: int
    trace: dict[str, torch.Tensor] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 120,
    processors: list | None = None,
    mask_fn=None,
    position_fn=None,
    seed: int = 0,
    stop_strings: list[str] | None = None,
    capture_trace: bool = False,
    greedy_if_processor: bool = True,
) -> GenOut:
    """Sample a batch of completions.

    ``processors`` transform scores before sampling. If the last processor is
    deterministic (Greedy/HybridPostTopP) sampling is a no-op argmax, which is
    still routed through the same loop so traces stay comparable.
    """
    device = pick_device()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    ids, am, _ = encode_prompts(tokenizer, prompts, device=device)
    B, L = ids.shape
    real_lens = am.sum(-1)  # (B,)
    eos_ids = stop_token_ids(tokenizer)

    cache = new_cache(model)
    gen = torch.Generator(device=device).manual_seed(seed)
    steps_done = 0
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    trace = {"top1_prob": [], "entropy": [], "chosen_prob": [], "chosen_rank": [], "chosen_logprob": []}
    all_ids = ids

    for step in range(max_new_tokens):
        q_len = L if step == 0 else 1
        inp = all_ids if step == 0 else all_ids[:, -1:]
        if step == 0:
            pos = (am.cumsum(-1) - 1).clamp(min=0)
        elif position_fn is not None:
            pos = position_fn(step, real_lens, device).view(B, 1)
        else:
            pos = (real_lens + step).view(B, 1)

        if mask_fn is not None:
            attn = mask_fn(step, q_len, all_ids.shape[1], real_lens, device)
        else:
            attn = torch.cat([am, torch.ones((B, step), dtype=am.dtype, device=device)], dim=1)

        out = model(
            input_ids=inp,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        raw_scores = out.logits[:, -1, :].float()
        scores = raw_scores
        for proc in processors or []:
            scores = proc(scores, all_ids, step)

        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        if probs.sum(-1).min() <= 0:
            probs = torch.where(probs.sum(-1, keepdim=True) > 0, probs, torch.ones_like(probs) / probs.shape[-1])

        next_tok = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        if capture_trace:
            # GLTR rank must be measured in the ORIGINAL model distribution, not
            # in the truncated one (where the chosen token is always rank 0).
            logp = F.log_softmax(raw_scores, dim=-1)
            p_raw = logp.exp()
            ent = -(p_raw * logp).sum(-1)
            tok_lp = logp.gather(-1, next_tok.unsqueeze(-1))
            trace["top1_prob"].append(p_raw.max(-1).values)
            trace["entropy"].append(ent)
            trace["chosen_prob"].append(tok_lp.exp().squeeze(-1))
            trace["chosen_logprob"].append(tok_lp.squeeze(-1))
            rank = (raw_scores > raw_scores.gather(-1, next_tok.unsqueeze(-1))).sum(-1)
            trace["chosen_rank"].append(rank)

        next_tok = torch.where(finished, torch.full_like(next_tok, tokenizer.eos_token_id), next_tok)
        for b in range(B):
            if int(next_tok[b]) in eos_ids:
                finished[b] = True
        all_ids = torch.cat([all_ids, next_tok.unsqueeze(-1)], dim=1)
        steps_done += 1
        if bool(finished.all()):
            break

    texts, kept = [], []
    for b in range(B):
        seq = all_ids[b, L:].tolist()
        cut = len(seq)
        for i, t in enumerate(seq):
            if t in eos_ids:
                cut = i
                break
        seq = seq[:cut]
        texts.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
        kept.append(seq)
    if stop_strings:
        cleaned = []
        for t in texts:
            for s in stop_strings:
                idx = t.find(s)
                if idx >= 0:
                    t = t[:idx]
            cleaned.append(t.strip())
        texts = cleaned

    tr = {}
    if capture_trace and trace["top1_prob"]:
        tr = {k: torch.stack(v) for k, v in trace.items()}
    return GenOut(texts=texts, token_ids=kept, prompt_len=L, trace=tr)
