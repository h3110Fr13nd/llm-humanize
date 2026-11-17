"""Logits processors: the decoding zoo, batched.

Every processor is called as ``proc(scores [B,V] float32, input_ids [B,L], step)``
and returns modified scores. The sampler samples once from the result.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

NEG = float("-inf")


def _fill(scores: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~keep_mask, NEG)


class Processor:
    def __call__(self, scores, input_ids, step):  # pragma: no cover
        raise NotImplementedError


class Greedy(Processor):
    def __call__(self, scores, input_ids, step):
        idx = scores.argmax(-1, keepdim=True)
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, idx, True)
        return _fill(scores, keep)


class Temperature(Processor):
    def __init__(self, t: float = 1.0):
        self.t = max(t, 1e-5)

    def __call__(self, scores, input_ids, step):
        return scores / self.t


class TopP(Processor):
    def __init__(self, p: float = 0.9):
        self.p = p

    def __call__(self, scores, input_ids, step):
        probs = F.softmax(scores, dim=-1)
        s, idx = torch.sort(probs, descending=True, dim=-1)
        cum = s.cumsum(-1) - s  # mass strictly before each token
        keep_sorted = cum <= self.p
        keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
        return _fill(scores, keep)


class MinP(Processor):
    """min-p: relative probability floor against the top token."""

    def __init__(self, p: float = 0.05, temperature: float = 1.0):
        self.p = p
        self.temperature = temperature

    def __call__(self, scores, input_ids, step):
        s = scores / self.temperature
        probs = F.softmax(s, dim=-1)
        top = probs.max(-1, keepdim=True).values
        return s.masked_fill(probs < self.p * top, NEG)


class Typical(Processor):
    """Typical decoding: keep tokens whose surprisal is closest to the
    conditional entropy, up to mass ``p``."""

    def __init__(self, p: float = 0.9):
        self.p = p

    def __call__(self, scores, input_ids, step):
        logp = F.log_softmax(scores, dim=-1)
        probs = logp.exp()
        ent = -(probs * logp).sum(-1, keepdim=True)
        dev = (ent + logp).abs()  # |surprisal - entropy|
        order = torch.argsort(dev, dim=-1)
        sorted_p = probs.gather(-1, order)
        cum = sorted_p.cumsum(-1) - sorted_p
        keep_sorted = cum <= self.p
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        return _fill(scores, keep)


class Eta(Processor):
    """eta-sampling (Hewitt et al. 2022, 'Truncation Sampling as LM
    Desensitization'): drop tokens with p_i < min(eta, sqrt(eta) * exp(-H))."""

    def __init__(self, eta: float = 1e-4):
        self.eta = eta

    def __call__(self, scores, input_ids, step):
        logp = F.log_softmax(scores, dim=-1)
        probs = logp.exp()
        ent = -(probs * logp).sum(-1, keepdim=True)  # nats
        thresh = torch.minimum(
            torch.full_like(ent, self.eta),
            torch.sqrt(torch.tensor(self.eta, device=scores.device)) * torch.exp(-ent),
        )
        keep = probs >= thresh
        keep[:, 0] = True
        return _fill(scores, keep)


class TopH(Processor):
    """top-h: smallest head whose cumulative entropy stays <= H nats."""

    def __init__(self, H: float = 2.0):
        self.H = H

    def __call__(self, scores, input_ids, step):
        logp = F.log_softmax(scores, dim=-1)
        probs = logp.exp()
        s, idx = torch.sort(probs, descending=True, dim=-1)
        sl = torch.sort(logp, descending=True, dim=-1).values
        ent_cum = -(s * sl).cumsum(-1)
        # keep the leading run that satisfies the budget, always keep >= 1
        keep_sorted = ent_cum <= self.H
        keep_sorted[:, 0] = True
        keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
        return _fill(scores, keep)


class MirostatV2(Processor):
    """Mirostat v2: feedback-controlled surprise target tau."""

    def __init__(self, tau: float = 3.0, eta: float = 0.1):
        self.tau = tau
        self.eta = eta
        self.mu: torch.Tensor | None = None

    def __call__(self, scores, input_ids, step):
        B = scores.shape[0]
        if self.mu is None:
            self.mu = torch.full((B, 1), 2 * self.tau, device=scores.device)
        logp = F.log_softmax(scores, dim=-1)
        surprisal = -logp / math.log(2)
        keep = surprisal <= self.mu
        keep[:, 0] = keep[:, 0] | (keep.sum(-1) == 0)
        return _fill(scores, keep)


class DRY(Processor):
    """DRY: exponential penalty on repeated suffixes of length 1..max_n."""

    def __init__(self, multiplier: float = 0.8, base: float = 1.75, allowed: int = 2, max_n: int = 4):
        self.multiplier = multiplier
        self.base = base
        self.allowed = allowed
        self.max_n = max_n

    def __call__(self, scores, input_ids, step):
        out = scores.clone()
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            L = len(seq)
            for n in range(1, self.max_n + 1):
                if L <= n:
                    continue
                suffix = tuple(seq[-n:])
                count = sum(1 for i in range(L - n) if tuple(seq[i : i + n]) == suffix)
                if count > self.allowed:
                    pen = self.multiplier * (self.base ** (count - self.allowed))
                    out[b, seq[-1]] -= math.log(max(pen, 1.0))
        return out


class RepetitionPenalty(Processor):
    def __init__(self, penalty: float = 1.15, window: int = 256):
        self.penalty = penalty
        self.window = window

    def __call__(self, scores, input_ids, step):
        out = scores.clone()
        for b in range(input_ids.shape[0]):
            ids = input_ids[b, -self.window :]
            vals = out[b, ids]
            out[b, ids] = torch.where(vals < 0, vals * self.penalty, vals / self.penalty)
        return out


class BanPhrases(Processor):
    """Ban the token that would complete any forbidden token sequence."""

    def __init__(self, phrases: list[tuple[int, ...]]):
        self.phrases = [tuple(p) for p in phrases if len(p) > 0]

    def __call__(self, scores, input_ids, step):
        out = scores.clone()
        for b in range(input_ids.shape[0]):
            tail = input_ids[b].tolist()
            for ph in self.phrases:
                k = len(ph)
                if k > 1 and tuple(tail[-(k - 1) :]) == ph[:-1]:
                    out[b, ph[-1]] = NEG
        return out


class AdaptiveTemperature(Processor):
    """Row-wise temperature chosen by the model's own confidence.

    E12 showed head escape is only reachable at temperatures that destroy the
    text, and that copying is decided at *confident* steps. This applies the
    heat only there: rows whose top-1 probability exceeds ``conf`` get
    ``high`` temperature, the rest keep ``base``.

    ``invert=True`` applies the heat to the *uncertain* rows instead — the
    control that shows the direction of the effect matters.
    """

    def __init__(self, base: float = 0.7, high: float = 1.5, conf: float = 0.6, invert: bool = False):
        self.base = base
        self.high = high
        self.conf = conf
        self.invert = invert
        self.hot_steps = 0
        self.total_steps = 0

    def __call__(self, scores, input_ids, step):
        p = F.softmax(scores / self.base, dim=-1)
        top1 = p.max(-1, keepdim=True).values
        hot = (top1 <= self.conf) if self.invert else (top1 > self.conf)
        t = torch.where(hot, torch.full_like(top1, self.high), torch.full_like(top1, self.base))
        self.hot_steps += int(hot.sum().item())
        self.total_steps += scores.shape[0]
        return scores / t

    def stats(self) -> dict:
        n = max(self.total_steps, 1)
        return {"hot_frac": self.hot_steps / n}


class HybridPostTopP(Processor):
    """Experiment 02/03 routing: greedy above tau, post-nucleus below."""

    def __init__(self, tau: float = 0.5, x: float = 0.4, record: bool = False):
        self.tau = tau
        self.x = x
        self.record = record
        self.decisions: list[list[bool]] = []

    def __call__(self, scores, input_ids, step):
        probs = F.softmax(scores, dim=-1)
        s, idx = torch.sort(probs, descending=True, dim=-1)
        greedy = s[:, 0] >= self.tau
        if self.record:
            self.decisions.append(greedy.tolist())
        cum = s.cumsum(-1)
        first = (cum > self.x).to(torch.int64).argmax(-1)
        post = torch.clamp(first + 1, max=scores.shape[-1] - 1)
        target = torch.where(greedy, torch.zeros_like(post), post)
        chosen = idx.gather(-1, target.unsqueeze(-1))
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, chosen, True)
        return _fill(scores, keep)


def window_mask_fn(window: int, source_full: bool = True, mask_source: bool = False):
    """4-D mask callable for :func:`humanize_lab.sampling.generate`.

    ``source_full=True`` is Experiment 05's design: every generated token sees
    the *whole* prompt prefix (uncompressed source attention) and only the last
    ``window`` generated tokens of its own stream.
    """

    def mask_fn(step, q_len, kv_len, real_lens, device):
        B = real_lens.shape[0]
        Lmax = int(real_lens.max().item())
        kv = torch.arange(kv_len, device=device).view(1, -1).expand(B, -1)
        prompt_pad = (Lmax - real_lens).view(B, 1)
        in_src = kv < Lmax
        kv_content = torch.where(in_src, kv - prompt_pad, real_lens.view(B, 1) + (kv - Lmax))
        valid = kv_content >= 0
        if q_len > 1:  # prompt pass: plain causal
            q_idx = torch.arange(q_len, device=device).view(-1, 1)
            keep = q_idx >= kv.view(B, 1, kv_len)
            keep = keep & valid.view(B, 1, kv_len)
            return keep.view(B, 1, q_len, kv_len)
        q_content = (real_lens + step).view(B, 1)
        keep = (in_src & source_full) | (kv_content > (q_content - window))
        if mask_source:
            keep = ~in_src & (kv_content > (q_content - window))
        keep = keep & valid
        return keep.view(B, 1, 1, kv_len)

    return mask_fn
