"""Experiment 11 — entity-pinned head escape.

Spawned by E10: every standard truncation family keeps the model inside the
top-10 head (GLTR 0.98-1.00 vs 0.80 for human text), while the only decoder
that leaves the head (repetition penalty) destroys entity recall.

Idea: instead of gating on *confidence* (E02/E03) gate on *token class*.

* fact tokens (capitalised words, numbers) that occur in the source -> force
  greedy, i.e. copy the anchor exactly;
* every other token -> sample from a mixture of the nucleus and the
  post-nucleus boundary token (Experiment 01's mechanism), controlled by q.

The mixture is implemented at the distribution level (log-softmax + log q),
so it stays deterministic given the sampler seed and shows up correctly in the
GLTR rank trace.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .processors import NEG, Processor


def fact_token_ids(tokenizer) -> set[int]:
    """Vocab ids that look like fact carriers: capitalised words or digits."""
    out = set()
    for tid in range(len(tokenizer)):
        s = tokenizer.convert_ids_to_tokens(tid)
        if not isinstance(s, str):
            continue
        t = s.strip().lstrip("▁Ġ##").strip()
        if not t:
            continue
        has_digit = any(ch.isdigit() for ch in t)
        cap = t[0].isupper() and t.isalpha() and len(t) > 1
        if has_digit or cap:
            out.add(tid)
    return out


class EntityPinnedRouter(Processor):
    """Copy fact tokens, head-escape everything else."""

    def __init__(
        self,
        fact_ids: set[int],
        top_p: float = 0.9,
        boundary_x: float = 0.6,
        post_prob: float = 0.5,
        temperature: float = 0.8,
    ):
        self.fact_ids = fact_ids
        self.top_p = top_p
        self.boundary_x = boundary_x
        self.post_prob = post_prob
        self.temperature = temperature
        self.greedy_steps = 0
        self.total_steps = 0
        self.post_steps = 0

    def __call__(self, scores, input_ids, step):
        s = scores / self.temperature
        B, V = s.shape
        logp = F.log_softmax(s, dim=-1)
        probs = logp.exp()
        top1 = s.argmax(-1)
        out = torch.full_like(s, NEG)
        # per-row source vocabulary (the prompt is part of input_ids); recomputed
        # every step so each document is gated against its own source only.
        src_sets = [set(row.tolist()) for row in input_ids]

        # nucleus (standard top-p on the temperature-scaled distribution)
        sp, idx = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(-1) - sp
        nucleus_sorted = cum <= self.top_p

        for b in range(B):
            t1 = int(top1[b])
            fact = t1 in self.fact_ids and t1 in src_sets[b]
            if fact:
                # anchors are copied verbatim
                out[b, t1] = 0.0
                self.greedy_steps += 1
            else:
                keep_ids = idx[b][nucleus_sorted[b]]
                # boundary token: first token outside the boundary_x nucleus
                cum_b = torch.cumsum(sp[b], -1)
                first = int((cum_b > self.boundary_x).to(torch.int64).argmax())
                post_id = int(idx[b][min(first + 1, V - 1)])
                q = self.post_prob
                out[b, keep_ids] = logp[b, keep_ids] + torch.log(torch.tensor(max(1 - q, 1e-6), device=s.device))
                if q > 0:
                    out[b, post_id] = torch.logaddexp(
                        out[b, post_id], logp[b, post_id] + torch.log(torch.tensor(q, device=s.device))
                    )
                    self.post_steps += 1
            self.total_steps += 1
        return out

    def stats(self) -> dict:
        t = max(self.total_steps, 1)
        return {"greedy_frac": self.greedy_steps / t, "post_frac": self.post_steps / t}


class PinnedAdaptive(Processor):
    """E11 x E13: copy the anchors, apply entropy-adaptive heat to the rest.

    E11 showed pinning fact tokens recovers entity recall but leaves head
    conformity at 0.99. E13 showed heating only confident steps reaches the
    human GLTR profile but destroys recall. This combines them: a fact-token
    top-1 that occurs in the source is copied greedily (no heat, no sampling),
    and every other position gets the adaptive temperature.
    """

    def __init__(
        self,
        fact_ids: set[int],
        base: float = 0.7,
        high: float = 1.3,
        conf: float = 0.6,
        pin: bool = True,
    ):
        self.fact_ids = fact_ids
        self.base = base
        self.high = high
        self.conf = conf
        self.pin = pin
        self.pinned = 0
        self.hot = 0
        self.total = 0

    def __call__(self, scores, input_ids, step):
        B = scores.shape[0]
        probe = F.softmax(scores / self.base, dim=-1)
        top1_p, top1 = probe.max(-1)
        src_sets = [set(row.tolist()) for row in input_ids] if self.pin else None
        out = torch.empty_like(scores)
        for b in range(B):
            t1 = int(top1[b])
            is_fact = self.pin and t1 in self.fact_ids and t1 in src_sets[b]
            if is_fact:
                out[b] = NEG
                out[b, t1] = 0.0
                self.pinned += 1
            else:
                hot = float(top1_p[b]) > self.conf
                out[b] = scores[b] / (self.high if hot else self.base)
                self.hot += int(hot)
            self.total += 1
        return out

    def stats(self) -> dict:
        n = max(self.total, 1)
        return {"pinned_frac": self.pinned / n, "hot_frac": self.hot / n}


class BanFirst(Processor):
    """Mask a set of ids at step 0 only (chat preamble suppression)."""

    def __init__(self, ids: set[int]):
        self.ids = ids

    def __call__(self, scores, input_ids, step):
        if step != 0:
            return scores
        out = scores.clone()
        for i in self.ids:
            out[:, i] = NEG
        return out


PREAMBLE_PHRASES = [
    "Here is", "Here's", "Here are", "Below is", "Sure,", "Certainly", "Of course",
    "The following", "This is a", "In summary", "In conclusion", "As an AI",
    "**", "###", "Step 1", "Step 2", "Rewritten", "Rewritten text",
]


def preamble_ban(tokenizer, phrases: list[str] | None = None):
    """Return (BanFirst, BanPhrases) processors covering the chat preamble."""
    from .processors import BanPhrases

    phrases = phrases or PREAMBLE_PHRASES
    seqs: list[tuple[int, ...]] = []
    first_ids: set[int] = set()
    for p in phrases:
        for variant in (p, " " + p):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            seqs.append(tuple(ids))
            first_ids.add(ids[0])
    return BanFirst(first_ids), BanPhrases(seqs)
