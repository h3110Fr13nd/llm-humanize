"""Text metrics for humanization research — no external NLP dependencies.

All metrics are deliberately cheap so that they can be recomputed for every
generation in every experiment on a laptop.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import torch
import torch.nn.functional as F

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")

FUNCTION_WORDS = set(
    """a an the and or but if because while of to in on at by for with from as is are was were be been
    being it its this that these those he she they them his her their we our you your i my me not no
    than then so such into over under between during after before about more most other some any""".split()
)

BOILERPLATE_MARKERS = [
    "here is", "here's", "below is", "the following is", "in summary", "in conclusion",
    "to summarize", "this summary", "rewritten", "rewrite", "as an ai", "certainly",
    "sure,", "of course", "note that", "overall,", "key points", "takeaway",
    "**", "###", "step 1", "step 4", "let me", "i hope this",
]

HEDGES = ["it is important", "it's important", "it should be noted", "arguably",
          "in many ways", "somewhat", "relatively", "essentially", "notably", "significantly"]


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def distinct_n(text: str, n: int) -> float:
    w = words(text)
    if len(w) < n:
        return 1.0
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(text: str, n: int = 4) -> float:
    return 1.0 - distinct_n(text, n)


def ttr(text: str) -> float:
    w = words(text)
    return len(set(w)) / max(len(w), 1)


def mtld(text: str, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy 2005), forward pass only."""
    w = words(text)
    if len(w) < 10:
        return 0.0
    factors, types, tokens = 0.0, set(), 0
    for tok in w:
        tokens += 1
        types.add(tok)
        if len(types) / tokens <= threshold:
            factors += 1
            types, tokens = set(), 0
    if tokens > 0:
        factors += (1 - (len(types) / tokens)) / (1 - threshold)
    return len(w) / max(factors, 1e-6)


def ngrams(w: list[str], n: int) -> list[tuple]:
    return [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]


def _ngrams(w: list[str], n: int) -> list[tuple]:
    return [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]


def copy_overlap(generated: str, source: str, n: int = 4) -> float:
    """Share of generated n-grams that occur verbatim in the source."""
    gw = words(generated)
    sw = words(source)
    g = _ngrams(gw, n)
    if not g:
        return 0.0
    src = set(_ngrams(sw, n))
    return sum(1 for x in g if x in src) / len(g)


def entity_recall(text: str, entities: list[str]) -> float:
    if not entities:
        return 0.0
    low = text.lower()
    return sum(1 for e in entities if e.lower() in low) / len(entities)


def boilerplate_rate(text: str) -> float:
    """Marker hits per 100 words."""
    low = text.lower()
    hits = sum(low.count(m) for m in BOILERPLATE_MARKERS)
    return 100.0 * hits / max(len(words(text)), 1)


def hedge_rate(text: str) -> float:
    low = text.lower()
    return 100.0 * sum(low.count(h) for h in HEDGES) / max(len(words(text)), 1)


def sentence_stats(text: str) -> dict[str, float]:
    lens = [len(words(s)) for s in sentences(text) if words(s)]
    if not lens:
        return {"sent_len_mean": 0.0, "sent_len_std": 0.0, "sent_len_cv": 0.0, "sent_len_max": 0.0, "n_sent": 0}
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    return {
        "sent_len_mean": mean,
        "sent_len_std": math.sqrt(var),
        "sent_len_cv": math.sqrt(var) / max(mean, 1e-6),
        "sent_len_max": float(max(lens)),
        "n_sent": float(len(lens)),
    }


def burstiness(text: str) -> float:
    """CV of sentence lengths — the standard 'burstiness' proxy."""
    return sentence_stats(text)["sent_len_cv"]


def compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(zlib.compress(b, 9)) / max(len(b), 1)


def function_word_distribution(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for w in words(t):
            if w in FUNCTION_WORDS:
                c[w] += 1
    total = sum(c.values()) or 1
    return Counter({k: v / total for k, v in c.items()})


def jsd(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    m = 0.0
    for k in keys:
        a, b = p.get(k, 0.0), q.get(k, 0.0)
        mid = (a + b) / 2
        if a > 0:
            m += 0.5 * a * math.log2(a / mid)
        if b > 0:
            m += 0.5 * b * math.log2(b / mid)
    return m


def style_jsd(text: str, human_dist: Counter) -> float:
    return jsd(function_word_distribution([text]), human_dist)


def gltr_buckets(trace: dict[str, torch.Tensor]) -> dict[str, float]:
    """GLTR-style rank buckets of the *chosen* tokens (fraction per bucket)."""
    if not trace or "chosen_rank" not in trace:
        return {}
    return _rank_buckets(trace["chosen_rank"].flatten())


def _rank_buckets(rank: torch.Tensor) -> dict[str, float]:
    n = rank.numel()
    if n == 0:
        return {}
    return {
        "gltr_top10": float((rank < 10).sum()) / n,
        "gltr_10_100": float(((rank >= 10) & (rank < 100)).sum()) / n,
        "gltr_100_1000": float(((rank >= 100) & (rank < 1000)).sum()) / n,
        "gltr_tail": float((rank >= 1000).sum()) / n,
    }


@torch.no_grad()
def gltr_from_texts(model, tokenizer, texts: list[str], max_len: int = 512) -> dict[str, float]:
    """GLTR rank profile of *real* tokens in human text under ``model``.

    This is the missing human baseline: if a decoder's generated tokens all sit
    in the top-10 head but human text does not, head conformity is measurable.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ranks = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc.input_ids.to(model.device)
        if ids.shape[1] < 3:
            continue
        logits = model(input_ids=ids).logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        scores_at = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        rank = (logits > scores_at.unsqueeze(-1)).sum(-1)
        ranks.append(rank.flatten().cpu())
    if not ranks:
        return {}
    return _rank_buckets(torch.cat(ranks))


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], max_len: int = 512, batch_size: int = 8) -> list[float]:
    """Per-text token-level perplexity under ``model`` (left-to-right)."""
    out: list[float] = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    for i in range(0, len(texts), batch_size):
        chunk = [t if t.strip() else " " for t in texts[i : i + batch_size]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ids = enc.input_ids.to(model.device)
        am = enc.attention_mask.to(model.device)
        logits = model(input_ids=ids, attention_mask=am).logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        mask = am[:, 1:].bool()
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for b in range(ids.shape[0]):
            valid = mask[b]
            if valid.sum() == 0:
                out.append(float("inf"))
                continue
            nll = -lp[b][valid].mean().item()
            out.append(math.exp(nll))
    return out


def summarize(text: str, source: str, entities: list[str], human_dist: Counter | None = None) -> dict[str, float]:
    d = {
        "n_words": len(words(text)),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "rep_4": repetition_rate(text, 4),
        "rep_5": repetition_rate(text, 5),
        "ttr": ttr(text),
        "mtld": mtld(text),
        "copy_4gram": copy_overlap(text, source, 4),
        "entity_recall": entity_recall(text, entities),
        "boilerplate": boilerplate_rate(text),
        "hedge": hedge_rate(text),
        "compression": compression_ratio(text),
    }
    d.update(sentence_stats(text))
    d["burstiness"] = d["sent_len_cv"]
    if human_dist is not None:
        d["style_jsd"] = style_jsd(text, human_dist)
    return d
