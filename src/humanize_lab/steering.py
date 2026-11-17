"""Activation steering utilities (Experiment 06).

Difference-in-means style vectors: the mean residual-stream state on
human-written text minus the mean on model-generated text, at a chosen layer.
Injected during generation with a forward hook.
"""

from __future__ import annotations

import contextlib

import torch


@torch.no_grad()
def mean_hidden(model, tokenizer, texts: list[str], layer: int, batch_size: int = 6) -> torch.Tensor:
    """Mean residual stream at ``layer`` (0 = embeddings) over real tokens."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    acc, n = None, 0
    for i in range(0, len(texts), batch_size):
        chunk = [t for t in texts[i : i + batch_size] if t.strip()]
        if not chunk:
            continue
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = enc.input_ids.to(model.device)
        am = enc.attention_mask.to(model.device)
        out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        h = out.hidden_states[layer].float()  # (B, L, H)
        m = am.unsqueeze(-1).float()
        acc = (h * m).sum((0, 1)) if acc is None else acc + (h * m).sum((0, 1))
        n += m.sum().item()
    return acc / max(n, 1.0)


def style_vector(model, tokenizer, human_texts: list[str], model_texts: list[str], layer: int) -> torch.Tensor:
    h = mean_hidden(model, tokenizer, human_texts, layer)
    m = mean_hidden(model, tokenizer, model_texts, layer)
    return (h - m).float()


@contextlib.contextmanager
def steer(model, layer: int, vector: torch.Tensor, alpha: float, residual_norm: float = 1.0):
    """Add ``alpha * unit(vector) * residual_norm`` after decoder layer ``layer``.

    ``alpha`` is therefore a *relative* strength: 0.2 means "20 % of the mean
    residual-stream norm at this layer". Absolute scaling is a trap — residual
    norms grow from ~150 (layer 6) to ~740 (layer 24) in this model, so an
    absolute delta of a few units is invisible.
    """
    v = vector / (vector.norm() + 1e-8)
    target = model.model.layers[layer]

    def hook(module, inputs, output):
        delta = (alpha * residual_norm * v).to(dtype=output.dtype, device=output.device)
        if isinstance(output, tuple):
            return (output[0] + delta,) + tuple(output[1:])
        return output + delta

    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def residual_norms(model, tokenizer, texts: list[str], layers: list[int]) -> dict[int, float]:
    """Mean per-token residual norm at each layer (the natural delta scale)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids.to(model.device),
                    attention_mask=enc.attention_mask.to(model.device), output_hidden_states=True)
    am = enc.attention_mask.to(model.device).unsqueeze(-1).float()
    res = {}
    for l in layers:
        h = out.hidden_states[l].float()
        n = (h.norm(dim=-1, keepdim=True) * am).sum() / am.sum().clamp_min(1)
        res[l] = float(n)
    return res


def projection(model, tokenizer, texts: list[str], vector: torch.Tensor, layer: int) -> torch.Tensor:
    """Per-text scalar projection of the residual stream onto the style vector."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids.to(model.device), attention_mask=enc.attention_mask.to(model.device),
                    output_hidden_states=True)
    h = out.hidden_states[layer].float()
    am = enc.attention_mask.to(model.device).unsqueeze(-1).float()
    mean = (h * am).sum(1) / am.sum(1).clamp_min(1)
    return (mean @ vector.to(mean.device)) / (vector.norm() + 1e-8)
