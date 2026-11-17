"""Minimal LoRA + DPO, hand-rolled in pure torch.

Why not `peft` + `trl`? This repo runs on Python 3.14 free-threading with
transformers 5.x; trl pulls a large dependency surface for what is, at 135 M
parameters, ~120 lines of maths. Everything here is explicit so the experiment
can report exactly what it trained.

LoRA
----
Every targeted ``nn.Linear`` gets ``y = W x + (alpha / r) B A x`` with ``A``
random and ``B`` zero, so the adapter starts as an exact identity. Because the
adapter is a *separate* module, the reference policy for DPO is simply the same
model with ``lora.enabled = False`` — no second copy of the weights.

DPO
---
    L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                              - (log pi(y_l) - log pi_ref(y_l)) ] )

with length-normalised sequence log-probabilities (the standard practice; raw
sums bias toward short completions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        h = self.drop(x).to(torch.float32) @ self.A.T @ self.B.T
        return out + (self.scale * h).to(out.dtype)


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class LoRAHandle:
    modules: list[LoRALinear] = field(default_factory=list)

    def parameters(self):
        for m in self.modules:
            yield m.A
            yield m.B

    def trainable(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def set_enabled(self, flag: bool) -> None:
        for m in self.modules:
            m.enabled = flag

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def apply_lora(model, cfg: LoRAConfig | None = None) -> LoRAHandle:
    """Swap every targeted Linear for a LoRALinear and freeze everything else."""
    cfg = cfg or LoRAConfig()
    for p in model.parameters():
        p.requires_grad_(False)
    handle = LoRAHandle()
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in cfg.targets:
                wrapped = LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout)
                setattr(module, child_name, wrapped)
                handle.modules.append(wrapped)
    return handle


# --------------------------------------------------------------------------- #
# sequence log-probabilities
# --------------------------------------------------------------------------- #


def sequence_logprob(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_len: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum of log p(label_t) for t >= prompt_len, per row.

    ``labels`` is ``input_ids`` with prompt positions set to -100. The attention
    mask must be passed whenever rows are left-padded: without it the pad tokens
    are treated as real context and every statistic is contaminated.
    """
    kw = {"attention_mask": attention_mask} if attention_mask is not None else {}
    out = model(input_ids=input_ids, **kw).logits.float()
    logp = F.log_softmax(out[:, :-1, :], dim=-1)
    tgt = labels[:, 1:]
    gathered = logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = tgt != -100
    return (gathered * mask).sum(-1), mask.sum(-1).clamp_min(1)


def tokenize_pair(tokenizer, prompt: str, completion: str, max_len: int = 512):
    """Left-padded prompt + completion → (input_ids, labels)."""
    p = tokenizer(prompt, add_special_tokens=False)
    c = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)
    ids = p["input_ids"] + c["input_ids"]
    labels = [-100] * len(p["input_ids"]) + list(c["input_ids"])
    if len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]
    return ids, labels


def pad_batch(rows: list[tuple[list[int], list[int]]], pad_id: int):
    L = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    am = torch.zeros((len(rows), L), dtype=torch.long)
    # left padding, so the completion always ends at the last position
    for i, (a, b) in enumerate(rows):
        n = len(a)
        ids[i, L - n :] = torch.tensor(a, dtype=torch.long)
        lab[i, L - n :] = torch.tensor(b, dtype=torch.long)
        am[i, L - n :] = 1
    return ids, lab, am


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Length-normalised DPO loss. Inputs are (logprob_sum, n_tokens)."""
    pc, nc = policy_chosen
    pr, nr = policy_rejected
    rc, _ = ref_chosen
    rr, _ = ref_rejected
    logp_c = pc / nc
    logp_r = pr / nr
    ref_c = rc / nc
    ref_r = rr / nr
    margin = (logp_c - ref_c) - (logp_r - ref_r)
    loss = -F.logsigmoid(beta * margin).mean()
    acc = (margin > 0).float().mean()
    return loss, {
        "dpo_loss": float(loss.detach()),
        "reward_margin": float(margin.detach().mean()),
        "pref_acc": float(acc),
    }


class DPOData:
    """A list of (prompt, chosen, rejected) triples."""

    def __init__(self, triples: list[tuple[str, str, str]]):
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def batches(self, tokenizer, batch_size: int, shuffle: bool = True, seed: int = 0, max_len: int = 512):
        idx = list(range(len(self.triples)))
        if shuffle:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(idx), generator=g).tolist()
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for i in range(0, len(idx), batch_size):
            chunk = [self.triples[j] for j in idx[i : i + batch_size]]
            ch = [tokenize_pair(tokenizer, p, c, max_len) for p, c, _ in chunk]
            rj = [tokenize_pair(tokenizer, p, r, max_len) for p, _, r in chunk]
            yield pad_batch(ch, pad_id), pad_batch(rj, pad_id)


@torch.no_grad()
def evaluate_preference(model, handle: LoRAHandle, tokenizer, data: DPOData, max_len: int = 512) -> dict:
    """Held-out preference accuracy of the trained policy (no reference needed)."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    handle.set_enabled(True)
    correct, margins = 0, []
    for (ids_c, lab_c, am_c), (ids_r, lab_r, am_r) in data.batches(tokenizer, batch_size=8, shuffle=False):
        ids_c, lab_c, am_c = ids_c.to(model.device), lab_c.to(model.device), am_c.to(model.device)
        ids_r, lab_r, am_r = ids_r.to(model.device), lab_r.to(model.device), am_r.to(model.device)
        lc, nc = sequence_logprob(model, ids_c, lab_c, 0, am_c)
        lr, nr = sequence_logprob(model, ids_r, lab_r, 0, am_r)
        m = lc / nc - lr / nr
        correct += int((m > 0).sum())
        margins.extend(m.tolist())
    n = max(len(data), 1)
    return {"heldout_pref_acc": correct / n, "heldout_margin": sum(margins) / max(len(margins), 1)}


def train_dpo(
    model,
    handle: LoRAHandle,
    tokenizer,
    train: DPOData,
    test: DPOData,
    *,
    beta: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_len: int = 512,
    log_every: int = 10,
) -> list[dict]:
    """Run DPO, returning one history row per optimisation step."""
    opt = torch.optim.AdamW(handle.trainable(), lr=lr, weight_decay=0.0)
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    total = epochs * ((steps_per_epoch + grad_accum - 1) // grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total, 1), pct_start=0.1)
    history: list[dict] = []
    step = 0
    model.train()
    for epoch in range(epochs):
        for bi, ((ids_c, lab_c, am_c), (ids_r, lab_r, am_r)) in enumerate(
            train.batches(tokenizer, batch_size, shuffle=True, seed=epoch)
        ):
            ids_c, lab_c, am_c = ids_c.to(model.device), lab_c.to(model.device), am_c.to(model.device)
            ids_r, lab_r, am_r = ids_r.to(model.device), lab_r.to(model.device), am_r.to(model.device)

            handle.set_enabled(True)
            pc = sequence_logprob(model, ids_c, lab_c, 0, am_c)
            pr = sequence_logprob(model, ids_r, lab_r, 0, am_r)
            with torch.no_grad():
                handle.set_enabled(False)
                rc = sequence_logprob(model, ids_c, lab_c, 0, am_c)
                rr = sequence_logprob(model, ids_r, lab_r, 0, am_r)
            handle.set_enabled(True)

            loss, info = dpo_loss(pc, pr, rc, rr, beta=beta)
            (loss / grad_accum).backward()
            if (bi + 1) % grad_accum == 0 or bi == steps_per_epoch - 1:
                torch.nn.utils.clip_grad_norm_(handle.trainable(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == total:
                    history.append({"step": step, "epoch": epoch, **info})
                    print(f"    step {step:3d}/{total}  loss {info['dpo_loss']:.4f}  "
                          f"margin {info['reward_margin']:+.4f}  acc {info['pref_acc']:.2f}", flush=True)
    model.eval()
    handle.set_enabled(True)
    hist_test = evaluate_preference(model, handle, tokenizer, test, max_len)
    history.append({"step": step, "epoch": epochs, "eval": True, **hist_test})
    print(f"    held-out preference accuracy: {hist_test['heldout_pref_acc']:.3f} "
          f"(margin {hist_test['heldout_margin']:+.4f})")
    return history
