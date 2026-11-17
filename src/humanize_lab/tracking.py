"""Small helpers for reproducible experiment bookkeeping."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import torch


def env_fingerprint() -> dict:
    try:
        import transformers

        tf = transformers.__version__
    except Exception:  # pragma: no cover
        tf = "?"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "nogit"
    fps = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        fps = {"gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1)}
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": tf,
        "cuda": torch.cuda.is_available(),
        "cpus": torch.get_num_threads(),
        "commit": commit,
        **fps,
    }


def save_rows(rows: list[dict], path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if meta is not None:
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def md_table(rows: list[dict], cols: list[str] | None = None, digits: int = 3) -> str:
    if not rows:
        return "_(no rows)_"
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def group_mean(rows: list[dict], group_key: str, value_keys: list[str]) -> list[dict]:
    """Mean of every value key per group, keeping row order of first appearance."""
    order, acc = [], {}
    for r in rows:
        g = r[group_key]
        if g not in acc:
            acc[g] = {k: [] for k in value_keys}
            order.append(g)
        for k in value_keys:
            if isinstance(r.get(k), (int, float)):
                acc[g][k].append(float(r[k]))
    out = []
    for g in order:
        row = {group_key: g}
        for k in value_keys:
            vals = acc[g][k]
            row[k] = sum(vals) / len(vals) if vals else 0.0
        out.append(row)
    return out
