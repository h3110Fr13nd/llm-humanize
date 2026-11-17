"""Model loading, chat rendering and prompt encoding (device aware)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
BASE_MODEL = "HuggingFaceTB/SmolLM-135M"

_TOKENIZERS: dict[str, AutoTokenizer] = {}
_MODELS: dict[tuple, AutoModelForCausalLM] = {}


def pick_device() -> str:
    """CUDA when present (RTX 3050 Laptop, 4 GB here), otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str | None = None) -> torch.dtype:
    device = device or pick_device()
    return torch.float16 if device == "cuda" else torch.float32


def get_tokenizer(model_id: str = DEFAULT_MODEL) -> AutoTokenizer:
    if model_id not in _TOKENIZERS:
        _TOKENIZERS[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZERS[model_id]


def load(
    model_id: str = DEFAULT_MODEL,
    device: str | None = None,
    attn_implementation: str | None = None,
    dtype: torch.dtype | None = None,
):
    """Load (and memoise) a model. Pass ``attn_implementation='eager'`` when
    you intend to feed explicit 4-D attention masks (window experiments)."""
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    key = (model_id, device, attn_implementation, str(dtype))
    if key in _MODELS:
        return _TOKENIZERS[model_id], _MODELS[key]
    tok = get_tokenizer(model_id)
    kwargs = {"dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    _MODELS[key] = model
    return tok, model


def render_chat(tokenizer: AutoTokenizer, system: str, user: str, add_generation_prompt: bool = True) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def encode_prompts(tokenizer: AutoTokenizer, prompts: list[str], device: str | None = None):
    """Left-padded batch encoding.

    Left padding keeps every real token contiguous at the end of the row, so a
    single ``arange`` position id is valid for all rows and generated tokens
    simply continue the sequence.
    """
    device = device or pick_device()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    ids = enc.input_ids.to(device)
    am = enc.attention_mask.to(device)
    # position ids identical to HF generate(): pads collapse to 0.
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    return ids, am, pos


def new_cache(model) -> DynamicCache:
    return DynamicCache(config=model.config)


def stop_token_ids(tokenizer: AutoTokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id}
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(int(tid))
    ids.discard(None)
    return {int(i) for i in ids}
