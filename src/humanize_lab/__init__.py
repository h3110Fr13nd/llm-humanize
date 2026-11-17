"""Shared local experiment harness for the Humanize-Experiments repo.

Everything here is CPU/4GB-GPU friendly: the model under test is always
``HuggingFaceTB/SmolLM-135M`` (base, for scoring) or ``...-Instruct`` (for
generation). No network calls are made after the weights are cached.
"""

from .models import BASE_MODEL, DEFAULT_MODEL, encode_prompts, load, pick_device, render_chat
from .sampling import GenOut, generate

__all__ = [
    "BASE_MODEL",
    "DEFAULT_MODEL",
    "GenOut",
    "encode_prompts",
    "generate",
    "load",
    "pick_device",
    "render_chat",
]
