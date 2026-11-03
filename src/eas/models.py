"""Model + tokenizer loading, with attention actually exposed.

The `attn_implementation="eager"` flag is load-bearing: the default SDPA /
FlashAttention kernels return `attentions=None` silently.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    name: str,
    revision: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """Load a causal LM in eval mode with eager attention.

    Args:
        name: HF model id, e.g. "EleutherAI/pythia-410m".
        revision: commit SHA to pin (strongly recommended for reported runs).
        device: "cuda", "cpu", or None (leave on default).
        dtype: float32 recommended for exact, reproducible attention.
    """
    if revision in (None, "FILL_ME"):
        print(
            f"[warn] loading {name} without a pinned revision — set the commit "
            "SHA in config/models.yaml before any run you intend to report."
        )
        revision = None

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{name} did not load a fast tokenizer; offset mapping (needed for "
            "span location) requires one."
        )
    if tokenizer.pad_token is None:
        # Pythia has no pad token. Reuse EOS; always pass attention_mask so this
        # never leaks into scores.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    model.eval()
    if device:
        model.to(device)
    return model, tokenizer
