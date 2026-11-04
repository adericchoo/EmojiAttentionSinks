"""Model selection, reusing the pinned revisions in config/models.yaml (owned by C1)."""
from __future__ import annotations

import os
from typing import Tuple

import yaml

from . import ROOT

# CLI aliases -> config/models.yaml keys
ALIASES = {"pythia": "primary", "qwen": "replication", "primary": "primary", "replication": "replication"}
# short tags used in output filenames
TAGS = {"primary": "pythia410m", "replication": "qwen25_05b"}


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "models.yaml")) as f:
        return yaml.safe_load(f)


def model_entry(model: str) -> Tuple[str, str, str]:
    """'pythia' / 'primary' -> (config key, HF name, pinned revision)."""
    key = ALIASES.get(model)
    if key is None:
        raise SystemExit(f"unknown model {model!r}; use one of {sorted(ALIASES)}")
    entry = load_config()[key]
    return key, entry["name"], entry.get("revision")


def model_tag(model: str) -> str:
    return TAGS[model_entry(model)[0]]


def load_tokenizer(model: str):
    """Tokenizer only (the audit and probe generator never need the weights).

    Same pinned revision that eas.load_model uses for the full model.
    """
    from transformers import AutoTokenizer

    _, name, rev = model_entry(model)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    if not tok.is_fast:
        raise RuntimeError(f"{name}: fast tokenizer required for offset mappings")
    return tok
