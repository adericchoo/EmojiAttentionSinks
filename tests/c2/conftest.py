"""Shared fixtures for C2 tests.

Tokenizer- and model-based tests use only files already in the local
HuggingFace cache (local_files_only) and are skipped otherwise, so plain
`pytest -q` still works offline with no download.
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

from c2.config import model_entry  # noqa: E402


def _tok(model):
    from transformers import AutoTokenizer
    _, name, rev = model_entry(model)
    try:
        return AutoTokenizer.from_pretrained(name, revision=rev, local_files_only=True)
    except Exception as e:  # not cached
        pytest.skip(f"{name} tokenizer not in local HF cache ({type(e).__name__})")


@pytest.fixture(scope="session")
def pythia_tok():
    return _tok("pythia")


@pytest.fixture(scope="session")
def qwen_tok():
    return _tok("qwen")


@pytest.fixture(scope="session", params=["pythia", "qwen"])
def any_tok(request):
    return _tok(request.param)


@pytest.fixture(scope="session")
def pythia():
    """(model, tokenizer) loaded through the C1 loader, from the local cache only."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _tok("pythia")
    from eas import load_model
    _, name, rev = model_entry("pythia")
    try:
        return load_model(name, revision=rev)
    except Exception as e:
        pytest.skip(f"{name} weights not in local HF cache ({type(e).__name__})")
