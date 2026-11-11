"""Fixtures for the metric-validation tests.

Pure-arithmetic tests need only numpy. The model-backed integration tests reuse
files already in the local HuggingFace cache and the generated probe files, and
skip cleanly when either is absent, so `pytest -q` still works offline.
"""
import os
import sys
import warnings

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
sys.path.insert(0, _SRC)
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


@pytest.fixture(scope="session")
def pythia_ready():
    """(model, tokenizer, probes) if Pythia weights and dev probes are available."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pytest.importorskip("torch")
    from metric_validation import common

    tag = common.model_tag("pythia")
    try:
        _path, probes = common.load_probes(tag, "dev")
    except SystemExit as exc:
        pytest.skip(f"probe file not available: {exc}")
    try:
        _key, _name, _rev, mdl, tok = common.load_pinned_model("pythia", "cpu")
    except Exception as exc:
        pytest.skip(f"pythia weights not in local HF cache ({type(exc).__name__})")
    return mdl, tok, probes
