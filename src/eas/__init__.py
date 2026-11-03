"""emoji-attention-sinks (eas) — C1 pipeline: load, capture, locate, validate.

The public names are exposed lazily (PEP 562 module ``__getattr__``) so that
``import eas.span`` or ``import eas.validate`` does NOT drag in the heavy,
model-loading modules. This keeps the span/validation logic testable with only
its own light dependencies, and defers importing ``transformers`` until you
actually load a model.
"""

__all__ = [
    "AttentionCapture",
    "SpanLoc",
    "load_model",
    "capture_attention",
    "build_stimulus",
    "locate_span_by_offsets",
    "validate_span",
    "SpanNotFound",
]


def __getattr__(name):
    # Imported on first access, not at package import time.
    if name in ("AttentionCapture", "SpanLoc"):
        from . import types
        return getattr(types, name)
    if name in ("locate_span_by_offsets", "validate_span", "SpanNotFound"):
        from . import span
        return getattr(span, name)
    if name == "load_model":
        from .models import load_model
        return load_model
    if name in ("capture_attention", "build_stimulus"):
        from . import capture
        return getattr(capture, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
