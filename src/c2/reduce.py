"""Streaming attention reduction with forward hooks.

eas.capture_attention keeps every layer's [H, n, n] map (fine for checks and
the diagnostic subset). For the sweep we instead hook each attention module:
as soon as a layer's eager attention weights exist, we reduce them to
  a_S    [H, n]  attention each query gives the probe span
  recv_Q [H, n]  attention each key receives from the query block
plus two validity numbers (row-sum error, causal leak), and then replace the
weights in the module output with None so HuggingFace never accumulates them.

Requires the model loaded with attn_implementation="eager" (eas.load_model).
Tested against eas.capture_attention in tests/c2/test_c2_model.py.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch


def attention_modules(model) -> List[torch.nn.Module]:
    """The per-layer attention modules for GPT-NeoX (Pythia) and Qwen2."""
    if hasattr(model, "gpt_neox"):
        return [layer.attention for layer in model.gpt_neox.layers]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [layer.self_attn for layer in model.model.layers]
    raise NotImplementedError(f"don't know where the attention modules are in {type(model).__name__}")


class SpanReducer:
    """Context manager: hooks installed on enter, removed on exit."""

    def __init__(self, model, keep_full: bool = False):
        self.model = model
        self.keep_full = keep_full
        self.handles = []
        self.span: Sequence[int] = ()
        self.queries: Sequence[int] = ()
        self._reset()

    def _reset(self):
        self.a_S, self.recv_Q, self.rowsum_err, self.causal_leak, self.full = [], [], [], [], []

    def set_probe(self, span: Sequence[int], queries: Sequence[int]) -> None:
        self.span, self.queries = list(span), list(queries)
        self._reset()

    def _hook(self, module, args, output):
        idx = None
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[-1] == o.shape[-2]:
                idx = i
                break
        if idx is None:
            raise RuntimeError("no attention weights in module output; load with attn_implementation='eager' "
                               "and call the model with output_attentions=True")
        att = output[idx][0].float()                                   # [H, n, n]  (batch size 1)
        n = att.shape[-1]
        s = torch.as_tensor(self.span, device=att.device)
        q = torch.as_tensor(self.queries, device=att.device)
        self.a_S.append(att.index_select(-1, s).sum(-1).cpu().numpy())        # [H, n]
        self.recv_Q.append(att.index_select(-2, q).sum(-2).cpu().numpy())     # [H, n]
        self.rowsum_err.append((att.sum(-1) - 1).abs().max().item())
        future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=att.device), diagonal=1)
        self.causal_leak.append(att[:, future].abs().max().item() if n > 1 else 0.0)
        if self.keep_full:
            self.full.append(att.cpu().numpy())
        return tuple(None if i == idx else o for i, o in enumerate(output))

    def __enter__(self):
        for m in attention_modules(self.model):
            self.handles.append(m.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def run(self, input_ids: Sequence[int], span: Sequence[int], queries: Sequence[int],
            device: Optional[str] = None) -> dict:
        """One forward pass at batch size 1 -> reduced arrays, [L, H, n] each."""
        self.set_probe(span, queries)
        ids = torch.tensor([list(input_ids)], device=device)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             output_attentions=True, use_cache=False)
        res = {
            "a_S": np.stack(self.a_S), "recv_Q": np.stack(self.recv_Q),
            "rowsum_err": max(self.rowsum_err), "causal_leak": max(self.causal_leak),
            "logits_last": out.logits[0, -1].float().cpu().numpy(),
        }
        if self.keep_full:
            res["full"] = np.stack(self.full)
        return res
