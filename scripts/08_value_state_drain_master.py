"""Leg 3c: is the emoji F0 9F sink a value-state drain, or just high-attention?

    python scripts/08_value_state_drain.py              # primary model (Pythia)
    python scripts/08_value_state_drain.py replication  # Qwen

Established attention sinks (the first token / BOS) are "value-state drains": the
sink position's VALUE vector has a small norm, so attending to it moves almost
nothing into the residual stream — a no-op parking spot. Legs 3/3a/3b showed the
emoji F0 9F fragment attracts a large, head-specific share of attention. This leg
asks the mechanism question: does that fragment also carry a near-zero value (a
genuine drain like BOS), or a normal-size value (an attention sink WITHOUT a value
drain)?

output_attentions returns attention WEIGHTS, not value states, so the value vectors
are captured here with forward hooks. That capture path is new and lives entirely in
this script. Two architectures are handled explicitly:
  - GPT-NeoX / Pythia: fused query_key_value projection -> slice the value third and
    reshape to [heads, seq, head_dim].
  - Qwen2 / Qwen2.5: separate v_proj -> [kv_heads, seq, head_dim], then expand the
    grouped-query kv heads up to the query heads exactly as the model does.

This is additive. It reuses, without modifying:
  - model loading                                    (eas.load_model)
  - the structured probe + span bookkeeping          (c2.probes.make_probe)
  - script 05's received-attention reduction and
    script 06's F0 9F lead-fragment tagging          (05 via 06)
  - the emoji conditions and ordinary-word baseline  (config/master_stimuli.yaml)

Following scripts 04/05/06/07, ONE model is run per invocation (primary by default,
"replication" for Qwen); run it twice to cover both models.
"""
import importlib.util
import os
import sys

from _bootstrap import load_config, select_model

import numpy as np
import torch
import yaml
from c2.probes import MAX_TOKENS, Rejected, build_contexts, make_probe
from c2.tokaudit import token_bytes
from eas import load_model
from eas.span import SpanNotFound

# Make emoji printable on consoles whose default code page is not UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Bars (kept together at the top)
# ---------------------------------------------------------------------------
# A head counts as a "sink head" for the F0 9F fragment when it attends more than
# this multiple of the ordinary-word baseline to that fragment (reuses 06's
# fragment-vs-baseline received-attention comparison, with a heavy-attention ratio).
SINK_HEAD_RATIO = 3.0
# The value-state drain is confirmed when the drain_ratio falls below this: the
# sink's value vector is much smaller than an ordinary token's.
DRAIN_CONFIRM_BAR = 0.5


def _load_byte_structure_module():
    """Import script 06 (name begins with a digit -> load by path). It carries
    script 05's reduction as `frag` and the F0 9F lead-byte tagging."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "06_byte_structure_master.py")
    spec = importlib.util.spec_from_file_location("byte_structure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs06 = _load_byte_structure_module()
frag = bs06.frag  # script 05's module (fragment_received, etc.)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_stimuli():
    """The byte-structure stimulus set (group A emoji + R_reference words)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "config", "master_stimuli.yaml")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Architecture description and value-capture forward hooks
# ---------------------------------------------------------------------------
def model_arch(model):
    """(model_type, n_layers, n_query_heads, n_kv_heads, head_dim) from the config."""
    config = model.config
    model_type = config.model_type
    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads))
    return model_type, n_layers, n_heads, n_kv_heads, head_dim


def _neox_value_hook(store, layer_index, n_heads, head_dim):
    """GPT-NeoX: query_key_value output is [batch, seq, 3 * n_heads * head_dim];
    reshape to per-head thirds and keep the value third."""
    def hook(module, inputs, output):
        batch = output.shape[0]
        seq = output.shape[1]
        reshaped = output.view(batch, seq, n_heads, 3 * head_dim)
        value = reshaped[:, :, :, 2 * head_dim:3 * head_dim]  # [batch, seq, heads, head_dim]
        value = value[0].permute(1, 0, 2).contiguous()        # [heads, seq, head_dim]
        store[layer_index] = value.detach().to("cpu", torch.float32)
    return hook


def _qwen_value_hook(store, layer_index, n_heads, n_kv_heads, head_dim):
    """Qwen2 (grouped-query): v_proj output is [batch, seq, n_kv_heads * head_dim];
    reshape to kv heads, then expand each kv head to its query heads as repeat_kv
    does (query head q is served by kv head q // n_rep)."""
    def hook(module, inputs, output):
        seq = output.shape[1]
        reshaped = output[0].view(seq, n_kv_heads, head_dim)   # [seq, kv_heads, head_dim]
        reshaped = reshaped.permute(1, 0, 2).contiguous()      # [kv_heads, seq, head_dim]
        n_rep = n_heads // n_kv_heads
        expanded = torch.zeros(n_heads, seq, head_dim)
        for query_head in range(n_heads):
            kv_head = query_head // n_rep
            expanded[query_head] = reshaped[kv_head]
        store[layer_index] = expanded.detach().to("cpu", torch.float32)
    return hook


def register_value_hooks(model, arch):
    """Register per-layer value hooks for the supported architectures.

    Returns (store, handles). `store` fills in during the forward pass as
    {layer_index: [n_query_heads, seq, head_dim]}.
    """
    model_type, n_layers, n_heads, n_kv_heads, head_dim = arch
    store = {}
    handles = []
    if model_type == "gpt_neox":
        for layer_index in range(n_layers):
            module = model.gpt_neox.layers[layer_index].attention.query_key_value
            handle = module.register_forward_hook(
                _neox_value_hook(store, layer_index, n_heads, head_dim)
            )
            handles.append(handle)
    elif model_type == "qwen2":
        for layer_index in range(n_layers):
            module = model.model.layers[layer_index].self_attn.v_proj
            handle = module.register_forward_hook(
                _qwen_value_hook(store, layer_index, n_heads, n_kv_heads, head_dim)
            )
            handles.append(handle)
    else:
        raise RuntimeError(f"value capture not implemented for architecture {model_type!r}")
    return store, handles


def _stack_values(store, n_layers):
    layer_list = []
    for layer_index in range(n_layers):
        layer_list.append(store[layer_index])
    return torch.stack(layer_list, dim=0)  # [n_layers, n_heads, seq, head_dim]


def _check_value_shape(values_tensor, n_layers, n_heads, seq, head_dim):
    shape = tuple(values_tensor.shape)
    if len(shape) != 4:
        raise RuntimeError(f"value tensor must be [L, H, seq, head_dim]; got {shape}")
    if shape[0] != n_layers or shape[1] != n_heads or shape[2] != seq or shape[3] != head_dim:
        raise RuntimeError(
            f"value tensor shape {shape} != expected "
            f"({n_layers}, {n_heads}, {seq}, {head_dim})"
        )


def capture_with_values(model, tok, text, arch):
    """ONE forward pass with output_attentions AND the value hooks live.

    Returns (input_ids, attentions [L, H, n, n], values [L, H, n, head_dim]).
    Tokenised with add_special_tokens=False, matching make_probe, so token indices
    line up with the probe's token spans.
    """
    model_type, n_layers, n_heads, n_kv_heads, head_dim = arch
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    seq = enc["input_ids"].shape[1]
    if seq > MAX_TOKENS:
        raise SpanNotFound(f"sequence length {seq} exceeds max_tokens {MAX_TOKENS}")
    store, handles = register_value_hooks(model, arch)
    try:
        with torch.no_grad():
            out = model(**enc, output_attentions=True, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if out.attentions is None:
        raise RuntimeError(
            "model returned attentions=None — load it with attn_implementation='eager'."
        )
    attentions = torch.stack([a[0] for a in out.attentions]).to("cpu", torch.float32)
    values_tensor = _stack_values(store, n_layers)
    _check_value_shape(values_tensor, n_layers, n_heads, seq, head_dim)
    ids = enc["input_ids"][0].tolist()
    return ids, attentions, values_tensor


def value_pass(model, tok, contexts, filler, arch):
    """Build the structured probe for `filler`, run the value+attention pass, and
    return (ids, attentions, values, span_indices, far_queries).

    Tries carriers in order; raises SpanNotFound if none build and retokenise
    cleanly. The span and far-block indices come straight from make_probe.
    """
    last_error = None
    for ctx in contexts:
        try:
            probe = make_probe(tok, ctx, filler)
        except Rejected as error:
            last_error = error
            continue
        ids, attentions, values_tensor = capture_with_values(model, tok, probe["text"], arch)
        if ids != probe["input_ids"]:
            last_error = SpanNotFound("text did not retokenise to the probe input_ids")
            continue
        span_indices = []
        for index in range(probe["token_span"][0], probe["token_span"][1]):
            span_indices.append(index)
        queries = []
        for query in range(probe["query_span"][0], probe["query_span"][1]):
            queries.append(query)
        return ids, attentions, values_tensor, span_indices, queries
    if last_error is None:
        raise SpanNotFound("no carrier available")
    raise SpanNotFound(str(last_error))


# ---------------------------------------------------------------------------
# Value-norm reduction and the drain ratio (explicit loops)
# ---------------------------------------------------------------------------
def value_norms_at(values_tensor, position):
    """Per layer per head, the L2 norm over head_dim of the value vector at one
    position. Returns an [L, H] numpy array."""
    n_layers = values_tensor.shape[0]
    n_heads = values_tensor.shape[1]
    norms = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            vector = values_tensor[layer, head, position, :]
            norms[layer, head] = float(torch.linalg.vector_norm(vector))
    return norms


def sink_head_mask(received_lh, baseline_received_lh, ratio):
    """Heads that attend heavily to the F0 9F fragment: received attention above
    `ratio` x the ordinary-word baseline on that head. A near-zero baseline head
    counts only when the fragment actually receives something there."""
    n_layers = received_lh.shape[0]
    n_heads = received_lh.shape[1]
    mask = np.zeros((n_layers, n_heads), dtype=bool)
    for layer in range(n_layers):
        for head in range(n_heads):
            got = float(received_lh[layer, head])
            base = float(baseline_received_lh[layer, head])
            if got <= 0.0:
                continue
            if base <= 0.0:
                mask[layer, head] = True
            elif got > ratio * base:
                mask[layer, head] = True
    return mask


def drain_ratio(sink_norms, ordinary_norms, mask):
    """Mean over the selected heads of value_norm(sink) / value_norm(ordinary).

    `mask` None means all heads. Heads with a non-positive ordinary norm are
    skipped so the ratio stays defined.
    """
    n_layers = sink_norms.shape[0]
    n_heads = sink_norms.shape[1]
    ratios = []
    for layer in range(n_layers):
        for head in range(n_heads):
            if mask is not None and not mask[layer, head]:
                continue
            denominator = float(ordinary_norms[layer, head])
            if denominator <= 0.0:
                continue
            ratios.append(float(sink_norms[layer, head]) / denominator)
    return _mean(ratios)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _mean(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


def _mean_arrays(arrays):
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0)


def _cells_mean(array_lh):
    total = 0.0
    count = 0
    for layer in range(array_lh.shape[0]):
        for head in range(array_lh.shape[1]):
            total = total + float(array_lh[layer, head])
            count = count + 1
    if count == 0:
        return float("nan")
    return total / count


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(model_name, summary):
    print(f"\nVALUE-STATE DRAIN MAP  —  {model_name}")
    print("L2 value-vector norm per layer per head; drain_ratio = value_norm(sink) / value_norm(ordinary word)")
    print(f"  sink heads = heads attending > {SINK_HEAD_RATIO:g}x the ordinary baseline to the F0 9F fragment "
          f"(mean {summary['mean_sink_heads']:.1f} of {summary['n_cells']} cells)\n")
    print(f"  {'quantity':38} {'value':>12}")
    print("  " + "-" * 52)
    print(f"  {'value_norm(sink F0 9F)':38} {summary['vn_sink']:>12.4f}")
    print(f"  {'value_norm(ordinary word)':38} {summary['vn_ord']:>12.4f}")
    print(f"  {'value_norm(BOS / first token)':38} {summary['vn_bos']:>12.4f}")
    print(f"  {'drain_ratio sink/ordinary (SINK heads)':38} {summary['drain_sink']:>12.4f}")
    print(f"  {'drain_ratio sink/ordinary (ALL heads)':38} {summary['drain_all']:>12.4f}")
    print(f"  {'drain_ratio BOS/ordinary  (ALL heads)':38} {summary['drain_bos']:>12.4f}")


def print_read(model_name, summary):
    drain_sink = summary["drain_sink"]
    drain_bos = summary["drain_bos"]
    print(f"\n  READ [{model_name}]:")
    bos_drained = drain_bos < DRAIN_CONFIRM_BAR
    emoji_drained = drain_sink < DRAIN_CONFIRM_BAR
    if bos_drained:
        print(f"    calibration: BOS/ordinary = {drain_bos:.3f} << 1 — BOS is a value-state drain, as expected.")
    else:
        print(f"    calibration WEAK: BOS/ordinary = {drain_bos:.3f} did not land well below 1; "
              "read the emoji result with care.")
    if emoji_drained and bos_drained:
        print(f"    CONFIRMED: the emoji F0 9F fragment is a value-state drain like BOS "
              f"(sink-head drain_ratio {drain_sink:.3f} << 1).")
    elif bos_drained and not emoji_drained:
        print(f"    NOT a value-state drain: the emoji F0 9F fragment carries a normal-size value "
              f"(sink-head drain_ratio {drain_sink:.3f} ~ 1), unlike BOS.")
        print("    So it is an attention sink WITHOUT a value drain — a different mechanism from the BOS sink.")
    else:
        print(f"    INCONCLUSIVE: emoji sink-head drain_ratio {drain_sink:.3f}, BOS drain_ratio "
              f"{drain_bos:.3f} (BOS calibration did not land << 1).")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect_ordinary(model, tok, contexts, reference_group, arch):
    """Ordinary-word value norms, their received-attention baseline, and BOS value
    norms — all from the same value passes. Returns
    (ordinary_norm [L,H], baseline_received [L,H], bos_value_list)."""
    ordinary_value_list = []
    baseline_received_list = []
    bos_value_list = []
    for item in reference_group["items"]:
        if item.get("kind") != "ordinary_word":
            continue
        word = item["text"]
        try:
            ids, attentions, values_tensor, span_indices, queries = value_pass(
                model, tok, contexts, word, arch
            )
        except SpanNotFound as error:
            print(f"  [skip] ordinary {word!r}: {error}")
            continue
        lead = span_indices[0]
        ordinary_value_list.append(value_norms_at(values_tensor, lead))
        received = frag.fragment_received(attentions, lead, queries).detach().numpy()
        baseline_received_list.append(received)
        bos_value_list.append(value_norms_at(values_tensor, 0))
    if len(ordinary_value_list) == 0:
        raise SpanNotFound("no ordinary-word reference could be captured")
    ordinary_norm = _mean_arrays(ordinary_value_list)
    baseline_received = _mean_arrays(baseline_received_list)
    return ordinary_norm, baseline_received, bos_value_list


def run_for_model(name, revision, stimuli):
    model, tok = load_model(name, revision=revision)
    arch = model_arch(model)
    model_type, n_layers, n_heads, n_kv_heads, head_dim = arch
    contexts = build_contexts(tok)
    groups = stimuli["groups"]

    print(f"\nVALUE-STATE DRAIN CHARACTERISATION  —  {name}  ({model_type})")
    print("forward-hook value capture + output_attentions in ONE pass; structured probe, fixed far block\n")

    ordinary_norm, baseline_received, bos_value_list = collect_ordinary(
        model, tok, contexts, groups["R_reference"], arch
    )

    sink_value_list = []
    drain_sink_list = []
    drain_all_list = []
    sink_head_counts = []
    for item in groups["A_smp_pictographs"]["items"]:
        emoji = item["emoji"]
        try:
            ids, attentions, values_tensor, span_indices, queries = value_pass(
                model, tok, contexts, emoji, arch
            )
        except SpanNotFound as error:
            print(f"  [skip] {emoji!r}: {error}")
            continue
        sink_position = span_indices[0]
        raw_bytes = token_bytes(tok, ids[sink_position])
        tag = bs06.lead_byte_tag(raw_bytes)
        if tag != "F0 9F":
            print(f"  [skip] {emoji!r}: lead fragment is {tag!r}, not F0 9F")
            continue
        sink_norm = value_norms_at(values_tensor, sink_position)
        received = frag.fragment_received(attentions, sink_position, queries).detach().numpy()
        mask = sink_head_mask(received, baseline_received, SINK_HEAD_RATIO)
        sink_value_list.append(sink_norm)
        drain_sink_list.append(drain_ratio(sink_norm, ordinary_norm, mask))
        drain_all_list.append(drain_ratio(sink_norm, ordinary_norm, None))
        count = 0
        for layer in range(mask.shape[0]):
            for head in range(mask.shape[1]):
                if mask[layer, head]:
                    count = count + 1
        sink_head_counts.append(count)
        bos_value_list.append(value_norms_at(values_tensor, 0))

    if len(sink_value_list) == 0:
        raise SpanNotFound("no F0 9F emoji fragment could be captured")

    sink_norm_mean = _mean_arrays(sink_value_list)
    bos_norm_mean = _mean_arrays(bos_value_list)

    summary = {
        "n_cells": n_layers * n_heads,
        "mean_sink_heads": _mean(sink_head_counts),
        "vn_sink": _cells_mean(sink_norm_mean),
        "vn_ord": _cells_mean(ordinary_norm),
        "vn_bos": _cells_mean(bos_norm_mean),
        "drain_sink": _mean(drain_sink_list),
        "drain_all": _mean(drain_all_list),
        "drain_bos": drain_ratio(bos_norm_mean, ordinary_norm, None),
    }
    print_table(name, summary)
    print_read(name, summary)
    print()


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    stimuli = load_stimuli()
    run_for_model(name, revision, stimuli)


if __name__ == "__main__":
    main()
