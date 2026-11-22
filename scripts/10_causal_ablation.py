"""Leg: causal ablation of the emoji F0 9F sink — is the attention justified?

    python scripts/10_causal_ablation.py              # primary model (Pythia-410m)
    python scripts/10_causal_ablation.py replication  # Qwen (NOT supported, see note)

Leg 3c (script 08) showed the emoji F0 9F fragment is an attention sink WITHOUT a
value-state drain: heads attend heavily to it AND it carries a normal-size value, so
those heads appear to READ from it. This leg asks whether that read does real work.

We block attention TO the F0 9F lead-byte key in the identified sink heads only (set
its pre-softmax score to -inf, so its post-softmax weight goes to 0 and every query
row renormalises over the remaining keys) and measure the change in next-token
cross-entropy loss, SPLIT by what is being predicted:
  - continuation bytes : the emoji's own within-emoji tokens after F0 9F
  - following words     : the real word tokens AFTER the emoji span ends
A large loss increase on continuation bytes means local emoji assembly (the read is
justified); a large increase on following words means a forward "emoji here" signal;
near-zero on both means the attention is largely wasted despite a normal value.

CONTROLS
  - random-head  : block the same F0 9F key in a matched COUNT of random NON-sink
                   heads; the sink-head effect must exceed this or it is not
                   head-specific.
  - ordinary-key : block an ORDINARY following-word key (matched slot) in the SINK
                   heads, to show the effect is specific to the F0 9F key.
Optional robustness variant: mean-ablation — overwrite the F0 9F VALUE vector with
the ordinary-token mean instead of masking the edge; reported alongside.

This is additive. It reuses, without modifying:
  - model loading, one-pass capture, span location/validation       (eas)
  - the received-attention reference reduction                      (eas.metric_reference)
  - script 06's byte-level lead-token tagging / token_bytes          (06 -> c2.tokaudit)
The attention-edge patch lives entirely here: each GPT-NeoX attention module's
`_attn` is rebound to a faithful copy that injects a per-head, per-key additive mask
before the softmax. `src/eas` and scripts 04-09 are untouched.

MODELS: Pythia-410m primary. Pythia-2.8b would use the identical GPT-NeoX patch if it
were pinned in config/models.yaml. The edge-ablation path is GPT-NeoX-specific, so a
non-NeoX replication model (Qwen) is reported as unsupported rather than run wrong.
"""
import importlib.util
import os
import random
import sys
import types

from _bootstrap import load_config, select_model

import numpy as np
import torch

from eas import capture_attention, load_model
from eas.metric_reference import received_attention
from eas.span import SpanNotFound, locate_span_by_offsets, validate_span
from eas.types import AttentionCapture, SpanLoc

# Make emoji printable on consoles whose default code page is not UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Bars / knobs (kept together at the top)
# ---------------------------------------------------------------------------
# A head counts as a "sink head" when, aggregated over the stimuli, it attends more
# than this multiple of the ordinary-token baseline to the F0 9F key (the same
# fragment-vs-baseline received-attention rule scripts 08/09 use).
SINK_HEAD_RATIO = 3.0
# Reproducible choice of the random non-sink control heads, averaged over several
# independent draws so the control is not hostage to one unlucky head set.
RANDOM_SEED = 20240611
RANDOM_DRAWS = 5
# A per-stimulus mean Delta-loss below this (in nats) is treated as negligible.
NEGLIGIBLE_DELTA = 0.02

# Target classes for the loss split.
CLASS_CONTINUATION = "continuation"
CLASS_FOLLOWING = "following"
CLASS_OTHER = "other"


def _load_byte_structure_module():
    """Import script 06 (name begins with a digit -> load by path). It carries the
    F0 9F lead-byte tagging and, through c2.tokaudit, token_bytes."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "06_byte_structure.py")
    spec = importlib.util.spec_from_file_location("byte_structure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs06 = _load_byte_structure_module()
from c2.tokaudit import token_bytes  # noqa: E402  (c2 is on the path via 06's import)


# ---------------------------------------------------------------------------
# Natural carrier sentences. A byte-level BPE tokenizer MERGES a preceding space
# into the emoji's first token (producing a "20 F0 9F" token, not the clean "F0 9F"
# lead the earlier legs localised). So the emoji begins a new line: a lead-in clause,
# a newline, then EMOJI + " " + real words after it. This keeps the F0 9F lead clean
# AND leaves genuine following-word targets, exactly as script 09's slot does.
# {emoji} is the only slot; everything else is fixed ordinary text.
# ---------------------------------------------------------------------------
CARRIER_TEMPLATES = [
    "The crew finished packing the gear.\n{emoji} was loaded and we left at dawn the next morning.",
    "Here is the note my sister wrote on the card.\n{emoji} means we should call her back this evening.",
    "The teacher pointed at the board and paused.\n{emoji} was the answer the whole class had missed.",
    "He opened the box and looked inside.\n{emoji} sat on top of the folded paper and old photographs.",
    "The message arrived while we were at lunch.\n{emoji} appeared right before the rest of the text.",
    "She sketched a few shapes in her notebook.\n{emoji} stood out among the plain pencil drawings.",
    "The sign by the road was easy to read.\n{emoji} marked the turn toward the new bridge downtown.",
    "They decorated the hall for the party.\n{emoji} hung above the door for everyone to see clearly.",
    "I scrolled back through the old chat log.\n{emoji} was the first thing he sent that whole week.",
    "The report had a small mark in the margin.\n{emoji} flagged the line the editor wanted us to fix.",
]


def build_sentence(template, emoji):
    """Return (text, emoji_char_start, emoji_char_end) for one filled template."""
    marker = "{emoji}"
    position = template.index(marker)
    before = template[:position]
    after = template[position + len(marker):]
    text = before + emoji + after
    char_start = len(before)
    char_end = len(before) + len(emoji)
    return text, char_start, char_end


# ---------------------------------------------------------------------------
# Locating the emoji span and tagging the three target-position classes
# ---------------------------------------------------------------------------
def locate_emoji_span(tok, cap, char_start, char_end, emoji):
    """Validated SpanLoc for the emoji inside an already-captured sentence."""
    idx = locate_span_by_offsets(cap.offsets, char_start, char_end)
    validate_span(tok, cap.input_ids, idx, emoji)
    return SpanLoc(start=idx[0], end=idx[-1] + 1, token_indices=idx)


def position_classes(cap, span, f0_index):
    """Classify every token POSITION (0..n-1) as continuation / following / other.

    - continuation : a within-emoji token after the F0 9F lead (the emoji's own
      trailing byte-tokens).
    - following    : any token strictly after the emoji span ends (the real words).
    - other        : everything else (the prefix, and the F0 9F lead token itself).
    Returns a list of length n.
    """
    n = cap.n
    classes = []
    for position in range(n):
        classes.append(CLASS_OTHER)
    for token_index in span.token_indices:
        if token_index == f0_index:
            continue
        classes[token_index] = CLASS_CONTINUATION
    for position in range(span.end, n):
        classes[position] = CLASS_FOLLOWING
    return classes


# ---------------------------------------------------------------------------
# Per-token cross-entropy loss and the loss split (pure, explicit loops)
# ---------------------------------------------------------------------------
def per_token_loss(logits, input_ids):
    """Next-token cross-entropy at each predicted position.

    logits: [n, vocab] (batch already removed). input_ids: list of length n.
    Returns a list `loss` of length n where loss[t] is the CE of predicting token t
    from positions < t, for t in 1..n-1, and loss[0] is NaN (nothing predicts it).
    """
    n = len(input_ids)
    loss = []
    for position in range(n):
        loss.append(float("nan"))
    log_probs = torch.log_softmax(logits, dim=-1)
    for target in range(1, n):
        token_id = input_ids[target]
        loss[target] = -float(log_probs[target - 1, token_id])
    return loss


def split_loss_by_class(loss, classes):
    """Collect per-token losses into the three target classes.

    loss and classes are both length n; position 0 (NaN loss) is skipped. Returns
    {class_name: [loss values]}.
    """
    collected = {}
    collected[CLASS_CONTINUATION] = []
    collected[CLASS_FOLLOWING] = []
    collected[CLASS_OTHER] = []
    for position in range(len(loss)):
        value = loss[position]
        if value != value:  # skip NaN (position 0)
            continue
        class_name = classes[position]
        collected[class_name].append(value)
    return collected


def mean_of(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


def std_of(values):
    if len(values) < 2:
        return float("nan")
    average = mean_of(values)
    total = 0.0
    for value in values:
        total = total + (value - average) * (value - average)
    return (total / (len(values) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# The attention-edge intervention: a patched GPT-NeoX _attn
# ---------------------------------------------------------------------------
class EdgeController:
    """Holds the live ablation state read by every patched attention layer.

    enabled    : when False, the patched _attn is identical to the original.
    key_index  : the key POSITION whose column is masked (the F0 9F token, or an
                 ordinary token for the specificity control).
    head_flags : [n_layers, n_heads] bool; True means "mask the key in this head".
    """

    def __init__(self, n_layers, n_heads):
        self.enabled = False
        self.key_index = None
        self.head_flags = np.zeros((n_layers, n_heads), dtype=bool)

    def clear(self):
        # Replace the array rather than mutating it in place: set_mask keeps a copy,
        # but never zero a shared buffer the caller might still hold.
        self.enabled = False
        self.key_index = None
        self.head_flags = np.zeros_like(self.head_flags)

    def set_mask(self, key_index, head_flags):
        # Copy the caller's mask so clear() can never mutate the caller's array.
        self.enabled = True
        self.key_index = key_index
        self.head_flags = np.array(head_flags, dtype=bool, copy=True)


def edge_masked_softmax(attn_scores, key_index, layer_head_flags):
    """Softmax over keys after forcing one key's score to -inf for flagged heads.

    attn_scores : [batch, heads, query, key]. layer_head_flags : length-heads bool.
    Adds -inf at [:, flagged_head, :, key_index] so that key's post-softmax weight is
    exactly 0 and each query row renormalises over the remaining keys. Heads whose
    flag is False, and every other key, are untouched.
    """
    n_heads = attn_scores.shape[1]
    biased = attn_scores
    if key_index is not None:
        additive = torch.zeros_like(attn_scores)
        any_flag = False
        for head in range(n_heads):
            if layer_head_flags[head]:
                additive[:, head, :, key_index] = float("-inf")
                any_flag = True
        if any_flag:
            biased = attn_scores + additive
    return torch.nn.functional.softmax(biased, dim=-1)


def _patched_neox_attn(self, query, key, value, attention_mask=None, head_mask=None):
    """A faithful copy of GPTNeoXAttention._attn with one injected step.

    The only change from the stock eager _attn is that the final softmax goes
    through `edge_masked_softmax`, which masks the controller's key column for this
    layer's flagged heads before normalising. With the controller disabled (or no
    flagged head on this layer) the result is bit-for-bit the original softmax.
    """
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    if key_length > self.bias.shape[-1]:
        self._init_bias(key_length, device=key.device)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=self.norm_factor,
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    mask_value = torch.finfo(attn_scores.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    controller = self._eas_controller
    layer_index = self._eas_layer_index
    if controller.enabled:
        layer_flags = controller.head_flags[layer_index]
        attn_weights = edge_masked_softmax(attn_scores, controller.key_index, layer_flags)
    else:
        attn_weights = edge_masked_softmax(attn_scores, None, None)
    attn_weights = attn_weights.to(value.dtype)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = self.attention_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def install_edge_patch(model):
    """Rebind `_attn` on every GPT-NeoX attention layer and return the controller.

    Raises if the model is not GPT-NeoX, since the edge patch is architecture
    specific. The patch is a no-op until the controller is enabled.
    """
    model_type = model.config.model_type
    if model_type != "gpt_neox":
        raise RuntimeError(
            f"edge-ablation patch supports GPT-NeoX (Pythia) only; got {model_type!r}."
        )
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    controller = EdgeController(n_layers, n_heads)
    for layer_index in range(n_layers):
        module = model.gpt_neox.layers[layer_index].attention
        module._eas_controller = controller
        module._eas_layer_index = layer_index
        module._attn = types.MethodType(_patched_neox_attn, module)
    return controller


# ---------------------------------------------------------------------------
# Mean-ablation (optional robustness): overwrite the F0 9F VALUE with a baseline.
# ---------------------------------------------------------------------------
def _neox_value_overwrite_hook(key_index, mean_value_per_layer, layer_index, n_heads, head_dim):
    """GPT-NeoX query_key_value hook: replace the value third at `key_index` with a
    per-head mean baseline vector, leaving query and key untouched."""
    def hook(module, inputs, output):
        reshaped = output.view(output.shape[0], output.shape[1], n_heads, 3 * head_dim)
        for head in range(n_heads):
            replacement = mean_value_per_layer[layer_index][head]
            reshaped[0, key_index, head, 2 * head_dim:3 * head_dim] = replacement
        return reshaped.view(output.shape)
    return hook


def capture_values_neox(model, enc, n_layers, n_heads, head_dim):
    """One forward pass capturing the per-layer VALUE third for every position.

    Returns values[layer] = tensor [n_heads, seq, head_dim] (float32, CPU).
    """
    store = {}
    handles = []

    def make_capture(layer_index):
        def hook(module, inputs, output):
            reshaped = output.view(output.shape[0], output.shape[1], n_heads, 3 * head_dim)
            value = reshaped[:, :, :, 2 * head_dim:3 * head_dim]
            value = value[0].permute(1, 0, 2).contiguous()
            store[layer_index] = value.detach().to("cpu", torch.float32)
        return hook

    for layer_index in range(n_layers):
        module = model.gpt_neox.layers[layer_index].attention.query_key_value
        handles.append(module.register_forward_hook(make_capture(layer_index)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return store


def mean_value_over_positions(values_store, positions, n_layers, n_heads, head_dim):
    """Per-layer per-head mean value vector over the given token positions.

    Returns mean_value[layer] = tensor [n_heads, head_dim].
    """
    mean_value = {}
    for layer_index in range(n_layers):
        layer_values = values_store[layer_index]
        per_head = torch.zeros(n_heads, head_dim)
        for head in range(n_heads):
            accum = torch.zeros(head_dim)
            count = 0
            for position in positions:
                accum = accum + layer_values[head, position, :]
                count = count + 1
            if count > 0:
                per_head[head] = accum / count
        mean_value[layer_index] = per_head
    return mean_value


# ---------------------------------------------------------------------------
# One forward pass -> (logits, attentions), honouring the live controller
# ---------------------------------------------------------------------------
def forward_logits_and_attention(model, enc):
    """Run the model (controller state is read inside the patched _attn) and return
    (logits [n, vocab] float32 CPU, attentions [L, H, n, n] float32 CPU)."""
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    if out.attentions is None:
        raise RuntimeError("attentions=None — load the model with attn_implementation='eager'.")
    logits = out.logits[0].detach().to("cpu", torch.float32)
    attentions = torch.stack([a[0] for a in out.attentions]).to("cpu", torch.float32)
    return logits, attentions


# ---------------------------------------------------------------------------
# Sink-head identification (the existing received-attention ratio rule)
# ---------------------------------------------------------------------------
def received_to_key(attentions, key_index, query_indices):
    """[L, H] received attention from the query positions to a single key position,
    via the project's received_attention reduction on a one-token span."""
    span = SpanLoc(start=key_index, end=key_index + 1, token_indices=[key_index])
    cap = AttentionCapture(text="", input_ids=[], tokens=[], offsets=[], attentions=attentions)
    return received_attention(cap, span, query_indices).detach().numpy()


def sink_head_mask(emoji_received, baseline_received, ratio):
    """Heads attending heavily to the F0 9F key: received above `ratio` x the
    ordinary-token baseline (a near-zero baseline head counts only when the emoji
    key actually receives something). Same rule as scripts 08/09."""
    n_layers = emoji_received.shape[0]
    n_heads = emoji_received.shape[1]
    mask = np.zeros((n_layers, n_heads), dtype=bool)
    for layer in range(n_layers):
        for head in range(n_heads):
            got = float(emoji_received[layer, head])
            base = float(baseline_received[layer, head])
            if got <= 0.0:
                continue
            if base <= 0.0:
                mask[layer, head] = True
            elif got > ratio * base:
                mask[layer, head] = True
    return mask


def count_true(mask):
    count = 0
    for layer in range(mask.shape[0]):
        for head in range(mask.shape[1]):
            if mask[layer, head]:
                count = count + 1
    return count


def random_nonsink_mask(sink_mask, target_count, seed):
    """A mask of `target_count` randomly chosen heads drawn from the NON-sink cells."""
    n_layers = sink_mask.shape[0]
    n_heads = sink_mask.shape[1]
    candidates = []
    for layer in range(n_layers):
        for head in range(n_heads):
            if not sink_mask[layer, head]:
                candidates.append((layer, head))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[:target_count]
    mask = np.zeros((n_layers, n_heads), dtype=bool)
    for layer, head in chosen:
        mask[layer, head] = True
    return mask


# ---------------------------------------------------------------------------
# Stimulus preparation (phase 1): clean pass, locate, classify, keep per stimulus
# ---------------------------------------------------------------------------
def prepare_stimulus(model, tok, template, emoji):
    """Clean pass + bookkeeping for one (template, emoji). Returns a record or None.

    The record carries the tokenised input, the F0 9F key index, the following-word
    key index (for the ordinary-key control), the per-position classes, the clean
    per-token loss, and the clean received-attention arrays for the sink-head rule.
    """
    text, char_start, char_end = build_sentence(template, emoji)
    cap = capture_attention(model, tok, text)

    try:
        span = locate_emoji_span(tok, cap, char_start, char_end, emoji)
    except SpanNotFound as error:
        print(f"  [skip] {emoji!r} in {template[:28]!r}...: {error}")
        return None

    f0_index = span.token_indices[0]
    raw_bytes = token_bytes(tok, cap.input_ids[f0_index])
    tag = bs06.lead_byte_tag(raw_bytes)
    if tag != "F0 9F":
        print(f"  [skip] {emoji!r}: lead fragment is {tag!r}, not F0 9F")
        return None

    if span.end >= cap.n:
        print(f"  [skip] {emoji!r}: no following-word tokens after the emoji span")
        return None

    classes = position_classes(cap, span, f0_index)
    following_key = span.end  # first real word token after the emoji
    query_indices = []
    for position in range(span.end, cap.n):
        query_indices.append(position)

    enc = tok(text, return_tensors="pt")
    logits, attentions = forward_logits_and_attention(model, enc)
    clean_loss = per_token_loss(logits, cap.input_ids)

    emoji_received = received_to_key(attentions, f0_index, query_indices)
    # Ordinary-token baseline: the following-word key positions in this sentence.
    baseline_list = []
    for position in query_indices:
        baseline_list.append(received_to_key(attentions, position, _without(query_indices, position)))
    baseline_received = _mean_arrays(baseline_list)

    record = {
        "emoji": emoji,
        "text": text,
        "enc": enc,
        "input_ids": cap.input_ids,
        "f0_index": f0_index,
        "following_key": following_key,
        "query_indices": query_indices,
        "classes": classes,
        "clean_loss": clean_loss,
        "clean_attentions": attentions,
        "emoji_received": emoji_received,
        "baseline_received": baseline_received,
        "n_continuation": _class_count(classes, CLASS_CONTINUATION),
        "n_following": _class_count(classes, CLASS_FOLLOWING),
    }
    return record


def _without(indices, drop):
    kept = []
    for index in indices:
        if index != drop:
            kept.append(index)
    return kept


def _class_count(classes, class_name):
    count = 0
    for name in classes:
        if name == class_name:
            count = count + 1
    return count


def _mean_arrays(arrays):
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0)


# ---------------------------------------------------------------------------
# Delta-loss for one ablation condition on one stimulus
# ---------------------------------------------------------------------------
def delta_loss_for_condition(model, record, controller, key_index, head_flags):
    """Run one ablated pass, split the loss, and return per-class mean Delta-loss.

    Returns (deltas, ablated_attentions) where deltas maps class -> mean(ablated -
    clean) over that class's target positions in this stimulus.
    """
    controller.set_mask(key_index, head_flags)
    logits, attentions = forward_logits_and_attention(model, record["enc"])
    controller.clear()

    ablated_loss = per_token_loss(logits, record["input_ids"])
    clean_split = split_loss_by_class(record["clean_loss"], record["classes"])
    ablated_split = split_loss_by_class(ablated_loss, record["classes"])

    deltas = {}
    for class_name in (CLASS_CONTINUATION, CLASS_FOLLOWING, CLASS_OTHER):
        clean_mean = mean_of(clean_split[class_name])
        ablated_mean = mean_of(ablated_split[class_name])
        if clean_mean != clean_mean or ablated_mean != ablated_mean:
            deltas[class_name] = float("nan")
        else:
            deltas[class_name] = ablated_mean - clean_mean
    return deltas, attentions


def _mean_random_delta(model, record, controller, random_masks):
    """Per-class mean Delta-loss for the F0 9F key, averaged over the random-head
    draws. Each draw ablates a matched COUNT of non-sink heads; averaging keeps one
    unlucky head set from dominating the control."""
    per_class = {}
    for class_name in (CLASS_CONTINUATION, CLASS_FOLLOWING, CLASS_OTHER):
        per_class[class_name] = []
    for mask in random_masks:
        deltas, _ = delta_loss_for_condition(model, record, controller, record["f0_index"], mask)
        for class_name in (CLASS_CONTINUATION, CLASS_FOLLOWING, CLASS_OTHER):
            value = deltas[class_name]
            if value == value:
                per_class[class_name].append(value)
    averaged = {}
    for class_name in (CLASS_CONTINUATION, CLASS_FOLLOWING, CLASS_OTHER):
        averaged[class_name] = mean_of(per_class[class_name])
    return averaged


def mean_ablation_delta(model, record, arch):
    """Optional robustness: overwrite the F0 9F VALUE with the ordinary-token mean
    (computed over the following-word positions) and measure Delta-loss. Masks no
    attention edge — a different mechanism for the same question."""
    model_type, n_layers, n_heads, n_kv_heads, head_dim = arch
    values_store = capture_values_neox(model, record["enc"], n_layers, n_heads, head_dim)
    mean_value = mean_value_over_positions(
        values_store, record["query_indices"], n_layers, n_heads, head_dim
    )
    handles = []
    for layer_index in range(n_layers):
        module = model.gpt_neox.layers[layer_index].attention.query_key_value
        hook = _neox_value_overwrite_hook(
            record["f0_index"], mean_value, layer_index, n_heads, head_dim
        )
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = model(**record["enc"], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    logits = out.logits[0].detach().to("cpu", torch.float32)
    ablated_loss = per_token_loss(logits, record["input_ids"])
    clean_split = split_loss_by_class(record["clean_loss"], record["classes"])
    ablated_split = split_loss_by_class(ablated_loss, record["classes"])
    deltas = {}
    for class_name in (CLASS_CONTINUATION, CLASS_FOLLOWING, CLASS_OTHER):
        clean_mean = mean_of(clean_split[class_name])
        ablated_mean = mean_of(ablated_split[class_name])
        if clean_mean != clean_mean or ablated_mean != ablated_mean:
            deltas[class_name] = float("nan")
        else:
            deltas[class_name] = ablated_mean - clean_mean
    return deltas


# ---------------------------------------------------------------------------
# Intervention validation (print): the F0 9F key weight is ~0 in sink heads only
# ---------------------------------------------------------------------------
def earliest_sink_layer(sink_mask):
    """Index of the first layer that holds any sink head (or n_layers if none)."""
    n_layers = sink_mask.shape[0]
    n_heads = sink_mask.shape[1]
    for layer in range(n_layers):
        for head in range(n_heads):
            if sink_mask[layer, head]:
                return layer
    return n_layers


def validate_intervention(record, clean_attentions, ablated_attentions, sink_mask):
    """Check the edge mask did exactly what it should, clean vs ablated.

    Two separate claims, because blocking attention in an early layer changes the
    residual stream and so LEGITIMATELY shifts heads in LATER layers:
      - DIRECT: the mean weight from the later queries to the F0 9F key goes to ~0
        on the sink heads (the masked edges).
      - UNAFFECTED: non-sink heads in layers BEFORE the earliest ablated layer are
        bit-for-bit unchanged (their input never changed). Non-sink heads at or
        after that layer are expected to move, via the residual, and are reported
        separately as the downstream cascade — not a failure.
    """
    f0 = record["f0_index"]
    queries = record["query_indices"]
    first_layer = earliest_sink_layer(sink_mask)
    sink_clean = []
    sink_ablated = []
    upstream_clean = []
    upstream_ablated = []
    downstream_clean = []
    downstream_ablated = []
    n_layers = clean_attentions.shape[0]
    n_heads = clean_attentions.shape[1]
    for layer in range(n_layers):
        for head in range(n_heads):
            clean_weight = _mean_weight_to_key(clean_attentions, layer, head, queries, f0)
            ablated_weight = _mean_weight_to_key(ablated_attentions, layer, head, queries, f0)
            if sink_mask[layer, head]:
                sink_clean.append(clean_weight)
                sink_ablated.append(ablated_weight)
            elif layer < first_layer:
                upstream_clean.append(clean_weight)
                upstream_ablated.append(ablated_weight)
            else:
                downstream_clean.append(clean_weight)
                downstream_ablated.append(ablated_weight)
    summary = {
        "first_layer": first_layer,
        "sink_clean": mean_of(sink_clean),
        "sink_ablated": mean_of(sink_ablated),
        "upstream_max_abs_change": _max_abs_change(upstream_clean, upstream_ablated),
        "downstream_max_abs_change": _max_abs_change(downstream_clean, downstream_ablated),
    }
    return summary


def _mean_weight_to_key(attentions, layer, head, queries, key_index):
    total = 0.0
    count = 0
    for query in queries:
        total = total + float(attentions[layer, head, query, key_index])
        count = count + 1
    if count == 0:
        return float("nan")
    return total / count


def _max_abs_change(clean_list, ablated_list):
    largest = 0.0
    for index in range(len(clean_list)):
        change = abs(ablated_list[index] - clean_list[index])
        if change > largest:
            largest = change
    return largest


# ---------------------------------------------------------------------------
# Architecture tuple for the optional value capture
# ---------------------------------------------------------------------------
def model_arch(model):
    config = model.config
    model_type = config.model_type
    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads))
    return model_type, n_layers, n_heads, n_kv_heads, head_dim


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_delta_table(model_name, rows, sink_head_count, n_cells):
    print(f"\nCAUSAL-ABLATION DELTA-LOSS TABLE  —  {model_name}")
    print("block attention to the F0 9F key; Delta-loss = loss_ablated - loss_clean (nats), "
          "mean over stimuli (std)")
    print(f"sink-head set: {sink_head_count} of {n_cells} (layer, head) cells "
          f"(received > {SINK_HEAD_RATIO:g}x ordinary baseline)\n")
    header = (f"  {'condition':22} {'continuation bytes':>24} {'following words':>22}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        cont = f"{row['continuation_mean']:+.4f} ({row['continuation_std']:.4f})"
        foll = f"{row['following_mean']:+.4f} ({row['following_std']:.4f})"
        print(f"  {row['name']:22} {cont:>24} {foll:>22}")


def print_read(model_name, sink_row, random_row, ordinary_row, validation, mean_row):
    print(f"\n  READ [{model_name}]:")
    print(f"    intervention check: weight to F0 9F key on SINK heads "
          f"{validation['sink_clean']:.4f} -> {validation['sink_ablated']:.4f} (target ~0); "
          f"non-sink heads above layer {validation['first_layer']} unchanged "
          f"(max |change| {validation['upstream_max_abs_change']:.2e}); downstream non-sink "
          f"heads move via the residual (max |change| {validation['downstream_max_abs_change']:.2e}, expected).")

    cont = sink_row["continuation_mean"]
    foll = sink_row["following_mean"]
    cont_big = cont > NEGLIGIBLE_DELTA
    foll_big = foll > NEGLIGIBLE_DELTA
    if cont_big and cont > foll:
        print(f"    continuation Delta-loss {cont:+.4f} >> following {foll:+.4f}: LOCAL EMOJI ASSEMBLY — "
              "the read into F0 9F is used to predict the emoji's own continuation bytes (justified).")
    elif foll_big and foll >= cont:
        print(f"    following Delta-loss {foll:+.4f} is the larger effect ({cont:+.4f} on continuation): "
              "a forward 'emoji here' signal into the following words.")
    else:
        print(f"    both Delta-losses are near zero (continuation {cont:+.4f}, following {foll:+.4f}): "
              "the sink attention is largely WASTED — blocking it changes next-token loss little, "
              "despite the fragment carrying a normal value.")

    sink_vs_random_cont = sink_row["continuation_mean"] - random_row["continuation_mean"]
    sink_vs_random_foll = sink_row["following_mean"] - random_row["following_mean"]
    specific = sink_vs_random_cont > 0.0 or sink_vs_random_foll > 0.0
    print(f"    sink vs random-head: continuation {sink_row['continuation_mean']:+.4f} vs "
          f"{random_row['continuation_mean']:+.4f}; following {sink_row['following_mean']:+.4f} vs "
          f"{random_row['following_mean']:+.4f}.")
    if specific:
        print("      -> the effect is larger in the identified sink heads than in matched random "
              "heads: it is head-specific to the sink circuit.")
    else:
        print("      -> the sink heads do not exceed matched random heads: no head-specific effect.")

    print(f"    sink vs ordinary-key (same sink heads, a following-word key): continuation "
          f"{sink_row['continuation_mean']:+.4f} vs {ordinary_row['continuation_mean']:+.4f}; "
          f"following {sink_row['following_mean']:+.4f} vs {ordinary_row['following_mean']:+.4f}.")

    if mean_row is not None:
        print(f"    robustness (mean-ablation of the F0 9F VALUE): continuation "
              f"{mean_row['continuation_mean']:+.4f}, following {mean_row['following_mean']:+.4f} "
              "(same question, value replaced instead of edge masked).")

    # Synthesis: reconcile the three interventions.
    edge_small = abs(cont) <= NEGLIGIBLE_DELTA * 10 and abs(foll) <= NEGLIGIBLE_DELTA * 10
    random_bigger = random_row["continuation_mean"] > cont + NEGLIGIBLE_DELTA
    value_matters = mean_row is not None and mean_row["continuation_mean"] > NEGLIGIBLE_DELTA * 10
    if edge_small and random_bigger and value_matters:
        print("    SYNTHESIS: the F0 9F token IS read — destroying its value wrecks continuation "
              f"prediction ({mean_row['continuation_mean']:+.4f}) and even ordinary non-sink heads "
              f"need its key ({random_row['continuation_mean']:+.4f}) — but the heavy long-range SINK "
              "heads are NOT where that read happens: blocking their edges is a near-no-op. The emoji "
              "is assembled by the value and by ordinary local heads; the sink attraction itself is "
              "largely idle for next-token prediction.")


def summarise_condition(name, per_stimulus):
    """Mean and std over stimuli of the per-stimulus mean Delta-loss, per class."""
    continuation_values = []
    following_values = []
    for deltas in per_stimulus:
        cont = deltas[CLASS_CONTINUATION]
        foll = deltas[CLASS_FOLLOWING]
        if cont == cont:
            continuation_values.append(cont)
        if foll == foll:
            following_values.append(foll)
    row = {
        "name": name,
        "continuation_mean": mean_of(continuation_values),
        "continuation_std": std_of(continuation_values),
        "following_mean": mean_of(following_values),
        "following_std": std_of(following_values),
    }
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_for_model(name, revision):
    model, tok = load_model(name, revision=revision)
    arch = model_arch(model)
    model_type, n_layers, n_heads, n_kv_heads, head_dim = arch

    print(f"\nCAUSAL ABLATION OF THE F0 9F SINK  —  {name}  ({model_type})")
    print("natural carrier sentences; block attention to the F0 9F key in the sink heads;")
    print("Delta next-token loss split by target: continuation bytes vs following words\n")

    if model_type != "gpt_neox":
        print(f"  [unsupported] the edge-ablation patch is GPT-NeoX-specific; {model_type!r} is not run.")
        print("  Pin a GPT-NeoX model (e.g. Pythia-2.8b) as the replication entry to use this leg.")
        return

    controller = install_edge_patch(model)

    # ---- Phase 1: clean passes, locate + classify, build the sink-head set ----
    records = []
    emoji_items = bs06.load_stimuli()["groups"]["A_smp_pictographs"]["items"]
    template_index = 0
    for item in emoji_items:
        emoji = item["emoji"]
        template = CARRIER_TEMPLATES[template_index % len(CARRIER_TEMPLATES)]
        template_index = template_index + 1
        record = prepare_stimulus(model, tok, template, emoji)
        if record is not None:
            records.append(record)

    if len(records) == 0:
        raise SpanNotFound("no F0 9F stimulus could be prepared")

    emoji_received_list = []
    baseline_received_list = []
    for record in records:
        emoji_received_list.append(record["emoji_received"])
        baseline_received_list.append(record["baseline_received"])
    emoji_received_mean = _mean_arrays(emoji_received_list)
    baseline_received_mean = _mean_arrays(baseline_received_list)
    sink_mask = sink_head_mask(emoji_received_mean, baseline_received_mean, SINK_HEAD_RATIO)
    sink_count = count_true(sink_mask)
    n_cells = n_layers * n_heads
    if sink_count == 0:
        raise SpanNotFound("no sink heads identified; cannot run the ablation")
    random_masks = []
    for draw in range(RANDOM_DRAWS):
        random_masks.append(random_nonsink_mask(sink_mask, sink_count, RANDOM_SEED + draw))

    print(f"  prepared {len(records)} F0 9F stimuli; sink-head set = {sink_count} of {n_cells} cells")
    print(f"  random-head control averaged over {RANDOM_DRAWS} independent non-sink draws\n")

    # ---- Phase 2: ablations + controls, per stimulus ----
    sink_deltas = []
    random_deltas = []
    ordinary_deltas = []
    mean_deltas = []
    validation_summary = None

    for record in records:
        sink_delta, ablated_attentions = delta_loss_for_condition(
            model, record, controller, record["f0_index"], sink_mask
        )
        sink_deltas.append(sink_delta)
        if validation_summary is None:
            validation_summary = validate_intervention(
                record, record["clean_attentions"], ablated_attentions, sink_mask
            )

        random_delta = _mean_random_delta(model, record, controller, random_masks)
        random_deltas.append(random_delta)

        ordinary_delta, _ = delta_loss_for_condition(
            model, record, controller, record["following_key"], sink_mask
        )
        ordinary_deltas.append(ordinary_delta)

        mean_delta = mean_ablation_delta(model, record, arch)
        mean_deltas.append(mean_delta)

    sink_row = summarise_condition("sink heads (F0 9F)", sink_deltas)
    random_row = summarise_condition("random heads (F0 9F)", random_deltas)
    ordinary_row = summarise_condition("sink heads (ordinary key)", ordinary_deltas)
    mean_row = summarise_condition("mean-ablation (value)", mean_deltas)

    print_delta_table(name, [sink_row, random_row, ordinary_row, mean_row], sink_count, n_cells)
    print_read(name, sink_row, random_row, ordinary_row, validation_summary, mean_row)
    print()


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    run_for_model(name, revision)


if __name__ == "__main__":
    main()
