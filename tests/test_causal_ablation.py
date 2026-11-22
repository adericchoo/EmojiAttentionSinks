"""Offline test for the causal-ablation loss split and the attention-edge mask.

No model is loaded and nothing is downloaded. Two properties are checked:
  - given fake logits and per-position class labels, the per-token loss is split
    into the continuation-byte and following-word classes correctly;
  - a constructed attention-score tensor has its F0 9F key column zeroed (post
    softmax) only for the target heads, and untouched for the rest.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert scripts first, then src, so src/ is at the FRONT: scripts/c2.py would
# otherwise shadow the src/c2 package (the numbered scripts avoid this because
# _bootstrap puts src/ at the front before c2 is first imported).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch


def _load_script_module():
    path = os.path.join(ROOT, "scripts", "10_causal_ablation.py")
    spec = importlib.util.spec_from_file_location("causal_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ca = _load_script_module()


def _one_hot_logits(input_ids, vocab, confident_id, confidence=10.0):
    """Logits [n, vocab] that put almost all mass on `confident_id` at every
    position, so the per-token loss is small when the next token IS confident_id
    and large otherwise — a controlled input for the split."""
    logits = torch.zeros(len(input_ids), vocab)
    for position in range(len(input_ids)):
        logits[position, confident_id] = confidence
    return logits


def test_loss_split_groups_positions_by_class():
    # Sequence of 6 tokens. Classes label each POSITION; position 0 has no loss.
    vocab = 8
    input_ids = [1, 2, 3, 4, 5, 2]
    classes = [
        ca.CLASS_OTHER,          # 0 — skipped (no token predicts it)
        ca.CLASS_OTHER,          # 1 — the F0 9F lead itself
        ca.CLASS_CONTINUATION,   # 2 — a continuation byte
        ca.CLASS_CONTINUATION,   # 3 — a continuation byte
        ca.CLASS_FOLLOWING,      # 4 — a following word
        ca.CLASS_FOLLOWING,      # 5 — a following word
    ]
    # Row p of the logits predicts token at position p+1. Make every row confident
    # that the next token is id 2: then any target position whose token IS id 2 has
    # near-zero loss, and every other target position has a large loss. Here only
    # the final following-word token (position 5) is id 2.
    logits = _one_hot_logits(input_ids, vocab, confident_id=2)
    loss = ca.per_token_loss(logits, input_ids)

    # position 0 carries NaN, every other position has a finite loss
    assert loss[0] != loss[0]
    for position in range(1, len(input_ids)):
        assert loss[position] == loss[position]

    split = ca.split_loss_by_class(loss, classes)
    # two continuation targets (positions 2, 3) and two following targets (4, 5)
    assert len(split[ca.CLASS_CONTINUATION]) == 2
    assert len(split[ca.CLASS_FOLLOWING]) == 2
    # position 1 is the only scored 'other'
    assert len(split[ca.CLASS_OTHER]) == 1

    # The split must route each position's loss value unchanged into its class.
    assert split[ca.CLASS_CONTINUATION] == [loss[2], loss[3]]
    assert split[ca.CLASS_FOLLOWING] == [loss[4], loss[5]]
    assert split[ca.CLASS_OTHER] == [loss[1]]

    # Only position 5's token is the confident id 2 -> its loss is ~0; the other
    # scored positions predict a non-confident token -> large loss.
    assert split[ca.CLASS_FOLLOWING][1] < 1e-3
    assert split[ca.CLASS_FOLLOWING][0] > 1.0
    assert split[ca.CLASS_CONTINUATION][0] > 1.0
    assert split[ca.CLASS_CONTINUATION][1] > 1.0


def test_edge_mask_zeroes_key_only_for_target_heads():
    # attn_scores [batch=1, heads=3, query=4, key=4]; uniform scores so an
    # untouched softmax row is flat (0.25 each).
    scores = torch.zeros(1, 3, 4, 4)
    key_index = 1
    # Flag heads 0 and 2; leave head 1 untouched.
    flags = [True, False, True]

    weights = ca.edge_masked_softmax(scores, key_index, flags)

    # Flagged heads: the masked key column is exactly 0 at every query row, and each
    # row still sums to 1 (renormalised over the remaining keys).
    for head in (0, 2):
        for query in range(4):
            assert float(weights[0, head, query, key_index]) == 0.0
            row_sum = float(weights[0, head, query].sum())
            assert abs(row_sum - 1.0) < 1e-6

    # Unflagged head: the key column keeps its ordinary uniform weight.
    for query in range(4):
        assert abs(float(weights[0, 1, query, key_index]) - 0.25) < 1e-6


def test_edge_mask_is_identity_when_no_head_flagged():
    scores = torch.randn(1, 2, 3, 3)
    plain = torch.nn.functional.softmax(scores, dim=-1)
    # key_index None -> no masking, identical to a plain softmax.
    same = ca.edge_masked_softmax(scores, None, None)
    assert torch.allclose(plain, same, atol=1e-6)
