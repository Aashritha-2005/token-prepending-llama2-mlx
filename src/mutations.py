"""Deliberately wrong TP variants for the S0.10 mutation test (ported from verified_checks/mutation_check.py).
Returns last-token states at every hidden_states index (0..n_layers, last one after the final norm)."""
import numpy as np
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from backend_mlx import PST_ID

VARIANTS = ["correct", "copy_after", "exit+1", "start-1", "pst_to_last"]


def run_variant(model, ids, variant, pst_vec, tp_start=1, tp_exit=7):
    ids = np.array(ids); pos = int(np.where(ids == PST_ID)[0][0]); ids = ids.copy(); ids[pos] = 0
    h = model.model.embed_tokens(mx.array(ids)[None])
    h = mx.concatenate([h[:, :pos], mx.array(pst_vec).astype(h.dtype)[None, None], h[:, pos + 1:]], 1)
    mask = create_attention_mask(h, None); out = [h[:, -1]]
    e = tp_exit + 1 if variant == "exit+1" else tp_exit
    s = tp_start - 1 if variant == "start-1" else tp_start
    for i, layer in enumerate(model.model.layers):
        rep = s <= i < e
        if rep and variant in ("correct", "exit+1", "start-1"):
            h = mx.concatenate([h[:, :pos], h[:, -1:], h[:, pos + 1:]], 1)
        if rep and variant == "pst_to_last":
            h = mx.concatenate([h[:, :-1], h[:, pos:pos + 1]], 1)
        h = layer(h, mask, cache=None)
        if rep and variant == "copy_after":
            h = mx.concatenate([h[:, :pos], h[:, -1:], h[:, pos + 1:]], 1)
        out.append(h[:, -1])
    out[-1] = model.model.norm(h)[:, -1]
    mx.eval(out)
    return np.stack([np.array(o.astype(mx.float32))[0] for o in out])
