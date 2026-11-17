# MLX backend. forward_states() is the plan's block
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

PST_ID = 32000

def forward_states(model, ids, out_layers, plan="vanilla", tp_start=1, tp_exit=7,
                   pst_vec=None, stop_after=None):
    """ids: list[int] for ONE prompt (BOS first). out_layers: iterable of 1-indexed layer numbers
    and/or 'final' (= layer 32 after the final RMSNorm = HF hidden_states[-1]); 0 = embedding output.
    Returns ({layer: np.float32 (hidden,)}, repl_idx) where repl_idx lists the 0-indexed layers before which <PST> was overwritten."""
    ids = np.array(ids)
    pos = None
    if plan == "tp":
        hits = np.where(ids == PST_ID)[0]
        assert len(hits) == 1, f"<PST> must occur exactly once, found {len(hits)}"
        pos = int(hits[0]); ids = ids.copy(); ids[pos] = 0       # any valid id; row overwritten next
    h = model.model.embed_tokens(mx.array(ids)[None])            # (1, T, d); works for quantised embeddings
    if pos is not None:
        h = mx.concatenate([h[:, :pos], mx.array(pst_vec).astype(h.dtype)[None, None], h[:, pos + 1:]], 1)
    mask = create_attention_mask(h, None)                       # exactly what LlamaModel.__call__ does
    want = set(out_layers); states = {}; repl_idx = []
    if 0 in want: states[0] = h[:, -1, :]
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):              # i is 0-indexed = repo's `index`
        if plan == "tp" and tp_start <= i < tp_exit:            # repo modeling_llama.py:565-568
            h = mx.concatenate([h[:, :pos], h[:, -1:, :], h[:, pos + 1:]], 1)   # last -> <PST>, BEFORE layer i
            repl_idx.append(i)
        h = layer(h, mask, cache=None)
        L = i + 1                                               # 1-indexed layer just computed
        if L in want and L < n_layers: states[L] = h[:, -1, :]  # raw output (HF hidden_states[L])
        if stop_after is not None and L == stop_after: break
    if "final" in want:
        assert stop_after is None or stop_after == n_layers, "'final' needs the full forward"
        states["final"] = model.model.norm(h)[:, -1, :]
    mx.eval(list(states.values()))                              # MLX is lazy: force computation
    end = stop_after if stop_after is not None else n_layers     # layers actually run
    assert plan != "tp" or repl_idx == list(range(tp_start, min(tp_exit, end)))
    return {k: np.array(v.astype(mx.float32))[0] for k, v in states.items()}, repl_idx


def forward_states_batch(model, ids_list, out_layers, plan="vanilla", tp_start=1, tp_exit=7,
                         pst_vec=None, stop_after=None):
    """Equal-length batch version of forward_states (S0.13 step 2): no padding, so the causal mask and
    the <PST> position are shared by all rows. Accepted only if it matches forward_states (cos >= 0.999)."""
    ids = np.array(ids_list)
    assert ids.ndim == 2, "all prompts in a batch must have the same token length"
    pos = None
    if plan == "tp":
        hits = [np.where(r == PST_ID)[0] for r in ids]
        assert all(len(x) == 1 for x in hits) and len({int(x[0]) for x in hits}) == 1, "<PST> position differs within batch"
        pos = int(hits[0][0]); ids = ids.copy(); ids[:, pos] = 0
    B = ids.shape[0]
    h = model.model.embed_tokens(mx.array(ids))
    if pos is not None:
        v = mx.broadcast_to(mx.array(pst_vec).astype(h.dtype)[None, None], (B, 1, h.shape[-1]))
        h = mx.concatenate([h[:, :pos], v, h[:, pos + 1:]], 1)
    mask = create_attention_mask(h, None)
    want = set(out_layers); states = {}; n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        if plan == "tp" and tp_start <= i < tp_exit:
            h = mx.concatenate([h[:, :pos], h[:, -1:, :], h[:, pos + 1:]], 1)
        h = layer(h, mask, cache=None)
        L = i + 1
        if L in want and L < n_layers: states[L] = h[:, -1, :]
        if stop_after is not None and L == stop_after: break
    if "final" in want:
        states["final"] = model.model.norm(h)[:, -1, :]
    mx.eval(list(states.values()))
    return {k: np.array(v.astype(mx.float32)) for k, v in states.items()}


def pst_equals_last_before_start(model, ids, tp_start, pst_vec):
    """S0.10 debug assertion: run layers 0..tp_start-1, apply the copy before layer tp_start
    exactly as forward_states does, and return whether the <PST> row equals the last row (exact)."""
    ids = np.array(ids); pos = int(np.where(ids == PST_ID)[0][0]); ids = ids.copy(); ids[pos] = 0
    h = model.model.embed_tokens(mx.array(ids)[None])
    h = mx.concatenate([h[:, :pos], mx.array(pst_vec).astype(h.dtype)[None, None], h[:, pos + 1:]], 1)
    mask = create_attention_mask(h, None)
    for i, layer in enumerate(model.model.layers):
        if i == tp_start:
            h = mx.concatenate([h[:, :pos], h[:, -1:, :], h[:, pos + 1:]], 1)
            mx.eval(h)
            return bool(np.array_equal(np.array(h[0, pos]), np.array(h[0, -1])))
        h = layer(h, mask, cache=None)
    raise ValueError("tp_start beyond the last layer")


class MLXBackend:
    """encode(prompts, layers, ...) -> {layer: (n, hidden) float32}. Tokenizer = HF AutoTokenizer + <PST>."""
    def __init__(self, model_path, tok_path, pst_vec):
        from transformers import AutoTokenizer
        self.model, _ = load(str(model_path))
        self.tok = AutoTokenizer.from_pretrained(str(tok_path))
        self.tok.pad_token_id = 0; self.tok.padding_side = "left"
        self.tok.add_tokens(["<PST>"])
        assert self.tok.convert_tokens_to_ids("<PST>") == PST_ID
        self.pst_vec = np.asarray(pst_vec, dtype=np.float32)

    def encode(self, prompts, layers, plan="vanilla", tp_start=1, tp_exit=7, stop_after=None):
        out = {L: [] for L in layers}
        for p in prompts:
            st, _ = forward_states(self.model, self.tok(p)["input_ids"], layers, plan, tp_start, tp_exit,
                                   self.pst_vec, stop_after)
            for L in layers: out[L].append(st[L])
        return {L: np.stack(v) for L, v in out.items()}

    def encode_batched(self, prompts, layers, plan="vanilla", tp_start=1, tp_exit=7, stop_after=None, bs=16):
        """Equal-length batches (no padding) via forward_states_batch; output rows in the order of `prompts`."""
        from collections import defaultdict
        ids = [self.tok(p)["input_ids"] for p in prompts]
        by_len = defaultdict(list)
        for j, x in enumerate(ids): by_len[len(x)].append(j)
        out = {L: [None] * len(prompts) for L in layers}
        for idx in by_len.values():
            for k in range(0, len(idx), bs):
                g = idx[k:k + bs]
                st = forward_states_batch(self.model, [ids[j] for j in g], layers, plan, tp_start, tp_exit, self.pst_vec, stop_after)
                for L in layers:
                    for r, j in enumerate(g): out[L][j] = st[L][r]
        return {L: np.stack(v) for L, v in out.items()}
