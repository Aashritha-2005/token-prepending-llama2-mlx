"""Chunked, resumable embedding cache and frozen-settings guard (plan Appendix F).

Deviation from Appendix F: freshly computed chunks are returned after the same float16 round trip
as chunks loaded from disk, so an interrupted-and-resumed run gives bit-identical results to an
uninterrupted one (gate S0.11).
"""
import json
import os
import time

import numpy as np

from runlog import ROOT, free_gb

MIN_FREE_GB = 4.0   # low-disk mode (user, 2026-11-03): refuse to compute a new chunk below this


class LowDiskError(RuntimeError):
    pass


def _lkey(L):
    return f"L{L}"


def run_cached(items, encode_fn, cache_dir, layers, chunk=500, meta=None, min_free_gb=MIN_FREE_GB, on_chunk=None):
    """items: list of (sentence_index, view, prompt). encode_fn(prompts, layers) -> {L: (n, d)}.
    Before computing each new chunk, raises LowDiskError if free disk < min_free_gb. Finished chunks
    stay on disk, so re-running the same call after freeing space resumes where it stopped.
    on_chunk(start, n_in_chunk, computed: bool, seconds) is called after every chunk (for progress output)."""
    os.makedirs(cache_dir, exist_ok=True)
    mpath = os.path.join(cache_dir, "meta.json")
    meta = dict(meta or {}, layers=[str(L) for L in layers], n_items=len(items), chunk=chunk)
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        assert old == meta, f"cache {cache_dir} was written with different settings:\n{old}\nvs\n{meta}"
    else:
        json.dump(meta, open(mpath, "w"), indent=1)
    out = {L: [] for L in layers}
    for start in range(0, len(items), chunk):
        f = os.path.join(cache_dir, f"chunk_{start:07d}.npz")
        t0 = time.perf_counter(); computed = not os.path.exists(f)
        if not computed:
            d = np.load(f); st = {L: d[_lkey(L)] for L in layers}
        else:
            free = free_gb()
            if free < min_free_gb:
                raise LowDiskError(f"Only {free:.1f} GB free (< {min_free_gb} GB) before chunk {start} of {cache_dir}. "
                                   f"Free disk space, then re-run the same cell: {start // chunk} finished chunks will be reused.")
            st = encode_fn([it[2] for it in items[start:start + chunk]], layers)
            st = {L: np.asarray(st[L]).astype(np.float16) for L in layers}
            tmp = f[:-4] + ".tmp.npz"
            np.savez(tmp, **{_lkey(L): st[L] for L in layers})
            os.replace(tmp, f)
        for L in layers: out[L].append(np.asarray(st[L], dtype=np.float32))
        if on_chunk is not None:
            on_chunk(start, len(items[start:start + chunk]), computed, time.perf_counter() - t0)
    return {L: np.concatenate(out[L]) for L in layers}


def require_frozen(name, settings, path=ROOT / "frozen_settings.json"):
    fs = json.load(open(path)) if os.path.exists(path) else {}
    if name not in fs or fs[name].get("settings") != settings:
        raise RuntimeError(f"Settings for '{name}' not frozen (or changed). Write them to {path} first.")
