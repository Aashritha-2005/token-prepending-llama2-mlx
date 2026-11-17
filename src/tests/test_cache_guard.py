"""Low-disk guard test: the guard stops a run before a new chunk, and resuming gives identical output."""
import os
import shutil
import tempfile

import numpy as np

import cache
from cache import LowDiskError, run_cached


def _enc(prompts, layers):
    return {L: np.stack([np.random.default_rng(abs(hash(p)) % 2**32).standard_normal(16) for p in prompts]) for L in layers}


def test_guard_and_resume():
    items = [(i, "fwd", f"prompt {i}") for i in range(1050)]   # 3 chunks of 500
    d_ref, d = tempfile.mkdtemp(), tempfile.mkdtemp()
    real_free = cache.free_gb
    try:
        ref = run_cached(items, _enc, d_ref, [27], chunk=500)
        readings = iter([100.0, 1.0])                  # plenty before chunk 0, "disk full" before chunk 1
        cache.free_gb = lambda: next(readings)
        raised = False
        try:
            run_cached(items, _enc, d, [27], chunk=500)
        except LowDiskError as e:
            raised = "finished chunks will be reused" in str(e)
        cache.free_gb = real_free
        written = sorted(f for f in os.listdir(d) if f.startswith("chunk_"))
        assert raised and written == ["chunk_0000000.npz"], (raised, written)
        out = run_cached(items, _enc, d, [27], chunk=500)   # resume after "freeing space"
        assert np.array_equal(out[27], ref[27])
        return {"raised_before_chunk_1": raised, "chunks_kept": written, "resume_identical": True}
    finally:
        cache.free_gb = real_free
        shutil.rmtree(d_ref); shutil.rmtree(d)
