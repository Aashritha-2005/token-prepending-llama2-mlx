"""Stage 1 helpers: one cached dev pass, per-layer scoring, per-pair cosine files."""
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np

from cache import run_cached
from evalsts import score
from runlog import ROOT, free_gb
from views import make_items

FLOOR_GB = 5


def disk(tag):
    """Print df -h ~ and stop cleanly below the 5 GB floor."""
    line = subprocess.run(["df", "-h", str(Path.home())], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"[df {tag}] {line}", flush=True)
    if free_gb() < FLOOR_GB:
        raise SystemExit(f"STOP: only {free_gb():.1f} GB free (< {FLOOR_GB} GB).")


def run_pass(be, table, name, plan, layers, tp_start=0, tp_exit=0, stop_after=None, batched=False, model_id=""):
    """One pass over every unique sentence of `table` (view fwd). Returns ({layer: (n, d)}, seconds).
    Cached in cache/dev/<name>; a finished cache is reloaded, not recomputed."""
    items = make_items(table.words, ["fwd"], plan)
    if batched:
        enc = partial(be.encode_batched, plan=plan, tp_start=tp_start, tp_exit=tp_exit, stop_after=stop_after)
    else:
        enc = partial(be.encode, plan=plan, tp_start=tp_start, tp_exit=tp_exit, stop_after=stop_after)
    meta = dict(model=model_id, plan=plan, tp_start=tp_start, tp_exit=tp_exit, stop_after=stop_after, view="fwd",
                split=table.name, batched=batched, pst="ref/pst_row_3407.npy" if plan == "tp" else None)
    disk(f"before {name}")
    t0 = time.perf_counter()
    E = run_cached(items, enc, str(ROOT / "cache" / "dev" / name), layers, chunk=500, meta=meta)
    secs = time.perf_counter() - t0
    disk(f"after {name}")
    return E, secs


def score_layers(table, E, name):
    """Spearman x100 per stored layer; saves per-pair cosines to results/dev/pairs_<name>.npz."""
    out, pairs = {}, {}
    for L, M in E.items():
        s, ps = score(table, M)
        out[L] = s
        pairs[f"L{L}"] = np.concatenate([ps[k][0] for k in ps])
    pairs["gold"] = np.concatenate([g for _, (_, _, g) in table.pairs.items()])
    d = ROOT / "results" / "dev"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"pairs_{name}.npz", **pairs)
    return out
