"""S2.8 CP-alone dev grid (AGENT-RUN; plan 10.2, notes/s2_8_preregistration.md). STS-B dev only. Logs no gate, freezes nothing.

Grid: loop index (= CP act_layer) in {2,3,4,5,6} x alpha in {1,2,4}; vanilla, layer 27, stop after 27, batch 1, 8-bit model.
Auxiliary prompt = ref/cp_repo/evaluate_intervention.py:201 (CP cleaning, no <PST>); B = last-position o_proj input of
layers[index] from the auxiliary pass (stop_after = index + 1), computed once per index and cached.
Steering = src/cp_mlx.py (parity-tested in S2.7): last-position o_proj input <- alpha * (A - B).
Caches: cache/dev/cp_aux_i<idx>_B (B vectors), cache/dev/cp_i<idx>_a<alpha>_L27 (states). Scores: results/dev/S2.8_cp_grid.json
and .md, plus one results.jsonl line per pass (stage "S2.8-grid", dev). Resumable: finished chunks are reused.

Usage (project root, venv active):
  caffeinate -dims python scripts/s2_8_cp_dev_grid.py > logs/S2.8_grid.log 2>&1
  python scripts/s2_8_cp_dev_grid.py --extension IDX 0.5    (the one pre-registered extension, only if the best alpha is 1)
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True                      # importing nothing from ref/cp_repo, but keep caches out of it regardless
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from runlog import check_env, append_result, require_gates, free_gb, RESULTS
from evalsts import SentenceTable, score, DEV_TASK
from views import make_items, to_str_words
from cache import run_cached
from stage1 import disk

IDX, ALPHAS = [2, 3, 4, 5, 6], [1.0, 2.0, 4.0]
MODEL, M = "models/llama2_7b_q8_meta", 27


def fa(a):
    return f"{a:g}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extension", nargs=2, metavar=("IDX", "ALPHA"), help="one extra pass (pre-registered: alpha 0.5 at the best index)")
    a = ap.parse_args()
    check_env()
    require_gates(["S2.6", "S2.7"])
    import mlx.core as mx
    from backend_mlx import MLXBackend, forward_states
    import cp_mlx
    mx.set_cache_limit(1 << 30)

    r2_dev = json.load(open(ROOT / "gates.json"))["S1.1"]["numbers"]["r2_dev"]
    table = SentenceTable(DEV_TASK)
    words = [to_str_words(w) for w in table.words]
    n = len(words)
    main_t, aux_t = cp_mlx.cp_templates()
    items = make_items(words, ["fwd"], "vanilla")
    assert all(it[2] == main_t.replace("*sent 0*", cp_mlx.cp_clean(" ".join(w))).strip() for it, w in zip(items, words)), "main prompt != CP :194"
    aux_items = [(i, "cp_aux", cp_mlx.cp_aux_prompt(w, aux_t)) for i, w in enumerate(words)]
    idx_of = {it[2]: it[0] for it in items}
    aux_idx_of = {it[2]: it[0] for it in aux_items}
    assert len(idx_of) == len(aux_idx_of) == n
    aux_sha = hashlib.sha256(aux_t.encode()).hexdigest()[:12]
    print(f"S2.8 grid start {datetime.datetime.now():%Y-%m-%d %H:%M:%S}; dev n = {n}; r2_dev {r2_dev:.4f}; aux template sha {aux_sha}", flush=True)
    print(f"aux template (CP :201): {aux_t}", flush=True)

    be = MLXBackend(ROOT / MODEL, ROOT / "models/llama2_tok", np.load(ROOT / "ref/pst_row_3407.npy"))
    tok, model = be.tok, be.model

    # ---- pre-registered sanity: un-steered wrapper == S1.1 vanilla layer-27 dev states (first 20)
    w = cp_mlx.install(model, 4)
    ref = np.load(ROOT / "cache/dev/vanilla_L20-32/chunk_0000000.npz")["L27"][:20].astype(np.float32)
    off = np.stack([forward_states(model, tok(it[2])["input_ids"], [M], "vanilla", stop_after=M)[0][M] for it in items[:20]])
    c = (off * ref).sum(1) / np.linalg.norm(off, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"sanity: un-steered wrapper vs S1.1 cache, 20 sentences: min cos {c.min():.8f} (need >= 0.99999)", flush=True)
    assert c.min() >= 0.99999, "STOP: wrapper (mode off) changes the vanilla states"
    cp_mlx.uninstall(model, 4)

    out_path = ROOT / "results/dev/S2.8_cp_grid.json"
    res = json.load(open(out_path)) if out_path.exists() else dict(r2_dev=r2_dev, aux_template=aux_t, scores={}, seconds={}, sanity_min_cos=None)
    res["sanity_min_cos"] = float(c.min())
    seen = {json.loads(l)["run"] for l in open(RESULTS)}

    def on_chunk(tag):
        return lambda s, k, comp, t: (mx.clear_cache(), print(f"  {tag} {s + k}/{n} ({'computed' if comp else 'reused'}, {t:.0f} s)", flush=True))

    plan = [(int(a.extension[0]), [float(a.extension[1])])] if a.extension else [(i, ALPHAS) for i in IDX]
    for idx, alphas in plan:
        w = cp_mlx.install(model, idx)
        name = f"cp_aux_i{idx}_B"
        disk(f"before {name}"); t0 = time.time()
        meta = dict(model=MODEL, plan="vanilla", view="cp_aux", split=DEV_TASK, batched=False, act_layer=idx, stop_after=idx + 1,
                    aux_template_sha=aux_sha, tensor="o_proj input, last position")
        B = run_cached(aux_items, lambda prompts, layers: {"B": np.stack([cp_mlx.capture_B(model, tok, w, idx, q) for q in prompts])},
                       str(ROOT / "cache/dev" / name), ["B"], chunk=500, meta=meta, min_free_gb=5.0, on_chunk=on_chunk(name))["B"]
        res["seconds"][name] = time.time() - t0
        disk(f"after {name}")
        assert B.shape == (n, 4096) and np.isfinite(B).all()
        for alpha in alphas:
            name = f"cp_i{idx}_a{fa(alpha)}_L27"
            disk(f"before {name}"); t0 = time.time()
            meta = dict(model=MODEL, plan="vanilla", view="fwd", split=DEV_TASK, batched=False, stop_after=M,
                        cp=dict(act_layer=idx, alpha=alpha, strategy="scaled", aux_template_sha=aux_sha))
            enc = lambda prompts, layers: {M: np.stack([cp_mlx.steered_states(model, tok, w, q, B[idx_of[q]], alpha, [M], M)[M] for q in prompts])}
            E = run_cached(items, enc, str(ROOT / "cache/dev" / name), [M], chunk=500, meta=meta, min_free_gb=5.0, on_chunk=on_chunk(name))[M]
            res["seconds"][name] = time.time() - t0
            assert np.isfinite(E).all()
            s = score(table, E)[0]
            res["scores"][f"i{idx}|a{alpha:g}"] = s
            print(f"RESULT index {idx} (paper l {idx + 1}) alpha {alpha:g}: dev {s:.4f}  (r2_dev {r2_dev:.4f}, {s - r2_dev:+.2f})", flush=True)
            run = f"S2.8-grid-cp-i{idx}-a{alpha:g}-L27"
            if run not in seen:
                append_result(dict(run=run, stage="S2.8-grid", split="dev", model=MODEL, plan="vanilla", cp=dict(act_layer=idx, alpha=alpha, strategy="scaled"),
                                   layers=[str(M)], scores={str(M): {"STS-B dev": s}}, seconds=res["seconds"][name], cache=f"cache/dev/{name}"))
            out_path.write_text(json.dumps(res, indent=1))
            disk(f"after {name}")
        cp_mlx.uninstall(model, idx)

    # numbers-only table (no selection here; selection is scripts/s2_8_select.py, USER-RUN)
    al = sorted({float(k.split("|a")[1]) for k in res["scores"]})
    md = [f"# S2.8 CP-alone dev grid (STS-B dev, Spearman x100; r2_dev {r2_dev:.2f})", "",
          "| loop index (paper l) | " + " | ".join(f"alpha {x:g}" for x in al) + " |", "|---|" + "---|" * len(al)]
    for i in sorted({int(k.split("|")[0][1:]) for k in res["scores"]}):
        md.append(f"| {i} ({i + 1}) | " + " | ".join(f"{res['scores'][f'i{i}|a{x:g}']:.2f}" if f"i{i}|a{x:g}" in res["scores"] else "-" for x in al) + " |")
    md += ["", f"Sanity (un-steered wrapper vs S1.1 cache, 20 sentences): min cos {res['sanity_min_cos']:.8f}.", "No selection applied here."]
    (ROOT / "results/dev/S2.8_cp_grid.md").write_text("\n".join(md) + "\n")
    out_path.write_text(json.dumps(res, indent=1))
    print("\n".join(md), flush=True)
    print(f"S2.8 grid done {datetime.datetime.now():%Y-%m-%d %H:%M:%S}; free {free_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
