import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

import runlog
from runlog import check_env, append_result, free_gb

IDX, ALPHAS, PILOT = [2, 3, 4, 5, 6], [1.0, 2.0, 4.0], (4, 2.0)
MODEL, M, TIE_D1, N_DEV = "models/llama2_7b_q8_meta", 27, 0.05, 2909
SEVEN_B = ("run_pass.py", "s4_2_cptp_parity.py", "s4_2_make_caches.py", "s2_8_cp_dev_grid.py", "s4_grid.py", "llama2_7b")


def fa(a):
    return f"{a:g}".replace(".", "p")


def d1s(tp):
    return "tp" if tp else "notp"


def now():
    return f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"


def log(msg):
    print(f"{now()} {msg}", flush=True)


def other_7b():
    """Other processes running a 7B script (or naming the 7B model), excluding this process and its parents."""
    me = {os.getpid(), os.getppid()}
    ps = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout.splitlines()
    out = []
    for l in ps:
        parts = l.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, cmd = int(parts[0]), parts[2]
        if pid in me or "ps -axo" in cmd or "caffeinate" in cmd.split()[0]:
            continue
        if "python" in cmd and any(k in cmd for k in SEVEN_B):
            out.append(f"{pid} {cmd[:80]}")
    return out


def gate_passed(g):
    return json.load(open(runlog.GATES)).get(g, {}).get("passed", False) if runlog.GATES.exists() else False


def wait_for(g, poll=60, every=600):
    t0, last = time.time(), -every
    while True:
        ok_g, others, free = gate_passed(g), other_7b(), free_gb()
        if ok_g and not others and free >= 5.0:
            log(f"conditions met after {(time.time() - t0) / 60:.1f} min: {g} passed, no other 7B process, free {free:.1f} GB")
            return
        if time.time() - last >= every:
            log(f"waiting: {g} passed {ok_g}; other 7B processes {len(others)}{' ' + str(others) if others else ''}; free {free:.1f} GB")
            last = time.time()
        time.sleep(poll)


def estimate():
    """Minutes, from measured timings: TP batch-1 main pass (S4.2b cache) and S2.8 aux seconds per index (aux cost grows with index)."""
    s28 = json.load(open(ROOT / "results/dev/S2.8_cp_grid.json"))["seconds"]
    aux = {i: s28[f"cp_aux_i{i}_B"] for i in IDX}
    b2 = runlog.LOGS / "S4.2b_caches.json"
    main = json.load(open(b2))["seconds"]["tp_e7_b1_L27"] if b2.exists() else float(np.mean([v for k, v in s28.items() if "_L27" in k]))
    total = 2 * aux[4] + 2 * main + sum(aux[i] for i in IDX if i != 4) + 14 * main
    return dict(main_s=main, aux_s=aux, pilot_min=(2 * aux[4] + 2 * main) / 60, total_min=total / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-gate", metavar="GATE")
    a = ap.parse_args()
    check_env()
    est = estimate()
    log(f"S4 grid started (pid {os.getpid()}); estimate {est['total_min']:.0f} min of compute once it runs "
        f"(main pass {est['main_s']:.0f} s x 16, aux {', '.join(f'i{i} {s:.0f} s' for i, s in est['aux_s'].items())}; pilot {est['pilot_min']:.0f} min)")
    if a.wait_for_gate:
        wait_for(a.wait_for_gate)
    assert gate_passed("S4.2"), "STOP: S4.2 not passed"
    pre = (ROOT / "notes/s4_preregistration.md").read_bytes()
    log(f"pre-registration sha256 {hashlib.sha256(pre).hexdigest()}")
    import mlx.core as mx
    from backend_mlx import MLXBackend, forward_states
    from evalsts import SentenceTable, score, DEV_TASK
    from views import make_items, to_str_words
    from cache import run_cached
    from stage1 import disk
    import cp_mlx
    mx.set_cache_limit(1 << 30)

    r3_dev = json.load(open(runlog.GATES))["S3.2"]["numbers"]["r3_dev"]
    table = SentenceTable(DEV_TASK)
    words = [to_str_words(w) for w in table.words]
    n = len(words)
    T = cp_mlx.cptp_templates()
    items = make_items(words, ["fwd"], "tp")
    assert all(it[2] == T["main_tp"].replace("*sent 0*", cp_mlx.cp_clean(" ".join(w))).strip() for it, w in zip(items, words))
    idx_of = {it[2]: it[0] for it in items}
    aux_items = {tp: [(i, "cptp_aux", cp_mlx.cp_aux_prompt(w, T["aux_tp" if tp else "aux"])) for i, w in enumerate(words)] for tp in (True, False)}
    be = MLXBackend(ROOT / MODEL, ROOT / "models/llama2_tok", np.load(ROOT / "ref/pst_row_3407.npy"))
    tok, model, pst = be.tok, be.model, be.pst_vec

    # sanity: wrapper installed, mode off == TP batch-1 cache (first 20)
    w = cp_mlx.install(model, 4)
    ref = np.load(ROOT / "cache/dev/tp_e7_b1_L27/chunk_0000000.npz")[f"L{M}"][:20].astype(np.float32)
    off = np.stack([forward_states(model, tok(it[2])["input_ids"], [M], "tp", 1, 7, pst, stop_after=M)[0][M] for it in items[:20]])
    c = (off * ref).sum(1) / np.linalg.norm(off, axis=1) / np.linalg.norm(ref, axis=1)
    log(f"sanity: wrapper off vs cache/dev/tp_e7_b1_L27, 20 sentences: min cos {c.min():.8f} (need >= 0.99999)")
    assert c.min() >= 0.99999, "STOP: wrapper (mode off) changes the TP states"
    cp_mlx.uninstall(model, 4)

    gpath, ppath = ROOT / "results/dev/S4.4_cptp_grid.json", ROOT / "results/dev/S4.3_cptp_pilot.json"
    res = json.load(open(gpath)) if gpath.exists() else dict(r3_dev=r3_dev, scores={}, seconds={}, d1=None)
    res["sanity_min_cos"] = float(c.min())
    seen = {json.loads(l)["run"] for l in open(runlog.RESULTS)}
    Bc = {}

    def on_chunk(tag):
        return lambda s, k, comp, t: (mx.clear_cache(), log(f"  {tag} {s + k}/{n} ({'computed' if comp else 'reused'}, {t:.0f} s)"))

    def aux_pass(idx, tp, wr):
        if (idx, tp) in Bc:
            return Bc[(idx, tp)]
        name = f"cptp_aux_i{idx}_{d1s(tp)}_B"
        disk(f"before {name}"); t0 = time.time()
        tmpl = T["aux_tp" if tp else "aux"]
        meta = dict(model=MODEL, plan="tp" if tp else "vanilla", tp_start=1, tp_exit=7, view="cptp_aux", split=DEV_TASK, batched=False,
                    act_layer=idx, stop_after=idx + 1, aux_tp=tp, aux_template_sha=hashlib.sha256(tmpl.encode()).hexdigest()[:12],
                    tensor="o_proj input, last position", pst="ref/pst_row_3407.npy")
        enc = lambda prompts, layers: {"B": np.stack([cp_mlx.capture_B(model, tok, wr, idx, q, "tp" if tp else "vanilla", 1, 7, pst) for q in prompts])}
        B = run_cached(aux_items[tp], enc, str(ROOT / "cache/dev" / name), ["B"], chunk=500, meta=meta, min_free_gb=5.0, on_chunk=on_chunk(name))["B"]
        res["seconds"][name] = time.time() - t0
        disk(f"after {name}")
        assert B.shape == (n, 4096) and np.isfinite(B).all()
        Bc[(idx, tp)] = B
        return B

    def main_pass(idx, alpha, tp, wr, stage):
        name = f"cptp_i{idx}_a{fa(alpha)}_{d1s(tp)}_L27"
        B = aux_pass(idx, tp, wr)
        disk(f"before {name}"); t0 = time.time()
        meta = dict(model=MODEL, plan="tp", tp_start=1, tp_exit=7, stop_after=M, view="fwd", split=DEV_TASK, batched=False, pst="ref/pst_row_3407.npy",
                    cptp=dict(act_layer=idx, alpha=alpha, aux_tp=tp, strategy="scaled",
                              aux_template_sha=hashlib.sha256(T["aux_tp" if tp else "aux"].encode()).hexdigest()[:12]))
        enc = lambda prompts, layers: {M: np.stack([cp_mlx.steered_states(model, tok, wr, q, B[idx_of[q]], alpha, [M], M, "tp", 1, 7, pst)[M]
                                                    for q in prompts])}
        E = run_cached(items, enc, str(ROOT / "cache/dev" / name), [M], chunk=500, meta=meta, min_free_gb=5.0, on_chunk=on_chunk(name))[M]
        res["seconds"][name] = time.time() - t0
        assert np.isfinite(E).all()
        s = score(table, E)[0]
        log(f"RESULT {stage} index {idx} (paper l {idx + 1}) alpha {alpha:g} aux {'with' if tp else 'without'} TP: dev {s:.4f} "
            f"(r3_dev {r3_dev:.4f}, {s - r3_dev:+.2f})")
        run = f"{stage}-cptp-i{idx}-a{alpha:g}-{d1s(tp)}-L27"
        if run not in seen:
            append_result(dict(run=run, stage=stage, split="dev", model=MODEL, plan="tp", tp_start=1, tp_exit=7, batched=False,
                               cptp=meta["cptp"], layers=[str(M)], scores={str(M): {"STS-B dev": s}}, seconds=res["seconds"][name],
                               cache=f"cache/dev/{name}"))
        disk(f"after {name}")
        return s

    # ---- S4.3 pilot: D1
    w = cp_mlx.install(model, PILOT[0])
    pil = {tp: main_pass(PILOT[0], PILOT[1], tp, w, "S4.3-pilot") for tp in (True, False)}
    diff = pil[True] - pil[False]
    d1 = bool(True if abs(diff) < TIE_D1 else diff > 0)
    pilot = dict(index=PILOT[0], alpha=PILOT[1], dev_with_tp=pil[True], dev_without_tp=pil[False], diff_with_minus_without=diff, r3_dev=r3_dev,
                 rule="higher dev; |diff| < 0.05 -> with TP (notes/s4_preregistration.md section 3)", d1_aux_tp=d1, date=now())
    ppath.write_text(json.dumps(pilot, indent=1))
    log(f"S4.3 D1: with TP {pil[True]:.4f}, without TP {pil[False]:.4f}, diff {diff:+.4f} -> aux {'WITH' if d1 else 'WITHOUT'} TP "
        f"({'tie rule' if abs(diff) < TIE_D1 else 'higher dev'})")
    res["d1"] = dict(aux_tp=d1, pilot=pilot)
    res["scores"][f"i{PILOT[0]}|a{PILOT[1]:g}"] = pil[d1]
    gpath.write_text(json.dumps(res, indent=1))
    cp_mlx.uninstall(model, PILOT[0])

    # ---- S4.4 grid at the chosen D1
    for idx in IDX:
        w = cp_mlx.install(model, idx)
        for alpha in ALPHAS:
            k = f"i{idx}|a{alpha:g}"
            if k in res["scores"] and (idx, alpha) == PILOT:
                continue
            res["scores"][k] = main_pass(idx, alpha, d1, w, "S4.4-grid")
            gpath.write_text(json.dumps(res, indent=1))
        cp_mlx.uninstall(model, idx)
        Bc.clear()

    md = [f"# S4.4 CP+TP dev grid (STS-B dev, Spearman x100; aux {'with' if d1 else 'without'} TP by the S4.3 D1 rule; r3_dev {r3_dev:.2f})", "",
          f"S4.3 pilot (index 4, alpha 2): aux with TP {pil[True]:.2f}, without TP {pil[False]:.2f} (diff {diff:+.2f}).", "",
          "| loop index (paper l) | alpha 1 | alpha 2 | alpha 4 |", "|---|---|---|---|"]
    for i in IDX:
        md.append(f"| {i} ({i + 1}) | " + " | ".join(f"{res['scores'][f'i{i}|a{x:g}']:.2f}" for x in ALPHAS) + " |")
    best = max(res["scores"], key=res["scores"].get)
    bi, ba = int(best.split("|")[0][1:]), float(best.split("|a")[1])
    edge = bool(not (3 <= bi <= 5 and ba in (1.0, 2.0)))
    md += ["", f"Sanity (wrapper off vs TP batch-1 cache, 20 sentences): min cos {res['sanity_min_cos']:.8f}.",
           f"Edge check (pre-registered: best must have index 3-5 and alpha 1-2): {'AT AN EDGE -> STOP, report to the user; no extension' if edge else 'not at an edge'}.",
           "No selection, gate or freeze here (scripts/s4_5_select.py, USER-RUN)."]
    res["edge"] = edge
    (ROOT / "results/dev/S4.4_cptp_grid.md").write_text("\n".join(md) + "\n")
    gpath.write_text(json.dumps(res, indent=1))
    print("\n".join(md), flush=True)
    log(f"S4 grid done; free {free_gb():.1f} GB" + ("; STOP: edge rule" if edge else ""))


if __name__ == "__main__":
    main()
