"""Screen-sized evidence printout of SAVED results (no model pass, nothing recomputed except Spearman checks on saved pair files).

Fits a 100-column x 40-row terminal with the command line and prompt (the printout itself is kept to <= 38 lines;
the last line reports the measured size). Every number comes from the files it lists with sha256 and modification time.

Usage (project root, venv active):
  python scripts/show_result.py --evidence E07_r3_tp_e7_test_vs_paper --run r3_tp_e7
  python scripts/show_result.py --evidence E11_run5_and_controls_compare --compare r2 r3 run4 run5 6a 6b 6c
  python scripts/show_result.py --evidence E12_bootstrap_C2 --bootstrap C2
  python scripts/show_result.py --evidence E01_stage0_gates --gates S0
  python scripts/show_result.py --evidence E03_s012_model_load_8bit --gate S0.12
  python scripts/show_result.py --evidence E05_s11_s14_dev_results --dev
  python scripts/show_result.py --evidence E00_env_python311_kernel_tp-mlx --env
Runs: r1 r2 r3 r3p run4 run5 6a 6b 6c, or file stems r1r2_vanilla r3_tp_e7 r3p_tp_e8 r4_reverseeol r5_enh1_fwd_rev_tp
r6a_fwd_shuf_tp r6b_fwd_fwdrevprompt_tp r6c_rev_only_tp.
"""
import argparse
import datetime
import hashlib
import importlib.metadata as im
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.stats import spearmanr

from refnums import load_refs

W, H = 100, 38
OUT = []
T = ROOT / "results/test"


def p(s=""):
    OUT.append(s)


# ---------------------------------------------------------------------------------------------- run registry
RUNS = {  # alias: scores file, layer key, pairs file, logs, reference entry, label
    "r1": ("scores_r1r2_vanilla.json", "final", "pairs_r1r2_vanilla.npz", ["logs/r1r2_vanilla.log"], "paper_prompteol", "r1 PromptEOL, layer 32"),
    "r2": ("scores_r1r2_vanilla.json", "27", "pairs_r1r2_vanilla.npz", ["logs/r1r2_vanilla.log"], "preprint_prompteol_l27", "r2 PromptEOL, layer 27"),
    "r3": ("scores_r3_tp_e7.json", "27", "pairs_r3_tp_e7.npz", ["logs/r3_tp_e7.log"], "paper_prompteol_tp", "r3 TP e=7, layer 27"),
    "r3p": ("scores_r3p_tp_e8.json", "27", "pairs_r3p_tp_e8.npz", ["logs/r3p_tp_e8.log"], "paper_prompteol_tp", "r3' TP e=8, layer 27"),
    "run4": ("scores_r4_reverseeol.json", "27", "pairs_r4_reverseeol.npz", ["logs/r4_rev_keep_final.log", "logs/S2.4.log"],
             "preprint_reverseeol", "run 4 ReverseEOL (no TP)"),
    "run5": ("scores_r5_enh1_fwd_rev_tp.json", "27", "pairs_r5_enh1_fwd_rev_tp.npz", ["logs/r5_tp_rev_keep_final.log", "logs/S3.4.log"],
             None, "run 5 fwd+rev, TP"),
    "6a": ("scores_r6a_fwd_shuf_tp.json", "27", "pairs_r6a_fwd_shuf_tp.npz", ["logs/r6a_tp_shuf.log", "logs/S3.4.log"], None, "6a fwd+shuffled, TP"),
    "6b": ("scores_r6b_fwd_fwdrevprompt_tp.json", "27", "pairs_r6b_fwd_fwdrevprompt_tp.npz", ["logs/r6b_tp_fwd_revprompt.log", "logs/S3.4.log"],
           None, "6b fwd+fwd(rev. wording), TP"),
    "6c": ("scores_r6c_rev_only_tp.json", "27", "pairs_r6c_rev_only_tp.npz", ["logs/r5_tp_rev_keep_final.log", "logs/S3.4.log"], None, "6c rev only, TP"),
}
STEMS = {"r1r2_vanilla": ["r1", "r2"], "r3_tp_e7": ["r3"], "r3p_tp_e8": ["r3p"], "r4_reverseeol": ["run4"], "r5_enh1_fwd_rev_tp": ["run5"],
         "r6a_fwd_shuf_tp": ["6a"], "r6b_fwd_fwdrevprompt_tp": ["6b"], "r6c_rev_only_tp": ["6c"]}


def scores(r):
    f, L = RUNS[r][0], RUNS[r][1]
    return json.load(open(T / f))["scores"][L]


def meta_of(r):
    """-> dict(model, split, plan, tp, view, layer, batching) read from the saved files."""
    f, L = RUNS[r][0], RUNS[r][1]
    js = json.load(open(T / f))
    lay = "32 (after final norm)" if L == "final" else L
    if "meta" in js:                                   # run_pass.py output
        m = js["meta"]
        tp = f"s {m['tp_start']} e {m['tp_exit']}" if m["plan"] == "tp" else "-"
        return dict(model=m["model"], split=m["split"], plan=m["plan"], tp=tp, view=m["view"], layer=lay,
                    batching="equal-length batches" if m["batched"] else "batch 1")
    if "caches" in js:                                 # S2.4 combine (run 4)
        fm, rm, fz = js["caches"]["fwd"]["meta"], js["caches"]["rev"]["meta"], js["frozen_used"]
        return dict(model=fm["model"], split=fm["split"], plan=fm["plan"], tp="-",
                    view=f"fwd + rev ({rm['punct']}), alpha {fz['alpha']} on fwd, norm {'on' if fz['normalize'] else 'off'}",
                    layer=lay, batching="batch 1" if not (fm["batched"] or rm["batched"]) else "mixed")
    cm = [json.load(open(ROOT / c / "meta.json")) for c in js["cache"]]   # S3.4 combine (run 5, 6a-6c)
    v = " + ".join(js["views"]) + (f" ({js['punct']})" if js.get("punct") else "")
    if js.get("alpha") is not None:
        v += f", alpha {js['alpha']} on fwd, norm {'on' if js['normalize'] else 'off'}"
    return dict(model=js["model"], split=js["split"], plan=js["plan"], tp=f"s {js['tp_start']} e {js['tp_exit']}", view=v, layer=lay,
                batching="equal-length batches" if all(m["batched"] for m in cm) else "batch 1")


# ---------------------------------------------------------------------------------------------- helpers
def finfo(path):
    q = ROOT / path
    if not q.exists():
        return f"  {'(missing)':12s} {'':>10s} {'':20s}  {path}"
    h = hashlib.sha256(q.read_bytes()).hexdigest()[:12]
    st = q.stat()
    t = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%b %e %H:%M:%S %Y")
    return f"  {h}  {st.st_size:>10,d}  {t}  {path}"


def files_block(paths):
    p("files (sha256[:12], bytes, modified):")
    for x in dict.fromkeys(paths):
        p(finfo(x))


def kernel_line():
    k = Path.home() / "Library/Jupyter/kernels/tp-mlx/kernel.json"
    if not k.exists():
        return "kernel tp-mlx: NOT FOUND"
    exe = json.load(open(k))["argv"][0]
    same = Path(exe).resolve() == Path(sys.executable).resolve()
    return f"kernel tp-mlx -> {exe.replace(str(ROOT) + '/', '')} (same interpreter as this printout: {'yes' if same else 'NO'})"


def header(name):
    now = datetime.datetime.now().astimezone()
    p("#" * W)
    p(f"##  EVIDENCE  {name}".ljust(W - 2) + "##")
    p("#" * W)
    p(f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}  |  {ROOT}  |  Python {platform.python_version()}")
    p(kernel_line())
    p("SAVED RESULTS: printed from the files listed below (timestamps = when they were written); not a live run")


def df_line():
    out = subprocess.run(["df", "-h", str(Path.home())], capture_output=True, text=True).stdout.strip().splitlines()
    p("df -h ~:  " + " ".join(out[-1].split()[:5]) + "  (fs, size, used, avail, capacity)")


COLS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STS-B", "SICK-R", "Avg"]
LW = 34


def trow(label, vals, signed=False):
    f = "{:+8.2f}" if signed else "{:8.2f}"
    return label[:LW].ljust(LW) + "".join(f.format(vals[c]) for c in COLS)


def thead():
    p("".ljust(LW) + "".join(f"{c:>8s}" for c in COLS))
    p("-" * (LW + 8 * len(COLS)))


def verify_pairs(r):
    """Recompute Spearman x100 from the saved pair cosines and compare with the saved score (max abs diff)."""
    f, L, pf = RUNS[r][0], RUNS[r][1], RUNS[r][2]
    z, s = np.load(T / pf), scores(r)
    return max(abs(100 * spearmanr(z[f"{d}|L{L}"], z[f"{d}|gold"]).correlation - s[d]) for d in COLS[:-1])


# ---------------------------------------------------------------------------------------------- modes
def mode_run(name):
    runs = STEMS.get(name) or ([name] if name in RUNS else None)
    if not runs:
        sys.exit(f"unknown run {name}; use one of {sorted(STEMS) + sorted(RUNS)}")
    E = load_refs()["entries"]
    m = meta_of(runs[0])
    p(f"run {name}: model {m['model']} | split {m['split']} | plan {m['plan']} | TP {m['tp']}")
    p(f"  view {m['view']} | batching {m['batching']}")
    p(f"  layer(s): " + "; ".join(f"{r} = {meta_of(r)['layer']}" for r in runs))
    p("  (27 = hidden_states[27], last token; 32 = layer-32 output after the final RMSNorm)")
    p()
    thead()
    for r in runs:
        ref = RUNS[r][4]
        if ref:
            p(trow(f"{E[ref]['label']} {E[ref]['tag']}", E[ref]["scores"]))
        s = scores(r)
        p(trow(f"{RUNS[r][5]} (ours)", s))
        if ref:
            p(trow(f"  delta ours - {E[ref]['tag']}", {c: s[c] - E[ref]["scores"][c] for c in COLS}, True))
    p("-" * (LW + 8 * len(COLS)))
    p("check: Spearman from saved pair cosines vs saved scores, max |diff|: "
      + ", ".join(f"{r} {verify_pairs(r):.1e}" for r in runs))
    p("[PAPER]/[PREPRINT] values: notes/reference_numbers.json, checked against plan 2.3 and the S2.1 spec")
    p()
    files_block([f"results/test/{RUNS[runs[0]][0]}", f"results/test/{RUNS[runs[0]][2]}"] + RUNS[runs[0]][3])


def mode_compare(runs):
    E = load_refs()["entries"]
    bad = [r for r in runs if r not in RUNS]
    if bad:
        sys.exit(f"unknown runs {bad}; use {sorted(RUNS)}")
    p("side-by-side, test split, Spearman x100; path per run in brackets")
    p()
    thead()
    for ref in ("paper_prompteol_tp", "preprint_reverseeol"):
        p(trow(f"{E[ref]['label']} {E[ref]['tag']}", E[ref]["scores"]))
    p("-" * (LW + 8 * len(COLS)))
    for r in runs:
        m = meta_of(r)
        tag = "TP batched" if m["plan"] == "tp" and "batches" in m["batching"] else f"{m['plan']} {m['batching']}"
        p(trow(f"{RUNS[r][5][:22]} [{tag}]", scores(r)))
    p("-" * (LW + 8 * len(COLS)))
    p("check: max |recomputed - saved| over runs = " + f"{max(verify_pairs(r) for r in runs):.1e}")
    p()
    files_block([f"results/test/{RUNS[r][0]}" for r in runs])


def mode_bootstrap(cid):
    rows = [json.loads(l) for l in open(ROOT / "results/results.jsonl")]
    hit = [l for l in rows if l.get("kind") == "bootstrap" and l["run"].split("-")[1:2] == [cid]]
    if not hit:
        sys.exit(f"no bootstrap line with id {cid} in results/results.jsonl")
    b = hit[-1]
    a, c = b["run"].split("-", 2)[2].split("_vs_")
    agree = b["n_pos_signs"] if b["diff"] > 0 else 7 - b["n_pos_signs"]
    ex0 = b["ci95"][0] > 0 or b["ci95"][1] < 0
    exb = b["ci_bonferroni"][0] > 0 or b["ci_bonferroni"][1] < 0
    p(f"bootstrap {cid}: {a} vs {c}   role: {b['role'].upper()}   stage {b['stage']}   split {b['split']}")
    p(f"  B = {b['n_boot']}, seed {b['seed']}, paired, per-dataset resampling, difference of the 7-set average")
    p()
    p(f"  point difference        {b['diff']:+.2f}")
    p(f"  95% CI                  [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]   excludes 0: {'yes' if ex0 else 'no'}")
    p(f"  Bonferroni CI           [{b['ci_bonferroni'][0]:+.2f}, {b['ci_bonferroni'][1]:+.2f}]   excludes 0: {'yes' if exb else 'no'}")
    p(f"  positive signs          {b['n_pos_signs']}/7   (agreeing with the sign of the difference: {agree}/7)")
    if b["role"] == "primary":
        word = "improves" if (ex0 and agree >= 5 and b["diff"] > 0) else "no resolvable difference" if not ex0 else "differs"
        p(f"  plan 8.4 reading        {word}{', robust' if word == 'improves' and exb else ''}")
    p()
    thead()
    p(trow(f"per-dataset diff ({a} - {c})", dict(b["per_dataset_diff"], Avg=b["diff"]), True))
    p()
    files_block(["results/results.jsonl"] + b.get("sources", []))


def gate_rows(prefix):
    g = json.load(open(ROOT / "gates.json"))
    key = lambda k: [int(x) for x in k[1:].split(".")]
    return [(k, g[k]) for k in sorted((k for k in g if k.startswith(prefix + ".")), key=key)]


def mode_gates(stage):
    rows = gate_rows(stage)
    if not rows:
        sys.exit(f"no gates for {stage} in gates.json")
    p(f"gates.json, stage {stage}: {sum(v['passed'] for _, v in rows)}/{len(rows)} PASS")
    p()
    p(f"{'gate':7s}{'result':8s}{'date':21s}note (first characters)")
    p("-" * W)
    for k, v in rows:
        note = " ".join(str(v.get("note", "")).split())
        p(f"{k:7s}{'PASS' if v['passed'] else 'FAIL':8s}{v.get('date', ''):21s}{note[:W - 36]}")
    p("-" * W)
    p()
    files_block(["gates.json", "gate_log.md"])


def mode_gate(gid):
    g = json.load(open(ROOT / "gates.json"))
    if gid not in g:
        sys.exit(f"{gid} not in gates.json")
    v = g[gid]
    p(f"gate {gid}: {'PASS' if v['passed'] else 'FAIL'}   logged {v.get('date', '')}")
    p("-" * W)
    budget = H - 20 - len(v.get("evidence", []))
    lines = []
    for k, x in v["numbers"].items():
        s = json.dumps(x, ensure_ascii=False) if not isinstance(x, (int, float, str)) else (f"{x:.6g}" if isinstance(x, float) else str(x))
        lines.append(f"  {k}: {s}"[:W])
    for l in lines[:budget]:
        p(l)
    if len(lines) > budget:
        p(f"  ... {len(lines) - budget} more fields in gates.json")
    note = " ".join(str(v.get("note", "")).split())
    p(f"note: {note[:W - 6]}")
    p("-" * W)
    files_block(["gates.json"] + v.get("evidence", []))


def mode_dev():
    rows = [json.loads(l) for l in open(ROOT / "results/results.jsonl")]
    cur = {"vanilla": {}, "tp e=7": {}, "tp e=8": {}}
    sweep = {}
    for l in rows:
        L = l.get("output_layer")
        if L is None:
            continue
        L = 32 if L == "final" else int(L)
        if l.get("stage") == "S1.1":
            cur["vanilla"][L] = l["score"]
        elif l.get("stage") == "S1.2":
            if l["tp_exit"] in (7, 8):
                cur[f"tp e={l['tp_exit']}"][L] = l["score"]
            if L == 27:
                sweep[l["tp_exit"]] = l["score"]
    p("S1.1-S1.4 dev results: STS-B dev, Spearman x100, vanilla batch 1, TP s=1 equal-length batches")
    p("M = output layer (27 = hidden_states[27], last token; 32 = after the final RMSNorm)")
    p()
    p(f"{'M':>4s}" + "".join(f"{k:>12s}" for k in cur) + "      TP end layer sweep (layer 27)")
    ks = sorted(sweep)
    for i, L in enumerate(range(20, 33)):
        extra = f"      e={ks[i]:<2d} {sweep[ks[i]]:7.2f}" if i < len(ks) else ""
        p(f"{L:>4d}" + "".join(f"{cur[k][L]:12.2f}" for k in cur) + extra)
    g = json.load(open(ROOT / "gates.json"))
    p()
    p("gates: " + "  ".join(f"{k} {'PASS' if g[k]['passed'] else 'FAIL'}" for k in ("S1.1", "S1.2", "S1.3", "S1.4")))
    p()
    files_block(["results/results.jsonl", "results/dev/S1.4_exit_curve.md", "results/figures/S1.4_exit_curve.png",
                 "results/dev/pairs_vanilla_L20-32.npz", "results/dev/pairs_tp_s1_e7_L20-32.npz"])


def mode_env():
    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or subprocess.run(cmd, capture_output=True, text=True).stderr.strip()
    p(f"$ python --version      -> {sh([sys.executable, '--version'])}")
    p(f"$ which python          -> {shutil.which('python')}")
    p(f"  sys.executable        -> {sys.executable}")
    p("$ jupyter kernelspec list")
    for l in sh([str(Path(sys.executable).parent / "jupyter"), "kernelspec", "list"]).splitlines():
        p("  " + l.strip()[:W - 2])
    import mlx.core as mx
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    p(f"mlx default device      -> {mx.default_device()}  {info.get('device_name', '')}  "
      f"memory {info.get('memory_size', 0) / 2**30:.0f} GB")
    p()
    pins = {}
    for l in (ROOT / "env/requirements-pinned.txt").read_text().splitlines():
        if "==" in l:
            n, v = l.strip().split("==", 1); pins[n.lower()] = v
    p(f"{'package':16s}{'pinned':12s}{'installed':12s}match")
    for n in ("mlx", "mlx-lm", "mlx-metal", "transformers", "tokenizers", "torch", "numpy", "scipy", "ipykernel", "matplotlib"):
        try:
            inst = im.version(n)
        except im.PackageNotFoundError:
            inst = "-"
        p(f"{n:16s}{pins.get(n, '-'):12s}{inst:12s}{'yes' if pins.get(n) == inst else 'NO'}")
    p()
    files_block(["env/requirements-pinned.txt"])


# ---------------------------------------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--evidence", default="(unnamed)", help="evidence name from results/figures/evidence/EVIDENCE_PLAN.md")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--run"); g.add_argument("--compare", nargs="+"); g.add_argument("--bootstrap")
    g.add_argument("--gates"); g.add_argument("--gate"); g.add_argument("--dev", action="store_true"); g.add_argument("--env", action="store_true")
    args = a.parse_args()
    header(args.evidence)
    p()
    if args.run: mode_run(args.run)
    elif args.compare: mode_compare(args.compare)
    elif args.bootstrap: mode_bootstrap(args.bootstrap)
    elif args.gates: mode_gates(args.gates)
    elif args.gate: mode_gate(args.gate)
    elif args.dev: mode_dev()
    else: mode_env()
    df_line()
    width = max(len(l) for l in OUT)
    p(f"[printout {len(OUT) + 1} lines x {max(width, 0)} cols; fits 100x40 with prompt: "
      f"{'yes' if len(OUT) + 1 <= H and width <= W else 'NO'}]")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
