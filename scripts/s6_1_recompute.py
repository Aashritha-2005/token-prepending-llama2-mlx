import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from stats import paired_bootstrap

OUT = ROOT / "results/figures/report_v3"
DS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STS-B", "SICK-R"]
TOL = 1e-9
# alias -> (pairs file, layer key)
PAIRS = {"r1": ("pairs_r1r2_vanilla.npz", "Lfinal"), "r2": ("pairs_r1r2_vanilla.npz", "L27"), "r3": ("pairs_r3_tp_e7.npz", "L27"),
         "r3p": ("pairs_r3p_tp_e8.npz", "L27"), "run4": ("pairs_r4_reverseeol.npz", "L27"), "run5": ("pairs_r5_enh1_fwd_rev_tp.npz", "L27"),
         "6a": ("pairs_r6a_fwd_shuf_tp.npz", "L27"), "6b": ("pairs_r6b_fwd_fwdrevprompt_tp.npz", "L27"),
         "6c": ("pairs_r6c_rev_only_tp.npz", "L27"), "run7": ("pairs_r7_cp.npz", "L27"), "run8": ("pairs_r8_cptp.npz", "L27"),
         "run9": ("pairs_r9_cptp_sent.npz", "L27"), "run10": ("pairs_r10_comb.npz", "L27")}
# logged id -> (A, B); the logged run name in results.jsonl is matched by its id
COMP = {"C1": ("r3", "r2"), "D1": ("r3", "r1"), "D2": ("r2", "r1"), "D3": ("r3", "r3p"), "S2.4": ("run4", "r2"),
        "C2": ("run5", "r3"), "E1": ("6a", "run5"), "E2": ("6b", "run5"), "E3": ("6c", "run5"), "E4": ("run5", "run4"),
        "E5": ("run7", "r2"), "C3": ("run8", "r3"), "E6": ("run8", "run7"), "E7": ("run9", "run8"), "E8": ("run8", "run5"),
        "C4": ("run10", "run5"), "E9": ("run10", "run8"), "E10": ("run10", "r3"), "E11": ("run10", "run4")}
MIN_SIGNS = 5


def load(alias):
    f, L = PAIRS[alias]
    z = np.load(ROOT / "results/test" / f)
    return {d: z[f"{d}|{L}"].astype(np.float64) for d in DS}, {d: z[f"{d}|gold"].astype(np.float64) for d in DS}


def logged():
    out = {}
    for line in open(ROOT / "results/results.jsonl"):
        d = json.loads(line)
        if d.get("kind") != "bootstrap":
            continue
        parts = d["run"].split("-")
        key = "S2.4" if d["run"].startswith("S2.4-bootstrap") else parts[1]
        assert key not in out, f"duplicate bootstrap line for {key}"
        out[key] = d
    return out


def reading(role, b):
    """Plan 8.4 wording, applied mechanically."""
    lo, hi = b["ci95"]
    alo, ahi = b["ci_bonferroni"]
    n = len(DS)
    pos = b["n_pos_signs"]
    agree = pos if b["diff"] > 0 else n - pos
    excl = lo > 0 or hi < 0
    if role == "primary":
        if excl and agree >= MIN_SIGNS and b["diff"] > 0:
            word = "improves" + (", robust" if (alo > 0 or ahi < 0) else "")
        else:
            word = "no resolvable difference"
        why = (f"95% CI {'excludes' if excl else 'includes'} 0; {agree}/{n} signs agree"
               f" ({'>=' if agree >= MIN_SIGNS else '<'} {MIN_SIGNS}); Bonferroni CI {'excludes' if (alo > 0 or ahi < 0) else 'includes'} 0")
        return word, why
    tag = "descriptive" if role == "descriptive" else "exploratory"
    return f"{tag}: no claim", f"95% CI {'excludes' if excl else 'includes'} 0; {agree}/{n} signs agree"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = logged()
    assert set(L) == set(COMP), f"logged ids {sorted(L)} != expected {sorted(COMP)}"
    cache, gold0 = {}, None
    rows, bad = [], []
    t0 = time.perf_counter()
    for cid, (a, b) in COMP.items():
        for r in (a, b):
            if r not in cache:
                cache[r] = load(r)
                g = cache[r][1]
                if gold0 is None:
                    gold0 = g
                assert all(np.array_equal(g[d], gold0[d]) for d in DS), f"gold differs in {r}"
        t = time.perf_counter()
        bt = paired_bootstrap(cache[a][0], cache[b][0], gold0, n_boot=2000, seed=0)
        lg = L[cid]
        assert lg["n_boot"] == 2000 and lg["seed"] == 0
        dif = dict(diff=abs(bt["diff"] - lg["diff"]),
                   ci95=max(abs(x - y) for x, y in zip(bt["ci95"], lg["ci95"])),
                   bonf=max(abs(x - y) for x, y in zip(bt["ci_adj"], lg["ci_bonferroni"])),
                   signs=abs(bt["n_pos_signs"] - lg["n_pos_signs"]))
        ok = dif["diff"] <= TOL and dif["ci95"] <= TOL and dif["bonf"] <= TOL and dif["signs"] == 0
        word, why = reading(lg["role"], lg)
        if lg.get("word") is not None:
            assert lg["word"] == word, f"{cid}: logged word {lg['word']!r} != rule {word!r}"
        rows.append(dict(id=cid, comparison=f"{a} vs {b}", role=lg["role"], stage=lg["stage"], logged_run=lg["run"],
                         diff=bt["diff"], ci95=list(bt["ci95"]), ci_bonferroni=list(bt["ci_adj"]), n_pos_signs=bt["n_pos_signs"],
                         n_datasets=len(DS), logged_diff=lg["diff"], logged_ci95=lg["ci95"], logged_ci_bonferroni=lg["ci_bonferroni"],
                         logged_n_pos_signs=lg["n_pos_signs"], max_abs_diff=max(dif["diff"], dif["ci95"], dif["bonf"]),
                         match=ok, reading=word, rule_applied=why, pairs=[PAIRS[a][0], PAIRS[b][0]], seconds=time.perf_counter() - t))
        print(f"{cid:5s} {a:>5s} - {b:<5s} recomputed {bt['diff']:+.4f} [{bt['ci95'][0]:+.4f}, {bt['ci95'][1]:+.4f}] "
              f"logged {lg['diff']:+.4f}  max|d| {rows[-1]['max_abs_diff']:.1e}  {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            bad.append(cid)
    js = dict(about="S6.1 recomputation from saved pair cosines (src/stats.paired_bootstrap, B 2000, seed 0) vs results.jsonl",
              tolerance=TOL, bonferroni_family="four primaries C1-C4; percentiles 0.625 / 99.375",
              coverage="test-pair sampling only; not dev selection, not the 8-bit model, not batching or seed",
              rows=rows, mismatches=bad, wall_s=time.perf_counter() - t0)
    json.dump(js, open(OUT / "S6.1_recompute.json", "w"), indent=1)
    with open(OUT / "S6.1_recompute.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "comparison", "role", "stage", "diff", "ci95_lo", "ci95_hi", "bonf_lo", "bonf_hi", "pos", "logged_diff",
                    "max_abs_diff", "match", "reading"])
        for r in rows:
            w.writerow([r["id"], r["comparison"], r["role"], r["stage"], f"{r['diff']:.12f}", f"{r['ci95'][0]:.12f}", f"{r['ci95'][1]:.12f}",
                        f"{r['ci_bonferroni'][0]:.12f}", f"{r['ci_bonferroni'][1]:.12f}", r["n_pos_signs"], f"{r['logged_diff']:.12f}",
                        f"{r['max_abs_diff']:.3e}", r["match"], r["reading"]])
    print(f"{len(rows)} comparisons, mismatches: {bad or 'none'}; wall {js['wall_s']:.0f} s")
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
