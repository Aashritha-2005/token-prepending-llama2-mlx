# LIVE combine-and-score, bootstrap and tolerance-gate printouts, for screenshots of steps as they run.

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path.home() / "Downloads" / "tp_project"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

import show_result as sr
from refnums import load_refs
from evalsts import SentenceTable, score, TEST_TASKS
from views import combine
from stats import paired_bootstrap

SHORT = {"STS12": "STS12", "STS13": "STS13", "STS14": "STS14", "STS15": "STS15", "STS16": "STS16",
         "STSBenchmark": "STS-B", "SICKRelatedness": "SICK-R"}
DS = [SHORT[t] for t in TEST_TASKS]
p = sr.p

# preset: label, [(cache dir, layer)], frozen key for alpha/normalisation (2 views), reference entry, saved scores (file, layer)
PRESETS = {
    "r1": ("r1 PromptEOL, layer 32", [("cache/test/r1r2_vanilla", "final")], None, "paper_prompteol", ("scores_r1r2_vanilla.json", "final")),
    "r2": ("r2 PromptEOL, layer 27", [("cache/test/r1r2_vanilla", "27")], None, "preprint_prompteol_l27", ("scores_r1r2_vanilla.json", "27")),
    "r3": ("r3 TP e=7, layer 27", [("cache/test/r3_tp_e7", "27")], None, "paper_prompteol_tp", ("scores_r3_tp_e7.json", "27")),
    "r3p": ("r3' TP e=8, layer 27", [("cache/test/r3p_tp_e8", "27")], None, "paper_prompteol_tp", ("scores_r3p_tp_e8.json", "27")),
    "run4": ("run 4 ReverseEOL (no TP)", [("cache/test/r1r2_vanilla", "27"), ("cache/test/r4_rev_keep_final", "27")], "reverseeol",
             "preprint_reverseeol", ("scores_r4_reverseeol.json", "27")),
    "run5": ("run 5 fwd+rev, TP", [("cache/test/r3_tp_e7", "27"), ("cache/test/r5_tp_rev_keep_final", "27")], "enh1",
             "paper_prompteol_tp", ("scores_r5_enh1_fwd_rev_tp.json", "27")),
    "6a": ("6a fwd+shuffled, TP", [("cache/test/r3_tp_e7", "27"), ("cache/test/r6a_tp_shuf", "27")], "enh1", None, ("scores_r6a_fwd_shuf_tp.json", "27")),
    "6b": ("6b fwd+fwd (rev. wording), TP", [("cache/test/r3_tp_e7", "27"), ("cache/test/r6b_tp_fwd_revprompt", "27")], "enh1", None,
           ("scores_r6b_fwd_fwdrevprompt_tp.json", "27")),
    "6c": ("6c rev only, TP", [("cache/test/r5_tp_rev_keep_final", "27")], None, None, ("scores_r6c_rev_only_tp.json", "27")),
    "run7": ("run 7 CP, no TP", [("cache/test/r7_cp", "27")], None, "published_cp_prompteol", ("scores_r7_cp.json", "27")),
    "run8": ("run 8 CP+TP", [("cache/test/r8_cptp", "27")], None, "paper_prompteol_tp", ("scores_r8_cptp.json", "27")),
    "run9": ("control 9 CP+TP sentiment aux", [("cache/test/r9_cptp_sent", "27")], None, None, ("scores_r9_cptp_sent.json", "27")),
    "run10": ("run 10 fwd+rev, CP+TP", [("cache/test/r8_cptp", "27"), ("cache/test/r10_cptp_rev", "27")], "comb", "paper_prompteol_tp",
              ("scores_r10_comb.json", "27")),
}
BOOTS = {"C1": ("r3", "r2", "primary"), "C2": ("run5", "r3", "primary"), "D1": ("r3", "r1", "descriptive"), "D2": ("r2", "r1", "descriptive"),
         "D3": ("r3", "r3p", "descriptive"), "E1": ("6a", "run5", "exploratory"), "E2": ("6b", "run5", "exploratory"),
         "E3": ("6c", "run5", "exploratory"), "E4": ("run5", "run4", "exploratory"), "E5": ("run7", "r2", "exploratory"),
         "C3": ("run8", "r3", "primary"), "E6": ("run8", "run7", "exploratory"), "E7": ("run9", "run8", "exploratory"),
         "E8": ("run8", "run5", "exploratory"), "C4": ("run10", "run5", "primary"), "E9": ("run10", "run8", "exploratory"),
         "E10": ("run10", "r3", "exploratory"), "E11": ("run10", "run4", "exploratory")}
STAGE_OF = {"E5": "S2.9", "C3": "S4.7", "E6": "S4.7", "E7": "S4.7", "E8": "S4.7", "C4": "S5.3", "E9": "S5.4", "E10": "S5.4", "E11": "S5.4"}
TIER_B = dict(avg=3.0, dataset=4.0)   # plan 7.3, CP row: average within +-3.0 of 75.27 [PUBLISHED] and run 7 > r2; datasets +-4.0
_TABLES = None
_REPRINT = None                       # set by --show: the time the step was logged


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = [SentenceTable(t) for t in TEST_TASKS]
    return _TABLES


# ---------------------------------------------------------------------------------------------- printing helpers
def live_header(name, what, dry=False):
    now = datetime.datetime.now().astimezone()
    p("#" * sr.W)
    tag = f"REPRINT (already logged at {_REPRINT})" if _REPRINT else "DRY RUN (trial, not evidence; nothing written)" if dry else "LIVE RUN"
    p(f"##  {tag}  {name}".ljust(sr.W - 2) + "##")
    p("#" * sr.W)
    p(f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}  |  {ROOT}  |  Python {sr.platform.python_version()}")
    p(sr.kernel_line())
    p(f"computing now: {what}"[:sr.W])


def live_table(rows, refs=()):
    """rows: [(label, scores dict)], refs: [reference entry keys]; prints the paper-style table with delta rows."""
    E = load_refs()["entries"]
    sr.thead()
    for key in refs:
        p(sr.trow(f"{E[key]['label']} {E[key]['tag']}", E[key]["scores"]))
    for i, (lab, s) in enumerate(rows):
        lab = lab if len(lab) <= sr.LW - 1 else lab.replace(" (live)", "")[:sr.LW - 8] + " (live)"
        p(sr.trow(lab, s))
        if i == 0:                      # delta only for the run under test, against the first reference row
            for key in refs[:1]:
                p(sr.trow(f"  delta - {E[key]['tag']}", {c: s[c] - E[key]["scores"][c] for c in sr.COLS}, True))
    p("-" * (sr.LW + 8 * len(sr.COLS)))


def gate_block(gid, checks):
    """checks: [(name, value text, rule text, ok)] -> PASS/FAIL block."""
    p(f"GATE {gid}: {'PASS' if all(c[3] for c in checks) else 'FAIL'}")
    for name, val, rule, ok in checks:
        p(f"  [{'x' if ok else ' '}] {name:28s} {val:>14s}   rule: {rule}"[:sr.W])


def footer(t0, files):
    p(f"computed in {time.perf_counter() - t0:.1f} s; inputs (sha256[:12], bytes, modified):")
    for f in dict.fromkeys(files):
        p(sr.finfo(f))
    sr.df_line()


# ---------------------------------------------------------------------------------------------- live computation
def load_view(cdir, layer):
    meta = json.load(open(ROOT / cdir / "meta.json"))
    files = sorted((ROOT / cdir).glob("chunk_*.npz"))
    assert [int(f.stem.split("_")[1]) for f in files] == list(range(0, meta["n_items"], meta["chunk"])), f"{cdir}: chunks incomplete"
    E = np.concatenate([np.load(f)[f"L{layer}"].astype(np.float32) for f in files])
    assert E.shape[0] == sum(len(t.sentences) for t in tables()), cdir
    return E, meta


def frozen(key):
    import runlog                                   # runlog.FROZEN = ROOT/frozen_settings.json unless TP_SCRATCH_STATE (tests)
    st = json.load(open(runlog.FROZEN))[key]["settings"]
    return (st["a_v"] if key == "comb" else st["alpha"]), st["normalize"]


def live_scores(views, frozen_key):
    """-> (scores dict incl. Avg, sims per dataset, gold per dataset, metas). Computed from the caches now."""
    Ev = [load_view(c, L) for c, L in views]
    if len(Ev) == 2:
        a, nm = frozen(frozen_key)
        X = combine({"v0": Ev[0][0], "v1": Ev[1][0]}, ["v0", "v1"], a, nm)
    else:
        X = Ev[0][0]
    s, sims, gold, off = {}, {}, {}, 0
    for tb in tables():
        m, d = len(tb.sentences), SHORT[tb.name]
        val, ps = score(tb, X[off:off + m])
        s[d], sims[d] = val, np.concatenate([ps[k][0] for k in ps])
        gold[d] = np.concatenate([g for _, (_, _, g) in tb.pairs.items()])
        off += m
    s["Avg"] = float(np.mean([s[d] for d in DS]))
    return s, sims, gold, [m for _, m in Ev]


def saved_match(preset, s):
    f, L = PRESETS[preset][4]
    q = ROOT / "results/test" / f
    if not q.exists():
        return "no saved scores file"
    saved = json.load(open(q))["scores"][L]
    return f"matches results/test/{f} to 1e-9: {'yes' if all(abs(saved[c] - s[c]) < 1e-9 for c in sr.COLS) else 'NO'}"


def cache_files(views):
    return [f"{c}/meta.json" for c, _ in views]


def describe(views, fk, metas):
    m = metas[0]
    tp = f"TP s {m['tp_start']} e {m['tp_exit']}" if m["plan"] == "tp" else "vanilla"
    comb = f", combined with frozen '{fk}' alpha {frozen(fk)[0]} norm {'on' if frozen(fk)[1] else 'off'}" if len(views) == 2 else ""
    return f"{tp}, {'equal-length batches' if m['batched'] else 'batch 1'}, layer {views[0][1]}{comb}"


# ---------------------------------------------------------------------------------------------- modes
def mode_score(a):
    if a.preset:
        lab, views, fk, ref, _ = PRESETS[a.preset]
    else:
        views = [tuple(x.rsplit(":", 1)) for x in a.cache]
        lab, fk, ref = a.label or "new run", a.frozen, a.ref
    live_header(a.evidence, f"Spearman on {len(DS)} test sets from {', '.join(c for c, _ in views)}", dry=a.dry)
    t0 = time.perf_counter()
    s, _, _, metas = live_scores(views, fk)
    p(f"  {describe(views, fk, metas)}")
    p()
    live_table([(f"{lab} (live)", s)], [ref] if ref else [])
    if a.preset:
        p("check: " + saved_match(a.preset, s))
    footer(t0, cache_files(views))


def mode_bootstrap(a):
    A, B, role = BOOTS[a.id]
    live_header(a.evidence, f"paired bootstrap {a.id}: {A} vs {B} ({role}), B = {a.n_boot}, seed {a.seed}, from the caches", dry=a.dry)
    t0 = time.perf_counter()
    sa, sima, gold, _ = live_scores(PRESETS[A][1], PRESETS[A][2])
    sb, simb, goldb, _ = live_scores(PRESETS[B][1], PRESETS[B][2])
    assert all(np.array_equal(gold[d], goldb[d]) for d in DS)
    bt = paired_bootstrap(sima, simb, gold, n_boot=a.n_boot, seed=a.seed)
    agree = bt["n_pos_signs"] if bt["diff"] > 0 else len(DS) - bt["n_pos_signs"]
    p()
    live_table([(f"{PRESETS[A][0]} (live)", sa), (f"{PRESETS[B][0]} (live)", sb)])
    p(sr.trow(f"  difference {A} - {B}", {c: sa[c] - sb[c] for c in sr.COLS}, True))
    p(f"{a.id} ({role.upper()}): diff {bt['diff']:+.2f}  95% CI [{bt['ci95'][0]:+.2f}, {bt['ci95'][1]:+.2f}]  "
      f"Bonferroni [{bt['ci_adj'][0]:+.2f}, {bt['ci_adj'][1]:+.2f}]  positive {bt['n_pos_signs']}/{len(DS)}")
    if a.id == "C3":                    # notes/s4_preregistration.md section 6 (control check is applied at S4.7)
        p(f"pre-registered C3 reading: {c3_reading(bt, True)[0]}")
    elif a.id == "C4":                  # notes/s5_preregistration.md section 5
        p(f"pre-registered C4 reading: {c4_word(bt)}")
    elif role == "primary":
        word = "improves" if bt["excludes0"] and agree >= 5 and bt["diff"] > 0 else "no resolvable difference" if not bt["excludes0"] else "differs"
        p(f"plan 8.4 reading: {word}{', robust' if word == 'improves' and bt['robust'] else ''}")
    else:
        p(f"exploratory: no claim. 95% CI excludes 0: {'yes' if bt['excludes0'] else 'no'}; signs agreeing with the difference: {agree}/{len(DS)}")
    if getattr(a, "record", False):
        p(record_boot(a.id, bt, sa, sb, sima, simb, gold, a.n_boot, a.seed, a.dry))
    footer(t0, cache_files(PRESETS[A][1] + PRESETS[B][1]))


def record_boot(bid, bt, sa, sb, sima, simb, gold, n_boot, seed, dry):
    """Append a bootstrap line to results.jsonl (once) and, for Stage 5 ids, save the per-pair cosines. Returns a status line."""
    import runlog
    from runlog import append_result
    A, B, role = BOOTS[bid]
    run = f"{STAGE_OF.get(bid, 'S?')}-{bid}-{A}_vs_{B}"
    pairs = ROOT / "results/test" / f"pairs_boot_{bid}.npz"
    s5 = STAGE_OF.get(bid, "").startswith("S5")
    if any(json.loads(l).get("run") == run for l in open(runlog.RESULTS)):
        return f"{run} already in results.jsonl; not appended again"
    if dry:
        return f"DRY RUN: would append {run} to results.jsonl" + (f" + {pairs.name}" if s5 else "") + " (not written)"
    if s5:
        np.savez_compressed(pairs, **{f"{d}|{A}": sima[d] for d in DS}, **{f"{d}|{B}": simb[d] for d in DS}, **{f"{d}|gold": gold[d] for d in DS})
    append_result(dict(kind="bootstrap", run=run, stage=STAGE_OF.get(bid, "S?"), split="test", role=role, diff=bt["diff"], ci95=bt["ci95"],
                       ci_bonferroni=bt["ci_adj"], n_pos_signs=bt["n_pos_signs"], per_dataset_diff={d: sa[d] - sb[d] for d in DS},
                       n_boot=n_boot, seed=seed, sources=[c for c, _ in PRESETS[A][1] + PRESETS[B][1]],
                       **(dict(pair_cosines=str(pairs.relative_to(ROOT)), word=c4_word(bt) if bid == "C4" else None) if s5 else {})))
    return f"appended {run} to results.jsonl" + (f"; pair cosines {pairs.relative_to(ROOT)}" if s5 else "")


def mode_gate(a):
    lab, views, fk, ref, _ = PRESETS[a.preset]
    ref = a.ref or ref
    live_header(a.evidence, f"tolerance gate for {a.preset} from the caches", dry=a.dry)
    t0 = time.perf_counter()
    s, _, _, _ = live_scores(views, fk)
    rows, checks, E = [(f"{lab} (live)", s)], [], load_refs()["entries"]
    if a.vs:
        sv, _, _, _ = live_scores(PRESETS[a.vs][1], PRESETS[a.vs][2])
        rows.append((f"{PRESETS[a.vs][0]} (live)", sv))
        if a.avg_tol is not None and not ref:
            d = s["Avg"] - sv["Avg"]
            checks.append((f"|{a.preset} - {a.vs}| avg", f"{d:+.2f}", f"within +-{a.avg_tol}", abs(d) <= a.avg_tol))
    if ref:
        if a.avg_tol is not None:
            d = s["Avg"] - E[ref]["scores"]["Avg"]
            checks.append((f"avg vs {E[ref]['tag']}", f"{d:+.2f}", f"within +-{a.avg_tol}", abs(d) <= a.avg_tol))
        if a.ds_tol is not None:
            worst = max(abs(s[c] - E[ref]["scores"][c]) for c in DS)
            checks.append(("max |dataset delta|", f"{worst:.2f}", f"<= {a.ds_tol}", worst <= a.ds_tol))
    if a.above:
        sb, _, _, _ = live_scores(PRESETS[a.above][1], PRESETS[a.above][2])
        rows.append((f"{PRESETS[a.above][0]} (live)", sb))
        checks.append((f"{a.preset} > {a.above} (avg)", f"{s['Avg']:.2f}>{sb['Avg']:.2f}", "higher average", s["Avg"] > sb["Avg"]))
    p()
    live_table(rows, [ref] if ref else [])
    gate_block(a.gid or a.preset, checks)
    footer(t0, cache_files(views))


def tier_checks(s7, sr2, E):
    ref = E["published_cp_prompteol"]["scores"]
    d_avg = s7["Avg"] - ref["Avg"]
    worst = max(abs(s7[d] - ref[d]) for d in DS)
    return [("avg vs 75.27 [PUBLISHED]", f"{d_avg:+.2f}", f"within +-{TIER_B['avg']}", abs(d_avg) <= TIER_B["avg"]),
            ("max |dataset delta|", f"{worst:.2f}", f"<= {TIER_B['dataset']}", worst <= TIER_B["dataset"]),
            ("run 7 > r2 (avg)", f"{s7['Avg']:.2f}>{sr2['Avg']:.2f}", "higher average", s7["Avg"] > sr2["Avg"])]


def mode_run7(a):
    """S2.9: run 7 scored live from cache/test/r7_cp, with [PAPER] PromptEOL, our r1 and r2, and the [PUBLISHED] CP row."""
    live_header(a.evidence, "S2.9 run 7 (CP, no TP): Spearman from cache/test/r7_cp + r1/r2 caches", dry=a.dry)
    t0, E = time.perf_counter(), load_refs()["entries"]
    s7, _, _, m7 = live_scores(PRESETS["run7"][1], None)
    s1, _, _, _ = live_scores(PRESETS["r1"][1], None)
    s2, _, _, _ = live_scores(PRESETS["r2"][1], None)
    cp = m7[0].get("cp", {})
    p(f"  run 7: CP act_layer {cp.get('act_layer')} (paper l {cp.get('act_layer', -1) + 1}), alpha {cp.get('alpha')}, {cp.get('strategy')}, "
      f"vanilla, batch 1, layer 27")
    p()
    sr.thead()
    for key in ("paper_prompteol", "published_cp_prompteol"):
        p(sr.trow(f"{E[key]['label']} {E[key]['tag']}", E[key]["scores"]))
    p(sr.trow("r1 PromptEOL, layer 32 (live)", s1))
    p(sr.trow("r2 PromptEOL, layer 27 (live)", s2))
    p(sr.trow("run 7 CP, layer 27 (live)", s7))
    p(sr.trow("  delta run 7 - [PUBLISHED]", {c: s7[c] - E["published_cp_prompteol"]["scores"][c] for c in sr.COLS}, True))
    p(sr.trow("  delta run 7 - r2", {c: s7[c] - s2[c] for c in sr.COLS}, True))
    p("-" * (sr.LW + 8 * len(sr.COLS)))
    checks = tier_checks(s7, s2, E)
    gate_block("S2.9 Tier B (recorded; the verdict is S2.10)", checks)
    p("check: " + saved_match("run7", s7))
    passed = all(np.isfinite([s7[c] for c in sr.COLS]))
    if a.dry:
        p("DRY RUN: gate S2.9 not logged")
    else:
        from runlog import log_gate
        log_gate("S2.9", passed, dict(run7=s7, r2=s2, r1=s1, tier_b={n: dict(value=v, rule=r, ok=ok) for n, v, r, ok in checks}, cp=cp),
                 ["cache/test/r7_cp", "results/test/scores_r7_cp.json", "scripts/live_step.py"],
                 "7 per-dataset numbers + average computed live from the cache; Tier B check recorded (plan 7.3).")
        p(f"logged gate S2.9: {'PASS' if passed else 'FAIL'} (7 numbers + tolerance check recorded)")
    footer(t0, cache_files(PRESETS["run7"][1] + PRESETS["r2"][1]))


def mode_verdict(a):
    """S2.10: verdict from numbers computed now (pre-registered rules, notes/s2_8_preregistration.md)."""
    from stats import paired_bootstrap
    live_header(a.evidence, "S2.10 CP verdict: run 7 and r2 from the caches, bootstrap B = 2000 seed 0", dry=a.dry)
    t0, E = time.perf_counter(), load_refs()["entries"]
    s7, sim7, gold, _ = live_scores(PRESETS["run7"][1], None)
    s2, sim2, gold2, _ = live_scores(PRESETS["r2"][1], None)
    assert all(np.array_equal(gold[d], gold2[d]) for d in DS)
    bt = paired_bootstrap(sim7, sim2, gold, n_boot=2000, seed=0)
    p()
    sr.thead()
    p(sr.trow(f"{E['published_cp_prompteol']['label']} {E['published_cp_prompteol']['tag']}", E["published_cp_prompteol"]["scores"]))
    p(sr.trow("r2 PromptEOL, layer 27 (live)", s2))
    p(sr.trow("run 7 CP, layer 27 (live)", s7))
    p(sr.trow("  delta run 7 - r2", {c: s7[c] - s2[c] for c in sr.COLS}, True))
    p("-" * (sr.LW + 8 * len(sr.COLS)))
    tier = tier_checks(s7, s2, E)
    within = tier[0][3] and tier[1][3]
    above = tier[2][3]
    ci_ok = bt["ci95"][0] > 0 and bt["n_pos_signs"] >= 5
    checks = tier + [("bootstrap run 7 - r2 (95% CI)", f"[{bt['ci95'][0]:+.2f},{bt['ci95'][1]:+.2f}]", "excludes 0, above", bt["ci95"][0] > 0),
                     ("positive dataset signs", f"{bt['n_pos_signs']}/{len(DS)}", ">= 5", bt["n_pos_signs"] >= 5)]
    verdict = "Reproduced" if within and above and ci_ok else ("Direction only" if above and ci_ok else "Not reproduced")
    gate_block("S2.10", checks)
    p(f"VERDICT: {verdict}  (run 7 - r2 {bt['diff']:+.2f}, Bonferroni [{bt['ci_adj'][0]:+.2f}, {bt['ci_adj'][1]:+.2f}]; exploratory)")
    p("note: the published +5.24 is vs the last-layer 70.03; the fair comparison is r2 (layer 27).")
    out = dict(verdict=verdict, run7=s7, r2=s2, tier_b={n: dict(value=v, rule=r, ok=ok) for n, v, r, ok in tier},
               bootstrap=dict(diff=bt["diff"], ci95=list(bt["ci95"]), ci_bonferroni=list(bt["ci_adj"]), n_pos_signs=bt["n_pos_signs"], role="exploratory"))
    if a.dry:
        p("DRY RUN: gate S2.10 not logged, nothing written")
    else:
        from runlog import log_gate
        (ROOT / "logs/S2.10.json").write_text(json.dumps(out, indent=1, default=float))
        log_gate("S2.10", verdict == "Reproduced", out, ["logs/S2.10.json", "notes/s2_8_preregistration.md", "cache/test/r7_cp", "scripts/live_step.py"],
                 f"Verdict: {verdict} (pre-registered rules). PASS only for Reproduced, as S2.5.")
        p(f"logged gate S2.10 ({'PASS' if verdict == 'Reproduced' else 'FAIL'}: {verdict}); logs/S2.10.json written")
    footer(t0, cache_files(PRESETS["run7"][1] + PRESETS["r2"][1]))


# ---------------------------------------------------------------------------------------------- Stage 4 (CP + TP)
def enh2():
    import runlog
    return json.load(open(runlog.FROZEN))["enh2"]


def cptp_meta_ok(meta, st, control):
    c = meta.get("cptp") or {}
    return ((c.get("act_layer"), c.get("alpha"), c.get("aux_tp"), c.get("control"), c.get("source"), meta["plan"], meta["tp_start"],
             meta["tp_exit"], meta["batched"], meta["split"]) ==
            (st["act_layer"], float(st["alpha"]), st["aux_tp"], control, "frozen enh2", "tp", st["tp_start"], st["tp_exit"], False, "test"))


def c3_reading(bt, ok_control):
    """notes/s4_preregistration.md section 6 (hashed)."""
    pos = bt["n_pos_signs"]
    if bt["ci95"][0] > 0 and pos >= 5:
        word = "improves, robust" if bt["ci_adj"][0] > 0 else "improves"
        text = "Run 8 > r3 (C3 differs): CP filters content TP does not."
    elif bt["ci95"][1] < 0 and len(DS) - pos >= 5:
        word, text = "hurts", "Run 8 < r3: CP steering harms TP in this setting. Negative result."
    else:
        word, text = "no resolvable difference", "Run 8 ~ r3: TP already captures what CP provides. Negative result."
    if not ok_control:
        text = "Run 9 >= run 8: pipeline or port suspect; re-examine S2.7/S4.2 before any claim. No claim."
    return word, text


def mode_s46(a):
    """S4.6: run 8 and control 9 scored live from their caches; checks against frozen enh2; logs S4.6."""
    live_header(a.evidence, "S4.6 run 8 (CP+TP) and control 9 (sentiment aux): Spearman from the test caches", dry=a.dry)
    t0, E = time.perf_counter(), load_refs()["entries"]
    st = enh2()["settings"]
    s8, _, _, m8 = live_scores(PRESETS["run8"][1], None)
    s9, _, _, m9 = live_scores(PRESETS["run9"][1], None)
    s3, _, _, _ = live_scores(PRESETS["r3"][1], None)
    s5, _, _, _ = live_scores(PRESETS["run5"][1], "enh1")
    s7, _, _, _ = live_scores(PRESETS["run7"][1], None)
    p(f"  enh2: index {st['act_layer']} (paper l {st['paper_l']}), alpha {st['alpha']:g}, aux {'with' if st['aux_tp'] else 'without'} TP, "
      f"scaled, TP s {st['tp_start']} e {st['tp_exit']}, layer 27, batch 1")
    p()
    sr.thead()
    pt = E["paper_prompteol_tp"]
    p(sr.trow(f"{pt['label']} {pt['tag']}", pt["scores"]))
    for lab, sc in (("r3 TP e=7 (live)", s3), ("run 5 fwd+rev, TP (live)", s5), ("run 7 CP, no TP (live)", s7),
                    ("run 8 CP+TP (live)", s8), ("control 9 sentiment aux (live)", s9)):
        p(sr.trow(lab, sc))
    p(sr.trow("  delta run 8 - r3", {c: s8[c] - s3[c] for c in sr.COLS}, True))
    p(sr.trow(f"  delta run 8 - {pt['tag']}", {c: s8[c] - pt["scores"][c] for c in sr.COLS}, True))
    p(sr.trow("  delta control 9 - run 8", {c: s9[c] - s8[c] for c in sr.COLS}, True))
    p("-" * (sr.LW + 8 * len(sr.COLS)))
    fin8, fin9 = all(np.isfinite([s8[c] for c in sr.COLS])), all(np.isfinite([s9[c] for c in sr.COLS]))
    ok8, ok9 = cptp_meta_ok(m8[0], st, None), cptp_meta_ok(m9[0], st, "sentiment")
    checks = [("run 8 numbers + meta = enh2", f"{s8['Avg']:.2f}", "7 finite; settings match", fin8 and ok8),
              ("control 9 numbers + meta", f"{s9['Avg']:.2f}", "7 finite; enh2 + sentiment", fin9 and ok9)]
    gate_block("S4.6", checks)
    worst = max(abs(s8[d] - pt["scores"][d]) for d in DS)
    p(f"info: run 8 avg vs 77.19 [PAPER] PromptEOL+TP {s8['Avg'] - pt['scores']['Avg']:+.2f}; max |dataset delta| {worst:.2f}")
    p(f"control check (pre-registered, used at S4.7): control 9 < run 8: {'yes' if s9['Avg'] < s8['Avg'] else 'NO'} "
      f"({s9['Avg']:.2f} vs {s8['Avg']:.2f})")
    p("check run 8: " + saved_match("run8", s8)); p("check run 9: " + saved_match("run9", s9))
    passed = fin8 and ok8 and fin9 and ok9
    if a.dry:
        p("DRY RUN: gate S4.6 not logged")
    else:
        from runlog import log_gate
        log_gate("S4.6", passed, dict(run8=s8, run9=s9, r3=s3, run5=s5, run7=s7, enh2=st | dict(prompts="(see frozen_settings.json)"),
                                      control_below_run8=bool(s9["Avg"] < s8["Avg"])),
                 ["cache/test/r8_cptp", "cache/test/r9_cptp_sent", "results/test/scores_r8_cptp.json", "results/test/scores_r9_cptp_sent.json",
                  "scripts/live_step.py"], "Run 8 and control 9: 7 numbers each, computed live; caches match frozen enh2.")
        p(f"logged gate S4.6: {'PASS' if passed else 'FAIL'}")
    footer(t0, cache_files(PRESETS["run8"][1] + PRESETS["run9"][1]))


def mode_s47(a):
    """S4.7: C3 (primary) and E6-E8 (exploratory) bootstraps computed now; the reading per plan 14.2; logs S4.7."""
    live_header(a.evidence, "S4.7 CP+TP reading: C3 + E6-E8 bootstraps (B = 2000, seed 0) from the caches", dry=a.dry)
    t0 = time.perf_counter()
    L = {k: live_scores(PRESETS[k][1], PRESETS[k][2]) for k in ("run8", "r3", "run9", "run7", "run5")}
    for k in L:
        assert all(np.array_equal(L[k][2][d], L["r3"][2][d]) for d in DS)
    bts = {i: paired_bootstrap(L[BOOTS[i][0]][1], L[BOOTS[i][1]][1], L["r3"][2], n_boot=2000, seed=0) for i in ("C3", "E6", "E7", "E8")}
    p()
    sr.thead()
    for k, lab in (("r3", "r3 TP e=7"), ("run5", "run 5 fwd+rev, TP"), ("run7", "run 7 CP, no TP"), ("run8", "run 8 CP+TP"),
                   ("run9", "control 9 sentiment aux")):
        p(sr.trow(f"{lab} (live)", L[k][0]))
    p(sr.trow("  difference run 8 - r3", {c: L["run8"][0][c] - L["r3"][0][c] for c in sr.COLS}, True))
    p("-" * (sr.LW + 8 * len(sr.COLS)))
    for i, bt in bts.items():
        A, B, role = BOOTS[i]
        p(f"{i} {'PRIMARY' if role == 'primary' else 'exploratory'} {A}-{B}: {bt['diff']:+.2f} 95% [{bt['ci95'][0]:+.2f}, {bt['ci95'][1]:+.2f}] "
          f"Bonf [{bt['ci_adj'][0]:+.2f}, {bt['ci_adj'][1]:+.2f}] pos {bt['n_pos_signs']}/{len(DS)}"[:sr.W])
    ok_control = L["run9"][0]["Avg"] < L["run8"][0]["Avg"]
    word, text = c3_reading(bts["C3"], ok_control)
    checks = [("C3 computed (run 8 vs r3)", word[:14], "pre-registered rule", True),
              ("control 9 below run 8", f"{L['run9'][0]['Avg']:.2f}<{L['run8'][0]['Avg']:.2f}", "else pipeline suspect", ok_control)]
    gate_block("S4.7", checks)
    p(f"C3 reading: {word}")
    p(f"READING (plan 14.2): {text}"[:sr.W])
    p("E6-E8 are exploratory: no claims. Caveat: r3 is batched TP, run 8 batch 1 (S4.2b reports the gap).")
    out = dict(C3=dict(bts["C3"], word=word), E6=bts["E6"], E7=bts["E7"], E8=bts["E8"], reading=text, control_below_run8=ok_control,
               scores={k: v[0] for k, v in L.items()})
    if a.dry:
        p("DRY RUN: gate S4.7 not logged; notes/enh2_reading.md not written")
    else:
        import runlog
        from runlog import log_gate
        (runlog.LOGS / "S4.7.json").write_text(json.dumps(out, indent=1, default=float))
        rd = ROOT / "notes/enh2_reading.md" if not runlog.SCRATCH else runlog.LOGS / "enh2_reading.md"
        b = bts["C3"]
        rd.write_text(f"# Enhancement 2 (CP + TP): reading (S4.7, {datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
                      f"- C3 (primary) run 8 - r3: {b['diff']:+.2f}, 95% CI [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}], Bonferroni "
                      f"[{b['ci_adj'][0]:+.2f}, {b['ci_adj'][1]:+.2f}], {b['n_pos_signs']}/7 positive -> **{word}**.\n"
                      + "".join(f"- {i} (exploratory) {BOOTS[i][0]} - {BOOTS[i][1]}: {bts[i]['diff']:+.2f} [{bts[i]['ci95'][0]:+.2f}, "
                                f"{bts[i]['ci95'][1]:+.2f}], {bts[i]['n_pos_signs']}/7 positive.\n" for i in ("E6", "E7", "E8"))
                      + f"- Control 9 below run 8: {'yes' if ok_control else 'NO'} ({L['run9'][0]['Avg']:.2f} vs {L['run8'][0]['Avg']:.2f}).\n"
                      f"- Reading (plan 14.2): {text}\n- Caveat: r3 is batched TP (equal-length batches), run 8 is batch 1.\n")
        log_gate("S4.7", ok_control, out, ["logs/S4.7.json", "notes/enh2_reading.md", "notes/s4_preregistration.md", "scripts/live_step.py"],
                 f"C3 {word}; reading written (pre-registered rules).")
        p(f"logged gate S4.7 ({'PASS' if ok_control else 'FAIL: control 9 not below run 8'}); logs/S4.7.json, enh2_reading.md written")
    footer(t0, cache_files(PRESETS["run8"][1] + PRESETS["r3"][1]))


def mode_s48(a):
    """S4.8 checklist and cost (plan 12.1); logs S4.8."""
    import re
    import runlog
    live_header(a.evidence, "S4.8 Stage 4 checklist and cost, from gates, caches and timings", dry=a.dry)
    t0 = time.perf_counter()
    g = json.load(open(runlog.GATES))
    ok = lambda k: g.get(k, {}).get("passed", False)
    pre = ROOT / "notes/s4_preregistration.md"
    import hashlib
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    m = re.search(r"s4_preregistration\.md written before any Stage 4 pass\. sha256 ([0-9a-f]{64}) \(recorded (\S+ \S+?);", (ROOT / "notes/decisions.md").read_text())
    rec_t = datetime.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S") if m else None
    s4_dev = sorted((ROOT / "cache/dev").glob("cptp_*")) + [ROOT / "cache/dev/tp_e7_b1_L27"]
    first = min(datetime.datetime.fromtimestamp((d / "meta.json").stat().st_mtime) for d in s4_dev if (d / "meta.json").exists())
    frz_t = datetime.datetime.fromisoformat(g["S4.5"]["date"]) if ok("S4.5") else None
    tmeta = {r: ROOT / f"cache/test/{r}/meta.json" for r in ("r8_cptp", "r9_cptp_sent")}
    test_t = {r: datetime.datetime.fromtimestamp(q.stat().st_mtime) for r, q in tmeta.items() if q.exists()}
    nchunks = {r: len(list((ROOT / f"cache/test/{r}").glob("chunk_*.npz"))) for r in tmeta}
    c3 = any(json.loads(l).get("run") == "S4.7-C3-run8_vs_r3" for l in open(runlog.RESULTS))
    rd = (ROOT / "notes/enh2_reading.md") if not runlog.SCRATCH else runlog.LOGS / "enh2_reading.md"
    items = [("S4.1 hash before any S4 pass", f"{m.group(2)[11:] if m else '-'} < {first:%H:%M:%S}",
              bool(m) and m.group(1) == sha and rec_t < first),
             ("S4.2a / S4.2b / S4.2 passed", f"{ok('S4.2a')}/{ok('S4.2b')}/{ok('S4.2')}", ok("S4.2a") and ok("S4.2b") and ok("S4.2")),
             ("S4.3-5 passed, frozen < test", f"S4.3-5 {ok('S4.3') and ok('S4.4') and ok('S4.5')}",
              ok("S4.3") and ok("S4.4") and ok("S4.5") and bool(test_t) and all(frz_t and frz_t < t for t in test_t.values())),
             ("runs 8, 9 on test: chunks", f"{nchunks['r8_cptp']} / {nchunks['r9_cptp_sent']}", ok("S4.6") and set(nchunks.values()) == {58}),
             ("C3 done, reading written", f"C3 {c3}, reading {rd.exists()}", c3 and ok("S4.7") and rd.exists())]
    passed = all(x[2] for x in items)
    gate_block("S4.8", [(n, v, "required", o) for n, v, o in items])
    sps = lambda f: json.load(open(ROOT / "results/test" / f)).get("s_per_sentence_computed")
    s8, s3 = sps("scores_r8_cptp.json"), sps("scores_r3_tp_e7.json")
    b2 = json.load(open(runlog.LOGS / "S4.2b_caches.json"))["s_per_sentence"]["tp_e7_b1_L27"] if (runlog.LOGS / "S4.2b_caches.json").exists() else None
    p(f"cost: run 8 {s8:.3f} s/sentence (batch 1, aux included); r3 {s3:.3f} (batched) -> {s8 / s3:.2f}x")
    if b2:
        p(f"      vs TP batch 1 (dev, S4.2b cache) {b2:.3f} s/sentence -> {s8 / b2:.2f}x  (plan expectation 1.15-1.20x)")
    out = dict(checklist={n: dict(value=v, ok=o) for n, v, o in items}, cost=dict(run8=s8, r3=s3, tp_b1_dev=b2, ratio_vs_r3=s8 / s3,
               ratio_vs_tp_b1=(s8 / b2) if b2 else None), passed=passed)
    if a.dry:
        p("DRY RUN: gate S4.8 not logged")
    else:
        from runlog import log_gate
        (runlog.LOGS / "S4.8.json").write_text(json.dumps(out, indent=1, default=str))
        log_gate("S4.8", passed, out, ["logs/S4.8.json", "gates.json", "notes/decisions.md"], "Stage 4 checklist and cost (plan 12.1 S4.8).")
        p(f"logged gate S4.8 ({'PASS' if passed else 'FAIL'}); logs/S4.8.json written. Stage 4 stops here.")
    footer(t0, ["notes/s4_preregistration.md", "frozen_settings.json"])


# ---------------------------------------------------------------------------------------------- Stage 5 (combination, run 10)
def comb():
    import runlog
    return json.load(open(runlog.FROZEN))["comb"]


def esha(entry):
    import hashlib
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def c4_word(bt):
    """notes/s5_preregistration.md section 5 (hashed). 'robust' on the same side for improves or hurts."""
    pos = bt["n_pos_signs"]
    if bt["ci95"][0] > 0 and pos >= 5:
        return "improves, robust" if bt["ci_adj"][0] > 0 else "improves"
    if bt["ci95"][1] < 0 and len(DS) - pos >= 5:
        return "hurts, robust" if bt["ci_adj"][1] < 0 else "hurts"
    return "no resolvable difference"


def c4_reading(word, e9):
    """notes/s5_preregistration.md section 6 (plan 14.2, Combination table). -> (reading, exploratory E9 note or None)."""
    if word.startswith("improves"):
        text = "Run 10 > both 5 and 8: complementary."
    elif word.startswith("hurts"):
        text = "Run 10 < run 5: adding CP to the forward-reverse combination hurts in this setting. Negative result."
    else:
        text = "Run 10 ~ run 5: reversal dominates."
    note = "run 10 ~ run 8 (E9, exploratory): CP dominates." if e9["ci95"][0] <= 0 <= e9["ci95"][1] else None
    return text, note


def rev_meta_ok(meta, st):
    c = meta.get("cptp") or {}
    return ((c.get("act_layer"), c.get("alpha"), c.get("aux_tp"), c.get("control"), c.get("strategy"), c.get("source"), meta["view"],
             meta.get("punct"), meta["plan"], meta["tp_start"], meta["tp_exit"], meta["batched"], meta["split"]) ==
            (st["act_layer"], st["alpha"], st["aux_tp"], None, "scaled", "frozen enh2", "rev", st["punct"], "tp", st["tp_start"],
             st["tp_exit"], False, "test"))


def mode_s53(a):
    """S5.3: run 10 = combine(run 8 forward, r10_cptp_rev) scored live with the comparison rows; checks; logs S5.3."""
    import runlog
    live_header(a.evidence, "S5.3 run 10 (fwd + rev, CP+TP): Spearman from cache/test/r8_cptp + r10_cptp_rev", dry=a.dry)
    t0, E = time.perf_counter(), load_refs()["entries"]
    g, cb = json.load(open(runlog.GATES)), comb()
    st = cb["settings"]
    s10, sim10, gold, m10 = live_scores(PRESETS["run10"][1], "comb")
    L = {k: live_scores(PRESETS[k][1], PRESETS[k][2])[0] for k in ("r3", "run4", "run5", "run8")}
    p(f"  comb: fwd = run 8; rev = CP+TP index {st['act_layer']} alpha {st['alpha']:g}, aux TP on s_rev, {st['punct']}; "
      f"a_v {st['a_v']}, norm {'on' if st['normalize'] else 'off'}")
    p()
    sr.thead()
    pt = E["paper_prompteol_tp"]
    p(sr.trow(f"{pt['label']} {pt['tag']}", pt["scores"]))
    for k, lab in (("r3", "r3 TP e=7"), ("run4", "run 4 ReverseEOL (no TP)"), ("run5", "run 5 fwd+rev, TP"), ("run8", "run 8 CP+TP")):
        p(sr.trow(f"{lab} (live)", L[k]))
    p(sr.trow("run 10 fwd+rev, CP+TP (live)", s10))
    for k, lab in (("run5", "run 5"), ("run8", "run 8"), ("r3", "r3")):
        p(sr.trow(f"  delta run 10 - {lab}", {c: s10[c] - L[k][c] for c in sr.COLS}, True))
    p(sr.trow(f"  delta run 10 - {pt['tag']}", {c: s10[c] - pt["scores"][c] for c in sr.COLS}, True))
    p("-" * (sr.LW + 8 * len(sr.COLS)))
    fwd_saved = saved_match("run8", L["run8"]).endswith("yes")
    nch = len(list((ROOT / "cache/test/r10_cptp_rev").glob("chunk_*.npz")))
    c_ok = g.get("S5.1", {}).get("passed", False) and esha(cb) == g["S5.1"]["numbers"].get("comb_entry_sha256")
    fin = all(np.isfinite([s10[c] for c in sr.COLS]))
    checks = [("7 numbers + average", f"{s10['Avg']:.2f}", "finite", fin),
              ("fwd = saved run 8 scores", str(fwd_saved), "to 1e-9 (scores_r8_cptp)", fwd_saved),
              ("rev cache meta = comb", str(rev_meta_ok(m10[1], st)), "CP+TP rev, keep_final, b1", rev_meta_ok(m10[1], st)),
              ("rev cache chunks", f"{nch}", "58", nch == 58),
              ("comb = S5.1 logged", str(bool(c_ok)), "sha256 equal", bool(c_ok))]
    passed = all(c[3] for c in checks)
    gate_block("S5.3", checks)
    if a.dry:
        p("DRY RUN: gate S5.3 not logged; scores, pairs and results line not written")
    else:
        from runlog import log_gate, append_result
        rdir = ROOT / "results/test"
        pairs = rdir / "pairs_r10_comb.npz"
        np.savez_compressed(pairs, **{f"{d}|L27": sim10[d] for d in DS}, **{f"{d}|gold": gold[d] for d in DS})
        (rdir / "scores_r10_comb.json").write_text(json.dumps(dict(run="r10_comb", scores={"27": s10}, comb=st | dict(prompts="(frozen comb)"),
                                                                   views=[c for c, _ in PRESETS["run10"][1]]), indent=1))
        if passed and not any(json.loads(l).get("run") == "r10_comb" for l in open(runlog.RESULTS)):
            append_result(dict(run="r10_comb", stage="S5.3", split="test", model=st["model"], plan="tp", tp_start=1, tp_exit=7, views=["fwd", "rev"],
                               layers=["27"], batched=False, scores={"27": s10}, cache=[c for c, _ in PRESETS["run10"][1]], comb=True,
                               pair_cosines=str(pairs.relative_to(ROOT))))
        log_gate("S5.3", passed, dict(run10=s10, r3=L["r3"], run4=L["run4"], run5=L["run5"], run8=L["run8"], fwd_matches_saved_run8=fwd_saved,
                                      rev_chunks=nch, comb_entry_sha256=esha(cb)),
                 ["cache/test/r8_cptp", "cache/test/r10_cptp_rev", "results/test/scores_r10_comb.json", "scripts/live_step.py"],
                 "Run 10: 7 numbers computed live from the run 8 forward and r10 reversed caches with frozen comb.")
        p(f"logged gate S5.3: {'PASS' if passed else 'FAIL'}; scores_r10_comb.json, pairs_r10_comb.npz"
          f"{', results.jsonl line' if passed else ''} written")
    footer(t0, cache_files(PRESETS["run10"][1]))


def pass_sps(run):
    import re
    q = ROOT / "logs" / f"{run}.log"
    m = re.findall(r"[ ,(](\d+\.\d+) s/sentence computed", q.read_text()) if q.exists() else []   # not "HH:MM:SS s/sentence computed:"
    return float(m[-1]) if m else None


def mode_s54(a):
    """S5.4: E9-E11 computed and recorded here, C4 recomputed (recorded at L32), reading (plan 14.2), cost, checklist; logs S5.4."""
    import re
    import hashlib
    import runlog
    live_header(a.evidence, "S5.4 combination reading: C4 + E9-E11 (B = 2000, seed 0), cost and checklist, from the caches", dry=a.dry)
    t0 = time.perf_counter()
    L = {k: live_scores(PRESETS[k][1], PRESETS[k][2]) for k in ("run10", "run5", "run8", "r3", "run4")}
    for k in L:
        assert all(np.array_equal(L[k][2][d], L["r3"][2][d]) for d in DS)
    bts = {i: paired_bootstrap(L[BOOTS[i][0]][1], L[BOOTS[i][1]][1], L["r3"][2], n_boot=2000, seed=0) for i in ("C4", "E9", "E10", "E11")}
    p("averages (live): " + " | ".join(f"{k} {L[k][0]['Avg']:.2f}" for k in ("r3", "run4", "run5", "run8", "run10")))
    for i, bt in bts.items():
        A, B, role = BOOTS[i]
        p(f"{i:3s} {'PRIMARY' if role == 'primary' else 'explor.'} {A}-{B}: {bt['diff']:+.2f} 95% [{bt['ci95'][0]:+.2f}, {bt['ci95'][1]:+.2f}] "
          f"Bonf [{bt['ci_adj'][0]:+.2f}, {bt['ci_adj'][1]:+.2f}] pos {bt['n_pos_signs']}/{len(DS)}"[:sr.W])
    rec = [record_boot(i, bts[i], L[BOOTS[i][0]][0], L[BOOTS[i][1]][0], L[BOOTS[i][0]][1], L[BOOTS[i][1]][1], L["r3"][2], 2000, 0, a.dry)
           for i in ("E9", "E10", "E11")]
    tag = lambda r: "+" if r.startswith("appended") else "=" if "already in" in r else "dry "
    p(f"E9-E11 --record to results.jsonl (+ appended, = already there): "
      f"{', '.join(tag(r) + i for r, i in zip(rec, ('E9', 'E10', 'E11')))}")
    word = c4_word(bts["C4"])
    text, note = c4_reading(word, bts["E9"])
    p(f"C4 reading (pre-registered): {word}")
    p(f"READING (plan 14.2): {text}"[:sr.W])
    p(f"  {note or 'E9 95% CI excludes 0: run 10 differs from run 8 (exploratory, no claim).'}"[:sr.W])
    sps = lambda f: json.load(open(ROOT / "results/test" / f)).get("s_per_sentence_computed")
    c3, c8, c10r, c5r = sps("scores_r3_tp_e7.json"), sps("scores_r8_cptp.json"), pass_sps("r10_cptp_rev"), pass_sps("r5_tp_rev_keep_final")
    p(f"{'cost (s/sentence, aux incl.)':34s} {'pass':>8s} {'run':>8s} {'vs r3':>7s}")
    for lab, ps_, run in (("r3 TP (batched, 1 view)", c3, c3), ("run 5 fwd+rev TP (batched)", c5r, (c3 + c5r) if c5r else None),
                          ("run 8 CP+TP fwd (batch 1)", c8, c8), ("run 10 = run 8 + rev pass (b1)", c10r, (c8 + c10r) if c10r else None)):
        f3 = lambda x: "-" if x is None else f"{x:.3f}"
        p(f"  {lab:32s} {f3(ps_):>8s} {f3(run):>8s} {'-' if not run else format(run / c3, '.2f') + 'x':>7s}")
    g = json.load(open(runlog.GATES))
    ok = lambda k: g.get(k, {}).get("passed", False)
    m = re.search(r"s5_preregistration\.md written before any Stage 5 pass\. sha256 ([0-9a-f]{64}) \(recorded (\S+ \S+?);",
                  (ROOT / "notes/decisions.md").read_text())
    pre_ok = bool(m) and m.group(1) == hashlib.sha256((ROOT / "notes/s5_preregistration.md").read_bytes()).hexdigest()
    rec_t = datetime.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S") if m else None
    metas = [ROOT / "cache/dev/cptp_i4_a4_tp_rev_keep_final_L27/meta.json", ROOT / "cache/test/r10_cptp_rev/meta.json"] + \
        sorted((ROOT / "cache/dryrun").glob("dry_r10*/meta.json"))
    first = min(datetime.datetime.fromtimestamp(q.stat().st_mtime) for q in metas if q.exists())
    t_test = datetime.datetime.fromtimestamp(metas[1].stat().st_mtime) if metas[1].exists() else None
    frz = datetime.datetime.fromisoformat(g["S5.1"]["date"]) if ok("S5.1") else None
    have = {json.loads(l).get("run") for l in open(runlog.RESULTS)}
    e_ok = all(f"S5.4-{i}-{BOOTS[i][0]}_vs_{BOOTS[i][1]}" in have for i in ("E9", "E10", "E11"))
    rd = ROOT / "notes/comb_reading.md" if not runlog.SCRATCH else runlog.LOGS / "comb_reading.md"
    items = [("S5.1 hash before any S5 pass", f"{m.group(2)[11:] if m else '-'} < {first:%H:%M:%S}", pre_ok and rec_t < first),
             ("S5.1 frozen before test pass", f"{frz:%H:%M:%S}" if frz else "-", bool(frz and t_test and frz < t_test)),
             ("S5.2 and S5.3 passed", f"{ok('S5.2')}/{ok('S5.3')}", ok("S5.2") and ok("S5.3")),
             ("C4 recorded (L32)", str("S5.3-C4-run10_vs_run5" in have), "S5.3-C4-run10_vs_run5" in have),
             ("E9-E11 recorded", "dry: would record" if a.dry and not _REPRINT else str(e_ok), (a.dry and not _REPRINT) or e_ok),
             ("reading written", "dry: would write" if a.dry and not _REPRINT else "notes/comb_reading.md",
              rd.exists() if _REPRINT else True)]       # a live run writes it below; a dry run would; a reprint checks it
    passed = all(x[2] for x in items)
    gate_block("S5.4", [(n, v, "required", o) for n, v, o in items])
    out = dict(C4=dict(bts["C4"], word=word), E9=bts["E9"], E10=bts["E10"], E11=bts["E11"], reading=text, e9_note=note,
               averages={k: v[0]["Avg"] for k, v in L.items()}, cost=dict(r3=c3, run5_rev_pass=c5r, run8=c8, run10_rev_pass=c10r,
               run10_total=(c8 + c10r) if c10r else None, ratio_vs_r3=((c8 + c10r) / c3) if c10r else None),
               checklist={n: dict(value=v, ok=o) for n, v, o in items}, passed=passed)
    if a.dry:
        p("DRY RUN: gate S5.4 not logged; notes/comb_reading.md and logs/S5.4.json not written")
    else:
        from runlog import log_gate
        b = bts["C4"]
        rd.write_text(f"# Combination (run 10 = Enhancement 1 + 2): reading (S5.4, {datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
                      f"- C4 (primary) run 10 - run 5: {b['diff']:+.2f}, 95% CI [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}], Bonferroni "
                      f"[{b['ci_adj'][0]:+.2f}, {b['ci_adj'][1]:+.2f}], {b['n_pos_signs']}/7 positive -> **{word}**.\n"
                      + "".join(f"- {i} (exploratory) {BOOTS[i][0]} - {BOOTS[i][1]}: {bts[i]['diff']:+.2f} [{bts[i]['ci95'][0]:+.2f}, "
                                f"{bts[i]['ci95'][1]:+.2f}], {bts[i]['n_pos_signs']}/7 positive.\n" for i in ("E9", "E10", "E11"))
                      + f"- Reading (plan 14.2): {text}\n" + (f"- {note}\n" if note else "")
                      + "- Caveats: not tuned; CP setting from dev (forward view, addendum 1); keep_final chosen after dev; run 4 batch-1 vanilla, "
                        "run 5 and r3 batched TP, run 8 and run 10 batch 1; CP x reversed view untested before the test pass.\n")
        (runlog.LOGS / "S5.4.json").write_text(json.dumps(out, indent=1, default=float))
        log_gate("S5.4", passed, out, ["logs/S5.4.json", "notes/comb_reading.md", "notes/s5_preregistration.md", "scripts/live_step.py"],
                 f"C4 {word}; E9-E11 recorded; reading, cost and checklist (plan 12.2).")
        p(f"logged gate S5.4 ({'PASS' if passed else 'FAIL'}); logs/S5.4.json, comb_reading.md written. Stage 5 stops here.")
    footer(t0, cache_files(PRESETS["run10"][1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", default="(unnamed)", help="screenshot name, e.g. L07_run5_scored")
    ap.add_argument("--dry", action="store_true", help="trial: banner says DRY RUN; nothing logged or written")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("score"); s.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    s.add_argument("--cache", action="append", help="DIR:LAYER (one view, or two to combine)"); s.add_argument("--frozen")
    s.add_argument("--ref"); s.add_argument("--label")
    b = sub.add_parser("bootstrap"); b.add_argument("id", choices=sorted(BOOTS)); b.add_argument("--record", action="store_true")
    sub.add_parser("run7"); sub.add_parser("verdict-s2.10")
    sub.add_parser("scored-s4.6"); sub.add_parser("gate-s4.7"); sub.add_parser("gate-s4.8")
    sub.add_parser("score-s5.3"); sub.add_parser("gate-s5.4")
    ap.add_argument("--show", action="store_true", help="read-only reprint of an already-logged S4.6 / S4.8 / S5.3 / C4 / S5.4 screen")
    b.add_argument("--n-boot", type=int, default=2000); b.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gate"); g.add_argument("preset", choices=sorted(PRESETS)); g.add_argument("--gid")
    g.add_argument("--ref"); g.add_argument("--vs", choices=sorted(PRESETS)); g.add_argument("--above", choices=sorted(PRESETS))
    g.add_argument("--avg-tol", type=float); g.add_argument("--ds-tol", type=float)
    a = ap.parse_args()
    if a.mode == "score" and not a.preset and not a.cache:
        ap.error("score needs a preset or --cache DIR:LAYER")
    if a.show:                          # read-only reprint: find when the step was logged; refuse if it was not
        global _REPRINT
        import runlog
        g = json.load(open(runlog.GATES))
        if a.mode == "scored-s4.6" and g.get("S4.6", {}).get("passed"):
            _REPRINT = g["S4.6"]["date"]
        elif a.mode == "gate-s4.8" and g.get("S4.8", {}).get("passed"):
            _REPRINT = g["S4.8"]["date"]
        elif a.mode == "score-s5.3" and g.get("S5.3", {}).get("passed"):
            _REPRINT = g["S5.3"]["date"]
        elif a.mode == "gate-s5.4" and g.get("S5.4", {}).get("passed"):
            _REPRINT = g["S5.4"]["date"]
        elif a.mode == "bootstrap" and a.id == "C4":
            hit = [json.loads(l) for l in open(runlog.RESULTS) if json.loads(l).get("run") == "S5.3-C4-run10_vs_run5"]
            _REPRINT = hit[-1]["date"] if hit else None
        elif a.mode not in ("scored-s4.6", "gate-s4.8", "score-s5.3", "gate-s5.4", "bootstrap") or (a.mode == "bootstrap" and a.id != "C4"):
            ap.error("--show is only for scored-s4.6, gate-s4.8, score-s5.3, bootstrap C4 and gate-s5.4")
        if not _REPRINT:
            print(f"REFUSED: --show reprints a step that has already logged; nothing logged yet for {a.mode}. Run it without --show.")
            sys.exit(2)
        a.dry = True                    # nothing is written or logged
    {"score": mode_score, "bootstrap": mode_bootstrap, "gate": mode_gate, "run7": mode_run7, "verdict-s2.10": mode_verdict,
     "scored-s4.6": mode_s46, "gate-s4.7": mode_s47, "gate-s4.8": mode_s48, "score-s5.3": mode_s53, "gate-s5.4": mode_s54}[a.mode](a)
    if _REPRINT:
        sr.OUT[:] = [f"REPRINT: nothing written or logged (already logged at {_REPRINT})" if l.startswith("DRY RUN:") else l for l in sr.OUT]
    width = max(len(l) for l in sr.OUT)
    p(f"[printout {len(sr.OUT) + 1} lines x {width} cols; fits 100x40 with prompt: {'yes' if len(sr.OUT) + 1 <= sr.H and width <= sr.W else 'NO'}]")
    print("\n".join(sr.OUT))


if __name__ == "__main__":
    main()
