# S3.2 Dev sweep + S3.3 freeze: Enhancement 1 (ReverseEOL + TP), plan Section 11. STS-B dev only.
import datetime
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from runlog import check_env, log_gate, append_result as _append, require_gates, RESULTS
from evalsts import SentenceTable, score, DEV_TASK
from views import make_items, combine, FWD, REV
from cache import run_cached
from stage1 import disk

_seen = {json.loads(l)["run"] for l in open(RESULTS)}


def append_result(rec):   # a rerun reuses the cached states; do not write the same dev line twice
    if rec["run"] not in _seen:
        _append(rec); _seen.add(rec["run"])


check_env()
require_gates(["S2.5", "S3.1"])
import mlx.core as mx
from backend_mlx import MLXBackend

mx.set_cache_limit(1 << 30)
MODEL = "models/llama2_7b_q8_meta"
g = json.load(open(ROOT / "gates.json"))
R2_DEV, R3_DEV = g["S1.1"]["numbers"]["r2_dev"], g["S1.2"]["numbers"]["tp27"]["7"]
tpf = json.load(open(ROOT / "frozen_settings.json"))["tp_reproduction"]["settings"]
S, E_STAR, M = tpf["tp_start"], tpf["tp_exit"], tpf["output_layer"]
ALPHAS, NORMS, PUNCTS = (0.4, 0.5, 0.6), (True, False), ("full", "keep_final")
DEFAULT = (0.5, True, "full")
print(f"s = {S}, e* = {E_STAR}, M = {M}; r2_dev {R2_DEV:.4f}, r3_dev {R3_DEV:.4f}")

table = SentenceTable(DEV_TASK)
words = table.words
n = len(words)

# ---- forward TP states from the S1.2 cache ----
fdir = ROOT / "cache/dev/tp_s1_e7_L20-32"
fmeta = json.load(open(fdir / "meta.json"))
assert (fmeta["plan"], fmeta["tp_start"], fmeta["tp_exit"], fmeta["view"], fmeta["n_items"]) == ("tp", S, E_STAR, "fwd", n)
assert str(M) in fmeta["layers"] and fmeta["batched"] is True and fmeta["pst"] == "ref/pst_row_3407.npy"
Ef = np.concatenate([np.load(fdir / f"chunk_{s:07d}.npz")[f"L{M}"].astype(np.float32) for s in range(0, n, fmeta["chunk"])])
r3_check = score(table, Ef)[0]
assert abs(r3_check - R3_DEV) < 1e-9, (r3_check, R3_DEV)
print(f"forward (S1.2 cache) layer {M}: {r3_check:.4f} == r3_dev")

# ---- TP passes: rev (2 punct rules), then the two controls ----
be = MLXBackend(ROOT / MODEL, ROOT / "models/llama2_tok", np.load(ROOT / "ref/pst_row_3407.npy"))
enc = partial(be.encode_batched, plan="tp", tp_start=S, tp_exit=E_STAR, stop_after=M)
PASSES = [("rev", "full"), ("rev", "keep_final"), ("fwd_revprompt", "full"), ("shuf", "full")]
Ev, secs = {}, {}
for view, punct in PASSES:
    name = f"tp_e{E_STAR}_rev_{punct}_L{M}" if view == "rev" else f"tp_e{E_STAR}_{view}_L{M}"
    items = make_items(words, [view], "tp", punct)
    assert all(it[2].count("<PST>") == 1 for it in items)
    meta = dict(model=MODEL, plan="tp", tp_start=S, tp_exit=E_STAR, stop_after=M, view=view, punct=punct,
                split=DEV_TASK, batched=True, pst="ref/pst_row_3407.npy")
    disk(f"before {name}")
    t0 = datetime.datetime.now()
    Ev[(view, punct)] = run_cached(items, enc, str(ROOT / "cache/dev" / name), [M], chunk=500, meta=meta, min_free_gb=5.0,
                                   on_chunk=lambda s, k, c, t, nm=name: (mx.clear_cache(), print(f"  {nm} {s + k}/{n} ({'computed' if c else 'reused'}, {t:.0f} s)", flush=True)))[M]
    secs[name] = (datetime.datetime.now() - t0).total_seconds()
    disk(f"after {name}")
    assert np.isfinite(Ev[(view, punct)]).all()

# ---- evaluate ----
nz = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
Er = {p: Ev[("rev", p)] for p in PUNCTS}
cos_fr = {p: float((nz(Ef) * nz(Er[p])).sum(1).mean()) for p in PUNCTS}
rev_only = {p: score(table, Er[p])[0] for p in PUNCTS}
res = {(a, nm, p): score(table, combine({"fwd": Ef, "rev": Er[p]}, ["fwd", "rev"], a, nm))[0]
       for p in PUNCTS for a in ALPHAS for nm in NORMS}
assert len(res) == 12
# DEVIATION (user decision 2026-11-04, after the dev results; gate_log.md): only punctuation rules whose rev-only (6c) check
# passes (>= r2_dev - 3) are eligible. The selection rule is otherwise unchanged. The unrestricted choice is still reported.
not_collapsed = {p: bool(rev_only[p] >= R2_DEV - 3) for p in PUNCTS}
eligible = [p for p in PUNCTS if not_collapsed[p]]
def select(keys):
    b = max(keys, key=res.get)
    t = [k for k in keys if res[b] - res[k] < 0.05]
    return b, t, min(t, key=lambda k: (k[0] != DEFAULT[0], k[1] != DEFAULT[1], k[2] != DEFAULT[2], -res[k]))
orig_best, orig_tied, orig_chosen = select(list(res))
print(f"original rule (all 12): best {orig_best} {res[orig_best]:.4f}; chosen {orig_chosen} {res[orig_chosen]:.4f}; "
      f"rev-only of its punct {rev_only[orig_chosen[2]]:.2f} vs floor {R2_DEV - 3:.2f}")
print(f"restricted rule: eligible punct {eligible}")
assert eligible, "no punctuation rule passes the rev-only check"
best_key, tied, chosen = select([k for k in res if k[2] in eligible])
a_c, n_c, p_c = chosen

ctrl = {}
for view, label in (("fwd_revprompt", "6b"), ("shuf", "6a")):
    X = Ev[(view, "full")]
    ctrl[label] = dict(view=view, single_view=score(table, X)[0], cos_fwd=float((nz(Ef) * nz(X)).sum(1).mean()),
                       grid={f"a{a}|norm_{'on' if nm else 'off'}": score(table, combine({"fwd": Ef, "x": X}, ["fwd", "x"], a, nm))[0]
                             for a in ALPHAS for nm in NORMS})
    ctrl[label]["at_chosen"] = ctrl[label]["grid"][f"a{a_c}|norm_{'on' if n_c else 'off'}"]

print(f"\nmean cos(e_fwd, e_rev) over dev: {cos_fr}")
print(f"rev-only (6c): {rev_only}   r2_dev {R2_DEV:.2f} (collapse floor {R2_DEV - 3:.2f})   r3_dev {R3_DEV:.2f}")
print("| punct | alpha | norm | dev Spearman | vs r3_dev |\n|---|---|---|---|---|")
for k in sorted(res, key=lambda k: (k[2] != "full", k[0], not k[1])):
    print(f"| {k[2]} | {k[0]} | {'on' if k[1] else 'off'} | {res[k]:.2f} | {res[k] - R3_DEV:+.2f} |"
          + (" <- best" if k == best_key else "") + (" <- chosen" if k == chosen else ""))
print(f"best {best_key} {res[best_key]:.4f}; tied within 0.05: {tied}; chosen {chosen} {res[chosen]:.4f}")
for lb, c in ctrl.items():
    print(f"control {lb} ({c['view']}, TP): single view {c['single_view']:.2f}; fwd + {c['view']} at chosen alpha/norm {c['at_chosen']:.2f}; "
          f"cos(fwd, view) {c['cos_fwd']:.4f}; grid {json.dumps({k: round(v, 2) for k, v in c['grid'].items()})}")

for p in PUNCTS:
    append_result(dict(run=f"S3.2-tp-rev-only-{p}-L{M}", stage="S3.2", split="dev", model=MODEL, plan="tp", tp_start=S, tp_exit=E_STAR,
                       view="rev", punct=p, layers=[str(M)], scores={str(M): {"STS-B dev": rev_only[p]}},
                       cache=f"cache/dev/tp_e{E_STAR}_rev_{p}_L{M}"))
for (a, nm, p), s in res.items():
    append_result(dict(run=f"S3.2-enh1-a{a}-norm{'on' if nm else 'off'}-{p}-L{M}", stage="S3.2", split="dev", model=MODEL, plan="tp",
                       tp_start=S, tp_exit=E_STAR, views=["fwd", "rev"], punct=p, alpha=a, normalize=nm, layers=[str(M)],
                       scores={str(M): {"STS-B dev": s}}, cache=[str(fdir.relative_to(ROOT)), f"cache/dev/tp_e{E_STAR}_rev_{p}_L{M}"]))
for lb, c in ctrl.items():
    append_result(dict(run=f"S3.2-control-{lb}-{c['view']}-L{M}", stage="S3.2", split="dev", model=MODEL, plan="tp", tp_start=S,
                       tp_exit=E_STAR, views=["fwd", c["view"]], layers=[str(M)], role="control (context only)",
                       scores={str(M): {"STS-B dev single view": c["single_view"], **{f"STS-B dev fwd+view {k}": v for k, v in c["grid"].items()}}},
                       cache=[str(fdir.relative_to(ROOT)), f"cache/dev/tp_e{E_STAR}_{c['view']}_L{M}"]))

cos_ok = all(0.2 < c < 0.995 for c in cos_fr.values())
passed = len(res) == 12 and cos_ok and not_collapsed[chosen[2]]
out = dict(s=S, e_star=E_STAR, M=M, r2_dev=R2_DEV, r3_dev=R3_DEV, rev_only=rev_only, rev_only_not_collapsed=not_collapsed,
           cos_fwd_rev=cos_fr, grid={f"a{a}|norm_{'on' if nm else 'off'}|{p}": s for (a, nm, p), s in res.items()},
           original_rule=dict(best=list(orig_best), chosen=list(orig_chosen), score=res[orig_chosen], tied=[list(k) for k in orig_tied]),
           restricted_rule=dict(eligible_punct=eligible, deviation="gate_log.md 2026-11-04, user decision after dev results"),
           best=dict(key=list(best_key), score=res[best_key]), tied=[list(k) for k in tied], chosen=dict(key=list(chosen), score=res[chosen]),
           best_minus_r3_dev=res[best_key] - R3_DEV, chosen_minus_r3_dev=res[chosen] - R3_DEV, controls=ctrl, seconds=secs,
           batching="equal-length batched (S0.13), same path as the S1.2 forward cache")
(ROOT / "results/dev/S3.2_enh1.json").write_text(json.dumps(out, indent=1, default=float))
log_gate("S3.2", passed, {k: v for k, v in out.items() if k != "controls"} | dict(controls={k: dict(single_view=v["single_view"], at_chosen=v["at_chosen"]) for k, v in ctrl.items()}),
         ["results/dev/S3.2_enh1.json", "results/results.jsonl", "logs/S3.2.log", "scripts/s3_2_enh1_dev.py"],
         f"DEVIATION: selection restricted to punct with rev-only >= r2_dev - 3 ({eligible}); original rule chose {orig_chosen} {res[orig_chosen]:.2f}. "
         f"Best eligible {res[best_key]:.2f} vs r3_dev {R3_DEV:.2f} ({res[best_key] - R3_DEV:+.2f}); chosen {chosen} {res[chosen]:.2f}. "
         f"cos(fwd, rev) {cos_fr} in (0.2, 0.995): {cos_ok}. rev-only {rev_only} >= r2_dev - 3 ({R2_DEV - 3:.2f}): {not_collapsed}. "
         f"Controls (context): 6b {ctrl['6b']['at_chosen']:.2f}, 6a {ctrl['6a']['at_chosen']:.2f} at the chosen alpha/norm.")
if not passed:
    sys.exit("STOP: S3.2 gate failed; nothing frozen. Debug reversed strings, <PST> in the reversed prompt, punctuation.")
if res[best_key] < R3_DEV:
    print(f"FINDING: best combined {res[best_key]:.2f} < r3_dev {R3_DEV:.2f}. Allowed by the gate; reported, nothing changed.")

# ---- S3.3 freeze ----
fs = json.load(open(ROOT / "frozen_settings.json"))
assert "enh1" not in fs, "enh1 already frozen"
fs["enh1"] = dict(date=datetime.date.today().isoformat(),
                  settings=dict(views=["fwd", "rev"], alpha=a_c, normalize=n_c, punct=p_c, plan="tp", tp_start=S, tp_exit=E_STAR,
                                output_layer=M, batched=True, pst="ref/pst_row_3407.npy", wording=dict(fwd=FWD, rev=REV)),
                  dev=dict(chosen=res[chosen], best=res[best_key], r3_dev=R3_DEV, r2_dev=R2_DEV, rev_only=rev_only[p_c],
                           controls_at_chosen={k: v["at_chosen"] for k, v in ctrl.items()}),
                  model="models/llama2_7b_q8_meta (8-bit, group 64)",
                  reason=f"S3.2 selection rule restricted to punct passing the rev-only check ({eligible}; user decision, gate_log.md); original rule chose {orig_chosen}. Restricted: best dev {best_key} = {res[best_key]:.4f}; ties < 0.05 go to defaults (0.5, on, full); "
                         f"tied set {tied} -> chosen {chosen}. alpha weights the forward view. e* shared across views (no re-tuning of e or M).")
json.dump(fs, open(ROOT / "frozen_settings.json", "w"), indent=1)
sha = hashlib.sha256((ROOT / "frozen_settings.json").read_bytes()).hexdigest()
enh1_sha = hashlib.sha256(json.dumps(fs["enh1"], sort_keys=True).encode()).hexdigest()
print("frozen 'enh1':", json.dumps(fs["enh1"]["settings"]))
print(f"frozen_settings.json sha256 {sha}; enh1 entry sha256 {enh1_sha}")
log_gate("S3.3", True, dict(enh1=fs["enh1"]["settings"], dev=fs["enh1"]["dev"], frozen_settings_sha256=sha, enh1_entry_sha256=enh1_sha),
         ["frozen_settings.json", "results/dev/S3.2_enh1.json", "scripts/s3_2_enh1_dev.py"],
         "Enhancement 1 settings frozen after S3.2 passed, before any Stage 3 test pass. File hash logged.")
