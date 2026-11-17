"""One embedding pass + scoring, same pipeline code as the notebooks (views -> backend_mlx -> cache -> evalsts).

Resumable: finished chunks live in cache/<split>/<run>/ and are reused when the same command is re-run.
Never deletes anything. Stops cleanly (exit code 3) if free disk < 5 GB before a chunk.

Examples (project root, venv active):
  python scripts/run_pass.py --run r1r2_vanilla --plan vanilla --split test --layers 27,final
  python scripts/run_pass.py --run r3_tp_e7 --plan tp --tp-start 1 --tp-exit 7 --split test --layers 27 --batched
  python scripts/run_pass.py --run dry_tp_e7 --plan tp --tp-start 1 --tp-exit 7 --split dev --layers 27 --batched --limit 50
  python scripts/run_pass.py --run r4_rev_full --plan vanilla --view rev --punct full --split test --layers 27   (S2.4)
  python scripts/run_pass.py --run r5_tp_rev_full --plan tp --tp-start 1 --tp-exit 7 --view rev --punct full --split test --layers 27 --batched   (S3.4)

Test passes of a non-fwd view (S2.4 onward) only cache states; the combined run is scored by its own step.
CP mode (S2.9, CP without TP): --cp-from-frozen reads act_layer, alpha and the auxiliary prompt ONLY from
frozen_settings.json["cp_alone"]; the auxiliary pass runs inside the same loop (B saved with the states, key LB).
  python scripts/run_pass.py --run r7_cp --plan vanilla --split test --layers 27 --cp-from-frozen          (S2.9)
  python scripts/run_pass.py --run dry_cp --plan vanilla --split dev --layers 27 --cp-from-frozen --limit 50
  python scripts/run_pass.py --run dev_cp_x --plan vanilla --split dev --layers 27 --cp-index 4 --cp-alpha 2  (dev only)
CP + TP mode (Stage 4, batch 1): --cptp-from-frozen reads index, alpha, D1 (aux with or without TP) and every prompt string ONLY
from frozen_settings.json["enh2"]; --control sentiment swaps the auxiliary prompt for the sentiment control (run 9).
  python scripts/run_pass.py --run r8_cptp --plan tp --tp-start 1 --tp-exit 7 --split test --layers 27 --cptp-from-frozen          (S4.6 run 8)
  python scripts/run_pass.py --run r9_cptp_sent --plan tp --tp-start 1 --tp-exit 7 --split test --layers 27 --cptp-from-frozen --control sentiment
  python scripts/run_pass.py --run dry_r8 --plan tp --tp-start 1 --tp-exit 7 --split dev --layers 27 --cptp-from-frozen --limit 50
  python scripts/run_pass.py --run dev_x --plan tp --tp-start 1 --tp-exit 7 --split dev --layers 27 --cptp-dev --cp-index 4 --cp-alpha 2 --cp-aux tp  (dev only)
CP + TP reversed view (Stage 5, run 10; notes/s5_preregistration.md): --cptp-from-frozen --view rev --punct keep_final. CP settings from
'enh2', punct from 'enh1'; the auxiliary prompt receives the reversed text. A test pass also needs S4.8 and S5.1 and the frozen 'comb'.
  python scripts/run_pass.py --run r10_cptp_rev --plan tp --tp-start 1 --tp-exit 7 --view rev --punct keep_final --split test --layers 27 --cptp-from-frozen
Every pass ends with a 12-line END-OF-PASS banner (13 for CP + TP; times, wall, s/sentence, chunks, disk, cache meta sha256, Python, kernel).
  python scripts/run_pass.py --banner-test        (50 dev sentences, vanilla, layer 27: prints the same banner)
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # no fork warning when the banner calls df
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from runlog import check_env, free_gb, append_result
from cache import run_cached, LowDiskError
from evalsts import SentenceTable, score, TEST_TASKS, DEV_TASK
from views import make_items

FLOOR_GB = 5.0
GATES_FOR_TEST = [f"S0.{i}" for i in range(14)] + ["S1.1", "S1.2", "S1.3", "S1.4"]
SHORT = {"STS12": "STS12", "STS13": "STS13", "STS14": "STS14", "STS15": "STS15", "STS16": "STS16",
         "STSBenchmark": "STS-B", "SICKRelatedness": "SICK-R", "STSBenchmark-dev": "STS-B dev"}


class Log:
    """Prints to stdout and appends to logs/<run>.log, unless stdout already is that file (nohup redirect)."""
    def __init__(self, run):
        self.path = ROOT / "logs" / f"{run}.log"; self.path.parent.mkdir(exist_ok=True)
        try:
            same = self.path.exists() and os.path.samestat(os.fstat(1), os.stat(self.path))
        except OSError:
            same = False
        self.fh = None if same else open(self.path, "a")

    def __call__(self, msg):
        line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n"); self.fh.flush()


def parse():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--run", help="run name (cache dir, log and result names)")
    a.add_argument("--plan", choices=["vanilla", "tp"])
    a.add_argument("--tp-start", type=int, default=0)
    a.add_argument("--tp-exit", type=int, default=0)
    a.add_argument("--view", default="fwd", choices=["fwd", "rev", "fwd_revprompt", "shuf"])
    a.add_argument("--punct", default="full", choices=["full", "keep_final"], help="punctuation rule for reversed views")
    a.add_argument("--split", choices=["dev", "test"])
    a.add_argument("--layers", help="comma list, e.g. 27,final  (final = layer 32 after the final norm)")
    a.add_argument("--batched", action="store_true", help="equal-length batches (TP only; validated in S0.13)")
    a.add_argument("--model", default="models/llama2_7b_q8_meta")
    a.add_argument("--limit", type=int, default=None, help="dry run: first N sentences, dev split only")
    a.add_argument("--chunk", type=int, default=500)
    a.add_argument("--banner-test", action="store_true", help="50 dev sentences, vanilla, layer 27; prints the end-of-pass banner")
    a.add_argument("--cp-from-frozen", action="store_true", help="CP steering with act_layer, alpha, aux prompt from frozen 'cp_alone'")
    a.add_argument("--cp-index", type=int, help="CP act_layer; dev only without --cp-from-frozen; must match the frozen entry if both")
    a.add_argument("--cp-alpha", type=float, help="CP alpha; dev only without --cp-from-frozen; must match the frozen entry if both")
    a.add_argument("--cptp-from-frozen", action="store_true", help="CP + TP (Stage 4) with every setting from frozen 'enh2'")
    a.add_argument("--cptp-dev", action="store_true", help="CP + TP with --cp-index/--cp-alpha/--cp-aux; dev split only")
    a.add_argument("--cp-aux", choices=["tp", "notp"], help="CP + TP dev only: auxiliary pass with or without TP (D1)")
    a.add_argument("--control", choices=["sentiment"], help="CP + TP: sentiment auxiliary prompt (control 9)")
    args = a.parse_args()
    if args.banner_test:   # a dev dry run with fixed settings; same code path as any --limit run
        args.run, args.plan, args.split, args.layers, args.limit = "banner_test", "vanilla", "dev", "27", 50
    missing = [f"--{k}" for k in ("run", "plan", "split", "layers") if getattr(args, k) is None]
    if missing:
        a.error("the following arguments are required: " + ", ".join(missing))
    return args


def banner(log, args, cdir, t_start, wall, sps, state, n_chunks):
    """12-line end-of-pass block (13 with CP + TP). Reporting only: reads the clock, the disk, meta.json and the kernel spec."""
    end = datetime.datetime.now()
    meta = cdir / "meta.json"
    msha = hashlib.sha256(meta.read_bytes()).hexdigest()[:12] if meta.exists() else "(no meta.json)"
    kspec = Path.home() / "Library/Jupyter/kernels/tp-mlx/kernel.json"
    kexe = json.load(open(kspec))["argv"][0] if kspec.exists() else "(kernel tp-mlx not found)"
    df = subprocess.run(["df", "-h", str(Path.home())], capture_output=True, text=True).stdout.strip().splitlines()[-1].split()
    cp, ct = getattr(args, "cp", None), getattr(args, "cptp", None)
    for line in ["=" * 88,
                 f"END OF PASS  {args.run}",
                 f"start {t_start:%Y-%m-%d %H:%M:%S}   end {end:%Y-%m-%d %H:%M:%S}   wall {wall / 60:.2f} min",
                 f"s/sentence computed: {sps:.4f}" if sps else "s/sentence computed: - (no chunk computed in this call)",
                 f"chunks: {state['comp_chunks']} computed, {state['done'] - state['comp_chunks']} reused, {n_chunks} total",
                 f"settings: plan {args.plan}, view {args.view}, split {args.split}, layers {args.layers}, batched {args.batched}, limit {args.limit}"
                 + (f", CP act_layer {cp['act_layer']} alpha {cp['alpha']:g} ({cp['source']})" if cp else ""),
                 *([f"  CP+TP index {ct['act_layer']} alpha {ct['alpha']:g} aux {'with' if ct['aux_tp'] else 'without'} TP"
                    f"{', sentiment control' if ct['control'] else ''} ({ct['source']})"] if ct else []),
                 f"free disk: {free_gb():.1f} GB  (df -h ~: {df[3]} avail of {df[1]})",
                 f"cache meta.json sha256[:12] {msha}  {cdir.relative_to(ROOT)}/meta.json",
                 f"python {platform.python_version()}  {sys.executable}",
                 f"kernel tp-mlx -> {kexe}",
                 f"log: logs/{args.run}.log",
                 "=" * 88]:
        log(line)


def cp_entry_sha(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def resolve_cp(args, log):
    """CP settings for this pass, or None. Refuses mismatching or out-of-scope arguments (exit 2) before the model loads."""
    manual = args.cp_index is not None or args.cp_alpha is not None
    if args.cptp_from_frozen or args.cptp_dev:
        if args.cp_from_frozen:
            log("REFUSED: CP arguments:"); log("  - --cp-from-frozen (CP alone) and CP + TP cannot be combined"); sys.exit(2)
        return None                                     # CP + TP is resolved by resolve_cptp
    if args.cp_aux or args.control:
        log("REFUSED: CP arguments:"); log("  - --cp-aux / --control need --cptp-from-frozen or --cptp-dev"); sys.exit(2)
    if not args.cp_from_frozen and not manual:
        return None
    sys.dont_write_bytecode = True                      # importing cp_mlx reads ref/cp_repo; never write .pyc there
    import cp_mlx
    main_t, aux_t = cp_mlx.cp_templates()
    bad = []
    if (args.plan, args.view, args.batched, args.layers, args.tp_start, args.tp_exit) != ("vanilla", "fwd", False, "27", 0, 0):
        bad.append("CP mode needs --plan vanilla, view fwd, batch 1, --layers 27, no TP")
    if args.cp_from_frozen:
        ent = json.load(open(ROOT / "frozen_settings.json")).get("cp_alone") if (ROOT / "frozen_settings.json").exists() else None
        if not ent:
            bad.append("frozen_settings.json has no 'cp_alone' (S2.8 selection freezes it)")
            cp = None
        else:
            st = ent["settings"]
            cp = dict(act_layer=st["act_layer"], alpha=float(st["alpha"]), aux=st["aux_prompt"], source="frozen cp_alone", entry_sha=cp_entry_sha(ent))
            if args.cp_index is not None and args.cp_index != st["act_layer"]:
                bad.append(f"--cp-index {args.cp_index} != frozen act_layer {st['act_layer']}")
            if args.cp_alpha is not None and args.cp_alpha != float(st["alpha"]):
                bad.append(f"--cp-alpha {args.cp_alpha:g} != frozen alpha {float(st['alpha']):g}")
            if st["aux_prompt"] != aux_t or st["main_prompt"] != main_t:
                bad.append("frozen prompt text differs from ref/cp_repo/evaluate_intervention.py:194/201")
            if (st["output_layer"], st["plan"], st["batched"], st["strategy"]) != (27, "vanilla", False, "scaled"):
                bad.append(f"frozen cp_alone is not (layer 27, vanilla, batch 1, scaled): {st}")
    else:
        cp = dict(act_layer=args.cp_index, alpha=args.cp_alpha, aux=aux_t, source="--cp-index/--cp-alpha (dev only)", entry_sha=None)
        if args.split != "dev":
            bad.append("--cp-index/--cp-alpha without --cp-from-frozen are for the dev split only")
        if args.cp_index is None or args.cp_alpha is None:
            bad.append("give both --cp-index and --cp-alpha")
    if bad:
        log("REFUSED: CP arguments:"); [log("  - " + m) for m in bad]
        sys.exit(2)
    return cp


def resolve_cptp(args, log):
    """CP + TP settings (Stage 4), or None. Refuses mismatching or out-of-scope arguments (exit 2) before the model loads."""
    if not args.cptp_from_frozen and not args.cptp_dev:
        return None
    sys.dont_write_bytecode = True
    import cp_mlx
    T = cp_mlx.cptp_templates()
    bad, ct = [], None
    if args.cptp_from_frozen and args.cptp_dev:
        bad.append("give --cptp-from-frozen or --cptp-dev, not both")
    if (args.plan, args.batched, args.layers) != ("tp", False, "27") or args.view not in ("fwd", "rev"):
        bad.append("CP + TP needs --plan tp, view fwd or rev, batch 1 (no --batched), --layers 27")
    if args.view == "rev":          # Stage 5: reversed view of run 10; nothing else may vary
        e1 = json.load(open(ROOT / "frozen_settings.json")).get("enh1") if (ROOT / "frozen_settings.json").exists() else None
        if not args.cptp_from_frozen or args.cptp_dev:
            bad.append("CP + TP view rev (Stage 5) needs --cptp-from-frozen; no --cptp-dev (nothing is tuned for the reversed view)")
        if args.control:
            bad.append("CP + TP view rev has no control (--control is for run 9, view fwd)")
        if not e1:
            bad.append("frozen_settings.json has no 'enh1' (punctuation rule of the reversed view)")
        elif args.punct != e1["settings"]["punct"]:
            bad.append(f"--punct {args.punct} != frozen enh1 punct {e1['settings']['punct']}")
    if args.cptp_from_frozen:
        if args.cp_index is not None or args.cp_alpha is not None or args.cp_aux:
            bad.append("no manual CP arguments with --cptp-from-frozen (--cp-index/--cp-alpha/--cp-aux)")
        ent = json.load(open(ROOT / "frozen_settings.json")).get("enh2") if (ROOT / "frozen_settings.json").exists() else None
        if not ent:
            bad.append("frozen_settings.json has no 'enh2' (S4.5 selection freezes it)")
        else:
            st = ent["settings"]
            ct = dict(act_layer=st["act_layer"], alpha=float(st["alpha"]), aux_tp=bool(st["aux_tp"]), control=args.control,
                      source="frozen enh2", entry_sha=cp_entry_sha(ent))
            if st["prompts"] != T:
                bad.append("frozen enh2 prompt strings differ from src/cp_mlx.cptp_templates() (CP :194/:201, pre-registered)")
            if (st["tp_start"], st["tp_exit"]) != (args.tp_start, args.tp_exit):
                bad.append(f"--tp-start/--tp-exit {args.tp_start}/{args.tp_exit} != frozen enh2 {st['tp_start']}/{st['tp_exit']}")
            if (st["output_layer"], st["stop_after"], st["batched"], st["strategy"], st["plan"]) != (27, 27, False, "scaled", "tp"):
                bad.append(f"frozen enh2 is not (layer 27, stop 27, batch 1, scaled, tp): {st}")
    if args.cptp_dev:
        ct = dict(act_layer=args.cp_index, alpha=args.cp_alpha, aux_tp=args.cp_aux == "tp", control=args.control,
                  source="--cptp-dev (dev only)", entry_sha=None)
        if args.split != "dev":
            bad.append("--cptp-dev is for the dev split only")
        if args.cp_index is None or args.cp_alpha is None or args.cp_aux is None:
            bad.append("--cptp-dev needs --cp-index, --cp-alpha and --cp-aux")
        if (args.tp_start, args.tp_exit) != (1, 7):
            bad.append("CP + TP dev passes use TP s 1, e 7 (pre-registered)")
    if bad:
        log("REFUSED: CP + TP arguments:"); [log("  - " + m) for m in bad]
        sys.exit(2)
    ct["aux_template"] = cp_mlx.cptp_aux_template(ct["aux_tp"], ct["control"])
    return ct


def frozen_key(args):
    if getattr(args, "cptp", None):
        if args.view == "rev":
            return "comb", None         # Stage 5 run 10 reversed view; checked field by field in check_test_allowed
        return "enh2", None             # checked field by field in resolve_cptp and check_test_allowed
    if getattr(args, "cp", None):
        return "cp_alone", None         # checked field by field in resolve_cp and check_test_allowed
    if args.view == "rev" and args.plan == "vanilla":
        return "reverseeol", None   # checked field by field in check_test_allowed
    if args.plan == "tp" and args.view in ("rev", "fwd_revprompt", "shuf"):
        return "enh1", None         # Stage 3 (run 5, controls 6b/6a); checked field by field in check_test_allowed
    if args.plan == "vanilla":
        return "vanilla", {"plan": "vanilla"}
    if (args.tp_start, args.tp_exit) == (1, 7):
        return "tp_reproduction", {"tp_start": 1, "tp_exit": 7, "output_layer": 27}
    if (args.tp_start, args.tp_exit) == (1, 8):
        return "tp_reproduction_e8", {"tp_start": 1, "tp_exit": 8, "output_layer": 27}
    return None, None


def check_test_allowed(args, log):
    """Refuse a test split unless settings are frozen and every required gate passed. Prints what is missing."""
    missing = []
    gates = json.load(open(ROOT / "gates.json")) if (ROOT / "gates.json").exists() else {}
    for g in GATES_FOR_TEST:
        if not gates.get(g, {}).get("passed"):
            missing.append(f"gate {g} not PASSED")
    fs = json.load(open(ROOT / "frozen_settings.json")) if (ROOT / "frozen_settings.json").exists() else {}
    key, want = frozen_key(args)
    if key == "reverseeol":
        for gt in ["S1.10", "S2.1", "S2.2", "S2.3"]:
            if not gates.get(gt, {}).get("passed"):
                missing.append(f"gate {gt} not PASSED")
        st = fs.get(key, {}).get("settings")
        if not st:
            missing.append("frozen_settings.json has no 'reverseeol' (S2.3 freezes it)")
        elif (st["plan"], st["punct"], [str(st["output_layer"])]) != (args.plan, args.punct, args.layers.split(",")):
            missing.append(f"--plan/--punct/--layers do not match frozen reverseeol {st}")
        if args.batched:
            missing.append("reverseeol test pass runs at batch 1 (vanilla)")
    elif key == "enh1":
        for gt in ["S2.5", "S3.1", "S3.2", "S3.3"]:
            if not gates.get(gt, {}).get("passed"):
                missing.append(f"gate {gt} not PASSED")
        ent = fs.get(key)
        st = (ent or {}).get("settings")
        if not st:
            missing.append("frozen_settings.json has no 'enh1' (S3.3 freezes it)")
        else:
            logged = gates.get("S3.3", {}).get("numbers", {}).get("enh1_entry_sha256")
            if hashlib.sha256(json.dumps(ent, sort_keys=True).encode()).hexdigest() != logged:
                missing.append("frozen 'enh1' entry changed since S3.3 (sha256 differs from the logged one)")
            if (st["tp_start"], st["tp_exit"], [str(st["output_layer"])], st["batched"]) != \
                    (args.tp_start, args.tp_exit, args.layers.split(","), args.batched):
                missing.append(f"--tp-start/--tp-exit/--layers/--batched do not match frozen enh1 {st}")
            if args.view == "rev" and args.punct != st["punct"]:
                missing.append(f"--punct {args.punct} != frozen enh1 punct {st['punct']}")
    elif key == "cp_alone":
        for gt in ["S2.6", "S2.7", "S2.8"]:
            if not gates.get(gt, {}).get("passed"):
                missing.append(f"gate {gt} not PASSED")
        if args.cp["source"] != "frozen cp_alone":
            missing.append("a CP test pass needs --cp-from-frozen")
        elif args.cp["entry_sha"] != gates.get("S2.8", {}).get("numbers", {}).get("cp_alone_entry_sha256"):
            missing.append("frozen 'cp_alone' entry changed since S2.8 (sha256 differs from the logged one)")
    elif key == "enh2":
        for gt in ["S4.2", "S4.5"]:
            if not gates.get(gt, {}).get("passed"):
                missing.append(f"gate {gt} not PASSED")
        if args.cptp["source"] != "frozen enh2":
            missing.append("a CP + TP test pass needs --cptp-from-frozen")
        elif args.cptp["entry_sha"] != gates.get("S4.5", {}).get("numbers", {}).get("enh2_entry_sha256"):
            missing.append("frozen 'enh2' entry changed since S4.5 (sha256 differs from the logged one)")
    elif key == "comb":
        for gt in ["S4.2", "S4.5", "S4.8", "S5.1"]:
            if not gates.get(gt, {}).get("passed"):
                missing.append(f"gate {gt} not PASSED")
        c = args.cptp
        if c["source"] != "frozen enh2":
            missing.append("the run 10 reversed pass needs --cptp-from-frozen")
        elif c["entry_sha"] != gates.get("S4.5", {}).get("numbers", {}).get("enh2_entry_sha256"):
            missing.append("frozen 'enh2' entry changed since S4.5 (sha256 differs from the logged one)")
        e1 = fs.get("enh1")
        if not e1 or cp_entry_sha(e1) != gates.get("S3.3", {}).get("numbers", {}).get("enh1_entry_sha256"):
            missing.append("frozen 'enh1' missing or changed since S3.3 (sha256 differs from the logged one)")
        cb = fs.get("comb")
        if not cb:
            missing.append("frozen_settings.json has no 'comb' (S5.1 freezes it)")
        else:
            if cp_entry_sha(cb) != gates.get("S5.1", {}).get("numbers", {}).get("comb_entry_sha256"):
                missing.append("frozen 'comb' entry changed since S5.1 (sha256 differs from the logged one)")
            st = cb["settings"]
            got = (c["act_layer"], c["alpha"], c["aux_tp"], args.punct, args.tp_start, args.tp_exit, args.batched, args.layers)
            want_c = (st["act_layer"], float(st["alpha"]), st["aux_tp"], st["punct"], st["tp_start"], st["tp_exit"], st["batched"], str(st["output_layer"]))
            if got != want_c:
                missing.append(f"arguments {got} != frozen comb {want_c}")
            if e1 and cb.get("enh1_entry_sha256") != cp_entry_sha(e1):
                missing.append("frozen 'comb' was built from a different 'enh1' entry")
            if cb.get("enh2_entry_sha256") != c["entry_sha"]:
                missing.append("frozen 'comb' was built from a different 'enh2' entry")
    elif key is None:
        missing.append(f"no frozen setting covers plan={args.plan} tp_start={args.tp_start} tp_exit={args.tp_exit} view={args.view}")
    elif fs.get(key, {}).get("settings") != want:
        missing.append(f"frozen_settings.json['{key}']['settings'] != {want} (found {fs.get(key, {}).get('settings')})")
    if args.view not in ("fwd", "rev") and key != "enh1":   # (comb is view rev only: resolve_cptp)
        missing.append(f"view {args.view} has no frozen test setting")
    if args.limit is not None:
        missing.append("--limit is for dev dry runs only")
    if missing:
        log("REFUSED: test split not allowed:"); [log("  - " + m) for m in missing]
        sys.exit(2)
    if key == "reverseeol":   # want is None here; report the frozen entry and the fields actually compared
        st = fs[key]["settings"]
        log(f"test split allowed: gates {GATES_FOR_TEST[0]}..{GATES_FOR_TEST[-1]} and S2.1..S2.3 passed; frozen 'reverseeol' = "
            f"plan {st['plan']}, punct {st['punct']}, layer {st['output_layer']}, alpha {st['alpha']}, normalize {st['normalize']} "
            f"(this pass checked plan, punct, layer and batch 1 against it; alpha and normalize are applied by the S2.4 combine step)")
    elif key == "cp_alone":
        c = args.cp
        log(f"test split allowed: gates {GATES_FOR_TEST[0]}..{GATES_FOR_TEST[-1]} and S2.6..S2.8 passed; frozen 'cp_alone' = act_layer "
            f"{c['act_layer']} (paper l {c['act_layer'] + 1}), alpha {c['alpha']:g}, scaled, layer 27, vanilla, batch 1; entry sha256 matches S2.8")
    elif key == "comb":
        c = args.cptp
        log(f"test split allowed: gates {GATES_FOR_TEST[0]}..{GATES_FOR_TEST[-1]}, S4.2, S4.5, S4.8 and S5.1 passed")
        log(f"  run 10 reversed view: punct {args.punct} (enh1); index {c['act_layer']}, alpha {c['alpha']:g}, aux TP on s_rev (enh2)")
        log("  enh1, enh2, comb entry sha256 = S3.3, S4.5, S5.1; TP s 1 e 7, layer 27, batch 1")
    elif key == "enh2":
        c = args.cptp
        log(f"test split allowed: gates {GATES_FOR_TEST[0]}..{GATES_FOR_TEST[-1]}, S4.2 and S4.5 passed")
        log(f"  frozen 'enh2' = index {c['act_layer']}, alpha {c['alpha']:g}, aux {'with' if c['aux_tp'] else 'without'} TP, scaled, "
            f"TP s {args.tp_start} e {args.tp_exit}, layer 27, batch 1")
        log(f"  entry sha256 matches S4.5{'; CONTROL 9: sentiment auxiliary prompt' if c['control'] else ''}")
    elif key == "enh1":
        st = fs[key]["settings"]
        log(f"test split allowed: gates {GATES_FOR_TEST[0]}..{GATES_FOR_TEST[-1]} and S2.5, S3.1..S3.3 passed; frozen 'enh1' = "
            f"tp s {st['tp_start']} e {st['tp_exit']}, layer {st['output_layer']}, batched {st['batched']}, punct {st['punct']}, "
            f"alpha {st['alpha']}, normalize {st['normalize']}; entry sha256 matches S3.3 (this pass checked s, e, layer, batching"
            f"{' and punct' if args.view == 'rev' else ''}; view {args.view}; alpha and normalize are applied by the S3.4 combine step)")
    else:
        log(f"test split allowed: gates {GATES_FOR_TEST[0]}..{GATES_FOR_TEST[-1]} passed; frozen setting '{key}' = {want}")


def main():
    args = parse()
    log = Log(args.run)
    check_env()
    args.cp = resolve_cp(args, log)
    args.cptp = resolve_cptp(args, log)
    layers = [L if L == "final" else int(L) for L in args.layers.split(",")]
    stop_after = None if "final" in layers else max(layers)
    if args.plan == "vanilla" and args.batched:
        log("REFUSED: vanilla runs at batch 1 (batching was not accepted for vanilla in S0.13)"); sys.exit(2)
    if args.split == "test":
        check_test_allowed(args, log)
    elif args.limit is None and args.run.startswith(("r1", "r2", "r3")):
        log("note: dev split")
    tasks = TEST_TASKS if args.split == "test" else [DEV_TASK]
    tables = [SentenceTable(t) for t in tasks]
    words = [w for tb in tables for w in tb.words]
    if args.limit:
        assert args.split == "dev", "--limit is for dev only"
        words = words[:args.limit]
    items = make_items(words, [args.view], args.plan, args.punct)
    n, n_chunks = len(items), -(-len(items) // args.chunk)

    from backend_mlx import MLXBackend
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)   # 1 GB buffer cache: varied batch shapes otherwise grow it until the 16 GB Mac swaps
    pst = np.load(ROOT / "ref/pst_row_3407.npy")
    be = MLXBackend(ROOT / args.model, ROOT / "models/llama2_tok", pst)
    enc = partial(be.encode_batched if args.batched else be.encode, plan=args.plan, tp_start=args.tp_start,
                  tp_exit=args.tp_exit, stop_after=stop_after)
    enc_layers = layers
    if args.cp:   # CP: auxiliary pass and steered main pass per sentence, in the same loop; B saved as LB
        import cp_mlx
        from views import to_str_words
        cpw = cp_mlx.install(be.model, args.cp["act_layer"])
        aux_of = {it[2]: cp_mlx.cp_aux_prompt(to_str_words(w), args.cp["aux"]) for it, w in zip(items, words)}

        def enc(prompts, _layers):
            Bs = [cp_mlx.capture_B(be.model, be.tok, cpw, args.cp["act_layer"], aux_of[q]) for q in prompts]
            st = [cp_mlx.steered_states(be.model, be.tok, cpw, q, B, args.cp["alpha"], layers, stop_after) for q, B in zip(prompts, Bs)]
            return {**{L: np.stack([x[L] for x in st]) for L in layers}, "B": np.stack(Bs)}
        enc_layers = layers + ["B"]
    if args.cptp:   # CP + TP (Stage 4): auxiliary pass (with or without TP, D1) and steered TP main pass per sentence; B saved as LB
        import cp_mlx
        from views import to_str_words
        ct = args.cptp
        cpw = cp_mlx.install(be.model, ct["act_layer"])
        from views import reverse_words
        aux_words = (lambda w: reverse_words(to_str_words(w), args.punct)) if args.view == "rev" else to_str_words   # S5: aux gets s_rev
        aux_of = {it[2]: cp_mlx.cp_aux_prompt(aux_words(w), ct["aux_template"]) for it, w in zip(items, words)}
        aux_plan = "tp" if ct["aux_tp"] else "vanilla"

        def enc(prompts, _layers):
            Bs = [cp_mlx.capture_B(be.model, be.tok, cpw, ct["act_layer"], aux_of[q], aux_plan, args.tp_start, args.tp_exit, be.pst_vec)
                  for q in prompts]
            st = [cp_mlx.steered_states(be.model, be.tok, cpw, q, B, ct["alpha"], layers, stop_after, "tp", args.tp_start, args.tp_exit,
                                        be.pst_vec) for q, B in zip(prompts, Bs)]
            return {**{L: np.stack([x[L] for x in st]) for L in layers}, "B": np.stack(Bs)}
        enc_layers = layers + ["B"]
    sub = "dryrun" if args.limit else args.split
    cdir = ROOT / "cache" / sub / args.run
    meta = dict(model=args.model, plan=args.plan, tp_start=args.tp_start, tp_exit=args.tp_exit, stop_after=stop_after,
                view=args.view, split=args.split, tasks=tasks, batched=args.batched, limit=args.limit,
                pst="ref/pst_row_3407.npy" if args.plan == "tp" else None)
    if args.view != "fwd":
        meta["punct"] = args.punct   # only for non-fwd views, so existing fwd caches keep their meta
    if args.cp:                      # only for CP runs, so existing caches keep their meta
        meta["cp"] = dict(act_layer=args.cp["act_layer"], alpha=args.cp["alpha"], strategy="scaled",
                          aux_template_sha=hashlib.sha256(args.cp["aux"].encode()).hexdigest()[:12], source=args.cp["source"])
    if args.cptp:                    # only for CP + TP runs
        ct = args.cptp
        meta["cptp"] = dict(act_layer=ct["act_layer"], alpha=ct["alpha"], aux_tp=ct["aux_tp"], control=ct["control"], strategy="scaled",
                            aux_template_sha=hashlib.sha256(ct["aux_template"].encode()).hexdigest()[:12], source=ct["source"])
    log(f"start {args.run}: {n} prompts, {n_chunks} chunks of {args.chunk}, layers {layers}, stop_after {stop_after}")
    log(f"  model {args.model}, batched {args.batched}, free {free_gb():.1f} GB")
    state = dict(done=0, comp_secs=0.0, comp_n=0, comp_chunks=0)

    def on_chunk(start, k, computed, secs):
        mx.clear_cache()
        state["done"] += 1
        if computed:
            state["comp_secs"] += secs; state["comp_n"] += k; state["comp_chunks"] += 1
        sps = state["comp_secs"] / state["comp_n"] if state["comp_n"] else float("nan")
        left = (n - (start + k)) * sps / 60 if state["comp_n"] else float("nan")
        log(f"{args.run} {state['done']}/{n_chunks} {'computed' if computed else 'reused'} | {sps:.3f} s/sent | "
            f"~{left:.0f} min left | free {free_gb():.0f} GB | mlx {mx.get_active_memory()/2**30:.1f}/{mx.get_peak_memory()/2**30:.1f} GB")

    t0, t_start = time.perf_counter(), datetime.datetime.now()
    try:
        E = run_cached(items, enc, str(cdir), enc_layers, chunk=args.chunk, meta=meta, min_free_gb=FLOOR_GB, on_chunk=on_chunk)
    except LowDiskError as e:
        log(f"STOPPED (disk): {e}"); sys.exit(3)
    wall = time.perf_counter() - t0
    sps = state["comp_secs"] / state["comp_n"] if state["comp_n"] else None

    if args.limit:   # dry run: no scoring, keep the states for the comparison
        np.savez(ROOT / "logs" / f"{args.run}_states.npz", **{f"L{L}": E[L] for L in enc_layers})
        log(f"dry run done: {n} prompts in {wall:.1f} s ({sps:.3f} s/sentence computed)" if sps else f"dry run done: {n} prompts (all reused)")
        banner(log, args, cdir, t_start, wall, sps, state, n_chunks); return
    if args.split == "test" and args.view != "fwd":   # two lines so the screen stays within 100 columns (output only)
        log(f"done {args.run}: wall {wall/60:.1f} min" + (f", {sps:.3f} s/sentence computed" if sps else ""))
        log(f"states cached in {cdir.relative_to(ROOT)}; not scored here (scored by its own step)")
        log(f"free disk {free_gb():.1f} GB")
        banner(log, args, cdir, t_start, wall, sps, state, n_chunks); return

    scores, pairs, off = {}, {}, 0
    for tb in tables:
        m = len(tb.sentences)
        for L in layers:
            s, ps = score(tb, E[L][off:off + m])
            scores.setdefault(str(L), {})[SHORT[tb.name]] = s
            pairs[f"{SHORT[tb.name]}|L{L}"] = np.concatenate([ps[k][0] for k in ps])
        pairs[f"{SHORT[tb.name]}|gold"] = np.concatenate([g for _, (_, _, g) in tb.pairs.items()])
        off += m
    for L in scores:
        if args.split == "test":
            scores[L]["Avg"] = float(np.mean([scores[L][SHORT[t]] for t in TEST_TASKS]))
    rdir = ROOT / "results" / args.split; rdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rdir / f"pairs_{args.run}.npz", **pairs)
    (rdir / f"scores_{args.run}.json").write_text(json.dumps(dict(run=args.run, scores=scores, meta=meta, seconds_wall=wall,
                                                                  s_per_sentence_computed=sps), indent=1))
    append_result(dict(run=args.run, stage=("S2.9" if args.cp else "S4.6" if args.cptp else "S1.6/S1.7") if args.split == "test" else
                       ("S5.2-dev" if args.cptp and args.view == "rev" else "dev"), split=args.split, model=args.model,
                       plan=args.plan, tp_start=args.tp_start, tp_exit=args.tp_exit, view=args.view, layers=[str(L) for L in layers],
                       batched=args.batched, scores=scores, seconds=wall, cache=str(cdir.relative_to(ROOT)), cp=meta.get("cp"), cptp=meta.get("cptp"),
                       pair_cosines=str((rdir / f"pairs_{args.run}.npz").relative_to(ROOT))))
    cols = [SHORT[t] for t in tasks] + (["Avg"] if args.split == "test" else [])
    log(f"done {args.run}: wall {wall/60:.1f} min" + (f", {sps:.3f} s/sentence computed" if sps else ""))
    log("| layer | " + " | ".join(cols) + " |")
    for L in scores:
        log(f"| {L} | " + " | ".join(f"{scores[L][c]:.2f}" for c in cols) + " |")
    log(f"free disk {free_gb():.1f} GB")
    banner(log, args, cdir, t_start, wall, sps, state, n_chunks)


if __name__ == "__main__":
    main()
