"""One-command runner for the rest of Stage 5 (USER-RUN): bash scripts/run_stage5.sh

Runs, in order, each step only if it has not already passed / been recorded (resume logic):
  0 preflight   1 S5.1 freeze   2 S5.2 dev gate   3 dry run + comparison   4 run 10 test pass (caffeinate, foreground, resumable)
  5 PAUSE L30   6 score-s5.3 (logs S5.3) + PAUSE L31   7 bootstrap C4 --record + PAUSE L32   8 gate-s5.4 (logs S5.4) + PAUSE L36
  9 summary
Rules: stops at the first failure, REFUSED, non-PASS gate or non-zero exit and says what to send; never retries on its own; never
runs a step that logs a gate or appends to results.jsonl twice (a gate already logged as FAIL stops the runner). If a step already
logged but its screenshot was not filed (crash between step and pause), it reprints the screen with live_step.py --show (read-only,
labelled REPRINT) and says so. Everything shown is also appended to logs/stage5_run.log (screen output unchanged).
Rules and settings: notes/s5_preregistration.md (hashed). This runner changes none of them; it only calls the scripts.

Test options (scratch only; refused unless TP_SCRATCH_STATE is set): --dry-chain --root DIR [--force-fail 3|6] [--crash-after 6|7|8].
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REAL_ROOT = Path.home() / "Downloads" / "tp_project"
CLEAR = "\033[2J\033[3J\033[H"
N_TEST, N_CHUNKS = 28530, 58
R10 = ("--run r10_cptp_rev --plan tp --tp-start 1 --tp-exit 7 --view rev --punct keep_final --split test --layers 27 "
       "--cptp-from-frozen").split()
DRY = ("--run dry_r10u --plan tp --tp-start 1 --tp-exit 7 --view rev --punct keep_final --split dev --layers 27 "
       "--cptp-from-frozen --limit 50").split()
SHOTS = {"L30": "L30_r10_cptp_rev_done", "L31": "L31_r10_cptp_rev_scored", "L32": "L32_C4_comb_bootstrap_gate",
         "L36": "L36_S5.4_comb_reading_gate"}


class Runner:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root) if a.root else REAL_ROOT
        sys.path.insert(0, str(self.root / "src"))
        import runlog
        self.rl = runlog
        self.ev = Path(os.environ["EVIDENCE_DEST"]) if a.dry_chain else self.root / "results/figures/evidence"
        import io
        self.log = io.StringIO() if a.plan_only else open(self.root / "logs/stage5_run.log", "a")   # --plan-only writes nothing
        self.py = sys.executable

    # ------------------------------------------------------------------ output
    def out(self, msg="", clear=False):
        if clear:
            sys.stdout.write(CLEAR); sys.stdout.flush()
            self.log.write("\n[clear screen]\n"); self.log.flush()
            if not msg:
                return
        print(msg, flush=True)
        self.log.write(msg + "\n"); self.log.flush()

    def run(self, cmd, label):
        """Run cmd in the project root; echo every line to the screen and the run log. -> (exit code, lines)."""
        self.log.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] $ {' '.join(cmd)}\n"); self.log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        lines = []
        try:
            pr = subprocess.Popen(cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in pr.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                self.log.write(line); self.log.flush()
                lines.append(line.rstrip("\n"))
            rc = pr.wait()
        except KeyboardInterrupt:
            self.out("")
            self.out(f"INTERRUPTED during: {label}. Nothing is retried. Run  bash scripts/run_stage5.sh  again to resume"
                     + (" (finished chunks are reused)." if "test pass" in label else "."))
            sys.exit(130)
        return rc, lines

    def stop(self, what, send):
        self.out("")
        self.out("=" * 100)
        self.out(f"STOPPED: {what}"[:100])
        self.out(f"Send me: {send}"[:100])
        self.out("Nothing is retried automatically. Change nothing; we decide the next step together.")
        self.out("=" * 100)
        sys.exit(1)

    # ------------------------------------------------------------------ state
    def gates(self):
        return json.load(open(self.rl.GATES)) if self.rl.GATES.exists() else {}

    def gate(self, gid):
        """None = not logged; True / False = logged PASS / FAIL."""
        g = self.gates().get(gid)
        return None if g is None else bool(g.get("passed"))

    def recorded(self, run):
        return any(json.loads(l).get("run") == run for l in open(self.rl.RESULTS))

    def test_pass_done(self):
        c = self.root / "cache/test/r10_cptp_rev"
        lg = self.root / "logs/r10_cptp_rev.log"
        return (len(list(c.glob("chunk_*.npz"))) == N_CHUNKS and (c / "meta.json").exists() and lg.exists()
                and "END OF PASS  r10_cptp_rev" in lg.read_text())

    def filed(self, name):
        m = self.ev / "MANIFEST.md"
        return m.exists() and f"| {name} |" in m.read_text()

    def skipped(self, name):
        m = self.ev / "MANIFEST_NOTES.md"
        return m.exists() and f"{name} was skipped at the runner pause" in m.read_text()

    def shot_pending(self, key):
        return not (self.filed(SHOTS[key]) or self.skipped(SHOTS[key]))

    def free_gb(self):
        return shutil.disk_usage(Path.home()).free / 1e9

    def df(self):
        self.out("df -h ~: " + " ".join(subprocess.run(["df", "-h", str(Path.home())], capture_output=True, text=True).stdout.splitlines()[-1].split()))

    def esha(self, entry):
        return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------ steps
    def preflight(self):
        self.out("#" * 100)
        self.out(f"##  STAGE 5 RUNNER{'  (DRY CHAIN, scratch state, stand-in data)' if self.a.dry_chain else ''}".ljust(98) + "##")
        self.out("#" * 100)
        self.out(f"{datetime.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}  |  {self.root}  |  logs/stage5_run.log"[:100])
        me = {os.getpid(), os.getppid()}
        ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
        busy = [l.strip()[:90] for l in ps if int(l.split(None, 1)[0]) not in me and "python" in l
                and re.search(r"(run_pass|s4_grid|live_step)\.py", l)]
        g, fs = self.gates(), json.load(open(self.rl.FROZEN))
        dec = (self.root / "notes/decisions.md").read_text()
        h1 = self.esha(fs["enh1"]) if "enh1" in fs else "-"
        h2 = self.esha(fs["enh2"]) if "enh2" in fs else "-"
        in_dec = lambda h: h in dec or f"{h[:8]}\u2026{h[-4:]}" in dec        # decisions.md has the full value or e.g. 1041899f…e76e
        rows = [("no run_pass / s4_grid / live_step running", "none" if not busy else f"{len(busy)} running", not busy),
                ("free disk >= 5 GB", f"{self.free_gb():.1f} GB", self.free_gb() >= 5.0),
                ("S3.8 and S4.8 passed", f"{self.gate('S3.8')}/{self.gate('S4.8')}", bool(self.gate("S3.8") and self.gate("S4.8"))),
                ("enh1 hash = S3.3 log = decisions.md", h1[:12], h1 == g.get("S3.3", {}).get("numbers", {}).get("enh1_entry_sha256") and in_dec(h1)),
                ("enh2 hash = S4.5 log = decisions.md", h2[:12], h2 == g.get("S4.5", {}).get("numbers", {}).get("enh2_entry_sha256") and in_dec(h2))]
        self.out("PREFLIGHT")
        for n, v, ok in rows:
            self.out(f"  [{'x' if ok else ' '}] {n:42s} {v:>16s}")
        for b in busy:
            self.out(f"      running: {b}")
        if not all(r[2] for r in rows):
            self.stop("preflight failed (see the unticked row)", "this screen")
        sps = self.dev_sps()
        nch = len(list((self.root / "cache/test/r10_cptp_rev").glob("chunk_*.npz")))
        left = max(N_CHUNKS - nch, 0) / N_CHUNKS * N_TEST
        st = lambda done, todo="to run": "done, skip" if done else todo
        plan = [("1", "S5.1 freeze comb (logs S5.1)", "~5 s", st(self.gate("S5.1") is not None, "to run")),
                ("2", "S5.2 dev sanity gate (logs S5.2)", "~10 s", st(self.gate("S5.2") is not None)),
                ("3", "dry run (50 dev) + comparison", "~2 min", st(self.test_pass_done() or self.gate("S5.3") is not None)),
                ("4", f"run 10 test pass r10_cptp_rev ({nch}/{N_CHUNKS} chunks saved)",
                 f"~{left * 0.162 / 60:.0f}-{left * sps / 60:.0f} min", st(self.test_pass_done())),
                ("5", "PAUSE L30 screenshot (tail -17 of the pass log)", "you", st(not self.shot_pending("L30"))),
                ("6", "score run 10 (logs S5.3) + PAUSE L31", "~1 min", st(self.gate("S5.3") is not None and not self.shot_pending("L31"))),
                ("7", "C4 bootstrap --record + PAUSE L32", "~1 min", st(self.recorded("S5.3-C4-run10_vs_run5") and not self.shot_pending("L32"))),
                ("8", "S5.4 reading: E9-E11, cost, checklist (logs S5.4) + PAUSE L36", "~3 min",
                 st(self.gate("S5.4") is not None and not self.shot_pending("L36"))),
                ("9", "summary", "", "")]
        self.out("PLAN (each step runs only if not already passed or recorded)")
        for i, n, t, s in plan:
            self.out(f"  {i}. {n:62s} {t:>11s}  {s}")
        self.out(f"Total about 1 h 30 min to 1 h 45 min, nearly all of it step 4. Ctrl-C stops; the same command resumes.")
        self.df()

    def dev_sps(self):
        """s/sentence of the S5.2 dev pass ("done ...: wall 9.2 min, 0.190 s/sentence computed"; not the banner's HH:MM:SS line)."""
        q = self.root / "logs/cptp_i4_a4_tp_rev_keep_final_L27.log"
        m = re.findall(r"[ ,(](\d+\.\d+) s/sentence computed", q.read_text()) if q.exists() else []
        return float(m[-1]) if m else 0.19

    def gate_step(self, gid, cmd, label):
        state = self.gate(gid)
        if state is True:
            self.out(f"[{gid}] already PASSED at {self.gates()[gid]['date']}; skipped (never run twice).")
            return
        if state is False:
            self.stop(f"{gid} is already logged as FAIL ({self.gates()[gid]['date']}); the runner does not run it again",
                      f"the {gid} entry of gate_log.md")
        self.out("")
        self.out(f"[{gid}] {label}")
        rc, lines = self.run(cmd, label)
        if rc != 0 or not any(f"GATE {gid}: PASS" in l for l in lines) or self.gate(gid) is not True:
            self.stop(f"{label}: exit {rc}; 'GATE {gid}: PASS' {'seen' if any(f'GATE {gid}: PASS' in l for l in lines) else 'NOT seen'}; "
                      f"gate logged {self.gate(gid)}", "the screen above (also in logs/stage5_run.log)")

    def dry_step(self):
        if self.test_pass_done() or self.gate("S5.3") is not None:
            self.out("[dry] test pass already complete; dry run and comparison skipped.")
            return
        self.out("")
        self.out("[dry] run 10 dry run (50 dev sentences) and comparison with a direct recomputation")
        rc, lines = self.run([self.py, "scripts/run_pass.py"] + DRY, "dry run")
        if rc != 0 or any("REFUSED" in l for l in lines):
            self.stop(f"dry run: exit {rc}{'; REFUSED' if any('REFUSED' in l for l in lines) else ''}", "the screen above")
        vs = "dry_r10u" if self.a.force_fail == 3 else "dry_r8"       # test: comparing with itself must STOP (B identical)
        if self.a.force_fail == 3:
            self.out("[dry-chain] FORCED FAILURE at step 3: comparing the dry run with itself (reversed B cannot differ)")
        rc, lines = self.run([self.py, "scripts/compare_dryrun_cptp.py", "--run", "dry_r10u", "--view", "rev", "--vs-run", vs], "comparison")
        if rc != 0 or not any(l.startswith("COMPARISON: OK") for l in lines):
            self.stop(f"comparison: exit {rc}; 'COMPARISON: OK' not printed", "the screen above (logs/dry_r10u_compare.json)")

    def test_step(self):
        if self.test_pass_done():
            self.out("[test pass] r10_cptp_rev already complete (58 chunks, END OF PASS in its log); skipped.")
            return
        self.out("")
        self.out(f"[test pass] r10_cptp_rev, foreground, caffeinate; started {datetime.datetime.now():%H:%M:%S}. "
                 "Ctrl-C stops; rerun resumes.")
        self.df()
        if self.free_gb() < 5.0:
            self.stop(f"free disk {self.free_gb():.1f} GB < 5 GB before the test pass", "this screen")
        rc, lines = self.run(["caffeinate", "-dims", self.py, "scripts/run_pass.py"] + R10, "test pass r10_cptp_rev")
        self.df()
        if rc != 0 or any("REFUSED" in l or l.split(" ", 1)[-1].startswith("STOPPED") for l in lines) or not self.test_pass_done():
            self.stop(f"test pass: exit {rc}; complete {self.test_pass_done()}", "tail -30 logs/r10_cptp_rev.log")

    def pause(self, key, shown, lines_on_screen):
        name = SHOTS[key]
        if self.filed(name):
            self.out(f"[{key}] {name} already filed in MANIFEST.md; no pause.")
            return
        if self.skipped(name):
            self.out(f"[{key}] {name} was skipped earlier (MANIFEST_NOTES.md); no pause.")
            return
        width = max([len(l) for l in lines_on_screen] + [0])
        if len(lines_on_screen) + 3 > 40 or width > 100:
            self.out(f"WARNING: this screen is {len(lines_on_screen) + 3} rows x {width} cols (over 100x40)")
        self.out("Take the screenshot now (Cmd+Shift+4, Space, Option-click the window),")
        self.out(f"then press Enter within 10 minutes to file {key}  (s + Enter = skip this screenshot)")
        while True:
            try:
                ans = input()
            except EOFError:
                self.stop(f"no input at pause {key}", "nothing; run the same command again to resume at this pause")
            self.log.write(f"[input at {key}] {ans!r}\n")
            if ans.strip().lower() == "s":
                with open(self.ev / "MANIFEST_NOTES.md", "a") as f:
                    f.write(f"- Note ({datetime.datetime.now():%Y-%m-%d %H:%M:%S}, Stage 5 runner): {name} was skipped at the runner pause; "
                            f"no screenshot filed. The step's output is in logs/stage5_run.log.\n")
                self.out(f"[{key}] skipped; recorded in MANIFEST_NOTES.md.")
                return
            rc, out = self.run(["zsh", str(self.root / "scripts/save_evidence.sh"), name, shown], f"filing {name}")
            if rc == 0:
                self.out(f"[{key}] filed.")
                return
            why = next((l for l in out if "REFUSED" in l), f"exit {rc}")
            self.out(f"[{key}] filing refused: {why}"[:100])
            self.out("The step is NOT re-run. Take a new screenshot if needed, then Enter = file again, s = skip.")

    def live_step(self, key, mode_args, logged, label):
        """Run (or, if already logged and the screenshot is pending, reprint read-only) a screenshotted live_step screen."""
        name = SHOTS[key]
        cmd_args = ["--evidence", name] + mode_args
        if logged:
            if not self.shot_pending(key):
                self.out(f"[{key}] {label}: already logged and screenshot handled; skipped.")
                return
            self.out("")
            self.out(f"[{key}] {label} already logged, screenshot not filed: REPRINT with --show (read-only, writes nothing).")
            input_shown = f"python scripts/live_step.py --evidence {name} --show {' '.join(mode_args)}"
            self.out("Press Enter to clear the screen and show the reprint.")
            try:
                input()
            except EOFError:
                self.stop("no input", "nothing; run again to resume")
            self.out(clear=True)
            rc, lines = self.run([self.py, "scripts/live_step.py", "--show"] + cmd_args, f"reprint {label}")
            if rc != 0:
                self.stop(f"reprint of {label}: exit {rc}", "the screen above")
            self.pause(key, input_shown, lines)
            return
        shown = f"python scripts/live_step.py {' '.join(cmd_args)}"
        self.out(clear=True)
        rc, lines = self.run([self.py, "scripts/live_step.py"] + cmd_args, label)
        return rc, lines, shown

    def run_all(self):
        self.preflight()
        self.gate_step("S5.1", [self.py, "scripts/s5_1_freeze.py"], "S5.1 freeze of comb")
        self.gate_step("S5.2", [self.py, "scripts/s5_2_dev_gate.py"], "S5.2 dev sanity gate")
        self.dry_step()
        self.test_step()
        # 5. L30
        if self.shot_pending("L30"):
            self.out(clear=True)
            rc, lines = self.run(["tail", "-17", "logs/r10_cptp_rev.log"], "tail of the pass log")
            self.pause("L30", "tail -17 logs/r10_cptp_rev.log", lines)
        else:
            self.out("[L30] screenshot already handled; skipped.")
        # 6. S5.3
        st = self.gate("S5.3")
        if st is False:
            self.stop(f"S5.3 is already logged as FAIL ({self.gates()['S5.3']['date']}); not run again", "the S5.3 entry of gate_log.md")
        if self.a.force_fail == 6 and st is None:
            fz = json.load(open(self.rl.FROZEN)); fz["comb"]["settings"]["alpha"] = 2.0
            json.dump(fz, open(self.rl.FROZEN, "w"), indent=1)
            self.out("[dry-chain] FORCED FAILURE at step 6: scratch comb alpha changed to 2 (hash no longer = S5.1)")
        r = self.live_step("L31", ["score-s5.3"], st is True, "run 10 scored (S5.3)")
        if r:
            rc, lines, shown = r
            if rc != 0 or not any("GATE S5.3: PASS" in l for l in lines) or self.gate("S5.3") is not True:
                self.stop(f"score-s5.3: exit {rc}; GATE S5.3 PASS not seen or not logged (logged: {self.gate('S5.3')})",
                          "the screen above (also in logs/stage5_run.log)")
            if self.a.crash_after == 6:
                self.out("[dry-chain] simulated crash after step 6, before its pause"); sys.exit(9)
            self.pause("L31", shown, lines)
        # 7. C4
        rec = self.recorded("S5.3-C4-run10_vs_run5")
        r = self.live_step("L32", ["bootstrap", "C4", "--record"], rec, "C4 bootstrap (primary)")
        if r:
            rc, lines, shown = r
            if rc != 0 or not self.recorded("S5.3-C4-run10_vs_run5"):
                self.stop(f"bootstrap C4: exit {rc}; recorded {self.recorded('S5.3-C4-run10_vs_run5')}", "the screen above")
            if self.a.crash_after == 7:
                self.out("[dry-chain] simulated crash after step 7, before its pause"); sys.exit(9)
            self.pause("L32", shown, lines)
        # 8. S5.4
        st = self.gate("S5.4")
        if st is False:
            self.stop(f"S5.4 is already logged as FAIL ({self.gates()['S5.4']['date']}); not run again", "the S5.4 entry of gate_log.md")
        r = self.live_step("L36", ["gate-s5.4"], st is True, "S5.4 reading, cost, checklist")
        if r:
            rc, lines, shown = r
            if rc != 0 or not any("GATE S5.4: PASS" in l for l in lines) or self.gate("S5.4") is not True:
                self.stop(f"gate-s5.4: exit {rc}; GATE S5.4 PASS not seen or not logged (logged: {self.gate('S5.4')})",
                          "the screen above (also in logs/stage5_run.log)")
            if self.a.crash_after == 8:
                self.out("[dry-chain] simulated crash after step 8, before its pause"); sys.exit(9)
            self.pause("L36", shown, lines)
        self.summary()

    def summary(self):
        g = self.gates()
        self.out("")
        self.out("=" * 100)
        self.out("STAGE 5 SUMMARY")
        self.out("  gates: " + "  ".join(f"{k} {g.get(k, {}).get('passed', False)}" for k in ("S5.1", "S5.2", "S5.3", "S5.4")))
        sc = self.root / "results/test/scores_r10_comb.json"
        if sc.exists():
            s = json.load(open(sc))["scores"]["27"]
            self.out(f"  run 10 average {s['Avg']:.2f}")
        c4 = [json.loads(l) for l in open(self.rl.RESULTS) if json.loads(l).get("run") == "S5.3-C4-run10_vs_run5"]
        if c4:
            b = c4[-1]
            self.out(f"  C4 run 10 - run 5: {b['diff']:+.2f}  95% [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]  Bonferroni "
                     f"[{b['ci_bonferroni'][0]:+.2f}, {b['ci_bonferroni'][1]:+.2f}]  {b['n_pos_signs']}/7 positive -> {b.get('word')}")
        s54 = self.rl.LOGS / "S5.4.json"
        if s54.exists():
            self.out(f"  reading: {json.load(open(s54))['reading']}"[:100])
        rd = self.root / "notes/comb_reading.md" if not self.rl.SCRATCH else self.rl.LOGS / "comb_reading.md"
        files = [self.root / "cache/test/r10_cptp_rev", self.root / "logs/r10_cptp_rev.log", self.root / "results/test/scores_r10_comb.json",
                 self.root / "results/test/pairs_r10_comb.npz"] + [self.root / f"results/test/pairs_boot_{i}.npz" for i in ("C4", "E9", "E10", "E11")] \
            + [rd, s54] + [self.ev / f"{n}.png" for n in SHOTS.values()]
        self.out("  files written:")
        for f in files:
            try:
                rel = f.relative_to(self.root)
            except ValueError:
                rel = f
            self.out(f"    {'[x]' if f.exists() else '[ ]'} {rel}"[:100])
        self.df()
        self.out("Done. Send me the outputs (logs/stage5_run.log has everything shown here).")
        self.out("=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-only", action="store_true", help="read-only: print the preflight and the plan, then exit (runs nothing)")
    ap.add_argument("--dry-chain", action="store_true", help="test only: scratch state + stand-in data")
    ap.add_argument("--root", help="test only (with --dry-chain): a scratch project root with patched script copies")
    ap.add_argument("--force-fail", type=int, choices=[3, 6], help="test only: force a failure at step 3 or 6")
    ap.add_argument("--crash-after", type=int, choices=[6, 7, 8], help="test only: exit after that step's content, before its pause")
    a = ap.parse_args()
    scratch = os.environ.get("TP_SCRATCH_STATE")
    if a.dry_chain:
        if not (scratch and a.root and os.environ.get("EVIDENCE_DEST") and os.environ.get("SHOT_DIR")):
            ap.error("--dry-chain needs TP_SCRATCH_STATE, --root, EVIDENCE_DEST and SHOT_DIR (scratch only)")
        if Path(a.root).resolve() == REAL_ROOT.resolve():
            ap.error("--dry-chain must not use the real project root")
    else:
        if scratch or a.root or a.force_fail or a.crash_after or os.environ.get("EVIDENCE_DEST") or os.environ.get("SHOT_DIR"):
            ap.error("TP_SCRATCH_STATE / EVIDENCE_DEST / SHOT_DIR / --root / --force-fail / --crash-after are for --dry-chain tests only")
        if not a.plan_only:
            print("\033[8;40;100t", end="", flush=True)    # 40 rows x 100 columns
    r = Runner(a)
    if a.plan_only:
        r.preflight(); print("--plan-only: nothing was run or written."); return
    r.run_all()


if __name__ == "__main__":
    main()
