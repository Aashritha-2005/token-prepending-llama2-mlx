import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path.home() / "Downloads" / "tp_project"
VENV = ROOT / "env" / "mlx"
GATES = ROOT / "gates.json"
GATE_LOG = ROOT / "gate_log.md"
RESULTS = ROOT / "results" / "results.jsonl"
FROZEN = ROOT / "frozen_settings.json"
LOGS = ROOT / "logs"
SCRATCH = os.environ.get("TP_SCRATCH_STATE") 
if SCRATCH:
    GATES, GATE_LOG, RESULTS, FROZEN, LOGS = (Path(SCRATCH) / "gates.json", Path(SCRATCH) / "gate_log.md",
                                              Path(SCRATCH) / "results.jsonl", Path(SCRATCH) / "frozen_settings.json", Path(SCRATCH) / "logs")
    print(f"SCRATCH STATE: gates, gate log, results, frozen settings and logs in {SCRATCH}", file=sys.stderr)


def check_env():
    """Fail loudly unless we run on Python 3.11/3.12 from the project's env/mlx venv on arm64."""
    import platform
    assert sys.version_info[:2] in [(3, 11), (3, 12)], f"Wrong Python {sys.version}; use the tp-mlx kernel"
    assert Path(sys.prefix).resolve() == VENV.resolve(), f"sys.prefix={sys.prefix} is not {VENV}; use the tp-mlx kernel"
    assert str(Path(sys.executable)).startswith(str(VENV)), f"sys.executable={sys.executable} is outside {VENV}"
    assert platform.machine() == "arm64", f"platform.machine()={platform.machine()}, need arm64"
    return dict(python=sys.version.split()[0], executable=sys.executable, machine=platform.machine())


def _load():
    return json.load(open(GATES)) if GATES.exists() else {}


def log_gate(task, passed, numbers, evidence, note=""):
    g = _load()
    g[task] = dict(passed=bool(passed), numbers=numbers, evidence=evidence, note=note,
                   date=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(g, open(GATES, "w"), indent=1, default=str)
    with open(GATE_LOG, "a") as f:
        f.write(f"\n### {task}  {g[task]['date']}  {'PASS' if passed else 'FAIL'}\n"
                f"numbers: {json.dumps(numbers, default=str)}\nevidence: {evidence}\n{note}\n")
    print(f"logged {task}: {'PASS' if passed else 'FAIL'}")


def require_gates(tasks):
    g = _load()
    missing = [t for t in tasks if not g.get(t, {}).get("passed")]
    if missing:
        raise RuntimeError(f"STOP: gates not passed yet: {missing}. Finish them before running this notebook.")
    print("gates OK:", tasks)


def append_result(record):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record, date=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def save_text(relpath, text):
    """Save raw output under the project (logs/ or verified_checks/outputs/)."""
    p = ROOT / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return str(p)


def free_gb():
    """Free space on the startup disk (home volume) in GB."""
    import shutil
    return shutil.disk_usage(Path.home()).free / 1e9
