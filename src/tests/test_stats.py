"""S0.6 bootstrap tests (plan v3: equal-systems test uses fixed seeds 0-29 and needs >= 25 of 30)."""
import numpy as np

from stats import paired_bootstrap


def _synthetic(seed, noise_a, noise_b, n=400, k=3):
    rng = np.random.default_rng(seed)
    gold, A, B = {}, {}, {}
    for d in range(k):
        g = rng.uniform(0, 5, n)
        gold[f"d{d}"] = g
        A[f"d{d}"] = g + rng.normal(0, noise_a, n)
        B[f"d{d}"] = g + rng.normal(0, noise_b, n)
    return A, B, gold


def test_identical(n_boot=500):
    A, _, gold = _synthetic(123, 1.0, 1.0)
    r = paired_bootstrap(A, A, gold, n_boot=n_boot, seed=0)
    assert r["diff"] == 0 and r["ci95"] == (0.0, 0.0), r
    return {"diff": r["diff"], "ci95": r["ci95"]}


def test_better(n_boot=500):
    A, B, gold = _synthetic(7, 0.8, 2.0)
    r = paired_bootstrap(A, B, gold, n_boot=n_boot, seed=0)
    assert r["excludes0"] and r["ci95"][0] > 0 and r["n_pos_signs"] == r["n_datasets"], r
    return {"diff": round(r["diff"], 3), "ci95": [round(x, 3) for x in r["ci95"]], "n_pos_signs": r["n_pos_signs"]}


def test_equal_systems(n_trials=30, n_boot=1000, need=25):
    contains0 = 0
    for seed in range(n_trials):
        A, B, gold = _synthetic(1000 + seed, 1.5, 1.5)
        r = paired_bootstrap(A, B, gold, n_boot=n_boot, seed=seed)
        contains0 += int(not r["excludes0"])
    assert contains0 >= need, f"CI contained 0 in only {contains0}/{n_trials}"
    return {"contains0": contains0, "n_trials": n_trials, "need": need}
