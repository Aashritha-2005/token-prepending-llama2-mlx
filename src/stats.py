"""Evaluation helpers and paired bootstrap (plan Appendix C, Section 8)."""
import numpy as np
from scipy.stats import spearmanr


def cosine_pairs(E1, E2):
    return (E1 * E2).sum(1) / (np.linalg.norm(E1, axis=1) * np.linalg.norm(E2, axis=1))


def spearman100(sims, gold):
    return 100.0 * spearmanr(sims, gold).correlation


def pooled_spearman100(sims_by_subset, gold_by_subset):   # STS12-16 'all'
    return spearman100(np.concatenate(list(sims_by_subset.values())),
                       np.concatenate([gold_by_subset[k] for k in sims_by_subset]))


def paired_bootstrap(simA, simB, gold, n_boot=2000, seed=0, q=(2.5, 97.5, 0.625, 99.375)):
    """simA, simB, gold: dict dataset -> arrays (pairs pooled per dataset). Returns dict."""
    rng = np.random.default_rng(seed); keys = list(gold)
    sp = lambda a, b: spearmanr(a, b).correlation
    pt = 100 * np.mean([sp(simA[k], gold[k]) - sp(simB[k], gold[k]) for k in keys])
    signs = {k: float(np.sign(sp(simA[k], gold[k]) - sp(simB[k], gold[k]))) for k in keys}
    d = np.empty(n_boot)
    for b in range(n_boot):
        acc = 0.0
        for k in keys:
            n = len(gold[k]); idx = rng.integers(0, n, n)
            acc += sp(simA[k][idx], gold[k][idx]) - sp(simB[k][idx], gold[k][idx])
        d[b] = 100 * acc / len(keys)
    lo, hi, alo, ahi = np.percentile(d, q)
    return dict(diff=pt, ci95=(lo, hi), ci_adj=(alo, ahi), n_pos_signs=int(sum(s > 0 for s in signs.values())),
                n_datasets=len(keys), excludes0=bool(lo > 0 or hi < 0), robust=bool(alo > 0 or ahi < 0))
