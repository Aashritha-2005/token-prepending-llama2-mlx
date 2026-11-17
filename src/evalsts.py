"""STS loading and scoring (plan S0.5).

Splits are loaded through SentEval's own loader classes (no re-parsing). Each split becomes a
SentenceTable of unique sentences; scoring maps pairs to rows of an embedding matrix aligned with
table.sentences, and reproduces the repo's aggregation:
  STS12-16: Spearman of all subsets pooled ('all' -> 'spearman' -> 'all')
  STS-B test, SICK-R test: Spearman of the single 'test' subset
  STS-B dev: Spearman of the single 'dev' subset
"""
import sys

import numpy as np
from scipy.stats import spearmanr

from runlog import ROOT

SENTEVAL = ROOT / "ref" / "token_prepending" / "SentEval"
TASK_PATH = SENTEVAL / "data"             # SentEval appends /downstream/...
TEST_TASKS = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
DEV_TASK = 'STSBenchmark-dev'


def _senteval():
    if str(SENTEVAL) not in sys.path:
        sys.path.insert(0, str(SENTEVAL))
    import senteval.sts as sts
    return sts


def load_task(name):
    """-> (evaluation object, dict subset -> (sent1 word lists, sent2 word lists, gold list))"""
    sts = _senteval()
    d = str(TASK_PATH / "downstream")
    if name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        ev = getattr(sts, name + 'Eval')(f"{d}/STS/{name}-en-test", seed=1111)
    elif name == 'STSBenchmark':
        ev = sts.STSBenchmarkEval(f"{d}/STS/STSBenchmark", seed=1111)
    elif name == 'STSBenchmark-dev':
        ev = sts.STSBenchmarkEvalDev(f"{d}/STS/STSBenchmark", seed=1111)
    elif name == 'SICKRelatedness':
        ev = sts.SICKRelatednessEval(f"{d}/SICK", seed=1111)
    else:
        raise ValueError(name)
    return ev, {k: ev.data[k] for k in ev.datasets}


def key_of(words):
    """The repo batcher only sees ' '.join(words), so that string identifies a sentence."""
    return ' '.join(w.decode('utf-8') if isinstance(w, bytes) else w for w in words)


class SentenceTable:
    def __init__(self, name):
        self.name = name
        _, self.subsets = load_task(name)
        self.sentences, self.words, self.index = [], [], {}
        self.pairs = {}   # subset -> (idx1 array, idx2 array, gold array)
        for sub, (s1, s2, gold) in self.subsets.items():
            i1 = [self._add(w) for w in s1]
            i2 = [self._add(w) for w in s2]
            self.pairs[sub] = (np.array(i1), np.array(i2), np.array(gold, dtype=float))

    def _add(self, words):
        k = key_of(words)
        if k not in self.index:
            self.index[k] = len(self.sentences)
            self.sentences.append(k)
            self.words.append([w.decode('utf-8') if isinstance(w, bytes) else w for w in words])
        return self.index[k]

    @property
    def n_pairs(self):
        return sum(len(g) for _, _, g in self.pairs.values())


def cosine(u, v):   # SentEval utils.cosine with SentEval's nan_to_num wrapping
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    return np.nan_to_num(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def pair_sims(table, E):
    """E: (len(table.sentences), d) -> dict subset -> (sims, gold)"""
    E = np.asarray(E, dtype=np.float64)
    out = {}
    for sub, (i1, i2, gold) in table.pairs.items():
        out[sub] = (np.array([cosine(E[a], E[b]) for a, b in zip(i1, i2)]), gold)
    return out


def score(table, E):
    """-> (Spearman x100 with the repo's aggregation, dict subset -> (sims, gold))"""
    ps = pair_sims(table, E)
    sims = np.concatenate([s for s, _ in ps.values()])
    gold = np.concatenate([g for _, g in ps.values()])
    return 100.0 * spearmanr(sims, gold).correlation, ps
