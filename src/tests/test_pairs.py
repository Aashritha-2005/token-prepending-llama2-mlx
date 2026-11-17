"""S2.2 pairs-batcher integration tests (plan S2.2). Call the test_* functions directly (no pytest in the pinned env).

The dummy backend embeds each prompt string as a deterministic hash vector, so any mis-alignment between items and
embeddings changes the combined embeddings."""
import random
import tempfile
import zlib

import numpy as np

from cache import run_cached
from evalsts import SentenceTable, DEV_TASK
from views import make_items, gather, combine, to_str_words

VIEWS = ["fwd", "rev"]


def hash_vec(prompt, dim=64):
    return np.random.default_rng(zlib.crc32(prompt.encode("utf-8"))).standard_normal(dim).astype(np.float32)


def dummy_encode(prompts, layers):
    E = np.stack([hash_vec(p) for p in prompts])
    return {L: E for L in layers}


def _combined(items, n, cache_dir):
    E = run_cached(items, dummy_encode, cache_dir, [27], chunk=37, min_free_gb=0.0)[27]   # odd chunk: crosses view pairs
    Ev = gather(items, E, n, VIEWS)
    return {(a, nz): combine(Ev, VIEWS, a, nz) for a in (0.4, 0.5, 0.6) for nz in (True, False)}, Ev


def _dev_words(n=None):
    w = [to_str_words(x) for x in SentenceTable(DEV_TASK).words]
    return w if n is None else w[:n]


def test_shuffle_realign(n=100):
    words = _dev_words(n)
    out = {}
    for plan in ("vanilla", "tp"):
        for punct in ("full", "keep_final"):
            items = make_items(words, VIEWS, plan, punct)
            shuffled = list(items); random.Random(0).shuffle(shuffled)
            assert shuffled != items
            with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
                A, EvA = _combined(items, n, d1)
                B, EvB = _combined(shuffled, n, d2)
            for k in A:
                assert np.array_equal(A[k], B[k]), (plan, punct, k)
            # the re-aligned rows really are the embeddings of the right prompts
            for i in (0, n // 2, n - 1):
                for v in VIEWS:
                    p = [it[2] for it in items if it[0] == i and it[1] == v][0]
                    assert np.allclose(EvB[v][i], hash_vec(p).astype(np.float16).astype(np.float32))
            assert not np.array_equal(EvA["fwd"], EvA["rev"])
            out[f"{plan}/{punct}"] = "identical after shuffle + re-align (6 alpha x norm combinations)"
    return out


def test_revprompt_differs():
    """fwd_revprompt = forward text under the reversed wording; rev = reversed text; they must differ for every
    sentence with >= 2 words. Checked on all STS-B dev sentences, for both punctuation rules."""
    words = _dev_words()
    out = {}
    for punct in ("full", "keep_final"):
        rp = make_items(words, ["fwd_revprompt"], "vanilla", punct)
        rv = make_items(words, ["rev"], "vanilla", punct)
        fw = make_items(words, ["fwd"], "vanilla", punct)
        multi = [i for i, w in enumerate(words) if len(w) >= 2]
        same = [i for i in multi if rp[i][2] == rv[i][2]]
        assert all(rp[i][2].startswith("This reversed sentence : ") and rv[i][2].startswith("This reversed sentence : ") for i in multi)
        assert all(fw[i][2].replace("This sentence : ", "This reversed sentence : ", 1) == rp[i][2] for i in range(len(words)))
        out[punct] = dict(n_sentences=len(words), n_ge2_words=len(multi), n_identical=len(same),
                          identical_examples=[" ".join(words[i]) for i in same[:5]])
    assert out["full"]["n_identical"] == 0, out["full"]
    return out


def test_pst_counts():
    words = _dev_words()
    views = ["fwd", "rev", "fwd_revprompt", "shuf"]
    out = {}
    for plan, want in (("tp", 1), ("vanilla", 0)):
        for punct in ("full", "keep_final"):
            items = make_items(words, views, plan, punct)
            counts = {it[2].count("<PST>") for it in items}
            assert counts == {want}, (plan, punct, counts)
            out[f"{plan}/{punct}"] = f"<PST> count {want} in all {len(items)} prompts"
    return out


def test_item_count():
    words = _dev_words()
    out = {}
    for views in (["fwd"], ["rev"], VIEWS, ["fwd", "rev", "fwd_revprompt", "shuf"]):
        items = make_items(words, views, "vanilla")
        assert len(items) == len(words) * len(views)
        assert [(i, v) for i, v, _ in items] == [(i, v) for i in range(len(words)) for v in views]
        out["+".join(views)] = f"{len(items)} = {len(words)} x {len(views)}"
    return out
