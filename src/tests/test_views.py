"""S0.6 view tests. Each test returns a dict of numbers and raises AssertionError on failure."""
import random
import subprocess
import sys

from views import build_view, make_items, reverse_words, shuffle_words, gather

TP_PROMPT = 'This sentence : <PST> "*sent 0*" means in one word:"'
VAN_PROMPT = 'This sentence : "*sent 0*" means in one word:"'


def repo_batcher_strings(batch, plan):
    """Copied from evaluate.py batcher (bytes decode, join, 3 clean-up lines, prompt filling)."""
    if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        batch = [[word.decode('utf-8') for word in s] for s in batch]
    task_prompts = [TP_PROMPT] if plan == 'tp' else [VAN_PROMPT]
    sentences = [' '.join(s) for s in batch]
    new_sentences = []
    for i, s in enumerate(sentences):
        if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
        s = s.replace('"', '\'')
        if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
        for prompt in task_prompts:
            new_sentences.append(prompt.replace('*sent 0*', s).strip())
    return new_sentences


def test_repo_parity(dev_words, n=200, seed=0):
    sample = random.Random(seed).sample(dev_words, n)
    for plan in ['vanilla', 'tp']:
        ref = repo_batcher_strings(sample, plan)
        ours = [build_view(w, 'fwd', plan == 'tp') for w in sample]
        bad = [(a, b) for a, b in zip(ref, ours) if a != b]
        assert not bad, f"{plan}: {len(bad)} mismatches, e.g. {bad[:2]}"
    return {"n_sentences": n, "mismatches": 0}


def test_reverse():
    w = "A man is playing a guitar .".split()
    assert reverse_words(w, "full") == ". guitar a playing is man A".split()
    assert reverse_words(w, "keep_final") == "guitar a playing is man A .".split()
    q = "Is the cat on the mat ?".split()
    assert reverse_words(q, "full") == "? mat the on cat the Is".split()
    assert reverse_words(q, "keep_final") == "mat the on cat the Is ?".split()
    full = build_view(w, 'rev', True, "full"); keep = build_view(w, 'rev', True, "keep_final")
    assert full == 'This reversed sentence : <PST> ". guitar a playing is man A." means in one word:"', full
    assert keep == 'This reversed sentence : <PST> "guitar a playing is man A ." means in one word:"', keep
    return {"full": full, "keep_final": keep}


def test_shuffle_deterministic():
    w = "the quick brown fox jumps over the lazy dog".split()
    a, b = shuffle_words(w), shuffle_words(w)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from views import shuffle_words; "
            "print(' '.join(shuffle_words('the quick brown fox jumps over the lazy dog'.split())))")
    import views
    src = views.__file__.rsplit('/', 1)[0]
    other = subprocess.run([sys.executable, "-c", code, src], capture_output=True, text=True, check=True).stdout.strip()
    assert a == b and ' '.join(a) == other, (a, b, other)
    assert a != w
    return {"shuffled": ' '.join(a), "other_process_equal": True}


def test_pst_once(dev_words):
    views = ['fwd', 'rev', 'fwd_revprompt', 'shuf']
    for plan, expect in [('tp', 1), ('vanilla', 0)]:
        for _, v, p in make_items(dev_words, views, plan):
            assert p.count('<PST>') == expect, (plan, v, p)
    return {"n_prompts_checked": 2 * len(dev_words) * len(views)}


def test_edge_cases():
    cases = {
        "empty": ([], 'This sentence : <PST> "" means in one word:"'),
        "quote": ('He said "hi" .'.split(), "This sentence : <PST> \"He said 'hi' .\" means in one word:\""),
        "question": ("Is it ?".split(), 'This sentence : <PST> "Is it ." means in one word:"'),
        "apostrophe": ("the dogs '".split(), "This sentence : <PST> \"the dogs '\" means in one word:\""),
        "bytes": ([b"caf\xc3\xa9", b"open"], 'This sentence : <PST> "café open." means in one word:"'),
    }
    for name, (w, expect) in cases.items():
        got = build_view(w, 'fwd', True)
        assert got == expect, (name, got, expect)
        assert got == repo_batcher_strings([w], 'tp')[0] or name == "empty", name
    return {"cases": list(cases)}


def test_alignment(dev_words):
    import numpy as np
    views = ['fwd', 'rev']
    items = make_items(dev_words[:100], views, 'tp')
    emb = lambda p: np.random.default_rng(abs(hash(p)) % 2**32).standard_normal(8)
    E = np.stack([emb(p) for _, _, p in items])
    perm = np.random.default_rng(0).permutation(len(items))
    items_s = [items[i] for i in perm]; E_s = E[perm]
    a = gather(items, E, 100, views); b = gather(items_s, E_s, 100, views)
    assert all(np.array_equal(a[v], b[v]) for v in views)
    for w in dev_words[:100]:
        if len(w) >= 2 and w != w[::-1]:
            assert build_view(w, 'fwd_revprompt', True) != build_view(w, 'rev', True)
    assert len(items) == 100 * len(views)
    return {"n_items": len(items)}
