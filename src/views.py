"""Prompt/view builder (plan Appendix B). REV wording is a placeholder until S2.1."""
import random
import zlib

import numpy as np

FWD = 'This sentence : {PST}"{S}" means in one word:"'
REV = 'This reversed sentence : {PST}"{S}" means in one word:"'   # wording: replace per S2.1


def to_str_words(words):
    return [w.decode("utf-8") if isinstance(w, bytes) else w for w in words]


def clean(s):
    # identical to the repo batcher (evaluate.py)
    if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
    s = s.replace('"', "'")
    if len(s) > 0 and s[-1] == '?': s = s[:-1] + '.'
    return s


def reverse_words(words, punct="full"):
    if punct == "keep_final" and words and words[-1] in {'.', '?', '!'}:
        return words[-2::-1] + [words[-1]]
    return words[::-1]


def shuffle_words(words):
    w = list(words)
    random.Random(zlib.crc32(' '.join(words).encode('utf-8'))).shuffle(w)
    return w


def build_view(words, view, use_pst, punct="full"):
    words = to_str_words(words)
    pst = '<PST> ' if use_pst else ''
    if view == 'fwd':             tmpl, w = FWD, words
    elif view == 'rev':           tmpl, w = REV, reverse_words(words, punct)
    elif view == 'fwd_revprompt': tmpl, w = REV, words                 # control 6b
    elif view == 'shuf':          tmpl, w = REV, shuffle_words(words)  # control 6a
    else: raise ValueError(view)
    return tmpl.replace('{PST}', pst).replace('{S}', clean(' '.join(w))).strip()


def make_items(words_list, views, plan, punct="full"):
    use_pst = (plan == 'tp')
    return [(i, v, build_view(w, v, use_pst, punct))
            for i, w in enumerate(words_list) for v in views]


def gather(items, E, n, views):
    """E: array (len(items), dim) aligned with items -> dict view -> (n, dim)"""
    out = {v: [None] * n for v in views}
    for (i, v, _), e in zip(items, E): out[v][i] = e
    return {v: np.stack(out[v]) for v in views}


def combine(Ev, views, alpha=0.5, normalize=True):
    def nz(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
    A, B = (nz(Ev[views[0]]), nz(Ev[views[1]])) if normalize else (Ev[views[0]], Ev[views[1]])
    return alpha * A + (1 - alpha) * B


def word_vec(w, dim=64):
    if isinstance(w, bytes): w = w.decode('utf-8')
    return np.random.default_rng(zlib.crc32(w.encode('utf-8'))).standard_normal(dim)


def dummy_embed(words):
    return np.mean([word_vec(w) for w in words], 0) if len(words) else np.zeros(64)


def dummy_batcher(params, batch):
    return np.stack([dummy_embed(s) for s in batch])
