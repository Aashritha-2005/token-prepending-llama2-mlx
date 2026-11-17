import mlx.core as mx
import mlx.nn as nn


class SteerOProj(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.mode = "off"        # "off" | "capture" | "steer"
        self.captured = None     # (B, d_in) from the auxiliary pass
        self.B = None            # (B, d_in) or (d_in,) subtracted in the main pass
        self.alpha = 1.0
        self.sign = 1.0

    def __call__(self, x):       # x: (batch, T, d_in)
        if self.mode == "capture":
            self.captured = x[:, -1, :]
        elif self.mode == "steer":
            A = x[:, -1, :]
            B = self.B if self.B.ndim == 2 else self.B[None, :]
            new_last = self.alpha * (self.sign * (A - B))
            x = mx.concatenate([x[:, :-1, :], new_last[:, None, :].astype(x.dtype)], axis=1)
        return self.inner(x)


def install(model, act_layer):
    """Wrap model.model.layers[act_layer].self_attn.o_proj once; returns the wrapper."""
    attn = model.model.layers[act_layer].self_attn
    if not isinstance(attn.o_proj, SteerOProj):
        attn.o_proj = SteerOProj(attn.o_proj)
    assert type(model.model.layers[act_layer].self_attn.o_proj).__name__ == "SteerOProj"
    return attn.o_proj


def uninstall(model, act_layer):
    attn = model.model.layers[act_layer].self_attn
    if isinstance(attn.o_proj, SteerOProj):
        attn.o_proj = attn.o_proj.inner


# ---------------------------------------------------------------- CP prompts and encode helpers (S2.8, S2.9)
import ast
import sys
from pathlib import Path

import numpy as np

CP_REPO = Path.home() / "Downloads" / "tp_project" / "ref" / "cp_repo"


def cp_templates():
    """(PromptEOL template, auxiliary template) read from CP's evaluate_intervention.py lines 194 and 201."""
    lines = (CP_REPO / "evaluate_intervention.py").read_text().splitlines()
    assert lines[193].strip().startswith("task_prompts = [") and lines[200].strip().startswith("negative_prompt = "), "CP file lines moved"
    return ast.literal_eval(lines[193].split("=", 1)[1].strip())[0], ast.literal_eval(lines[200].split("=", 1)[1].strip())


def cp_clean(s):
    """CP batcher, evaluate_intervention.py:234-236; the caller strips after substitution (:238/:240)."""
    if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
    s = s.replace('"', '\'')
    if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
    return s


def cp_aux_prompt(words, template):
    return template.replace("*sent 0*", cp_clean(" ".join(words))).strip()


def capture_B(model, tok, wrap, act_layer, prompt, plan="vanilla", tp_start=1, tp_exit=7, pst_vec=None):
    """Auxiliary pass truncated after layers[act_layer]; returns the last-position o_proj input (d,) float32.
    plan "tp" (Stage 4, D1 = with TP): the same TP loop as the main pass, copies before layers tp_start..act_layer."""
    from backend_mlx import forward_states
    wrap.mode = "capture"
    forward_states(model, tok(prompt)["input_ids"], [act_layer + 1], plan, tp_start, tp_exit, pst_vec, stop_after=act_layer + 1)
    wrap.mode = "off"
    return np.array(wrap.captured.astype(mx.float32))[0]


def steered_states(model, tok, wrap, prompt, B, alpha, layers, stop_after, plan="vanilla", tp_start=1, tp_exit=7, pst_vec=None):
    """Main pass with the last-position o_proj input of the wrapped layer replaced by alpha * (A - B).
    plan "tp" (Stage 4): the wrapper sits inside the TP loop of forward_states, so the order is copy (before layer l) ->
    layer l with steering -> the next copy carries the steered last state into <PST> (D2, notes/s4_preregistration.md)."""
    from backend_mlx import forward_states
    wrap.mode, wrap.B, wrap.alpha, wrap.sign = "steer", mx.array(np.asarray(B, dtype=np.float32)), alpha, 1.0
    st, _ = forward_states(model, tok(prompt)["input_ids"], layers, plan, tp_start, tp_exit, pst_vec, stop_after=stop_after)
    wrap.mode = "off"
    return st


# ---------------------------------------------------------------- Stage 4 (CP + TP) prompt strings, notes/s4_preregistration.md section 2
SENTIMENT = 'The sentence : "*sent 0*" reflects the sentiment in one word:"'   # CP Table 4 p.7, U+00A8 -> ", repo spacing


def with_pst(template):
    """Insert '<PST> ' after ' : ' and before the opening quote, where the main TP prompt has it."""
    assert template.count(' : "*sent 0*"') == 1, template
    return template.replace(' : "*sent 0*"', ' : <PST> "*sent 0*"')


def cptp_templates():
    """The five pre-registered strings: main (TP, view fwd), aux without / with TP, sentiment without / with TP."""
    from views import FWD
    main_t, aux_t = cp_templates()
    main_tp = FWD.replace("{PST}", "<PST> ").replace("{S}", "*sent 0*")
    assert main_tp == with_pst(main_t) == 'This sentence : <PST> "*sent 0*" means in one word:"', main_tp
    return dict(main_tp=main_tp, aux=aux_t, aux_tp=with_pst(aux_t), sentiment=SENTIMENT, sentiment_tp=with_pst(SENTIMENT))


def cptp_aux_template(aux_tp, control=None):
    """Auxiliary template for a CP+TP pass: the CP prompt (or the sentiment control), in its with-TP form if aux_tp."""
    t = cptp_templates()
    base = "sentiment" if control == "sentiment" else "aux"
    assert control in (None, "sentiment"), control
    return t[base + ("_tp" if aux_tp else "")]
