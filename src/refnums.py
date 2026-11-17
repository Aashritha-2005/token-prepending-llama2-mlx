"""Published reference values (notes/reference_numbers.json), checked before use.

[PAPER] rows must equal the table in TP_Master_Plan_v3.md section 2.3; [PREPRINT] rows must equal notes/reverseeol_spec.md
field 7, and their averages must appear in the plan tagged [PREPRINT]. load_refs() raises AssertionError on any mismatch."""
import json
import re

from runlog import ROOT


def _floats(s):
    return [float(x) for x in re.findall(r"\d+\.\d+", s)]


def load_refs():
    ref = json.load(open(ROOT / "notes/reference_numbers.json"))
    E, cols = ref["entries"], ref["datasets"] + ["Avg"]
    plan = (ROOT / "TP_Master_Plan_v3.md").read_text()
    sec = plan[plan.index("### 2.3 Paper numbers"):plan.index("### 2.4")]
    for key, label in (("paper_prompteol", "| PromptEOL |"), ("paper_prompteol_tp", "| PromptEOL + TP |")):
        line = [l for l in sec.splitlines() if l.startswith(label)][0]
        assert E[key]["tag"] == "[PAPER]" and _floats(line) == [E[key]["scores"][c] for c in cols], (key, line)
    assert float(re.search(r"PromptEOL \+ TP ([\d.]+)×", sec).group(1)) == E["paper_time_ratio_tp"]["value"]
    f7 = [l for l in (ROOT / "notes/reverseeol_spec.md").read_text().splitlines() if l.startswith("| 7 |")][0]
    for key, label in (("preprint_prompteol_l27", "PromptEOL (layer 27):"), ("preprint_reverseeol", "PromptEOL w/ ReverseEOL:")):
        assert E[key]["tag"] == "[PREPRINT]" and _floats(f7.split(label, 1)[1])[:8] == [E[key]["scores"][c] for c in cols], key
        assert f"{E[key]['scores']['Avg']:.2f} **[PREPRINT]**" in plan, key
    if "published_cp_prompteol" in E:      # CP Table 1 row, checked against the S2.6 spec (field 10) and the plan
        e = E["published_cp_prompteol"]
        f10 = [l for l in (ROOT / "notes/cp_spec.md").read_text().splitlines() if l.startswith("| 10 |")][0]
        assert e["tag"] == "[PUBLISHED]" and _floats(f10.split("PromptEOL + CP-NS", 1)[1])[:8] == [e["scores"][c] for c in cols], "CP row"
        assert f"{e['scores']['Avg']:.2f} **[PUBLISHED]**" in plan, "CP avg not in plan"
    return ref
