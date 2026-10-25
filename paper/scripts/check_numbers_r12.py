"""Flag every decimal number in the manuscript that is not traceable to data/stats_r12.json or the protocol.

    python scripts/check_numbers_r12.py manuscript.tex      # prints unmatched numbers; exit code 1 if any

A number written with d decimals matches if some value v in stats_r12.json satisfies round(|v|, d) == x, or
round(100 |v|, d) == x for shares and fractions. Protocol constants (quotas, bounds, grid values, model sizes,
pacing quotas computed from the formula) are listed below. Generated tables (tables/*.tex) are checked too.
"""
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "data" / "stats_r12.json").read_text(encoding="utf-8"))

vals = set()


def half_up(x, d):
    """Conventional rounding (16.25 -> 16.3), in addition to Python's round-half-even."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        for d in (1, 2, 3):
            for x in (abs(o), abs(o) * 100):
                vals.add((d, round(x, d)))
                vals.add((d, half_up(x, d)))


walk(S)
# trajectory examples (scripts/make_qualitative_r12.py), if present
QF = ROOT / "data" / "qualitative" / "qualitative_r12.json"
if QF.exists():
    walk(json.loads(QF.read_text(encoding="utf-8")))
# protocol constants: quotas 0.6/0.8/1.0, FedBSS quotas 0.7/0.9, bounds 0.5/1.5 (and scans 0.3/0.7/1.3/1.7),
# floors 0.4/0.6/0.8, budget 2.4 n_i and 16.7%, cosine quotas for E = 5 (0.600 0.659 0.800 0.941 1.000),
# temperature scans 0.5 and 2, kinematic lag 0.5 s, pilot threshold 0.02 m, parameters 0.839 M,
# lane-change threshold 3 m, sampling 10 Hz / 25 Hz.
WHITE = """0.6 0.8 1.0 0.7 0.9 0.5 1.5 0.3 1.3 1.7 0.4 2.4 16.7 0.600 0.659 0.800 0.941 1.000 2.0 0.02 0.839
0.833 0.889"""
for w in WHITE.split():
    d = len(w.split(".")[1])
    vals.add((d, round(float(w), d)))


def check(path):
    tex = Path(path).read_text(encoding="utf-8")
    bad = []
    for m in re.finditer(r"(?<![\w.\\])[-+]?(\d+\.\d+)", tex):
        x = float(m.group(1))
        d = len(m.group(1).split(".")[1])
        if (d, round(x, d)) not in vals:
            line = tex.count("\n", 0, m.start()) + 1
            bad.append((Path(path).name, line, m.group(0), tex[max(0, m.start() - 60):m.end() + 20].replace("\n", " ")))
    return bad


files = [sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "manuscript.tex")]
files += [str(p) for p in sorted((ROOT / "tables").glob("*.tex"))]
bad = []
for f in files:
    bad += check(f)
for b in bad:
    print(b)
print("checked", len(files), "files; unmatched:", len(bad))
sys.exit(1 if bad else 0)
