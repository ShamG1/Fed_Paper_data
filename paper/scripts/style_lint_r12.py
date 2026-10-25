"""Style lint for the manuscript: wording that reads as machine-generated, and defensive hedging.

    python scripts/style_lint_r12.py manuscript.tex

Reports (line, rule, excerpt). Exit code 1 if any hard rule fires; soft rules (hedges) are listed with counts so
that a paragraph with several of them can be rewritten. Rules:
  hard  words and phrases typical of generated prose (delve, leverage, notably, crucially, furthermore,
        moreover, additionally, comprehensive, seamless, pivotal, underscore, showcase, intricate, landscape,
        realm, tapestry, paramount, holistic, groundbreaking, cutting-edge, state-of-the-art, novel,
        "it is worth noting", "it should be noted", "plays a crucial role", "not only ... but also",
        "in this paper, we", significant(ly) (no significance tests are reported), em dashes (---)
  soft  hedges and defensive formulas (may, might, could, possibly, potentially, likely, appears, seems,
        suggests, "we hypothesize", "does not establish", "cannot exclude", "caution", "we do not claim",
        "note that", "to our knowledge"); a sentence with two or more is reported as stacked
"""
import re
import sys
from pathlib import Path

HARD = [r"\bdelv\w*", r"\bleverag\w*", r"\bnotabl[ye]\b", r"\bcrucial\w*", r"\bfurthermore\b", r"\bmoreover\b",
        r"\badditionally\b", r"\bcomprehensive\b", r"\bseamless\w*", r"\bpivotal\b", r"\bunderscor\w*",
        r"\bshowcas\w*", r"\bintricate\b", r"\blandscape\b", r"\brealm\b", r"\btapestry\b", r"\bparamount\b",
        r"\bholistic\b", r"\bgroundbreaking\b", r"\bcutting-edge\b", r"\bstate-of-the-art\b", r"\bnovel\b",
        r"it is worth noting", r"it should be noted", r"plays? a (?:crucial|key|vital) role", r"not only\b.*\bbut also",
        r"in this paper, we", r"\bsignificant\w*", r"(?<!-)---(?!-)", r"\brobust(?:ly|ness)? (?:improv|perform)\w*",
        r"\bremarkabl\w*", r"\bimportantly\b", r"\binterestingly\b"]
SOFT = [r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibl[ye]\b", r"\bpotential(?:ly)?\b", r"\blikely\b",
        r"\bappears?\b", r"\bseems?\b", r"\bsuggests?\b", r"we hypothesi[sz]e", r"does not establish",
        r"cannot exclude", r"\bcaution\w*", r"we do not claim", r"\bnote that\b", r"to our knowledge"]


def strip_tex(t):
    t = re.sub(r"(?<!\\)%.*", "", t)                       # comments
    t = re.sub(r"\\(?:cite|ref|eqref|label|includegraphics|input|bibliography\w*)\{[^}]*\}", " ", t)
    t = re.sub(r"\$[^$]*\$", " MATH ", t)                  # inline math
    return t


def main(path):
    raw = Path(path).read_text(encoding="utf-8")
    body = raw.split(r"\begin{document}", 1)[-1]
    offset = raw[:len(raw) - len(body)].count("\n")
    hard_hits, soft_hits = [], []
    for k, line in enumerate(body.split("\n"), start=offset + 1):
        txt = strip_tex(line)
        low = txt.lower()
        for pat in HARD:
            for m in re.finditer(pat, low):
                hard_hits.append((k, pat, txt[max(0, m.start() - 40):m.end() + 40].strip()))
        for sent in re.split(r"(?<=[.;:])\s+", txt):
            s_low = sent.lower()
            found = [pat for pat in SOFT if re.search(pat, s_low)]
            if found:
                soft_hits.append((k, found, sent.strip()[:160]))
    for h in hard_hits:
        print("HARD", h)
    for h in soft_hits:
        tag = "STACKED" if len(h[1]) > 1 else "soft"
        print(tag, h[0], h[1], "|", h[2])
    print(f"hard: {len(hard_hits)}, soft sentences: {len(soft_hits)}, "
          f"stacked: {sum(len(h[1]) > 1 for h in soft_hits)}")
    return 1 if hard_hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "manuscript.tex")))
