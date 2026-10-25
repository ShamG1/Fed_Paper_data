"""Find large blank vertical gaps inside the text columns of every page (R11 check_gaps.py, PyMuPDF version).

    python scripts/check_gaps_r12.py manuscript.pdf [min_gap_pt]

Each page is rasterized at 72 dpi (1 px = 1 pt). Every column is scanned for runs of blank rows; a run longer
than min_gap_pt (default 28 pt, above the usual float separation and heading space) that is followed by more
ink in the same column is reported, as is blank space at the bottom of a column on any page but the last.
"""
import sys

import fitz
import numpy as np

pdf = sys.argv[1] if len(sys.argv) > 1 else "manuscript.pdf"
min_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 28.0
doc = fitz.open(pdf)
report = []
for pi, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=72, colorspace=fitz.csGRAY)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = a < 200
    H, W = ink.shape
    top, bot = 55, H - 40                      # skip the running head and the bottom margin
    for ci, (x0, x1) in enumerate([(40, W // 2 - 4), (W // 2 + 4, W - 40)]):
        rows = ink[top:bot, x0:x1].any(1)
        ys = np.where(rows)[0]
        if len(ys) == 0:
            continue
        run, start, last = 0, None, ys.max()
        for y in range(ys.min(), last + 1):
            if not rows[y]:
                if run == 0:
                    start = y
                run += 1
            else:
                if run >= min_gap:
                    report.append((pi, "LR"[ci], start + top, run))
                run = 0
        tail = (bot - top) - 1 - last
        if tail >= min_gap and pi < doc.page_count:
            report.append((pi, "LR"[ci] + "-bottom", last + top, tail))
for r in report:
    print("page %d col %s y=%d gap=%d pt" % r)
print("pages:", doc.page_count, "gaps:", len(report))
