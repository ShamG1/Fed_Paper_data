"""Shared figure specification for every data figure of the manuscript (R12; unchanged from R11 except the
method palette, which now names CQS, CQS+FedNova, CQS+UDR and the baselines of R12).

One typeface, one type size, one set of line weights and one method palette, so that
all figures read as one system (IEEE: type ~9-10 pt at final size, consistent across
graphics; widths 3.5 in / 7.16 in; vector output with embedded fonts).

Typeface  Nimbus Roman (URW Times clone, TrueType, embedded as Type 42); math in STIX.
Type size 9 pt for every label, tick label, legend entry and panel title at final size.
Lines     axes 0.6 pt, ticks 0.6 pt x 3 pt (outward), data 1.4 pt, references 0.9 pt,
          error bars 0.8 pt, grid 0.4 pt.
Markers   4.2 pt with a 0.5 pt white edge; every method has its own marker and line style
          so that the figures remain readable in grey scale.
Palette   validated with the dataviz validator (OKLab, CVD simulation, all pairs):
          Fed-SSBR #1E4E9C, CQS #D96C25, UDR #2A9D8F, FedBSS #8B6BBE pass every check;
          baselines and references are neutral (FedAvg slate grey, CV charcoal).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "fonts_figures"
for f in sorted(FONT_DIR.glob("NimbusRoman-*.ttf")):
    font_manager.fontManager.addfont(str(f))
# Prefer the bundled TrueType faces to same-named system CFF/OpenType faces.
# Otherwise Matplotlib can select the latter despite pdf.fonttype = 42.
font_manager.fontManager.ttflist.sort(
    key=lambda item: 0 if Path(item.fname).resolve().parent == FONT_DIR.resolve() else 1)

COL1, COL2 = 3.5, 7.16          # IEEE single / double column widths (in)
FS = 9                           # the one type size (pt)

INK = "#1A1A1A"                  # primary ink: text, axes
INK2 = "#5A6069"                 # secondary ink: annotations, reference labels
GRID = "#E3E6EA"                 # hairline grid
SHADE = "#EEF0F3"                # warm-up rounds and other context bands
REF = "#2F2F2F"                  # kinematic CV reference (charcoal)

# method identity: colour, marker, line style (fixed across all figures)
C = {
    "CQS": "#D96C25",            # proposed client rule, burnt orange
    "CQS+FedNova": "#1E4E9C",    # CQS with step-normalized aggregation, deep cobalt
    "CQS+UDR": "#2A9D8F",        # CQS with update-deviation reweighting, teal
    "UDR": "#2A9D8F",
    "FedNova": "#8B6BBE",        # amethyst
    "FedAvg": "#7B8089",         # baseline, slate grey
    "CV": REF,
    # tints of the same families for controls and variants
    "CQS-random": "#EBA97C",
    "CQS-anti": "#F3CDB0",
    "static": "#5E656F",
    "FedProx": "#B9BEC5",
    # trajectory figure (as in R11)
    "Truth": "#000000",
    "History": "#A7ACB3",
}
MARK = {"CQS": "D", "CQS+FedNova": "o", "CQS+UDR": "^", "UDR": "^", "FedNova": "v", "FedAvg": "s",
        "CQS-random": "d", "static": "x", "CV": "^"}
LS = {"CQS": (0, (5, 1.6)), "CQS+FedNova": "-", "CQS+UDR": (0, (1.2, 1.4)), "UDR": (0, (1.2, 1.4)),
      "FedNova": (0, (3, 1.5)), "FedAvg": (0, (6, 1.6, 1.2, 1.6)), "CQS-random": (0, (1.5, 1.5)),
      "static": "-", "CV": (0, (2.2, 1.6))}

LW = 1.4          # data lines
LW_REF = 0.9      # reference lines
LW_ERR = 0.8      # error bars
MS = 4.2          # marker size (pt)
MEW = 0.5         # marker edge width (pt)


def apply():
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "mathtext.fontset": "stix",
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "figure.titlesize": FS,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.pad": 2.2, "ytick.major.pad": 2.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 5, "axes.labelpad": 2.5,
        "lines.linewidth": LW, "lines.markersize": MS, "lines.markeredgewidth": MEW,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "errorbar.capsize": 2.0,
        "legend.frameon": False, "legend.handlelength": 2.2, "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.1, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.28,
        "legend.borderpad": 0.2,
        "grid.color": GRID, "grid.linewidth": 0.4,
        "hatch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150,
        "axes.unicode_minus": True,
        "path.simplify": False,
    })


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def panel_title(ax, label, text, x=0.0, y=1.0, pad_pt=5.0):
    """Bold panel label '(a)' followed by a regular-weight title, left-aligned above the axes."""
    from matplotlib.transforms import ScaledTranslation
    fig = ax.figure
    base = ax.transAxes + ScaledTranslation(0, pad_pt / 72.0, fig.dpi_scale_trans)
    t1 = ax.text(x, y, label, transform=base, ha="left", va="baseline", fontsize=FS, fontweight="bold", color=INK)
    fig.canvas.draw()
    bb = t1.get_window_extent(renderer=fig.canvas.get_renderer())
    w_in = bb.width / fig.dpi
    off = base + ScaledTranslation(w_in + 3.0 / 72.0, 0, fig.dpi_scale_trans)
    ax.text(x, y, text, transform=off, ha="left", va="baseline", fontsize=FS, color=INK)


def minus(v, fmt="{:+.1f}"):
    return fmt.format(v).replace("-", "−")


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "R12 figure script", "CreationDate": None}
    fig.savefig(out_dir / f"{name}.pdf", metadata=meta)
    fig.savefig(out_dir / f"{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)
