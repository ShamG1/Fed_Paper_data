"""Data figures of the R12 manuscript (Figs. 2, 4-6) from data/stats_r12.json and data/figdata_r12.json, and
the trajectory examples (fig_traj) from data/qualitative (written by make_qualitative_r12.py).

    python scripts/make_figures_r12.py   # figures/fig_clients, fig_val, fig_strata, fig_server, fig_traj (.pdf/.png)

Figs. 1 and 3 (scenario and CQS round) are the authors' drawings: figures/author_original/figure1.png and
figure3.png, flattened onto a white background as figures/fig_scenario.png and fig_framework.png.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as FS  # noqa: E402
from fig_style import C, LS, MARK, COL2, INK, INK2, SHADE, LW, LW_REF, LW_ERR, MS, MEW  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "data" / "stats_r12.json").read_text(encoding="utf-8"))
F = json.loads((ROOT / "data" / "figdata_r12.json").read_text(encoding="utf-8"))
OUT = ROOT / "figures"
FS.apply()

NGSIM_SHORT = {"US101-0750": "U0750", "US101-0805": "U0805", "US101-0820": "U0820",
               "I80-1600": "I1600", "I80-1700": "I1700", "I80-1715": "I1715"}


def fig_clients():
    D = F["difficulty"]
    fig = plt.figure(figsize=(COL2, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[6.6, 10], wspace=0.16, left=0.072, right=0.995, bottom=0.17,
                          top=0.875)
    panels = [("fedtraj_ngsim", "natural", "(a)", "NGSIM, six recording periods"),
              ("fedtraj_highd", "motion", "(b)", "highD, ten motion clusters")]
    for k, (ds, part, lab, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        info = D[ds][part]
        cl = info["clients"]
        n = np.array([c["n"] for c in cl], float)
        order = np.argsort(-n)
        share = n / n.sum()
        for j, i in enumerate(order):
            q = cl[i]["q"]
            ax.add_patch(Rectangle((j - 0.3, q[1]), 0.6, q[3] - q[1], fc="#DCE3EC", ec=INK, lw=0.6, zorder=2))
            ax.plot([j, j], [q[0], q[1]], color=INK, lw=0.6, zorder=1)
            ax.plot([j, j], [q[3], q[4]], color=INK, lw=0.6, zorder=1)
            ax.plot([j - 0.14, j + 0.14], [q[0], q[0]], color=INK, lw=0.6)
            ax.plot([j - 0.14, j + 0.14], [q[4], q[4]], color=INK, lw=0.6)
            ax.plot([j - 0.3, j + 0.3], [q[2], q[2]], color=C["CQS"], lw=1.6, zorder=3, solid_capstyle="butt")
        names = info["names"]
        short = [NGSIM_SHORT.get(names[i], names[i]) for i in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{s}\n{100 * share[i]:.1f}%" for s, i in zip(short, order)])
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 3e2)
        ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1, 1e1, 1e2]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", length=0, pad=3)
        if k == 0:
            ax.set_ylabel(r"CV loss per window (m$^2$)")
        FS.grid(ax)
        FS.panel_title(ax, lab, title)
    FS.save(fig, OUT, "fig_clients")


VAL_CFGS = [("fedavg", "FedAvg"), ("cqs-random", "CQS-random"), ("fednova", "FedNova"), ("cqs", "CQS"),
            ("cqs-nova", "CQS+FedNova")]


def fig_val():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.80, wspace=0.2)
    titles = {"ngsim": "NGSIM", "highd": "highD"}
    for k, (ax, key) in enumerate(zip(axes, ("ngsim", "highd"))):
        V = F["settings"][key]["val_curves"]
        lo, hi = np.inf, -np.inf
        for cfg, lab in VAL_CFGS:
            m = np.array(V[cfg]["mean"])
            r = np.arange(1, m.size + 1)
            ax.plot(r, m, color=C[lab], ls=LS[lab], lw=LW, label=lab, zorder=3)
            j = int(np.argmin(m))
            ax.plot(r[j], m[j], ls="none", marker=MARK[lab], color=C[lab], ms=MS + 1.0, mec="white", mew=MEW,
                    zorder=5)
            lo, hi = min(lo, m.min()), max(hi, m[-1], m[9:].max())
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - 1.6 * pad, hi + 2 * pad)
        ax.set_xlim(0.5, 30.5)
        ax.axvspan(0.5, 5.5, color=SHADE, lw=0, zorder=0)
        ax.text(3.0, lo - 0.75 * pad, "warm-up", ha="center", va="center", color=INK2, fontsize=FS.FS)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel("Communication round")
        if k == 0:
            ax.set_ylabel("Validation 5 s RMSE (m)")
        FS.grid(ax)
        FS.panel_title(ax, f"({'ab'[k]})", titles[key])
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=5, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False)
    FS.save(fig, OUT, "fig_val")


STRATA = ["Q1", "Q2", "Q3", "Q4", "Q5", "lateral"]
STRATA_LAB = ["Q1", "Q2", "Q3", "Q4", "Q5", "LC"]
STRATA_CFGS = [("cqs-random", "CQS-random", None), ("cqs", "CQS", None), ("fednova", "FedNova", "////"),
               ("cqs-nova", "CQS+FedNova", None)]


def fig_strata():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.80, wspace=0.2)
    titles = {"ngsim": "NGSIM", "highd": "highD"}
    w = 0.2
    for k, (ax, key) in enumerate(zip(axes, ("ngsim", "highd"))):
        SV = S["settings"][key]["strata_vs_fedavg"]
        for j, (cfg, lab, hatch) in enumerate(STRATA_CFGS):
            x = np.arange(len(STRATA)) + (j - 1.5) * w
            v = np.array([SV[cfg][q]["rel"] for q in STRATA])
            lo = np.array([SV[cfg][q]["rel_lo"] for q in STRATA])
            hi = np.array([SV[cfg][q]["rel_hi"] for q in STRATA])
            ax.bar(x, v, w * 0.9, color=C[lab], label=lab, hatch=hatch, edgecolor="white" if hatch else C[lab],
                   linewidth=0.0 if not hatch else 0.3, zorder=2)
            ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=LW_ERR, capsize=1.4,
                        capthick=LW_ERR, zorder=3)
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        ax.axvline(4.5, color=INK2, lw=0.4, ls=(0, (2, 2)), zorder=1)
        ax.set_xticks(range(len(STRATA)))
        ax.set_xticklabels(STRATA_LAB)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(STRATA) - 0.45)
        if k == 0:
            ax.set_ylabel("Change of 5 s RMSE vs. FedAvg (%)")
        FS.grid(ax)
        FS.panel_title(ax, f"({'ab'[k]})", titles[key])
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=4, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False)
    FS.save(fig, OUT, "fig_strata")


def ratio_axis(ax, key, lab, title, show_ylabel):
    sv = S["settings"][key]["server"]
    p = np.array(sv["client_share"])
    udr = np.array(sv["udr"]["alpha_over_p"])
    nova = np.array(sv["cqs-nova_factor"]["tau_eff_over_tau"])
    beta = sv["static"]["beta"]
    n = np.array(F["settings"][key]["n_train"], float)
    grid = np.geomspace(n.min() * 0.85, n.max() * 1.15, 200)
    static = grid ** (beta - 1) * n.sum() / np.sum(n ** beta)
    ax.plot(100 * grid / n.sum(), static, color=C["static"], lw=LW_REF, ls="-", zorder=2,
            label=r"static $n_i^{\beta}$")
    keff = 1.0 / np.sum(p ** 2)
    ax.plot(100 * grid / n.sum(), 1.0 / (keff * grid / n.sum()), color=C["FedNova"], lw=LW_REF, ls=(0, (1.2, 1.4)),
            zorder=2, label=r"$1/(K_{\rm eff}\,p_i)$")
    ax.plot(100 * p, nova, ls="none", marker=MARK["FedNova"], color=C["FedNova"], ms=MS + 0.8, mec="white",
            mew=MEW, zorder=4, label=r"FedNova $\tau_{\rm eff}/\tau_i$")
    ax.plot(100 * p, udr, ls="none", marker=MARK["UDR"], color=C["UDR"], ms=MS + 0.8, mec="white", mew=MEW,
            zorder=5, label=r"UDR $\alpha_i/p_i$")
    ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xt = [3, 5, 10, 20, 30] if key == "highd" else [12, 15, 20]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    yt = [0.5, 1, 2, 4] if key == "highd" else [0.8, 1.0, 1.25]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if key == "highd":
        ax.set_ylim(0.45, 5.0)
    else:
        ax.set_ylim(0.75, 1.4)
    ax.set_xlabel("Client share of windows (%)")
    if show_ylabel:
        ax.set_ylabel("Weight relative to sample share")
    FS.grid(ax, "both")
    FS.panel_title(ax, lab, title)


def fig_server():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.80, wspace=0.32)
    ax = axes[0]
    sc = F["settings"]["highd"]["scatter"]
    sv = S["settings"]["highd"]["server"]["r_norm_steps"]
    for cfg, lab in (("fedavg", "FedAvg"), ("cqs", "CQS")):
        pts = np.array([[q[1], q[2]] for q in sc[cfg]], float)
        ax.plot(pts[:, 0], pts[:, 1], ls="none", marker=MARK[lab], ms=2.2, mew=0, color=C[lab], alpha=0.35,
                zorder=2 if cfg == "fedavg" else 3, rasterized=True)
    ax.set_xlabel(r"Local optimizer steps $\tau_i^t$")
    ax.set_ylabel(r"Update norm $\|\Delta_i^t\|_2$")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    hl = [Line2D([], [], ls="none", marker=MARK[l], ms=MS, color=C[l], label=f"{l} ($r={sv[c]['mean']:.2f}$)")
          for c, l in (("fedavg", "FedAvg"), ("cqs", "CQS"))]
    ax.legend(handles=hl, loc="lower right", handletextpad=0.2)
    FS.grid(ax, "both")
    FS.panel_title(ax, "(a)", "highD, update size")
    ratio_axis(axes[1], "highd", "(b)", "highD, client weights", True)
    ratio_axis(axes[2], "ngsim", "(c)", "NGSIM, client weights", False)
    h, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h, l_, ncol=4, loc="upper center", bbox_to_anchor=(0.66, 1.0), frameon=False, columnspacing=0.9)
    FS.save(fig, OUT, "fig_server")


TRAJ_ROWS = [("ngsim", "Q5", "NGSIM Q5"), ("ngsim", "LC", "NGSIM LC"), ("highd", "Q5", "highD Q5"),
             ("highd", "LC", "highD LC")]
TRAJ_SERIES = [("cv", "CV"), ("fedavg", "FedAvg"), ("cqs", "CQS"), ("cqs-nova", "CQS+FedNova")]
TRAJ_MARK = {"CV": "^", "FedAvg": "s", "CQS": "D", "CQS+FedNova": "p"}


def fig_traj():
    """Selected test windows (rule: data/qualitative/SELECTION_RULE.md); left of the driving direction is up."""
    qdir = ROOT / "data" / "qualitative"
    files = [qdir / f"selected_trajectories_{s}.csv" for s in ("ngsim", "highd")]
    if not all(f.exists() for f in files):
        print("skip fig_traj: run make_qualitative_r12.py first")
        return
    Q = json.loads((qdir / "qualitative_r12.json").read_text(encoding="utf-8"))
    data = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["setting"], r["stratum"], int(r["percentile"]))
                data.setdefault(key, {}).setdefault(r["series"], []).append((float(r["lon_m"]),
                                                                              -float(r["lat_m"])))
    fig, axes = plt.subplots(4, 3, figsize=(COL2, 4.7))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.905, bottom=0.075, wspace=0.17, hspace=0.66)
    for i, (setting, stratum, rlab) in enumerate(TRAJ_ROWS):
        sel = {s["percentile"]: s for s in Q["settings"][setting]["strata"][stratum]["selected"]}
        for j, p in enumerate(PCT_COLS):
            ax = axes[i, j]
            d = {k: np.array(v) for k, v in data[(setting, stratum, p)].items()}
            pts = [d["history"]]
            ax.plot(d["history"][:, 0], d["history"][:, 1], color=C["History"], lw=LW, zorder=3,
                    solid_capstyle="round")
            ax.plot(d["history"][9::10, 0], d["history"][9::10, 1], ls="none", marker="o", ms=2.6,
                    color=C["History"], zorder=3)
            for name, lab in [("truth", "Truth")] + TRAJ_SERIES:
                a = np.vstack([[0.0, 0.0], d[name]])
                pts.append(a)
                z = {"Truth": 7, "CV": 4}.get(lab, 5)
                if lab == "Truth":
                    ax.plot(a[:, 0], a[:, 1], color=C[lab], lw=LW + 0.3, zorder=z, solid_capstyle="round")
                    ax.plot(a[10:50:10, 0], a[10:50:10, 1], ls="none", marker="o", ms=3.0, color=C[lab], zorder=z)
                    ax.plot(a[50, 0], a[50, 1], ls="none", marker="o", ms=4.6, color=C[lab], zorder=z)
                    continue
                ax.plot(a[:, 0], a[:, 1], color=C[lab], lw=LW, ls=LS[lab] if lab != "CQS+FedNova" else "-",
                        zorder=z)
                ax.plot(a[10:50:10, 0], a[10:50:10, 1], ls="none", marker=TRAJ_MARK[lab], ms=3.0, color=C[lab],
                        mec="white", mew=0.4, zorder=z + 1)
                ax.plot(a[50, 0], a[50, 1], ls="none", marker=TRAJ_MARK[lab], ms=4.8, color=C[lab],
                        mec="white", mew=0.5, zorder=z + 1)
            P = np.vstack(pts)
            x0, x1 = P[:, 0].min(), P[:, 0].max()
            ax.set_xlim(x0 - 0.03 * (x1 - x0), x1 + 0.03 * (x1 - x0))
            yc, ys = 0.5 * (P[:, 1].min() + P[:, 1].max()), max(np.ptp(P[:, 1]), 2.4) * 1.22
            ax.set_ylim(yc - ys / 2, yc + ys / 2)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            FS.grid(ax, "both")
            e = sel[p]["e5_m"]
            FS.panel_title(ax, f"({'abcdefghijkl'[3 * i + j]})",
                           f"P{p}: FedAvg {e['fedavg']:.2f}, CQS {e['cqs']:.2f} m", pad_pt=3.5)
            if j == 0:
                ax.set_ylabel(f"{rlab}\nLateral (m)", linespacing=1.15)
            if i == len(TRAJ_ROWS) - 1:
                ax.set_xlabel("Longitudinal (m)")
    hl = [Line2D([], [], color=C["History"], lw=LW, marker="o", ms=2.6, label="History"),
          Line2D([], [], color=C["Truth"], lw=LW + 0.3, marker="o", ms=3.0, label="Truth")]
    hl += [Line2D([], [], color=C[lab], lw=LW, ls=LS[lab] if lab != "CQS+FedNova" else "-", marker=TRAJ_MARK[lab],
                  ms=3.2, mec="white", mew=0.4, label=lab) for _, lab in TRAJ_SERIES]
    fig.legend(handles=hl, ncol=6, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False)
    FS.save(fig, OUT, "fig_traj")


PCT_COLS = (10, 50, 90)


if __name__ == "__main__":
    fig_clients()
    fig_val()
    fig_strata()
    fig_server()
    fig_traj()
