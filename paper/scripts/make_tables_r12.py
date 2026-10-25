"""Write the LaTeX table bodies of the R12 manuscript from data/stats_r12.json (tables/*.tex)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "data" / "stats_r12.json").read_text(encoding="utf-8"))
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)
ST = S["settings"]

MAIN_ROWS = [
    ("central", "Centralized"),
    None,
    ("fedavg", r"FedAvg~\cite{mcmahan2017fedavg}"),
    ("fedprox", r"FedProx~\cite{li2020fedprox}"),
    ("fednova", r"FedNova~\cite{wang2020fednova}"),
    ("fedlc", r"LGC~\cite{wang2021local}"),
    ("fedbss", r"FedBSS-fixed"),
    ("fedbss-mean", r"FedBSS-mean"),
    None,
    ("cqs-random", r"CQS-random"),
    ("cqs-anti", r"CQS-anti"),
    ("cqs", r"CQS (proposed)"),
    None,
    ("cqs-prox", r"CQS+FedProx"),
    ("cqs-nova", r"CQS+FedNova"),
    None,
    ("udr", r"FedAvg+UDR"),
    ("cqs-udr", r"CQS+UDR"),
    ("cqs-udrfs", r"CQS+UDR-FS"),
    ("cqs-static", r"CQS+static"),
]
VISITS = {"fedbss": 0.889, "fedbss-mean": 0.93}


def num(x, d=3):
    s = f"{x:.{d}f}"
    return s


def signed(x, d=2):
    return f"{x:+.{d}f}".replace("+", "+").replace("-", "-")


def main_table():
    cols = ["1s", "3s", "5s", "mean"]
    fed = [r[0] for r in MAIN_ROWS if r and r[0] != "central"]
    best = {}
    for key in ("ngsim", "highd"):
        C = ST[key]["configs"]
        for c in cols:
            best[(key, c)] = min(C[f][c]["mean"] for f in fed)
    lines = []
    cv = {k: S["datasets"][ds]["cv"]["test"] for k, ds in (("ngsim", "fedtraj_ngsim"), ("highd", "fedtraj_highd"))}
    cells = ["CV (no training)", "---", "---"]
    for key in ("ngsim", "highd"):
        cells += [f"${num(cv[key][c])}$" for c in cols]
    lines.append(" & ".join(cells) + r" \\")
    for row in MAIN_ROWS:
        if row is None:
            lines.append(r"\midrule")
            continue
        cfg, label = row
        n = ST["ngsim"]["configs"][cfg]["5s"]["n"]
        assert n == ST["highd"]["configs"][cfg]["5s"]["n"], cfg
        if cfg == "central":
            vis = "---"
        elif cfg in ("fedavg", "fedprox", "fednova", "fedlc", "udr"):
            vis = "1.00"
        elif cfg in VISITS:
            vis = f"{VISITS[cfg]:.2f}"
        else:
            vis = "0.83"
        cells = [label, str(n), vis]
        for key in ("ngsim", "highd"):
            C = ST[key]["configs"][cfg]
            for c in cols:
                m = C[c]["mean"]
                txt = num(m)
                if c == "5s":
                    txt = f"{num(m)}\\pm{num(C[c]['sd'])}"
                if cfg != "central" and abs(m - best[(key, c)]) < 5e-7:
                    txt = r"\mathbf{" + txt + "}"
                cells.append(f"${txt}$")
        lines.append(" & ".join(cells) + r" \\")
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


PAIR_ROWS = [
    ("Scheduler", None),
    ("cqs|fedavg", r"CQS $-$ FedAvg$^{\dagger}$"),
    ("cqs|cqs-random", r"CQS $-$ CQS-random$^{\dagger}$"),
    ("cqs|cqs-anti", r"CQS $-$ CQS-anti"),
    ("cqs-random|fedavg", r"CQS-random $-$ FedAvg"),
    ("cqs|fedbss", r"CQS $-$ FedBSS-fixed"),
    ("cqs|fedbss-mean", r"CQS $-$ FedBSS-mean"),
    ("cqs|fedprox", r"CQS $-$ FedProx"),
    ("cqs|fednova", r"CQS $-$ FedNova"),
    ("cqs|fedlc", r"CQS $-$ LGC"),
    (r"CQS added to existing rules$^{\ddagger}$", None),
    ("cqs-prox|fedprox", r"CQS+FedProx $-$ FedProx"),
    ("cqs-nova|fednova", r"CQS+FedNova $-$ FedNova"),
    ("Server layer", None),
    ("udr|fedavg", r"FedAvg+UDR $-$ FedAvg"),
    ("cqs-udr|cqs", r"CQS+UDR $-$ CQS$^{\dagger}$"),
    ("cqs-static|cqs", r"CQS+static $-$ CQS"),
    ("cqs-udr|cqs-static", r"CQS+UDR $-$ CQS+static"),
    ("cqs-udrfs|cqs", r"CQS+UDR-FS $-$ CQS"),
    ("cqs-nova|cqs-udr", r"CQS+FedNova $-$ CQS+UDR$^{\ddagger}$"),
    ("cqs-udr|fedavg", r"CQS+UDR $-$ FedAvg$^{\dagger}$"),
    ("cqs-udr|fedprox", r"CQS+UDR $-$ FedProx$^{\dagger}$"),
]


def pcell(p, bold_effect=True):
    d = f"{p['rel']:+.2f}"
    ci = f"[{p['rel_lo']:+.2f}, {p['rel_hi']:+.2f}]"
    if bold_effect and p["verdict"] == "effect":
        d = r"\mathbf{" + d + "}"
    return [f"${d}$", f"${ci}$", f"{p['lower']}/{p['n']}"]


def pair_table():
    lines = []
    for key, label in PAIR_ROWS:
        if label is None:
            if lines:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{7}{@{}l}{\textit{" + key + r"}} \\")
            continue
        cells = [label]
        for s in ("ngsim", "highd"):
            cells += pcell(ST[s]["pairs"][key]["5s"])
        lines.append(" & ".join(cells) + r" \\")
    (OUT / "tab_pairs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


EXTRA_ROWS = [
    (r"NGSIM-motion, ten clients", None, None),
    ("ngsim_motion", "pairs", "cqs|fedavg", r"CQS $-$ FedAvg"),
    ("ngsim_motion", "pairs", "cqs|cqs-random", r"CQS $-$ CQS-random"),
    ("ngsim_motion", "pairs", "udr|fedavg", r"FedAvg+UDR $-$ FedAvg"),
    ("ngsim_motion", "pairs", "cqs-udr|cqs", r"CQS+UDR $-$ CQS"),
    (r"highD-recording, six of 60 clients per round", None, None),
    ("highd_rec", "pairs", "cqs|fedavg", r"CQS $-$ FedAvg"),
    ("highd_rec", "pairs", "udr|fedavg", r"FedAvg+UDR $-$ FedAvg"),
    ("highd_rec", "pairs", "fednova|fedavg", r"FedNova $-$ FedAvg"),
    ("highd_rec", "pairs", "cqs|fednova", r"CQS $-$ FedNova"),
    ("highd_rec", "pairs", "cqs-udr|cqs", r"CQS+UDR $-$ CQS"),
    (r"NGSIM-natural, three of six clients per round", None, None),
    ("ngsim", "half", "cqs-frac0.5|fedavg-frac0.5", r"CQS $-$ FedAvg"),
    ("ngsim", "half", "fedavg-udr-frac0.5|fedavg-frac0.5", r"FedAvg+UDR $-$ FedAvg"),
    ("ngsim", "half", "cqs-udr-frac0.5|cqs-frac0.5", r"CQS+UDR $-$ CQS"),
    (r"highD-motion, five of ten clients per round", None, None),
    ("highd", "half", "cqs-frac0.5|fedavg-frac0.5", r"CQS $-$ FedAvg"),
    ("highd", "half", "fedavg-udr-frac0.5|fedavg-frac0.5", r"FedAvg+UDR $-$ FedAvg"),
    ("highd", "half", "cqs-udr-frac0.5|cqs-frac0.5", r"CQS+UDR $-$ CQS"),
    (r"NGSIM-natural, five local epochs", None, None),
    ("ngsim", "pacing", "cqs-cosine-E5|cqs-E5", r"Cosine $-$ linear quotas"),
]


def extra_table():
    lines = []
    for row in EXTRA_ROWS:
        if row[1] is None:
            if lines:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{@{}l}{\textit{" + row[0] + r"}} \\")
            continue
        s, blk, key, label = row
        p5, pm = ST[s][blk][key]["5s"], ST[s][blk][key]["mean"]
        c5, cm = pcell(p5), pcell(pm)
        lines.append(" & ".join([label, c5[0], c5[1], c5[2], cm[0]]) + r" \\")
    (OUT / "tab_extra.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


SENS_ROWS = [
    ("cqs-f0.4-udr", r"$q_{\min}=0.4$"), ("cqs-f0.8-udr", r"$q_{\min}=0.8$"),
    ("cqs-w0-udr", r"$T_w=0$"), ("cqs-w10-udr", r"$T_w=10$"),
    ("cqs-udr-t0.5", r"$\tau_d=0.5$"), ("cqs-udr-t2", r"$\tau_d=2$"),
    ("cqs-udr-b0.7-1.3", r"$[s_{\min},s_{\max}]=[0.7,1.3]$"), ("cqs-udr-b0.3-1.7", r"$[s_{\min},s_{\max}]=[0.3,1.7]$"),
]


def sens_table():
    lines = []
    cells = [r"Default, three runs"]
    for s in ("ngsim", "highd"):
        b = ST[s]["sensitivity"]["band"]
        cells += [f"${b['best_val_min']:.3f}$--${b['best_val_max']:.3f}$",
                  f"${b['test_5s_min']:.3f}$--${b['test_5s_max']:.3f}$"]
    lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for cfg, label in SENS_ROWS:
        cells = [label]
        for s in ("ngsim", "highd"):
            v = ST[s]["sensitivity"][cfg]
            cells += [f"${v['best_val']:.3f}$", f"${v['test_5s']:.3f}$"]
        lines.append(" & ".join(cells) + r" \\")
    (OUT / "tab_sens.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_table():
    N, H = S["datasets"]["fedtraj_ngsim"], S["datasets"]["fedtraj_highd"]
    DN, DH = S["difficulty"]["fedtraj_ngsim"]["natural"], S["difficulty"]["fedtraj_highd"]["motion"]
    pn, ph = N["partitions"]["natural"], H["partitions"]["motion"]

    def big(x):
        return f"{x:,}".replace(",", "{,}")

    rows = [
        ("Source", "US-101, I-80; six periods", "60 recordings; six sites"),
        ("Sampling", r"10\,Hz", r"25\,Hz resampled to 10\,Hz"),
        ("Vehicles", f"${big(N['vehicles'])}$", f"${big(H['vehicles'])}$"),
        ("Window stride", r"$1.5$\,s", r"$1.0$\,s"),
        ("Training windows", f"${big(pn['n_train'])}$", f"${big(ph['n_train'])}$"),
        ("Validation / test windows", f"${big(pn['n_valid'])}$ / ${big(pn['n_test'])}$",
         f"${big(ph['n_valid'])}$ / ${big(ph['n_test'])}$"),
        ("Clients", f"{pn['K']} periods", f"{ph['K']} motion clusters"),
        ("Client share of windows", f"${100 * pn['share_min']:.1f}$--${100 * pn['share_max']:.1f}\\%$",
         f"${100 * ph['share_min']:.1f}$--${100 * ph['share_max']:.1f}\\%$"),
        (r"$K_{\rm eff}$ of sample weights", f"${pn['keff_size']:.2f}$", f"${ph['keff_size']:.2f}$"),
        (r"Client median CV loss (m$^2$)", f"${DN['median_min']:.2f}$--${DN['median_max']:.2f}$",
         f"${DH['median_min']:.2f}$--${DH['median_max']:.2f}$"),
        (r"Within-client P95/P5 of CV loss", f"${DN['p95_over_p5_min']:.0f}$--${DN['p95_over_p5_max']:.0f}$",
         f"${DH['p95_over_p5_min']:.0f}$--${DH['p95_over_p5_max']:.0f}$"),
        (r"Loss share of top 20\% windows", f"${100 * DN['top20_share_min']:.0f}$--${100 * DN['top20_share_max']:.0f}\\%$",
         f"${100 * DH['top20_share_min']:.0f}$--${100 * DH['top20_share_max']:.0f}\\%$"),
        (r"CV test RMSE at $5$\,s (m)", f"${N['cv']['test']['5s']:.3f}$", f"${H['cv']['test']['5s']:.3f}$"),
    ]
    lines = [" & ".join(r) + r" \\" for r in rows]
    (OUT / "tab_data.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


main_table()
pair_table()
extra_table()
sens_table()
data_table()
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
