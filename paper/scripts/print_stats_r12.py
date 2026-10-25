"""Readable dump of data/stats_r12.json (python scripts/print_stats_r12.py > data/stats_r12_summary.txt)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "data" / "stats_r12.json").read_text(encoding="utf-8"))


def ms(c, k, d=4):
    v = c[k]
    sd = "" if v["sd"] is None else f"+-{v['sd']:.{d}f}"
    return f"{v['mean']:.{d}f}{sd}"


def pr(name, p):
    if "sd" not in p:
        return f"  {name:34s} n={p['n']} d={p['d']:+.4f} ({p['rel']:+.2f}%)"
    return (f"  {name:34s} n={p['n']} d={p['d']:+.4f}+-{p['sd']:.4f} ({p['rel']:+.2f}% "
            f"[{p['rel_lo']:+.2f},{p['rel_hi']:+.2f}]) [{p['lo']:+.4f},{p['hi']:+.4f}] "
            f"{p['lower']}/{p['n']} {p['verdict']}")


for key, st in S["settings"].items():
    print(f"\n######## {key}")
    for cfg, c in sorted(st["configs"].items()):
        ke = f" keff={c['keff_all']:.3f}" if "keff_all" in c else ""
        print(f"  {cfg:24s} n={c['5s']['n']} 1s {ms(c, '1s')} 3s {ms(c, '3s')} 5s {ms(c, '5s')} "
              f"mean {ms(c, 'mean')} Q5 {ms(c, 'Q5')} lat {ms(c, 'lateral')} worst {ms(c, 'worst')} "
              f"bestval {ms(c, 'best_val')} rounds {c['best_rounds']}{ke}")
    for blk in ("pairs", "half", "pacing"):
        for name, pk in st.get(blk, {}).items():
            print(f" [{blk}] {name}")
            for k, p in pk.items():
                print(pr(k, p))
    if "server" in st:
        print(" server:", json.dumps(st["server"], indent=None)[:3000])
        print(" gap:", json.dumps(st["gap_central"]))
        print(" visits:", json.dumps(st["visits_rel"]))
        print(" pilot:", json.dumps(st["pilot"]))
        print(" sens:", json.dumps(st["sensitivity"]))
        for cfg, q in st["strata_vs_fedavg"].items():
            print(f"  strata {cfg}: " + "  ".join(f"{k} {v['rel']:+.2f}% ({v['lower']}/{v['n']}, {v.get('verdict')})"
                                                  for k, v in q.items()))
print("\ntiming:", json.dumps(S["timing"], indent=1))
print("difficulty:", json.dumps(S["difficulty"], indent=1))
print("datasets:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'partitions'} for k, v in S["datasets"].items()}))
for ds, v in S["datasets"].items():
    for part, info in v["partitions"].items():
        print(ds, part, {k: info[k] for k in ("K", "n_train", "n_valid", "n_test", "share_min", "share_max", "keff_size")})
