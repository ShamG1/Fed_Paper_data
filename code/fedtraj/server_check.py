"""Check a fresh server before training: packages, GPU, dataset integrity, speed estimate.

    python -m fedtraj.server_check                # full SHA-256 check of the uploaded data (~1 min)
    python -m fedtraj.server_check --quick        # sizes only

Prints the measured time of one training step per dataset and the resulting estimate of one 30-round run
(single job on an otherwise idle GPU; concurrent jobs share the GPU).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_manifest(d, quick):
    from fedtraj.manifest import digest

    man_path = d / "MANIFEST.json"
    if not man_path.exists():
        return f"no MANIFEST.json in {d}"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    bad = []
    for name, info in man.items():
        p = d / name
        if not p.exists() or p.stat().st_size != info["bytes"]:
            bad.append(name)
        elif not quick and digest(p) != info["sha256"]:
            bad.append(name)
    return "OK" if not bad else f"MISMATCH {bad}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="*", default=["data/fedtraj_ngsim", "data/fedtraj_highd"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch

    print(f"python {platform.python_version()}  numpy {np.__version__}  torch {torch.__version__}  "
          f"CUDA build {torch.version.cuda}  cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch: install a CUDA build of PyTorch")
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name}  {prop.total_memory / 2 ** 30:.1f} GB")

    from fedtraj.data import FeatureTransform, WindowStore
    from fedtraj.model import build_model
    from fedtraj.train import scaled_loss, setup_torch

    setup_torch(1, 0, 2)
    dev = torch.device("cuda")
    ok = True
    for d in map(Path, args.data):
        status = check_manifest(d, args.quick)
        print(f"{d}: manifest {status}")
        if status != "OK":
            ok = False
            continue
        t0 = time.time()
        store = WindowStore(d, "natural", dev)
        tf = FeatureTransform(store)
        model = build_model().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        st = store.starts["train"]
        model.train()
        times = []
        for i in range(40):
            torch.cuda.synchronize()
            t = time.time()
            loss = scaled_loss(model, tf, st[i * 512:(i + 1) * 512])[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        step = float(np.median(times[10:]))
        steps_round = sum(int(np.ceil(n / 512)) for n in store.counts("train")) * 3
        est = 30 * steps_round * step * 1.15 / 60          # +15% for scoring and validation
        print(f"   loaded in {time.time() - t0:.0f} s, {store.frames.shape[0]} frames, "
              f"{st.numel()} training windows; GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        print(f"   one training step {step * 1000:.1f} ms -> about {est:.0f} min per 30-round run (single job)")
        del store, tf, model, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("ready" if ok else "NOT ready: fix the data upload first")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
