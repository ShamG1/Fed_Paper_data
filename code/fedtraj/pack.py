"""Build the two upload files for the training server.

    python -m fedtraj.pack                    # -> upload_fedtraj/fedtraj_code.tar.gz, upload_fedtraj/fedtraj_data.tar
    python -m fedtraj.pack --code-only        # after a code change: only the small code archive

On the server, unpack both in one project directory (``tar xzf fedtraj_code.tar.gz; tar xf fedtraj_data.tar``)
and run every command from there.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ["data/fedtraj_ngsim", "data/fedtraj_highd"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "upload_fedtraj"))
    ap.add_argument("--code-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    code = out / "fedtraj_code.tar.gz"
    with tarfile.open(code, "w:gz") as tar:
        for f in sorted((ROOT / "fedtraj").iterdir()):
            if f.is_file() and f.suffix in (".py", ".md"):
                tar.add(f, arcname=f"fedtraj/{f.name}")
    print(f"wrote {code} ({code.stat().st_size / 1024:.0f} KB)")
    if args.code_only:
        return

    data = out / "fedtraj_data.tar"
    with tarfile.open(data, "w") as tar:
        for d in DATA:
            src = ROOT / d
            if not (src / "MANIFEST.json").exists():
                raise SystemExit(f"{src} has no MANIFEST.json: run `python -m fedtraj.manifest write {d}` first")
            for f in sorted(src.iterdir()):
                if f.is_file():
                    tar.add(f, arcname=f"{d}/{f.name}")
    print(f"wrote {data} ({data.stat().st_size / 2 ** 30:.2f} GB)")


if __name__ == "__main__":
    main()
