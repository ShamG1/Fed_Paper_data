"""Write or verify MANIFEST.json (size + SHA-256 of every file) of a dataset directory.

    python -m fedtraj.manifest write  data/fedtraj_ngsim
    python -m fedtraj.manifest verify data/fedtraj_ngsim      # on the server after upload
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "verify"):
        raise SystemExit(__doc__)
    d = Path(sys.argv[2])
    files = sorted(p for p in d.iterdir() if p.is_file() and p.name not in ("MANIFEST.json", "stats_report.json"))
    if sys.argv[1] == "write":
        man = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
        (d / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        print(f"wrote {d / 'MANIFEST.json'} ({len(man)} files)")
        return
    man = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = []
    for name, info in man.items():
        p = d / name
        if not p.exists() or p.stat().st_size != info["bytes"] or digest(p) != info["sha256"]:
            bad.append(name)
    print("OK" if not bad else f"MISMATCH: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
