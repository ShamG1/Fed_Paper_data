"""Split a processed fedtraj dataset into tar.gz parts for Git LFS, or check and unpack the parts.

Run from the repository root:

    python code/tools/dataset_parts.py extract fedtraj_ngsim
        Checks the SHA-256 of every part of data/fedtraj_ngsim.tar.gz.part* against SHA256SUMS.txt, streams the
        concatenated parts through tar into code/data/fedtraj_ngsim/, and checks every unpacked file against the
        size and SHA-256 recorded in its MANIFEST.json. Training commands are then run from code/
        (`python -m fedtraj.train --data-dir data/fedtraj_ngsim ...`), as in code/fedtraj/README.md.

    python code/tools/dataset_parts.py pack <dataset dir> --out data
        How the parts were made: tar of every file of the dataset directory under data/<name>/, gzip level 6,
        split into parts of at most 256 MiB. Prints the SHA-256 of every part and of the joined archive.

On Linux and macOS the parts can also be unpacked with `cat data/<name>.tar.gz.part* | tar -xzf - -C code`.
Needs only the Python standard library (3.8 or later).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PART_BYTES = 256 * 2 ** 20
LFS_POINTER = b"version https://git-lfs"


class SplitWriter(io.RawIOBase):
    """Write-only stream that cuts its bytes into <prefix>.partNN files of at most ``size`` bytes."""

    def __init__(self, prefix: Path, size: int):
        self.prefix, self.size = prefix, size
        self.parts, self.fh, self.left = [], None, 0
        self.whole = hashlib.sha256()

    def writable(self):
        return True

    def _next(self):
        if self.fh:
            self.fh.close()
        p = self.prefix.with_name(f"{self.prefix.name}.part{len(self.parts) + 1:02d}")
        self.parts.append(p)
        self.fh, self.left = open(p, "wb"), self.size

    def write(self, b):
        b = memoryview(b)
        self.whole.update(b)
        done = 0
        while done < len(b):
            if self.fh is None or self.left == 0:
                self._next()
            k = min(self.left, len(b) - done)
            self.fh.write(b[done:done + k])
            self.left -= k
            done += k
        return len(b)

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None
        super().close()


class ConcatReader(io.RawIOBase):
    """Read-only stream over several files in order; hashes the bytes it returns."""

    def __init__(self, paths):
        self.paths, self.i, self.fh = list(paths), 0, None
        self.whole = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buf):
        while self.i < len(self.paths):
            if self.fh is None:
                self.fh = open(self.paths[self.i], "rb")
            n = self.fh.readinto(buf)
            if n:
                self.whole.update(memoryview(buf)[:n])
                return n
            self.fh.close()
            self.fh, self.i = None, self.i + 1
        return 0


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def clean(ti):
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o755 if ti.isdir() else 0o644
    return ti


def pack(src: Path, out: Path, part_bytes: int):
    name = src.name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{name}.tar.gz.part*"):
        old.unlink()
    sw = SplitWriter(out / f"{name}.tar.gz", part_bytes)
    with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=sw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for f in sorted(p for p in src.iterdir() if p.is_file()):
                tar.add(f, arcname=f"data/{name}/{f.name}", filter=clean)
    sw.close()
    for p in sw.parts:
        print(f"{sha256(p)}  data/{p.name}")
    print(f"{sw.whole.hexdigest()}  data/{name}.tar.gz  (joined parts, {sum(p.stat().st_size for p in sw.parts)} bytes)")


def read_sums(root: Path):
    sums = {}
    f = root / "SHA256SUMS.txt"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                h, path = line.split(None, 1)
                sums[path.strip().split("  (")[0]] = h
    return sums


def extract(name: str, root: Path, dest: Path):
    parts = sorted((root / "data").glob(f"{name}.tar.gz.part*"))
    if not parts:
        sys.exit(f"no parts data/{name}.tar.gz.part* under {root}")
    for p in parts:
        with open(p, "rb") as fh:
            if fh.read(len(LFS_POINTER)) == LFS_POINTER:
                sys.exit(f"{p.name} is a Git LFS pointer: run  git lfs pull --include \"data/{name}.tar.gz.part*\"")
    sums = read_sums(root)
    for p in parts:
        want = sums.get(f"data/{p.name}")
        if want is None:
            sys.exit(f"data/{p.name} is not listed in SHA256SUMS.txt")
        if sha256(p) != want:
            sys.exit(f"SHA-256 mismatch: data/{p.name}")
    print(f"{len(parts)} parts of {name}: SHA-256 OK")
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"data/{name}/"
    reader = ConcatReader(parts)
    with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 20), mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.startswith(prefix) or "/" in m.name[len(prefix):] or ".." in m.name:
                sys.exit(f"unexpected member {m.name!r}")
            if hasattr(tarfile, "data_filter"):
                tar.extract(m, dest, filter="data")
            else:
                tar.extract(m, dest)
    want = sums.get(f"data/{name}.tar.gz")
    if want and reader.whole.hexdigest() != want:
        sys.exit(f"SHA-256 mismatch of the joined archive data/{name}.tar.gz")
    target = dest / "data" / name
    man = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [f for f, info in man.items()
           if not (target / f).exists() or (target / f).stat().st_size != info["bytes"] or sha256(target / f) != info["sha256"]]
    if bad:
        sys.exit(f"files differ from MANIFEST.json: {bad}")
    print(f"unpacked {target}: {len(man)} files match MANIFEST.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack")
    a.add_argument("src")
    a.add_argument("--out", default=str(ROOT / "data"))
    a.add_argument("--part-mib", type=int, default=PART_BYTES // 2 ** 20)
    b = sub.add_parser("extract")
    b.add_argument("name", help="fedtraj_ngsim or fedtraj_highd")
    b.add_argument("--root", default=str(ROOT), help="repository root (parts in <root>/data, sums in <root>/SHA256SUMS.txt)")
    b.add_argument("--dest", default=None, help="directory that receives data/<name>/ (default <root>/code)")
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(Path(args.src), Path(args.out), args.part_mib * 2 ** 20)
    else:
        root = Path(args.root)
        extract(args.name, root, Path(args.dest) if args.dest else root / "code")


if __name__ == "__main__":
    main()
