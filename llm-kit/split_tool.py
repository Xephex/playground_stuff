"""Split large files into GitHub-safe chunks and reassemble them.

Chunks are plain byte slices named <file>.part000, .part001, ... with a
<file>.sha256 sidecar holding the whole-file digest. No compression, no
executables: the inverse is a plain Python concatenation.

    python split_tool.py split  <file> [--size-mb 95] [--out DIR]
    python split_tool.py join   <dir>            # rebuilds every *.part000 group in dir, recursively
    python split_tool.py verify <dir>            # checks every rebuilt file against its .sha256
"""

import argparse, hashlib, pathlib, re, sys

CHUNK_RE = re.compile(r"^(?P<stem>.+)\.part(?P<n>\d{3})$")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def split(path: pathlib.Path, size_mb: int, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    size = size_mb * 1024 * 1024
    digest = sha256(path)
    (out / f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n")
    n = 0
    with path.open("rb") as f:
        while True:
            block = f.read(size)
            if not block:
                break
            (out / f"{path.name}.part{n:03d}").write_bytes(block)
            n += 1
    print(f"split {path.name}: {n} parts, sha256 {digest[:12]}...")


def groups(root: pathlib.Path):
    seen = {}
    for p in root.rglob("*.part000"):
        stem = p.name[: -len(".part000")]
        parts = sorted(p.parent.glob(f"{stem}.part[0-9][0-9][0-9]"))
        seen[p.parent / stem] = parts
    return seen


def join(root: pathlib.Path, delete_parts: bool) -> int:
    bad = 0
    for target, parts in groups(root).items():
        nums = [int(CHUNK_RE.match(q.name)["n"]) for q in parts]
        if nums != list(range(len(parts))):
            print(f"GAP in parts for {target.name}: have {nums}")
            bad += 1
            continue
        if target.exists() and _matches(target):
            print(f"skip {target.name}: already assembled and verified")
            continue
        with target.open("wb") as out:
            for q in parts:
                out.write(q.read_bytes())
        if _matches(target):
            print(f"ok   {target.name} ({target.stat().st_size / 1e9:.2f} GB)")
            if delete_parts:
                for q in parts:
                    q.unlink()
        else:
            print(f"BAD  {target.name}: sha256 mismatch, parts kept")
            bad += 1
    return bad


def _matches(target: pathlib.Path) -> bool:
    side = target.with_name(target.name + ".sha256")
    if not side.exists():
        print(f"no .sha256 for {target.name}; cannot verify")
        return False
    want = side.read_text().split()[0]
    return sha256(target) == want


def verify(root: pathlib.Path) -> int:
    bad = 0
    for side in root.rglob("*.sha256"):
        target = side.with_name(side.name[: -len(".sha256")])
        if not target.exists():
            print(f"missing {target.name}")
            bad += 1
        elif _matches(target):
            print(f"ok   {target.name}")
        else:
            print(f"BAD  {target.name}")
            bad += 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("file", type=pathlib.Path)
    s.add_argument("--size-mb", type=int, default=95)
    s.add_argument("--out", type=pathlib.Path)
    j = sub.add_parser("join")
    j.add_argument("dir", type=pathlib.Path)
    j.add_argument(
        "--keep-parts",
        action="store_true",
        help="do not delete .partNNN files after a verified join",
    )
    v = sub.add_parser("verify")
    v.add_argument("dir", type=pathlib.Path)
    a = ap.parse_args()
    if a.cmd == "split":
        split(a.file, a.size_mb, a.out or a.file.parent)
    elif a.cmd == "join":
        sys.exit(1 if join(a.dir, delete_parts=not a.keep_parts) else 0)
    else:
        sys.exit(1 if verify(a.dir) else 0)


if __name__ == "__main__":
    main()
