"""Build and check the offline wheel/model bundle for the Windows laptop.

Runs on any OS with Python 3.14 (stdlib only) plus uv, either on PATH or
importable as `python -m uv`. Target: CPython 3.14, win_amd64.

    python llm-kit/tools/build_bundle.py lock            # requirements.in -> wheels/requirements.lock
    python llm-kit/tools/build_bundle.py fetch [--dry-run] [--only NAME ...]
    python llm-kit/tools/build_bundle.py models [--dry-run] [--only FILE ...]
    python llm-kit/tools/build_bundle.py verify [WHEELS_DIR]

Resolution always goes through uv with --python-platform x86_64-pc-windows-msvc
and --python-version 3.14, so environment markers (sys_platform == "win32",
python_version, ...) are evaluated for the laptop, not for the build machine.
`pip download --platform` does not do that, which is how colorama went missing
once. `fetch` downloads exactly the lock's pins with no dependency resolution
of its own.

Files over 95 MB are stored as .partNNN chunks plus a .sha256 sidecar
(split_tool.py). `verify` joins such chunks into a temporary directory and
never modifies the directory it checks.
"""

from __future__ import annotations

import argparse, ast, hashlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile
import urllib.error, urllib.parse, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(KIT))
import split_tool  # noqa: E402

WHEELS = KIT / "wheels"
MODELS = KIT / "models"
REQUIREMENTS_IN = HERE / "requirements.in"
LOCK = WHEELS / "requirements.lock"
SUMS_NAME = "SHA256SUMS"

PLATFORM = "x86_64-pc-windows-msvc"
PY_VERSION = "3.14"
PY_NODOT = "314"
SPLIT_MB = 95
SPLIT_BYTES = SPLIT_MB * 1024 * 1024

JAMEPENG_REPO = "JamePeng/llama-cpp-python"
JAMEPENG_TAG = "v0.4.2-cu128-win-20261003"
JAMEPENG_FIND_LINKS = f"https://github.com/{JAMEPENG_REPO}/releases/expanded_assets/{JAMEPENG_TAG}"
JAMEPENG_API = f"https://api.github.com/repos/{JAMEPENG_REPO}/releases/tags/{JAMEPENG_TAG}"
JAMEPENG_PACKAGES = {"llama-cpp-python"}

USER_AGENT = "llm-kit-build-bundle/1"


# ---------- small helpers ----------

def say(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    say(f"ERROR: {msg}")
    sys.exit(1)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(pathlib.Path.cwd()))
    except ValueError:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    return split_tool.sha256(path)


def sha256_parts(parts: list[pathlib.Path]) -> str:
    h = hashlib.sha256()
    for q in parts:
        with q.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
    return h.hexdigest()


def parts_of(directory: pathlib.Path, name: str) -> list[pathlib.Path]:
    return sorted(directory.glob(f"{glob_escape(name)}.part[0-9][0-9][0-9]"))


def glob_escape(name: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", name)


def uv_cmd() -> list[str]:
    exe = shutil.which("uv")
    if exe:
        return [exe]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True, text=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "uv"]
    die("uv not found. Install it (https://docs.astral.sh/uv/) or `python -m pip install uv`.")
    raise AssertionError


def run_uv(args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [*uv_cmd(), *args]
    say("  + uv " + " ".join(args))
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.setdefault("UV_LINK_MODE", "copy")
    return subprocess.run(cmd, env=env, text=True, capture_output=capture)


def http_get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)


def http_json(url: str) -> dict:
    headers = {"Accept": "application/json"}
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    with http_get(url, headers) as r:
        return json.load(r)


def download(url: str, dest: pathlib.Path, size: int | None, headers: dict | None = None) -> None:
    """Download url to dest, resuming a previous partial download (dest + '.download')."""
    tmp = dest.with_name(dest.name + ".download")
    have = tmp.stat().st_size if tmp.exists() else 0
    if size is not None and have > size:
        tmp.write_bytes(b"")
        have = 0
    if size is not None and have == size:
        tmp.replace(dest)
        return
    h = dict(headers or {})
    if have:
        h["Range"] = f"bytes={have}-"
    with http_get(url, h) as r:
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        total = size or int(r.headers.get("Content-Length") or 0) + have
        done, last = have, -1
        with tmp.open(mode) as out:
            for block in iter(lambda: r.read(1 << 20), b""):
                out.write(block)
                done += len(block)
                pct = int(done * 100 / total) if total else 0
                if total > 50_000_000 and pct != last and pct % 5 == 0:
                    say(f"    {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB ({pct}%)")
                    last = pct
    tmp.replace(dest)


# ---------- wheel tags ----------

WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<ver>[^-]+)(-(?P<build>\d[^-]*))?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$"
)


def wheel_tags(filename: str) -> list[tuple[str, str, str]] | None:
    m = WHEEL_RE.match(filename)
    if not m:
        return None
    return [(p, a, pl) for p in m["py"].split(".") for a in m["abi"].split(".") for pl in m["plat"].split(".")]


def supported_tags() -> list[tuple[str, str, str]]:
    """Tags CPython 3.14 on win_amd64 accepts, most specific first (same order as pip)."""
    cp = f"cp{PY_NODOT}"
    minor = int(PY_NODOT[1:])
    tags = [(cp, cp, "win_amd64")]
    tags += [(f"cp3{m}", "abi3", "win_amd64") for m in range(minor, 1, -1)]
    tags += [(cp, "none", "win_amd64")]
    pys = [f"py{PY_NODOT}", "py3"] + [f"py3{m}" for m in range(minor - 1, -1, -1)]
    tags += [(p, "none", "win_amd64") for p in pys]
    tags += [(cp, "none", "any")]
    tags += [(p, "none", "any") for p in pys]
    return tags


TAG_RANK = {t: i for i, t in enumerate(supported_tags())}


def wheel_rank(filename: str) -> int | None:
    """Lower is better; None if the wheel cannot be installed on cp314 win_amd64."""
    tags = wheel_tags(filename)
    if not tags:
        return None
    ranks = [TAG_RANK[t] for t in tags if t in TAG_RANK]
    return min(ranks) if ranks else None


def wheel_name_version(filename: str) -> tuple[str, str] | None:
    m = WHEEL_RE.match(filename)
    return (norm(m["name"]), m["ver"]) if m else None


# ---------- lock file ----------

def read_lock(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        die(f"{rel(path)} not found. Run: python {rel(HERE / 'build_bundle.py')} lock")
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$", line)
        if not m:
            die(f"{rel(path)}: cannot parse line {line!r} (expected name==version)")
        pins[norm(m[1])] = m[2]
    return pins


def read_requirements(path: pathlib.Path) -> list[str]:
    reqs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(line)
    return reqs


def kit_packages() -> list[str] | None:
    """PACKAGES from kit.py, read without importing it."""
    try:
        tree = ast.parse((KIT / "kit.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PACKAGES" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except ValueError:
                return None
    return None


# ---------- lock ----------

def cmd_lock(a) -> None:
    out = pathlib.Path(a.output)
    say(f"Resolving {rel(a.requirements)} for CPython {PY_VERSION} on {PLATFORM}")
    tmp = out.with_name(out.name + ".new")
    # uv keeps versions already pinned in --output-file unless told to upgrade.
    # Seed it with the current lock, or with the versions of the shipped wheels.
    seed = ""
    if not a.upgrade:
        if out.exists():
            seed = out.read_text()
            say(f"  keeping pins from {rel(out)} where possible (--upgrade to take newer versions)")
        elif (out.parent / SUMS_NAME).exists():
            lines = []
            for fn in read_sums(out.parent / SUMS_NAME):
                nv = wheel_name_version(fn)
                if nv:
                    lines.append(f"{nv[0]}=={nv[1]}\n")
            seed = "".join(sorted(lines))
            say(f"  keeping versions of the shipped wheels in {SUMS_NAME} where possible (--upgrade to take newer)")
    tmp.write_text(seed)
    r = run_uv([
        "pip", "compile", str(a.requirements),
        "--python-platform", PLATFORM, "--python-version", PY_VERSION,
        "--find-links", JAMEPENG_FIND_LINKS,
        "--no-build", "--no-config", "--quiet",
        "--custom-compile-command", "python llm-kit/tools/build_bundle.py lock",
        "--output-file", str(tmp),
    ])
    if r.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        die("uv could not resolve requirements.in for Windows/3.14 (message above). Lock file unchanged.")
    tmp.replace(out)
    pins = read_lock(out)
    say(f"Wrote {rel(out)}: {len(pins)} pinned packages.")


# ---------- fetch ----------

def pypi_wheel(name: str, version: str) -> dict:
    data = http_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/json")
    best = None
    for f in data.get("urls", []):
        if f.get("packagetype") != "bdist_wheel" or f.get("yanked"):
            continue
        rank = wheel_rank(f["filename"])
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, f)
    if not best:
        files = ", ".join(f["filename"] for f in data.get("urls", [])) or "none"
        raise LookupError(f"{name}=={version}: no wheel for cp{PY_NODOT} win_amd64 on PyPI (files: {files})")
    f = best[1]
    return {"filename": f["filename"], "url": f["url"], "size": f["size"], "sha256": f["digests"]["sha256"],
            "source": "PyPI"}


_jamepeng_cache: list | None = None


def jamepeng_wheel(name: str, version: str) -> dict:
    global _jamepeng_cache
    if _jamepeng_cache is None:
        _jamepeng_cache = http_json(JAMEPENG_API).get("assets", [])
    best = None
    for asset in _jamepeng_cache:
        nv = wheel_name_version(asset["name"])
        if not nv or nv != (norm(name), version):
            continue
        rank = wheel_rank(asset["name"])
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, asset)
    if not best:
        raise LookupError(f"{name}=={version}: no cp{PY_NODOT} win_amd64 wheel in {JAMEPENG_REPO} {JAMEPENG_TAG}")
    asset = best[1]
    digest = (asset.get("digest") or "").removeprefix("sha256:")
    if not digest:
        raise LookupError(f"{asset['name']}: release asset has no sha256 digest")
    return {"filename": asset["name"], "url": asset["browser_download_url"], "size": asset["size"],
            "sha256": digest, "source": f"{JAMEPENG_REPO} {JAMEPENG_TAG}"}


def stored_state(dest: pathlib.Path, filename: str, sha: str, size: int | None) -> str | None:
    """'whole' / 'chunks' if the file is already present and verified, else None."""
    whole = dest / filename
    if whole.exists() and (size is None or whole.stat().st_size == size) and sha256_file(whole) == sha:
        return "whole"
    side = dest / f"{filename}.sha256"
    parts = parts_of(dest, filename)
    if side.exists() and parts and side.read_text().split()[0] == sha:
        if (size is None or sum(p.stat().st_size for p in parts) == size) and sha256_parts(parts) == sha:
            return "chunks"
    return None


def cmd_fetch(a) -> None:
    dest = pathlib.Path(a.dest)
    pins = read_lock(pathlib.Path(a.lock))
    only = {norm(n) for n in a.only or []}
    unknown = only - set(pins)
    if unknown:
        die(f"--only names not in the lock: {', '.join(sorted(unknown))}")
    dest.mkdir(parents=True, exist_ok=True)
    plan, errors = [], []
    for name, version in sorted(pins.items()):
        try:
            info = jamepeng_wheel(name, version) if name in JAMEPENG_PACKAGES else pypi_wheel(name, version)
        except (LookupError, urllib.error.URLError, KeyError) as e:
            errors.append(str(e))
            continue
        info["name"] = name
        plan.append(info)
    if errors:
        for e in errors:
            say(f"  FAIL {e}")
        die(f"{len(errors)} pinned package(s) have no usable Windows cp{PY_NODOT} wheel. Nothing downloaded.")

    say(f"{len(plan)} pinned wheels for cp{PY_NODOT} win_amd64 -> {rel(dest)}")
    digests = {}
    total = 0
    for info in plan:
        fn, sha, size = info["filename"], info["sha256"], info["size"]
        selected = not only or info["name"] in only
        state = stored_state(dest, fn, sha, size) if selected or not a.dry_run else None
        big = size > SPLIT_BYTES
        if a.dry_run:
            if not selected:
                continue
            action = f"have ({state})" if state else ("download + split" if big else "download")
            say(f"  {action:<17} {fn}  {size / 1e6:,.1f} MB  sha256 {sha[:12]}  [{info['source']}]")
            say(f"                    {info['url']}")
            if not state:
                total += size
            continue
        if selected and not state:
            say(f"  get  {fn} ({size / 1e6:,.1f} MB) from {info['source']}")
            target = dest / fn
            download(info["url"], target, size)
            got = sha256_file(target)
            if got != sha:
                bad = target.with_name(fn + ".bad")
                target.replace(bad)
                die(f"{fn}: sha256 {got} does not match {info['source']} digest {sha} (kept as {rel(bad)})")
            state = "whole"
        elif selected:
            say(f"  have {fn} ({state}, verified)")
        if state == "whole" and big:
            split_tool.split(dest / fn, SPLIT_MB, dest)
            (dest / fn).unlink()
        digests[fn] = sha

    if a.dry_run:
        say(f"Would download {total / 1e6:,.1f} MB. (dry run: nothing written)")
        return

    # SHA256SUMS lists every pinned wheel. Ones not fetched in this run (--only)
    # must already be present; their digest is read from the files themselves.
    missing = []
    for info in plan:
        fn = info["filename"]
        if fn in digests:
            continue
        if stored_state(dest, fn, info["sha256"], info["size"]):
            digests[fn] = info["sha256"]
        else:
            missing.append(fn)
    if missing:
        die("not present or not matching the lock (fetch without --only to get them): " + ", ".join(missing))
    sums = dest / SUMS_NAME
    sums.write_text("".join(f"{digests[fn]}  {fn}\n" for fn in sorted(digests)))
    say(f"Wrote {rel(sums)} ({len(digests)} wheels).")

    pinned = {i["filename"] for i in plan}
    stale = sorted(p.name for p in dest.glob("*.whl") if p.name not in pinned)
    stale += sorted(p.name[:-len(".sha256")] for p in dest.glob("*.whl.sha256")
                    if p.name[:-len(".sha256")] not in pinned)
    if stale:
        say("Not in the lock any more (remove these by hand, with their .partNNN chunks if any):")
        for s in stale:
            say(f"  {s}")


# ---------- models ----------

def manifest_rows(path: pathlib.Path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        repo, name, size, sha = line.split("\t")
        rows.append((repo, name, int(size), sha))
    return rows


def model_state(folder: pathlib.Path, name: str, size: int, sha: str) -> str | None:
    return stored_state(folder, name, sha, size)


def cmd_models(a) -> None:
    root = pathlib.Path(a.models)
    rows = manifest_rows(root / "MANIFEST.txt")
    only = set(a.only or [])
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    total = 0
    for repo, name, size, sha in rows:
        folder = root / repo.split("/")[1]
        if only and name not in only and f"{folder.name}/{name}" not in only:
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(name)}"
        state = model_state(folder, name, size, sha)
        if a.dry_run:
            partial = folder / f"{name}.download"
            have = partial.stat().st_size if partial.exists() else 0
            action = f"have ({state})" if state else ("resume" if have else "download")
            say(f"  {action:<13} {folder.name}/{name}  {size / 1e9:.2f} GB  sha256 {sha[:12]}")
            say(f"                {url}")
            if not state:
                total += size - have
            continue
        if state:
            say(f"  have {folder.name}/{name} ({state}, verified)")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            say(f"  get  {folder.name}/{name} ({size / 1e9:.2f} GB)")
            target = folder / name
            download(url, target, size, headers)
            if target.stat().st_size != size:
                die(f"{name}: size {target.stat().st_size} != manifest {size}")
            got = sha256_file(target)
            if got != sha:
                bad = target.with_name(name + ".bad")
                target.replace(bad)
                die(f"{name}: sha256 {got} != manifest {sha} (kept as {rel(bad)})")
            state = "whole"
        if state == "whole" and size > SPLIT_BYTES:
            split_tool.split(folder / name, SPLIT_MB, folder)
            (folder / name).unlink()
    if a.dry_run:
        say(f"Would download {total / 1e9:.2f} GB. (dry run: nothing written)")


# ---------- verify ----------

def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_sums(path: pathlib.Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, fn = line.split(None, 1)
            sums[fn.strip().lstrip("*")] = digest.lower()
    return sums


def cmd_verify(a) -> None:
    src = pathlib.Path(a.wheels).resolve()
    reqs = pathlib.Path(a.requirements).resolve()
    if not src.is_dir():
        die(f"{src} is not a directory")
    sums_path = pathlib.Path(a.sums).resolve() if a.sums else src / SUMS_NAME
    lock_path = pathlib.Path(a.lock).resolve() if a.lock else src / LOCK.name
    errors: list[str] = []
    warnings: list[str] = []

    say(f"Checking wheels in {src}")
    say(f"  requirements: {reqs}")
    say(f"  checksums:    {sums_path}")

    kp = kit_packages()
    if kp is not None:
        listed = {norm(re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0]) for r in read_requirements(reqs)}
        absent = [p for p in kp if norm(p) not in listed]
        if absent:
            errors.append(f"kit.py PACKAGES not in {reqs.name}: {', '.join(absent)}")

    work_parent = pathlib.Path(a.work_dir).resolve() if a.work_dir else None
    if work_parent:
        work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llmkit-verify-", dir=work_parent) as tmp:
        tmp = pathlib.Path(tmp)
        flat = tmp / "wheels"
        flat.mkdir()
        # 1. Gather whole wheels; rebuild chunked ones in the temp dir.
        for w in sorted(src.glob("*.whl")):
            link_or_copy(w, flat / w.name)
        for first in sorted(src.glob("*.whl.part000")):
            fn = first.name[: -len(".part000")]
            if (flat / fn).exists():
                continue
            parts = parts_of(src, fn)
            nums = [int(split_tool.CHUNK_RE.match(q.name)["n"]) for q in parts]
            if nums != list(range(len(parts))):
                errors.append(f"{fn}: chunk numbers have gaps: {nums}")
                continue
            say(f"  joining {fn} from {len(parts)} chunks (temp copy)")
            with (flat / fn).open("wb") as out:
                for q in parts:
                    with q.open("rb") as f:
                        shutil.copyfileobj(f, out, 1 << 24)
            side = src / f"{fn}.sha256"
            if not side.exists():
                errors.append(f"{fn}: chunks have no .sha256 sidecar")
            elif sha256_file(flat / fn) != side.read_text().split()[0]:
                errors.append(f"{fn}: joined chunks do not match {side.name}")
        for side in sorted(src.glob("*.whl.sha256")):
            fn = side.name[: -len(".sha256")]
            if not (flat / fn).exists():
                errors.append(f"{fn}: has a .sha256 sidecar but neither the file nor its chunks are present")

        wheels = sorted(flat.glob("*.whl"))
        say(f"  {len(wheels)} wheels found")

        # 2. Tags: every wheel must install on CPython 3.14 / win_amd64.
        for w in wheels:
            if wheel_rank(w.name) is None:
                errors.append(f"{w.name}: tag is not installable on cp{PY_NODOT} win_amd64 "
                              f"(need cp{PY_NODOT}-cp{PY_NODOT}, cp3x-abi3, or py3-none, on win_amd64 or any)")

        # 3. Duplicate versions of one package make the result depend on resolver choice.
        seen: dict[str, list[str]] = {}
        for w in wheels:
            nv = wheel_name_version(w.name)
            if nv:
                seen.setdefault(nv[0], []).append(w.name)
        for name, files in sorted(seen.items()):
            if len(files) > 1:
                warnings.append(f"several wheels for {name}: {', '.join(files)}")

        # 4. Checksums.
        if not sums_path.exists():
            errors.append(f"{sums_path} not found")
        else:
            sums = read_sums(sums_path)
            for w in wheels:
                want = sums.get(w.name)
                if want is None:
                    errors.append(f"{w.name}: not listed in {SUMS_NAME}")
                elif sha256_file(w) != want:
                    errors.append(f"{w.name}: sha256 does not match {SUMS_NAME}")
            for fn in sorted(set(sums) - {w.name for w in wheels}):
                errors.append(f"{fn}: listed in {SUMS_NAME} but missing from the bundle")
            say(f"  checksums compared against {len(sums)} {SUMS_NAME} entries")

        # 5. Marker-correct offline install for the target.
        target = tmp / "site"
        r = run_uv([
            "pip", "install", "--target", str(target),
            "--python-platform", PLATFORM, "--python-version", PY_VERSION,
            "--python", sys.executable,
        "--no-index", "--find-links", str(flat), "--no-build", "--no-config", "--no-cache",
            "-r", str(reqs),
        ], capture=True)
        out = (r.stdout + r.stderr).strip()
        installed: dict[str, str] = {}
        if r.returncode != 0:
            errors.append("offline install for Windows/3.14 failed:\n" + "\n".join("      " + x for x in out.splitlines()))
        else:
            for d in target.glob("*.dist-info"):
                meta = (d / "METADATA").read_text(encoding="utf-8", errors="replace")
                name = re.search(r"^Name:\s*(.+)$", meta, re.M)[1].strip()
                ver = re.search(r"^Version:\s*(.+)$", meta, re.M)[1].strip()
                installed[norm(name)] = ver
            say(f"  offline install for Windows/3.14 OK: {len(installed)} packages")
            used = set(installed)
            unused = sorted(n for n in seen if n not in used)
            if unused:
                warnings.append("wheels not needed by requirements.in: " + ", ".join(unused))

        # 6. The bundle should be exactly the lock (if one is present).
        if lock_path.exists() and installed:
            pins = read_lock(lock_path)
            for name, ver in sorted(pins.items()):
                if name not in installed:
                    errors.append(f"{name}=={ver} is in {lock_path.name} but was not installed")
                elif installed[name] != ver:
                    errors.append(f"{name}: installed {installed[name]}, {lock_path.name} pins {ver}")
            for name in sorted(set(installed) - set(pins)):
                errors.append(f"{name}=={installed[name]} installed but not in {lock_path.name}")
        elif not lock_path.exists():
            warnings.append(f"no {lock_path.name} next to the wheels; lock comparison skipped")

    for w in warnings:
        say(f"  warning: {w}")
    if errors:
        say(f"\nFAILED: {len(errors)} problem(s):")
        for e in errors:
            say(f"  - {e}")
        sys.exit(1)
    say(f"\nOK: {len(wheels)} wheels verified; {len(installed)} packages install offline on CPython {PY_VERSION} win_amd64.")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lock", help="resolve requirements.in for Windows/3.14 into wheels/requirements.lock")
    p.add_argument("--requirements", type=pathlib.Path, default=REQUIREMENTS_IN)
    p.add_argument("--output", type=pathlib.Path, default=LOCK)
    p.add_argument("--upgrade", action="store_true", help="ignore existing pins and take the newest versions")
    p.set_defaults(fn=cmd_lock)

    p = sub.add_parser("fetch", help="download the lock's wheels for cp314 win_amd64 and rewrite SHA256SUMS")
    p.add_argument("--lock", type=pathlib.Path, default=LOCK)
    p.add_argument("--dest", type=pathlib.Path, default=WHEELS)
    p.add_argument("--only", nargs="+", metavar="NAME", help="fetch only these packages (others must be present)")
    p.add_argument("--dry-run", action="store_true", help="print what would be downloaded; write nothing")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("models", help="download, verify and split the models in models/MANIFEST.txt")
    p.add_argument("--models", type=pathlib.Path, default=MODELS)
    p.add_argument("--only", nargs="+", metavar="FILE", help="only these files (name or folder/name)")
    p.add_argument("--dry-run", action="store_true", help="print what would be downloaded; write nothing")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("verify", help="check a wheels dir installs offline on Windows/3.14 and matches SHA256SUMS")
    p.add_argument("wheels", nargs="?", type=pathlib.Path, default=WHEELS)
    p.add_argument("--requirements", type=pathlib.Path, default=REQUIREMENTS_IN)
    p.add_argument("--sums", type=pathlib.Path, help=f"checksum file (default: WHEELS/{SUMS_NAME})")
    p.add_argument("--lock", type=pathlib.Path, help=f"lock file (default: WHEELS/{LOCK.name})")
    p.add_argument("--work-dir", type=pathlib.Path, help="parent for the temporary directory")
    p.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
