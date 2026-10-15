"""llm-kit control tool. The only command you need:

    py -3.14 kit.py            # menu: chat, browser chat, API server, setup, update, status
    py -3.14 kit.py setup      # or one action: setup update status chat web image docs pi serve share tune ask

Runs entirely inside python.exe. Setup deletes the .exe launchers pip
creates for the kit's packages, so none exist to run. The only other
programs it starts are system-installed ones: py (the Python launcher),
git (for update) and nvidia-smi (read-only GPU info, in C:\\Windows\\System32).
"""

import argparse, importlib.metadata as md, json, os, pathlib, runpy, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
WHEELS = ROOT / "wheels"
MODELS = ROOT / "models"
STATE = ROOT / ".kit-state.json"
NEED = (3, 14)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))

# Packages setup installs from wheels\ (names as pip sees them).
PACKAGES = [
    "llama_cpp_python", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12",
    "huggingface_hub", "gradio", "fastapi", "uvicorn", "sse-starlette",
    "starlette-context", "pydantic-settings", "diskcache", "jinja2", "numpy", "pypdf",
]
# What each model is good for, shown in the picker.
NOTES = {
    "qwen3.5-4b": "daily driver, fast, reads images",
    "qwen3.5-9b": "best quality fully on the GPU",
    "gemma4-e4b": "images and audio input, long context",
    "granite-4.2-8b": "tool calling / agents",
    "gpt-oss-20b": "reasoning, partly in RAM",
    "qwen3.6-35b-a3b": "strongest coding model, partly in RAM",
}


# ---------- small helpers ----------

def say(msg=""):
    print(msg, flush=True)


def ask(prompt, default=""):
    try:
        got = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        say()
        sys.exit(0)
    return got or default


def py(*args, check=True):
    cmd = [sys.executable, *args]
    say("  + python " + " ".join(args))
    return subprocess.run(cmd, check=check).returncode


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    s = load_state()
    s.update(kw)
    STATE.write_text(json.dumps(s, indent=1))


def gb(n):
    return f"{n / 1e9:.1f} GB"


# ---------- python version ----------

def find_python_314():
    """Return a command list for an installed Python 3.14, or None."""
    cands = []
    if shutil.which("py"):
        cands.append(["py", "-3.14"])
    for name in ("python3.14", "python3", "python"):
        p = shutil.which(name)
        if p:
            cands.append([p])
    for c in cands:
        try:
            out = subprocess.run(
                [*c, "-c", "import sys;print(sys.version_info[:2]==(3,14))"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out == "True":
            return c
    return None


def ensure_python():
    if sys.version_info[:2] == NEED:
        return
    other = find_python_314()
    if other and not os.environ.get("LLMKIT_REEXEC"):
        say(f"You started this with Python {sys.version.split()[0]}; switching to Python 3.14.")
        env = dict(os.environ, LLMKIT_REEXEC="1")
        sys.exit(subprocess.run([*other, str(ROOT / "kit.py"), *sys.argv[1:]], env=env).returncode)
    say(f"This kit needs Python 3.14 (it ships cp314 wheels). You are on {sys.version.split()[0]},")
    say("and no Python 3.14 was found on this machine.")
    say("Install it from the Microsoft Store: open Store, search \"Python 3.14\", click Get.")
    say("Then run:  py -3.14 kit.py")
    sys.exit(1)


# ---------- inspection ----------

def manifest():
    rows = []
    for line in (MODELS / "MANIFEST.txt").read_text().splitlines()[1:]:
        repo, name, size, sha = line.split("\t")
        rows.append((repo.split("/")[1], name, int(size), sha))
    return rows


def file_state(folder, name, size):
    """'ready', 'parts' (needs setup), 'partial' (chunks missing; run update) or 'missing'."""
    f = MODELS / folder / name
    if f.exists() and f.stat().st_size == size:
        return "ready"
    parts = sorted((MODELS / folder).glob(name + ".part[0-9][0-9][0-9]"))
    if parts:
        have = sum(p.stat().st_size for p in parts)
        return "parts" if have == size else "partial"
    return "missing"


def model_states():
    from _common import REGISTRY
    by_file = {(folder, name): file_state(folder, name, size) for folder, name, size, _ in manifest()}
    sizes = {(folder, name): size for folder, name, size, _ in manifest()}
    out = {}
    for mid, (rel, proj, _) in REGISTRY.items():
        # Readiness follows the model file; the vision projector is only needed for images.
        files = [rel] + ([proj] if proj else [])
        out[mid] = (by_file.get(tuple(rel.split("/", 1)), "missing"),
                    sum(sizes.get(tuple(f.split("/", 1)), 0) for f in files))
    return out


def installed():
    have = {}
    for p in PACKAGES:
        try:
            have[p] = md.version(p)
        except md.PackageNotFoundError:
            have[p] = None
    return have


def gpu_info():
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def summary():
    st = load_state()
    pk = installed()
    missing = [p for p, v in pk.items() if v is None]
    ms = model_states()
    ready = sum(1 for s, _ in ms.values() if s == "ready")
    gpu_ok = st.get("gpu_ok_python") == sys.version.split()[0] and not missing
    return {
        "packages_ok": not missing, "missing": missing, "models": ms,
        "ready": ready, "gpu_ok": gpu_ok, "gpu_name": st.get("gpu_name"),
    }


# ---------- actions ----------

def do_status(_=None):
    s = summary()
    say(f"Python      {sys.version.split()[0]}  ({sys.executable})")
    g = gpu_info()
    say(f"GPU/driver  {g or 'nvidia-smi not found (driver missing, or not on PATH)'}")
    say(f"Packages    {'installed' if s['packages_ok'] else 'NOT installed: ' + ', '.join(s['missing'])}")
    say(f"GPU check   {'passed: ' + str(s['gpu_name']) if s['gpu_ok'] else 'not passed yet (run setup)'}")
    free = shutil.disk_usage(ROOT).free
    say(f"Disk free   {gb(free)}")
    say("Models:")
    for mid, (state, size) in s["models"].items():
        label = {"ready": "ready", "parts": "downloaded, needs setup", "partial": "incomplete, run update",
                 "missing": "not downloaded, run update"}[state]
        say(f"  {mid:<17} {gb(size):>8}  {label}")


def pip_target_args():
    """Install into the user site when the interpreter's own site-packages is read-only."""
    import sysconfig
    if sys.prefix != sys.base_prefix:
        return []
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    return [] if os.access(site, os.W_OK) else ["--user"]


def do_setup(_=None):
    say("Step 1/4  reassembling downloaded chunks (skips anything already done)")
    need = sum(p.stat().st_size for p in ROOT.rglob("*.part[0-9][0-9][0-9]"))
    free = shutil.disk_usage(ROOT).free
    if need and free < need * 1.05:
        say(f"  Not enough disk: joining needs about {gb(need)} free, you have {gb(free)}.")
        say("  Free up space and run setup again; finished files are kept.")
        return False
    import split_tool
    bad = split_tool.join(WHEELS, delete_parts=True) + split_tool.join(MODELS, delete_parts=True)
    if bad:
        say("  Some files failed verification (listed above). Run update to re-fetch them, then setup.")
        return False

    say("Step 2/4  checking pip")
    if py("-m", "pip", "--version", check=False) != 0:
        py("-m", "ensurepip", "--default-pip")

    say("Step 3/4  installing packages from wheels\\ (offline)")
    rc = py("-m", "pip", "install", "--no-index", "--find-links", str(WHEELS),
            "--disable-pip-version-check", *pip_target_args(), *PACKAGES, check=False)
    if rc != 0:
        say("  pip failed (see message above). Nothing was downloaded; fix the reported problem and rerun setup.")
        return False

    removed = remove_launchers()
    if removed:
        say(f"  removed {len(removed)} .exe launchers pip created (the kit never runs them)")

    make_shortcut()

    say("Step 4/4  GPU check")
    ok = do_gpucheck()
    if not ok and os.environ.get("LLMKIT_ALLOW_CPU") == "1":
        say("LLMKIT_ALLOW_CPU=1: continuing without a GPU (models run on the CPU).")
        return True
    return ok


def make_shortcut():
    """Desktop shortcut that opens the menu: runs the system py.exe on kit.py.

    Windows only; written with PowerShell's WScript.Shell COM object. Skipped
    silently if PowerShell is unavailable."""
    if os.name != "nt":
        return
    desktop = pathlib.Path(os.environ.get("USERPROFILE", "~")).expanduser() / "Desktop"
    link = desktop / "llm-kit.lnk"
    if link.exists() or not desktop.is_dir():
        return
    launcher = shutil.which("py") or sys.executable
    args = f'-3.14 "{ROOT / "kit.py"}"' if launcher.lower().endswith("py.exe") else f'"{ROOT / "kit.py"}"'
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:LLMKIT_LNK);"
        "$s.TargetPath=$env:LLMKIT_TARGET;$s.Arguments=$env:LLMKIT_ARGS;"
        "$s.WorkingDirectory=$env:LLMKIT_DIR;$s.Description='llm-kit menu';$s.Save()"
    )
    env = dict(os.environ, LLMKIT_LNK=str(link), LLMKIT_TARGET=launcher, LLMKIT_ARGS=args, LLMKIT_DIR=str(ROOT))
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           env=env, capture_output=True, timeout=30)
        if r.returncode == 0:
            say(f"  desktop shortcut created: {link.name} (double-click to open the menu)")
    except (OSError, subprocess.SubprocessError):
        pass


def kit_dist_names():
    """Normalized names of every package shipped in wheels\\."""
    import re
    names = set()
    for w in list(WHEELS.glob("*.whl")) + list(WHEELS.glob("*.whl.sha256")):
        names.add(re.sub(r"[-_.]+", "-", w.name.split("-")[0]).lower())
    return names


def remove_launchers():
    """Delete the Scripts\\*.exe launchers pip generates for the kit's packages.

    Only files listed in each kit package's own install RECORD and ending in
    .exe are touched; nothing else in Scripts\\ is.
    """
    import re
    ours = kit_dist_names()
    removed = []
    for dist in md.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"] or "").lower()
        if name not in ours:
            continue
        for f in dist.files or []:
            if str(f).lower().endswith(".exe"):
                path = pathlib.Path(f.locate())
                if path.is_file():
                    path.unlink()
                    removed.append(path)
    return removed


def do_gpucheck(_=None):
    r = subprocess.run([sys.executable, str(ROOT / "run" / "check_gpu.py")], capture_output=True, text=True)
    say(r.stdout.rstrip())
    if r.returncode == 0:
        name = r.stdout.strip().splitlines()[-1].removeprefix("GPU OK: ")
        save_state(gpu_ok_python=sys.version.split()[0], gpu_name=name)
        say("\nSetup complete. Pick 1 in the menu to start chatting.")
        return True
    say(r.stderr.rstrip()[-2000:])
    save_state(gpu_ok_python=None)
    say("\nGPU check failed. The lines above say why (they come from the NVIDIA driver itself).")
    say("Models still run on the CPU meanwhile, just much slower.")
    return False


def do_update(_=None):
    git = shutil.which("git")
    if not git:
        say("git is not on PATH. Use whatever you cloned with to pull the latest version, then run setup.")
        return False
    repo = ROOT.parent
    if not (repo / ".git").exists():
        say("This copy was not made with git clone (probably a ZIP download), so it cannot update itself.")
        say("Download the ZIP again from GitHub, unzip it over this folder, then run:  py -3.14 kit.py setup")
        return False
    say("Fetching updates from GitHub (only new or changed files download)...")
    # Setup deletes chunk files after joining them; a merge-style pull leaves those
    # deletions alone, a rebase-style pull refuses. Always merge.
    rc = subprocess.run([git, "-C", str(repo), "pull", "--no-rebase", "--no-edit"]).returncode
    if rc != 0:
        say("git pull failed (message above). Your files were not changed.")
        return False
    say("Update downloaded. Running setup to install anything new...")
    env = dict(os.environ, LLMKIT_REEXEC="1")
    # Re-launch so the freshly pulled kit.py runs the setup.
    return subprocess.run([sys.executable, str(ROOT / "kit.py"), "setup"], env=env).returncode == 0


def pick_model(default="qwen3.5-4b", only=None):
    ms = {k: v for k, v in model_states().items() if only is None or k in only}
    ids = list(ms)
    say("Models:")
    for i, mid in enumerate(ids, 1):
        state, size = ms[mid]
        tag = "" if state == "ready" else "   (not ready: run setup/update)"
        say(f"  {i}) {mid:<17} {gb(size):>8}  {NOTES.get(mid, '')}{tag}")
    got = ask(f"Choose 1-{len(ids)} [Enter = {default}]: ", "")
    mid = default if not got else (ids[int(got) - 1] if got.isdigit() and 0 < int(got) <= len(ids) else got)
    if mid not in ms:
        say(f"unknown choice '{got}'")
        return None
    if ms[mid][0] != "ready":
        say(f"{mid} is not ready yet. Run setup (and update if it says incomplete).")
        return None
    return mid


def ready_to_run():
    s = summary()
    if not s["packages_ok"]:
        say("Packages are not installed yet. Running setup first.\n")
        return do_setup()
    return True


def run_script(name, argv):
    sys.argv = [name, *argv]
    try:
        runpy.run_path(str(ROOT / "run" / name), run_name="__main__")
    except KeyboardInterrupt:
        say("\nstopped.")
    except SystemExit as e:
        if e.code not in (None, 0):
            say(f"\n{name} exited: {e.code}")


def do_chat(a=None):
    if not ready_to_run():
        return
    mid = (a and a.model) or pick_model()
    if mid:
        run_script("chat.py", ["--model", mid])


def do_web(a=None):
    if not ready_to_run():
        return
    mid = (a and a.model) or pick_model()
    if mid:
        say("Opening http://127.0.0.1:7860 in your browser. Press Ctrl+C here to stop.")
        run_script("webchat.py", ["--model", mid])


def do_serve(a=None):
    if not ready_to_run():
        return
    mid = (a and a.model) or pick_model("qwen3.5-9b")
    if mid:
        say("API at http://127.0.0.1:8080/v1 (model name: " + mid + "). Press Ctrl+C here to stop.")
        run_script("serve.py", ["--model", mid])


def lan_addresses():
    import socket
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outbound interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    # Skip loopback, link-local and the usual virtual-switch ranges (WSL, Hyper-V, Docker).
    skip = ("127.", "169.254.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.")
    return sorted(ip for ip in ips if not ip.startswith(skip))


def do_share(a=None):
    """API server reachable from other machines (n8n etc.), protected by a key."""
    if not ready_to_run():
        return
    mid = (a and a.model) or pick_model("granite-4.2-8b")
    if not mid:
        return
    import secrets, socket
    key = load_state().get("api_key")
    if not key:
        key = "llmkit-" + secrets.token_urlsafe(24)
        save_state(api_key=key)
    port = 8080
    say("\nSharing on the network. Anything that can reach this laptop and has the key can use the model.")
    say("Paste these into n8n (Credentials > New > OpenAI):")
    say(f"  API Key   {key}")
    for ip in lan_addresses() or ["<this laptop's IP>"]:
        say(f"  Base URL  http://{ip}:{port}/v1")
    say(f"  (laptop name: http://{socket.gethostname()}:{port}/v1)")
    say(f"Model name to use in n8n: {mid}")
    say("If Windows asks whether to allow Python on networks, choose Private networks.")
    say("Press Ctrl+C here to stop sharing.\n")
    run_script("serve.py", ["--model", mid, "--host", "0.0.0.0", "--port", str(port), "--api-key", key])


def do_ask(a=None):
    """One question, one answer, exit (also used by the automatic Windows test)."""
    if not ready_to_run():
        return False
    mid = (a and a.model) or "qwen3.5-4b"
    question = " ".join(a.text) if a and a.text else ask("question: ")
    run_script("chat.py", ["--model", mid, "--once", question] +
               (["--image", a.image] if a and a.image else []))


def do_image(a=None):
    if not ready_to_run():
        return
    vision = [m for m in model_states() if mmproj_exists(m)]
    mid = (a and a.model) or pick_model("qwen3.5-9b", only=vision)
    if not mid:
        return
    f = (a and a.image) or ask("Path to the image (or audio file for gemma4-e4b), drag it here: ").strip('"')
    if not pathlib.Path(f).is_file():
        say(f"file not found: {f}")
        return
    say("Ask about it below; /image FILE adds another one later.")
    run_script("chat.py", ["--model", mid, "--image", f])


def do_docs(a=None):
    if not ready_to_run():
        return
    mid = (a and a.model) or pick_model("qwen3.5-9b")
    if not mid:
        return
    folder = (a and a.folder) or ask(f"Folder with your documents [Enter = {ROOT / 'docs-inbox'}]: ", str(ROOT / "docs-inbox"))
    run_script("docs.py", ["--model", mid, "--folder", folder.strip('"')])


def do_tune(a=None):
    if not ready_to_run():
        return
    say("Tuning loads each model several times and measures its speed; it takes a few minutes per model.")
    say("Close other GPU-heavy programs first. Results are saved and used automatically from now on.\n")
    which = (a and a.model) or ask("Tune which? [Enter = the two big models, a = all]: ", "")
    if which == "a":
        run_script("tune.py", ["--all"])
    elif which:
        run_script("tune.py", ["--model", which])
    else:
        for mid in ("gpt-oss-20b", "qwen3.6-35b-a3b"):
            run_script("tune.py", ["--model", mid])


def do_pi(a=None):
    if not ready_to_run():
        return
    if not shutil.which("node"):
        say("Pi needs Node.js 20.6 or newer, and `node` was not found on PATH.")
        return
    mid = (a and a.model) or pick_model("qwen3.6-35b-a3b")
    if not mid:
        return
    say("Starting Pi (coding agent). It works in the folder you are in now:")
    say(f"  {pathlib.Path.cwd()}")
    say("Type /quit in Pi to leave.\n")
    run_script("pi.py", ["--model", mid])


def mmproj_exists(mid):
    from _common import mmproj
    return bool(mmproj(mid))


ACTIONS = {"setup": do_setup, "update": do_update, "status": do_status, "chat": do_chat,
           "web": do_web, "serve": do_serve, "share": do_share, "gpucheck": do_gpucheck,
           "ask": do_ask, "image": do_image, "docs": do_docs, "tune": do_tune, "pi": do_pi}


def menu():
    while True:
        s = summary()
        setup_line = "ready" if s["packages_ok"] and s["gpu_ok"] else "needs setup (choose 9)"
        say("\n================ llm-kit ================")
        say(f" Python {sys.version.split()[0]}   GPU: {s['gpu_name'] or 'not checked yet'}")
        say(f" Setup: {setup_line}   Models ready: {s['ready']}/{len(s['models'])}")
        say("------------------------------------------")
        say("  1) Chat here in this window")
        say("  2) Chat in the browser")
        say("  3) Ask about an image (or audio)")
        say("  4) Ask questions about your documents")
        say("  5) Pi coding agent")
        say("  6) API server for editors (this laptop only)")
        say("  7) Share API with n8n / other machines (needs key)")
        say("  8) Tune speed for this GPU")
        say("  9) Setup / repair")
        say(" 10) Update from GitHub, then setup")
        say(" 11) Status details")
        say("  q) Quit")
        c = ask("> ").lower()
        if c in ("q", "quit", "exit"):
            return
        fn = {"1": do_chat, "2": do_web, "3": do_image, "4": do_docs, "5": do_pi, "6": do_serve,
              "7": do_share, "8": do_tune, "9": do_setup, "10": do_update, "11": do_status}.get(c)
        if fn:
            say()
            fn()
            if c in ("8", "9", "10", "11"):
                ask("\nPress Enter to return to the menu...")
        elif c:
            say("pick 1-11 or q")


def main():
    ap = argparse.ArgumentParser(description="llm-kit: offline local LLMs. Run with no arguments for the menu.")
    ap.add_argument("action", nargs="?", choices=list(ACTIONS))
    ap.add_argument("text", nargs="*", help="for ask: the question")
    ap.add_argument("--model", help="model id (skips the picker)")
    ap.add_argument("--image", help="for ask/image: an image or audio file")
    ap.add_argument("--folder", help="for docs: the documents folder")
    a = ap.parse_args()
    ensure_python()
    if a.action:
        ok = ACTIONS[a.action](a)
        sys.exit(1 if ok is False else 0)
    menu()


if __name__ == "__main__":
    main()
