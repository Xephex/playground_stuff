r"""Run the Pi coding agent against the kit's local models.

python run\pi.py --model qwen3.6-35b-a3b              (interactive Pi in this window)
python run\pi.py --model granite-4.2-8b -p "task"     (one task, then exit)

Needs Node.js 20.6 or newer (`node --version`). Pi is bundled as pi\pi-*.zip
and unpacked to pi\app on first use. Pi runs as `node <pi>\dist\cli.js`;
its own launcher is not used. Its settings live in pi\agent (not your
personal ~/.pi), and its models.json points at the kit's API on this laptop.
If the kit's API server is not running, this starts it in the background with
the chosen model and stops it when Pi exits.

On Windows Pi runs shell commands through Git Bash (C:\Program Files\Git\bin\bash.exe),
which comes with Git for Windows.
"""

import argparse, json, os, pathlib, shutil, socket, subprocess, sys, time, urllib.request, zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import DEFAULTS, REGISTRY, ROOT

PI = ROOT / "pi"
APP = PI / "app"
AGENT = PI / "agent"
CLI = APP / "node_modules" / "@mariozechner" / "pi-coding-agent" / "dist" / "cli.js"
PORT = 8080
MIN_NODE = (20, 6)


def node_path():
    node = shutil.which("node")
    if not node:
        sys.exit("Node.js is not installed or not on PATH. Pi needs Node.js 20.6 or newer.")
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    ver = tuple(int(x) for x in out.lstrip("v").split(".")[:2])
    if ver < MIN_NODE:
        sys.exit(f"Node.js {out} is too old; Pi needs 20.6 or newer.")
    return node


def unpack():
    zips = sorted(PI.glob("pi-*-win-x64.zip"))
    if not zips:
        sys.exit(f"No Pi bundle in {PI}. Run update.")
    marker = APP / ".from"
    if CLI.exists() and marker.exists() and marker.read_text() == zips[-1].name:
        return
    print(f"unpacking {zips[-1].name} (first run only)...", flush=True)
    if APP.exists():
        shutil.rmtree(APP)
    APP.mkdir(parents=True)
    with zipfile.ZipFile(zips[-1]) as z:
        z.extractall(APP)
    marker.write_text(zips[-1].name)


def write_config(model, key):
    AGENT.mkdir(parents=True, exist_ok=True)
    models = []
    for mid in REGISTRY:
        ctx = REGISTRY[mid][2].get("n_ctx", DEFAULTS["n_ctx"])
        m = {"id": mid, "name": f"{mid} (local)", "contextWindow": ctx, "maxTokens": min(8192, ctx // 2)}
        if mid in ("qwen3.5-4b", "qwen3.5-9b", "gemma4-e4b", "qwen3.6-35b-a3b"):
            m["input"] = ["text", "image"]
        if mid == "gpt-oss-20b":
            m["reasoning"] = True
        models.append(m)
    cfg = {"providers": {"llm-kit": {
        "baseUrl": f"http://127.0.0.1:{PORT}/v1",
        "api": "openai-completions",
        "apiKey": key or "local",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
        "models": models,
    }}}
    (AGENT / "models.json").write_text(json.dumps(cfg, indent=2))
    settings = AGENT / "settings.json"
    s = json.loads(settings.read_text()) if settings.exists() else {}
    s.update({"defaultProvider": "llm-kit", "defaultModel": model})
    git_bash = pathlib.Path(r"C:\Program Files\Git\bin\bash.exe")
    if os.name == "nt" and git_bash.exists():
        s.setdefault("shellPath", str(git_bash))
    settings.write_text(json.dumps(s, indent=2))


def server_model(key):
    """Model id served on PORT, or None if nothing is listening."""
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return None


def port_busy():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6-35b-a3b", choices=list(REGISTRY))
    ap.add_argument("--key", help="API key, if the running server needs one")
    ap.add_argument("pi_args", nargs=argparse.REMAINDER, help="arguments passed to Pi (e.g. -p \"task\")")
    a = ap.parse_args()

    node = node_path()
    unpack()
    write_config(a.model, a.key)

    server = None
    running = server_model(a.key)
    if running and running != a.model:
        print(f"The kit's API is already serving {running}; Pi will use that model.")
        a.model = running
        write_config(a.model, a.key)
    elif not running:
        if port_busy():
            sys.exit(f"Port {PORT} is in use by something else. Stop it, or stop the kit's other server.")
        print(f"starting the kit's API with {a.model} in the background...", flush=True)
        log = open(PI / "server.log", "w")
        server = subprocess.Popen([sys.executable, str(ROOT / "run" / "serve.py"), "--model", a.model,
                                   "--port", str(PORT)] + (["--api-key", a.key] if a.key else []),
                                  stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if server_model(a.key):
                break
            if server.poll() is not None:
                sys.exit(f"The API server stopped while loading; see {PI / 'server.log'}")
            time.sleep(1)
        else:
            server.terminate()
            sys.exit("The API server did not come up within 10 minutes.")

    env = dict(os.environ, PI_CODING_AGENT_DIR=str(AGENT))
    args = [a for a in a.pi_args if a != "--"]
    cmd = [node, str(CLI), "--provider", "llm-kit", "--model", a.model, *args]
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except KeyboardInterrupt:
        rc = 130
    finally:
        if server:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
    sys.exit(rc)


if __name__ == "__main__":
    main()
