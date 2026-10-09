"""OpenAI-compatible API on http://127.0.0.1:8080/v1 for editors and tools
(Continue, Cline, Open Interpreter, any `openai` client with base_url set).

    python run\serve.py --model qwen3.5-9b
    python run\serve.py --model qwen3.6-35b-a3b --n-cpu-moe 30

Wraps `python -m llama_cpp.server` so you never need a console-script .exe.
"""

import argparse, pathlib, runpy, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import DEFAULTS, MODELS, REGISTRY, add_nvidia_dlls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(REGISTRY))
    ap.add_argument("--n-cpu-moe", type=int)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    rel, _, extra = REGISTRY[a.model]
    kw = {**DEFAULTS, **extra}
    if a.n_cpu_moe is not None:
        kw["n_cpu_moe"] = a.n_cpu_moe
    argv = [
        "llama_cpp.server",
        "--model",
        str(MODELS / rel),
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--n_gpu_layers",
        str(kw["n_gpu_layers"]),
        "--n_ctx",
        str(kw["n_ctx"]),
        "--n_threads",
        str(kw["n_threads"]),
        "--flash_attn",
        "true",
        "--type_k",
        str(kw["type_k"]),
        "--type_v",
        str(kw["type_v"]),
        "--model_alias",
        a.model,
    ]
    if kw.get("n_cpu_moe"):
        argv += ["--n_cpu_moe", str(kw["n_cpu_moe"])]
    add_nvidia_dlls()
    print("+ python -m", " ".join(argv), flush=True)
    sys.argv = argv
    runpy.run_module("llama_cpp.server", run_name="__main__")


if __name__ == "__main__":
    main()
