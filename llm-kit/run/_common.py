"""Shared model registry and loader for the run scripts."""

import os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# id -> (gguf relative to models/, mmproj or None, extra Llama kwargs)
# n_cpu_moe=99 keeps every MoE expert in system RAM; lower it to use more VRAM.
REGISTRY = {
    "qwen3.5-4b": (
        "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf",
        "Qwen3.5-4B-GGUF/mmproj-F16.gguf",
        {},
    ),
    "qwen3.5-9b": (
        "Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
        "Qwen3.5-9B-GGUF/mmproj-F16.gguf",
        {"n_ctx": 12288},
    ),
    "gemma4-e4b": (
        "gemma-4-E4B-it-qat-q4_0-gguf/gemma-4-E4B_q4_0-it.gguf",
        "gemma-4-E4B-it-qat-q4_0-gguf/gemma-4-E4B-it-mmproj.gguf",
        {},
    ),
    "granite-4.2-8b": ("granite-4.2-8b-GGUF/granite-4.2-8b-Q4_K_M.gguf", None, {}),
    "gpt-oss-20b": (
        "gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
        None,
        {"n_cpu_moe": 99, "n_ctx": 32768},
    ),
    "qwen3.6-35b-a3b": (
        "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf",
        {"n_cpu_moe": 99, "n_ctx": 32768},
    ),
}

DEFAULTS = dict(
    n_gpu_layers=-1,
    n_ctx=16384,
    flash_attn=True,
    type_k=8,
    type_v=8,
    n_threads=max(4, (os.cpu_count() or 8) // 2),
    verbose=False,
)


def add_nvidia_dlls() -> None:
    """Make pip-installed CUDA DLLs loadable (Windows only; no-op elsewhere)."""
    if os.name != "nt":
        return
    for site in sys.path:
        nv = pathlib.Path(site) / "nvidia"
        if nv.is_dir():
            for d in nv.glob("*/bin"):
                os.add_dll_directory(str(d))


def load(model_id: str, **overrides):
    add_nvidia_dlls()
    from llama_cpp import Llama

    if model_id not in REGISTRY:
        sys.exit(f"unknown model '{model_id}'. choices: {', '.join(REGISTRY)}")
    rel, _, extra = REGISTRY[model_id]
    path = MODELS / rel
    if not path.exists():
        sys.exit(
            f"{path} not found. run: python setup_env.py (or split_tool.py join models)"
        )
    kw = {**DEFAULTS, **extra, **overrides}
    print(
        f"loading {model_id} ({path.stat().st_size / 1e9:.1f} GB) with {kw}", flush=True
    )
    return Llama(model_path=str(path), **kw)


def mmproj(model_id: str):
    rel = REGISTRY[model_id][1]
    return str(MODELS / rel) if rel else None
