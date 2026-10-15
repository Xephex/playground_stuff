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
    flash_attn_type=1,  # 1 = on (llama-cpp-python 0.4.x; the old flash_attn=True is ignored)
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
                # llama.cpp loads ggml-cuda.dll with plain LoadLibraryW, which ignores
                # add_dll_directory and searches PATH, so the CUDA DLLs must be on PATH too.
                if str(d) not in os.environ["PATH"].split(os.pathsep):
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


STATE = ROOT / ".kit-state.json"


def tuned(model_id: str) -> dict:
    """Settings saved by `kit.py tune` for this model (empty if never tuned)."""
    import json

    try:
        return json.loads(STATE.read_text()).get("tuned", {}).get(model_id, {}).get("kwargs", {})
    except (OSError, ValueError):
        return {}


def load(model_id: str, vision: bool = False, **overrides):
    """Load a model with the kit's defaults, then any saved tuning, then overrides.

    vision=True attaches the model's image/audio projector (if it has one).
    """
    add_nvidia_dlls()
    from llama_cpp import Llama

    if model_id not in REGISTRY:
        sys.exit(f"unknown model '{model_id}'. choices: {', '.join(REGISTRY)}")
    rel, _, extra = REGISTRY[model_id]
    path = MODELS / rel
    if not path.exists():
        sys.exit(f"{path} not found. Run: py -3.14 kit.py setup")
    kw = {**DEFAULTS, **extra, **tuned(model_id), **overrides}
    if vision and "chat_handler" not in kw:
        kw["chat_handler"] = media_handler(model_id)
    shown = {k: v for k, v in kw.items() if k != "chat_handler"}
    print(f"loading {model_id} ({path.stat().st_size / 1e9:.1f} GB) with {shown}", flush=True)
    return Llama(model_path=str(path), **kw)


def mmproj(model_id: str):
    rel = REGISTRY[model_id][1]
    return str(MODELS / rel) if rel and (MODELS / rel).exists() else None


def media_handler(model_id: str):
    """Chat handler that lets the model read images (and audio for Gemma 4)."""
    proj = mmproj(model_id)
    if not proj:
        sys.exit(f"{model_id} cannot read images: no vision file (choose qwen3.5-4b, qwen3.5-9b, gemma4-e4b or qwen3.6-35b-a3b)")
    from llama_cpp import llama_multimodal as mm

    if model_id.startswith("gemma4"):
        return mm.Gemma4ChatHandler(mmproj_path=proj, verbose=False, enable_thinking=False)
    if model_id.startswith(("qwen3.5", "qwen3.6")):
        return mm.Qwen35ChatHandler(mmproj_path=proj, verbose=False, enable_thinking=False)
    return mm.GenericMTMDChatHandler(chat_format=None, mmproj_path=proj, verbose=False)


def media_part(path) -> dict:
    """An OpenAI-style message part for an image or audio file."""
    import base64, mimetypes

    p = pathlib.Path(path)
    data = base64.b64encode(p.read_bytes()).decode()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    if mime.startswith("audio/"):
        fmt = {"audio/mpeg": "mp3", "audio/x-wav": "wav", "audio/wav": "wav"}.get(mime, p.suffix.lstrip("."))
        return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
