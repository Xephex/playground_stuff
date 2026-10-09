"""One-shot offline setup: reassemble chunks, install wheels, smoke-test the GPU.

    python setup_env.py            # does everything
    python setup_env.py --no-join  # skip chunk reassembly (already done)

Runs entirely inside python.exe. Never launches another executable.
"""

import argparse, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
WHEELS = ROOT / "wheels"
MODELS = ROOT / "models"


def run_py(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-join", action="store_true")
    a = ap.parse_args()

    if sys.version_info[:2] != (3, 12):
        sys.exit(
            f"This bundle ships cp312 wheels; you are on Python {sys.version.split()[0]}. "
            "Install Python 3.12 (python.org or Microsoft Store) and rerun with it."
        )
    if os.name != "nt":
        print(
            "warning: wheels are win_amd64; on Linux/macOS install llama-cpp-python from pip instead."
        )

    if not a.no_join:
        run_py(str(ROOT / "split_tool.py"), "join", str(WHEELS))
        run_py(str(ROOT / "split_tool.py"), "join", str(MODELS))

    # pip itself is a module, so this is allowed even when Scripts\pip.exe is not.
    run_py(
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(WHEELS),
        "--upgrade",
        "pip",
    )
    run_py(
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(WHEELS),
        "llama_cpp_python",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "huggingface_hub",
        "gradio",
        "fastapi",
        "uvicorn",
        "sse-starlette",
        "starlette-context",
        "pydantic-settings",
        "diskcache",
        "jinja2",
        "numpy",
    )

    run_py(str(ROOT / "run" / "check_gpu.py"))
    print("\nsetup complete. next: python run\\chat.py --model qwen3.5-4b")


if __name__ == "__main__":
    main()
