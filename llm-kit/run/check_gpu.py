"""Prove the CUDA backend loads and sees the GPU. Exit 1 if it falls back to CPU."""

import ctypes, os, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import add_nvidia_dlls

add_nvidia_dlls()
import llama_cpp
from llama_cpp import _ggml

print("llama-cpp-python", llama_cpp.__version__)
lib_dir = pathlib.Path(llama_cpp.__file__).resolve().parent / "lib"

# The CUDA backend is a plug-in (ggml-cuda.dll) that llama.cpp loads at runtime,
# the same way Llama() does before loading a model.
llama_cpp.llama_backend_init()
_ggml.ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))

devs = []
for i in range(_ggml.ggml_backend_dev_count()):
    d = _ggml.ggml_backend_dev_get(i)
    name = _ggml.ggml_backend_dev_name(d).decode()
    desc = _ggml.ggml_backend_dev_description(d).decode()
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    _ggml.ggml_backend_dev_memory(d, ctypes.byref(free), ctypes.byref(total))
    devs.append((name, desc))
    mem = f"  {total.value / 2**30:.1f} GB" if total.value else ""
    print(f"device {i}: {name}  {desc}{mem}")

gpu = [d for d in devs if d[0].upper().startswith(("CUDA", "GPU"))]
if gpu:
    print(f"\nGPU OK: {gpu[0][1]}")
    sys.exit(0)

# No GPU device: find out why ggml-cuda.dll did not load.
print("\nNO CUDA DEVICE.")
cuda = lib_dir / "ggml-cuda.dll"
if os.name == "nt" and cuda.exists():
    try:
        ctypes.WinDLL(str(cuda))
        print("ggml-cuda.dll loads, but it found no usable GPU: the NVIDIA driver is probably older than 570.")
    except OSError as e:
        print(f"ggml-cuda.dll failed to load: {e}")
        for dep in ("nvcuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
            try:
                ctypes.WinDLL(dep)
                print(f"  {dep:<20} ok")
            except OSError as e2:
                hint = {
                    "nvcuda.dll": "NVIDIA driver not installed",
                    "cudart64_12.dll": "nvidia-cuda-runtime-cu12 wheel missing; rerun setup",
                    "cublas64_12.dll": "nvidia-cublas-cu12 wheel missing; rerun setup",
                    "cublasLt64_12.dll": "nvidia-cublas-cu12 wheel missing; rerun setup",
                }[dep]
                print(f"  {dep:<20} MISSING ({hint}): {e2}")
        print("If every dependency is ok, your organization is blocking DLL loads from this folder.")
else:
    print("Check: NVIDIA driver >= 570, nvidia-*-cu12 wheels installed, Python 3.14.")
sys.exit(1)
