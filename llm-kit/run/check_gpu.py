"""Prove the CUDA backend loads and sees the GPU. Exit 1 if it falls back to CPU."""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import add_nvidia_dlls

add_nvidia_dlls()
import llama_cpp

print("llama-cpp-python", llama_cpp.__version__)
llama_cpp.llama_backend_init()
devs = []
try:
    n = llama_cpp.ggml_backend_dev_count()
    for i in range(n):
        d = llama_cpp.ggml_backend_dev_get(i)
        name = llama_cpp.ggml_backend_dev_name(d).decode()
        desc = llama_cpp.ggml_backend_dev_description(d).decode()
        devs.append((name, desc))
        print(f"device {i}: {name}  {desc}")
except AttributeError as e:
    print("device enumeration API not exposed in this build:", e)
    print("supports_gpu_offload:", llama_cpp.llama_supports_gpu_offload())
    sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)

gpu = [d for d in devs if d[0].startswith(("CUDA", "GPU"))]
if not gpu:
    print(
        "\nNO CUDA DEVICE. Check: NVIDIA driver >= 570, nvidia-*-cu12 wheels installed, Python 3.12."
    )
    sys.exit(1)
print(f"\nGPU OK: {gpu[0][1]}")
