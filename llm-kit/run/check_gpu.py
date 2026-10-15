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

cuda = lib_dir / "ggml-cuda.dll"

# No GPU device: ask CUDA itself why, instead of guessing.
print("\nNO CUDA DEVICE. Asking the NVIDIA driver directly:")
win = os.name == "nt"
CU_ERR = {
    100: "no CUDA-capable GPU visible. The NVIDIA GPU is switched off or hidden: check the laptop's "
         "graphics mode (set Hybrid/Discrete, not Integrated/Eco/iGPU-only), that the laptop is not in "
         "battery saver, and that the GPU is enabled in Device Manager > Display adapters.",
    35: "the NVIDIA driver is too old for this CUDA version. Update the driver (570 or newer).",
    3: "the driver could not initialize. Reboot; if it persists, reinstall the NVIDIA driver.",
    999: "unknown driver error. Reboot; if it persists, reinstall the NVIDIA driver.",
}


def cu(lib, fn, *args):
    return getattr(lib, fn)(*args)


try:
    drv = ctypes.WinDLL("nvcuda.dll") if win else ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print(f"  NVIDIA driver library not found ({e}).")
    print("  The NVIDIA driver is not installed, or this is not the NVIDIA GPU's Windows session.")
    sys.exit(1)

ver = ctypes.c_int()
cu(drv, "cuDriverGetVersion", ctypes.byref(ver))
print(f"  driver supports CUDA {ver.value // 1000}.{ver.value % 1000 // 10}")
rc = cu(drv, "cuInit", 0)
if rc != 0:
    print(f"  cuInit failed with error {rc}: {CU_ERR.get(rc, 'see the CUDA error list')}")
    sys.exit(1)
n = ctypes.c_int()
cu(drv, "cuDeviceGetCount", ctypes.byref(n))
print(f"  driver sees {n.value} GPU(s)")
for i in range(n.value):
    dev, name = ctypes.c_int(), ctypes.create_string_buffer(256)
    cu(drv, "cuDeviceGet", ctypes.byref(dev), i)
    cu(drv, "cuDeviceGetName", name, 256, dev)
    print(f"    {i}: {name.value.decode()}")
if n.value == 0:
    print(f"  {CU_ERR[100]}")
    sys.exit(1)

try:
    rt = ctypes.WinDLL("cudart64_12.dll") if win else ctypes.CDLL("libcudart.so.12")
    rver, cnt = ctypes.c_int(), ctypes.c_int()
    cu(rt, "cudaRuntimeGetVersion", ctypes.byref(rver))
    err = cu(rt, "cudaGetDeviceCount", ctypes.byref(cnt))
    print(f"  CUDA runtime {rver.value // 1000}.{rver.value % 1000 // 10}: error {err}, {cnt.value} GPU(s)")
    if err:
        print(f"  {CU_ERR.get(err, 'runtime error; send this output')}")
except OSError as e:
    print(f"  CUDA runtime DLL failed to load ({e}): rerun setup.")
print("  The driver sees the GPU but llama.cpp's CUDA plug-in did not register it.")
if win:
    # Load the plug-in the way llama.cpp does (plain LoadLibraryW) to get the real Windows error.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.LoadLibraryW.restype = ctypes.c_void_p
    if k32.LoadLibraryW(str(cuda)):
        print("  ggml-cuda.dll loads the way llama.cpp loads it; send this whole output.")
    else:
        code = ctypes.get_last_error()
        print(f"  Loading ggml-cuda.dll the way llama.cpp does fails with Windows error {code}"
              + (" (a DLL it needs is not on PATH)." if code == 126 else "."))
        print("  PATH entries for NVIDIA DLLs:", [p for p in os.environ["PATH"].split(os.pathsep) if "nvidia" in p.lower()] or "none")
print("Send this whole output.")
sys.exit(1)
