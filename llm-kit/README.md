# Offline local-LLM kit for a locked-down Windows laptop

Everything needed to run open-weight models on an NVIDIA RTX PRO 2000 Blackwell laptop (8 GB VRAM, 64 GB RAM) using only `python.exe`. No installers, no `.exe`/`.msi`, no internet after cloning.

Large files are stored as `*.partNNN` chunks (under 95 MB each) because GitHub rejects files over 100 MB. `setup_env.py` reassembles them.

## Requirements on the laptop

- Python **3.12** (the bundled wheels are `cp312-win_amd64`). Check with `python --version`.
- NVIDIA driver 570 or newer (`nvidia-smi` lives in `C:\Windows\System32`, so it is allowed to run).
- ~55 GB free disk during setup (chunks + reassembled files; delete the `.part` files afterwards, which `setup_env.py` does once each file verifies).

## Setup (once)

```
git clone https://github.com/Xephex/playground_stuff
cd playground_stuff\llm-kit
python setup_env.py
```

`setup_env.py` joins the chunks, verifies every file against its `.sha256`, installs the wheels from `wheels\` with `python -m pip install --no-index`, and runs `run\check_gpu.py`, which must print a CUDA device. If it prints `NO CUDA DEVICE`, the driver is too old or the CUDA DLLs did not load; see Troubleshooting.

Prefer a venv? `python -m venv .venv` then `.venv\Scripts\python.exe setup_env.py --no-join` after a first plain `python setup_env.py`. Always invoke the venv's `python.exe` directly; `Activate.ps1` and the `Scripts\*.exe` shims may be blocked.

## Use

| What | Command |
|---|---|
| Terminal chat | `python run\chat.py --model qwen3.5-4b` |
| Browser chat (model picker, http://127.0.0.1:7860) | `python run\webchat.py` |
| OpenAI-compatible API for editors (http://127.0.0.1:8080/v1) | `python run\serve.py --model qwen3.5-9b` |
| Big MoE model, experts in RAM | `python run\chat.py --model qwen3.6-35b-a3b --n-cpu-moe 99` |
| Same, faster once you know VRAM headroom | `--n-cpu-moe 30` (lower = more experts on GPU; raise if you get out-of-memory) |
| Image input | `python run\chat.py --model gemma4-e4b --image screenshot.png` |

Every command is `python <script>`; nothing in this kit is launched by its own executable.

## Models included

| id | File | Size | Fits | Use for |
|---|---|---|---|---|
| `qwen3.5-4b` | Qwen3.5-4B Q4_K_M + vision projector | 2.7 + 0.7 GB | GPU | daily driver; chat, code, images; 40+ tok/s |
| `qwen3.5-9b` | Qwen3.5-9B Q4_K_M + vision projector | 5.7 + 0.9 GB | GPU, tight | best quality that stays on the GPU; 22-28 tok/s; context capped at 12K |
| `gemma4-e4b` | Gemma 4 E4B QAT q4_0 + projector | 5.2 + ? GB | GPU | text, images **and audio** input; 128K context |
| `granite-4.2-8b` | Granite 4.2 8B Q4_K_M | 5.4 GB | GPU | tool calling, agents, thinking mode |
| `gpt-oss-20b` | gpt-oss-20b MXFP4 | 12.1 GB | RAM offload | reasoning with adjustable effort; ~16-18 tok/s |
| `qwen3.6-35b-a3b` | Qwen3.6-35B-A3B UD-Q4_K_M + projector | ~20 GB | RAM offload | strongest coding/agent model here; ~30 tok/s with `--n-cpu-moe` |

Licenses: Qwen, Gemma 4, Granite, gpt-oss are all Apache 2.0. Each model folder keeps its upstream `LICENSE`/README where the repo provided one.

Speeds are from published runs on 8 GB RTX 4060 / 2080 cards; this GPU has more bandwidth, so expect similar or better on mains power. Full research notes with sources: `docs/local-models-rtx-pro-2000-blackwell.md`.

## How the GPU path works without executables

- `llama_cpp_python-0.4.2+cu128-cp312-win_amd64.whl` is the [JamePeng fork](https://github.com/JamePeng/llama-cpp-python) built for CUDA 12.8; it contains `llama.dll` and `ggml-cuda.dll`, which `python.exe` loads as libraries.
- `nvidia_cublas_cu12` and `nvidia_cuda_runtime_cu12` wheels supply `cublas64_12.dll` and `cudart64_12.dll`; `run\_common.py` adds their folders to the DLL search path at import time.
- AppLocker's default rules block process launches from the user profile but do **not** evaluate DLL loads (the DLL rule collection is off by default). If your organization enabled DLL rules or strict WDAC, this kit will fail at `check_gpu.py` with a load error, and you will need an exception.

Be aware of what you are doing: the llama-cpp wheel is a community build, not signed by NVIDIA or Microsoft. Its SHA-256 is in `wheels/SHA256SUMS`. If your workplace policy treats unsigned native code as prohibited regardless of mechanism, ask before using it.

## Troubleshooting

- `check_gpu.py` says `NO CUDA DEVICE`: run `nvidia-smi`; driver must be >= 570. Then `python -c "import nvidia.cublas, os; print(os.listdir(os.path.dirname(nvidia.cublas.__file__)+'/bin'))"` should list `cublas64_12.dll`.
- `ImportError: DLL load failed` on `import llama_cpp`: DLL rules are on, or Python is not 3.12 x64.
- Out of memory loading `qwen3.5-9b`: close the browser, or add `--n-ctx 8192`.
- MoE model slow: lower `--n-cpu-moe` in steps of 5 until it fails to load, then go back one step.
- Adding another model: drop a `.gguf` under `models\<folder>\`, add a line to `REGISTRY` in `run\_common.py`.

## Layout

```
llm-kit/
  setup_env.py        reassemble + install + GPU check
  split_tool.py       split/join/verify (python split_tool.py --help)
  wheels/             win_amd64 cp312 wheels (+ .part chunks for the big one), SHA256SUMS
  models/<name>/      .gguf files as .partNNN chunks + .sha256
  run/                chat.py, webchat.py, serve.py, check_gpu.py, _common.py
  docs/               research notes
```
