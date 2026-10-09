# Local open-weight models for an RTX PRO 2000 Blackwell laptop

Target machine: NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU (8 GB GDDR7, 128-bit, ~384 GB/s, 5th-gen Tensor Cores, 35-115 W TGP) with 64 GB system RAM. Compiled 2026-10-09.

## How to read this

- **VRAM budget.** 8 GB total. The GPU driver, desktop compositor and browser take 0.5-1.5 GB, so plan on ~6.5 GB for weights + KV cache. A model "fits" here if its 4-bit file is at or under ~5.5 GB. 6-7.5 GB is "borderline": it loads, but only with short context or quantized KV cache. Anything larger is "offload": part of the model lives in system RAM.
- **Why 64 GB RAM matters.** Mixture-of-Experts (MoE) models (gpt-oss-20b, Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, Gemma 4 26B-A4B) activate only 3-4B parameters per token. llama.cpp's `--n-cpu-moe` keeps the attention layers and router on the GPU and the expert weights in RAM. With 64 GB this makes 20-35B MoE models run at 15-30 tokens/s on an 8 GB GPU, which is usable. Dense models above ~9B do not get this benefit: once they spill to RAM they drop to 5-10 tokens/s.
- **Speed figures** are from published runs on 8 GB RTX cards (RTX 4060 8 GB, RTX 2080 8 GB, RTX 3070 8 GB). The RTX PRO 2000 Blackwell has more memory bandwidth (384 GB/s vs 272 GB/s on a 4060) so expect equal or slightly better numbers on mains power. On battery or in a thin chassis the TGP limit will cut these by 20-40%.
- 15 tokens/s is the floor for interactive use; 25+ feels instant. Prompt processing (reading your input) is a separate, much faster number.

## Tier 1: fits fully in VRAM, fast

| Model | Params | Released | License | 4-bit file | Input | Best at | Speed evidence (8 GB RTX) | Get it |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-4B** | 4B dense | 2026-03-02 | Apache 2.0 | 3.3 GB (Q4_K_M) | text + image | best small all-rounder: chat, reasoning, coding, vision, 256K ctx | 40+ t/s [1] | `ollama pull qwen3.5:4b` |
| **Gemma 4 E4B-it** | 4.5B eff. (8B w/ embeddings) | 2026-04-02 | Apache 2.0 | 5.15 GB (official QAT q4_0) | text + image + **audio** | multimodal assistant, speech-to-text input, 128K ctx; MMLU-Pro 69.4 | not published; same class as 8B dense (~25-30 t/s) | `llama-server -hf google/gemma-4-E4B-it-qat-q4_0-gguf` (not on Ollama yet) |
| **Granite 4.2 8B** | 8B dense | 2026-08-25 | Apache 2.0 | 5.35 GB (Q4_K_M) | text | agentic/tool-calling, thinking mode, 128K ctx; AIME 86.7, SWE-bench 47.7 | 8B class: 25-32 t/s [1] | `llama-server -hf ibm-granite/granite-4.2-8b-GGUF` |
| **Ministral 3 8B** | 8B dense | 2025-12-02 | Apache 2.0 | 5.0 GB (Q4_K_M) | text + image | balanced chat/code/vision, 256K ctx | 8B class: 25-32 t/s [1] | `ollama pull ministral-3:8b` |
| **DeepSeek-R1-0528-Qwen3-8B** | 8B dense | 2025-05-28 | MIT | ~4.0 GB (Q4_K_M) | text | math and reasoning (AIME24 86%) | 8B class [1][3] | `ollama pull deepseek-r1:8b` |
| **Nemotron-Nano-9B-v2** | 9B hybrid Mamba | 2025 | NVIDIA Open Model License | ~4.5 GB (Q4_K_M) | text | long-context reasoning, high throughput, 128K ctx | not published for 8 GB | `ollama pull nemotron-nano:9b-v2` |
| **Gemma 4 E2B-it** | 2.3B eff. | 2026-04-02 | Apache 2.0 | 3.4 GB (QAT q4_0) | text + image + audio | fastest multimodal option, audio input | not published | `llama-server -hf google/gemma-4-E2B-it-qat-q4_0-gguf` |
| **Granite 4.2 3B** | 3B dense | 2026-08-25 | Apache 2.0 | 2.1 GB | text | tool calling with thinking at very low cost | not published | `llama-server -hf ibm-granite/granite-4.2-3b-GGUF` |
| **Phi-4-mini-reasoning** | 3.8B dense | 2025-04 | MIT | 2.5 GB | text | math/chain-of-thought specialist | not published | `ollama pull phi4-mini-reasoning` |
| **Phi-4-multimodal-instruct** | 5.6B | 2025-02 | MIT | ~3.8 GB | text + image + speech | speech recognition/translation + vision in one model | not published | HF `microsoft/Phi-4-multimodal-instruct` (Transformers/vLLM; no Ollama tag) |
| **SmolLM3-3B** | 3B dense | 2025 | Apache 2.0 | 1.8 GB | text | tiny model with thinking mode, 92% BFCL tool calling | not published | HF `HuggingFaceTB/SmolLM3-3B` |
| **Qwen3.5-2B / 0.8B** | 2B / 0.8B | 2026-03-02 | Apache 2.0 | 2.7 / 1.2 GB | text + image | autocomplete, background tasks, batch jobs | very fast | `ollama pull qwen3.5:2b` |

## Tier 2: borderline (loads, needs care)

| Model | Params | License | 4-bit file | Notes | Speed evidence |
|---|---|---|---|---|---|
| **Qwen3.5-9B** | 9B hybrid | Apache 2.0 | 6.6 GB (Q4_K_M; Ollama tag 6.6-7.6 GB) | Strongest sub-10B model: coding, reasoning, vision. Use `--cache-type-k q8_0 --cache-type-v q8_0` and keep context at 8-16K, or close the browser. The NVFP4 tag (`qwen3.5:9b-nvfp4`) is the right pick on Blackwell if you want it fully resident. | 22-28 t/s at ~7 GB VRAM [1]; 57.9 t/s on RTX 3070 8 GB [3] |
| **Gemma 4 12B Unified** | 12B dense | Apache 2.0 | 7.0 GB (QAT q4_0) | Much stronger than E4B (MMLU-Pro 77.2, LiveCodeBench 72) and has audio input, but 7 GB leaves almost nothing for KV cache. Expect to offload 4-8 layers to RAM; then 10-15 t/s. | not published |
| **Qwen3-8B** (2025) | 8B dense | Apache 2.0 | ~5.2 GB | Older than Qwen3.5-9B; only pick if you need its specific finetunes. | 8B class [1] |
| **Ministral 3 14B** | 14B dense | Apache 2.0 | 8.5 GB | Dense, so partial offload is slow (8-15 t/s). Prefer an MoE from Tier 3 instead. | not published |

## Tier 3: MoE offload (uses the 64 GB RAM, still usable)

Run these with llama.cpp: `llama-server -m <model>.gguf -ngl 99 --n-cpu-moe 99 --flash-attn -c 32768 --cache-type-k q8_0 --cache-type-v q8_0`. Then lower `--n-cpu-moe` (e.g. 35, 30, 25) until VRAM is full; each step puts more experts on the GPU and adds speed. Ollama and LM Studio do partial offload automatically but 20-40% slower than the explicit flag.

| Model | Params | Released | License | 4-bit file | RAM needed | Best at | Measured speed on 8 GB GPU |
|---|---|---|---|---|---|---|---|
| **gpt-oss-20b** | 21B total / 3.6B active | 2025-08-05 | Apache 2.0 | 12.8 GB (official MXFP4) | ~14 GB | reasoning with adjustable effort, tool use, agents | 15.5-17.9 t/s on RTX 4060 8 GB + CPU experts [4] |
| **Qwen3.6-35B-A3B** | 35B total / 3B active | 2026-04-16 | Apache 2.0 | 19.2 GB (Q4_K_S) / ~20 GB (Q4_K_M) | ~22 GB | best local coding/agent model in this class; SWE-bench and Terminal-Bench tuned; text + image | 30-33 t/s generation, 300-1500 t/s prompt on RTX 2080 8 GB with `--n-cpu-moe 99` [2] |
| **Qwen3-Coder-30B-A3B-Instruct** | 30B / 3.3B active | 2025-07 | Apache 2.0 | ~18 GB (Q4_K_M) | ~20 GB | repository-scale coding, 256K ctx | 27 t/s full-GPU reference; same architecture as above so expect ~30 t/s with CPU experts | 
| **Gemma 4 26B-A4B** | 25.2B / 3.8B active | 2026-04-02 | Apache 2.0 | 14.4 GB (QAT q4_0) | ~16 GB | near-flagship reasoning (MMLU-Pro 82.6, AIME 88.3, LiveCodeBench 77.1) with vision, 256K ctx | no published 8 GB number; author of [2] reports it runs with `--n-cpu-moe 99` but denser routing means slower than Qwen 35B-A3B |
| **Qwen3.5-35B-A3B** | 35B / 3B active | 2026-02-24 | Apache 2.0 | 22 GB (Ollama) | ~24 GB | superseded by Qwen3.6-35B-A3B for most uses | 7-15 t/s via Ollama auto-offload [1]; use llama.cpp flag instead |

Not recommended on this hardware even with 64 GB RAM: Devstral Small 2 24B and Qwen3.6-27B / Qwen3.8-27B (dense, so offload runs 5-10 t/s), Llama 4 Scout (109B total, 60+ GB at 4-bit), Nemotron-Nano-12B-v2 (measured 10.9 t/s on an 8 GB 3070).

## Picks by job

- **Daily driver, fully on GPU:** Qwen3.5-4B. Step up to Qwen3.5-9B (NVFP4 tag) when you want quality over headroom.
- **Best coding/agent model the laptop can run:** Qwen3.6-35B-A3B Q4_K_S with `--n-cpu-moe`. 30 t/s measured on a weaker 8 GB card.
- **Reasoning with tool use:** gpt-oss-20b (16-18 t/s, adjustable reasoning effort) or Granite 4.2 8B (fully on GPU, faster).
- **Voice input / transcription:** Gemma 4 E4B (audio-in, fits) or Whisper large-v3-turbo (809M, MIT, ~1.5 GB) via faster-whisper. Phi-4-multimodal handles speech + vision in one model.
- **Vision / OCR / screenshots:** Gemma 4 E4B, Qwen3.5-4B (vision built in), or Qwen3-VL-2B (~1.1 GB) for a dedicated small VLM. Florence-2 (MIT, 0.5-1.7 GB) for detection/captioning.
- **Text-to-speech:** Kokoro-82M (Apache 2.0, 0.2 GB). Qwen3-TTS and Chatterbox exist but licenses were not confirmed.

## Blackwell runtime notes

- llama.cpp has native NVFP4 kernels for Blackwell; the `*-nvfp4` Ollama tags for Qwen3.5 (0.8b, 2b, 4b, 9b) use them. NVFP4 mainly speeds up prompt processing; generation speed is close to Q4_K_M.
- Ollama, LM Studio (0.3.15+, CUDA 12.8 backend) and llama.cpp CUDA builds all support sm_120. NVIDIA driver 570-series or newer is required.
- TensorRT-LLM and NVIDIA NIM target the RTX PRO Blackwell server/desktop parts; there are no published laptop RTX PRO 2000 results. llama.cpp is the tested path.
- Ollama's gemma4 library currently only has the 31B tag; pull Gemma 4 E2B/E4B/12B/26B QAT GGUFs from Hugging Face (`google/gemma-4-*-it-qat-q4_0-gguf`).

## Running them on a locked-down laptop (no .exe allowed, Python allowed)

Assumption: Windows 11 with application control (AppLocker or WDAC) that blocks executables the user downloads, while `python.exe` and `pip` are permitted. Verify the actual policy before building on this; the behavior below is the Microsoft-documented default.

### What the lockdown actually blocks

- AppLocker's default executable rules allow programs under `C:\Windows` and `C:\Program Files*` and block anything launched from the user profile (`%LOCALAPPDATA%`, `%APPDATA%`, Downloads). That is where LM Studio, Ollama and GPT4All install.
- The DLL rule collection is **off by default**. A `.dll` or `.pyd` that Python loads into its own process is not a process launch and is not blocked. This is the loophole that makes everything below work: llama.cpp's CUDA code and NVIDIA's cuBLAS ship as DLLs that `python.exe` loads.
- pip-created console scripts (`Scripts\foo.exe`) **are** executables and will be blocked. Always launch tools as `python -m <module>`, never by their shim name.
- If IT has turned DLL rules on, or uses WDAC in a strict mode, none of this works and you need an exception request. Test early: `python -c "import ctypes, nvidia.cublas"` after step 2 below tells you in ten seconds.

### The desktop apps

| App | Verdict | Why |
|---|---|---|
| **LM Studio** | Blocked | NSIS `.exe` installer into `%LOCALAPPDATA%\Programs`; the app and its llama.cpp backend are separate executables that must each be allowed. No Microsoft Store build. The `lms` CLI and `lmstudio` pip package only talk to a running desktop app. |
| **Ollama** | Blocked | `OllamaSetup.exe` installer, plus `ollama.exe serve` and `ollama app.exe` child processes. The portable zip still needs `ollama.exe` to run. |
| **GPT4All** | Blocked | `.exe` installer only; no portable or Store build. |
| **Jan** | Maybe | Jan is on the Microsoft Store as an MSIX package (`apps.microsoft.com/detail/xpdcnfn5cpzlqb`). Store apps fall under AppLocker's separate "Packaged app" rule collection, which many corporate policies leave open. If the Store works on the laptop, try Jan first: it bundles llama.cpp, runs GGUF models from Hugging Face, and exposes an OpenAI-compatible server on `localhost:1337`. Whether its bundled llama.cpp sidecar is also allowed is `cannot_verify` without testing. |

Short answer to "can I use LM Studio": not unless IT whitelists it. Jan via the Store is the only app-shaped option with a plausible path; otherwise go pure Python.

### Recommended: llama.cpp through Python, no compiler, no CUDA Toolkit

This gives you the same engine and speeds as LM Studio/Ollama, including `--n-cpu-moe` for the Tier 3 models, entirely inside `python.exe`.

1. **Check the driver.** Blackwell needs NVIDIA driver 570 or newer (580+ for the CUDA 13 wheels). `nvidia-smi` should run (it lives under `C:\Windows\System32`, so it is allowed) and show the RTX PRO 2000.
2. **Install CUDA runtime DLLs from pip** (no toolkit installer):
   ```
   python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
   ```
3. **Install a prebuilt Blackwell wheel of llama-cpp-python.** Use the JamePeng fork: it is current (0.4.2, 2026-10-03), ships Windows `cu128` wheels for Python 3.10-3.14, has `cpu_moe` / `n_cpu_moe` as direct `Llama()` arguments, and handles Gemma 4 audio/vision, Qwen3-VL and Qwen3-ASR through its multimodal handler. Download the wheel for your Python version from https://github.com/JamePeng/llama-cpp-python/releases/tag/v0.4.2-cu128-win-20261003 (e.g. `llama_cpp_python-0.4.2+cu128-cp312-cp312-win_amd64.whl`, ~148 MB) and:
   ```
   python -m pip install llama_cpp_python-0.4.2+cu128-cp312-cp312-win_amd64.whl
   python -c "from llama_cpp import Llama; print('ok')"
   ```
   Alternative: dougeeai publishes a single `0.3.20+cuda13.0.sm100.sm120.blackwell-py3-none-win_amd64.whl` compiled specifically for sm_120 and lists the RTX PRO 2000 Blackwell Laptop as a target, but it needs CUDA 13 DLLs (`cublas64_13.dll`), so pair it with `python -m pip install nvidia-cuda-runtime nvidia-cublas` (the unsuffixed CUDA 13 packages) and driver 580+. The upstream abetlen wheels stop at CUDA 12.5 and do not advertise sm_120; avoid them here.
4. **Make the DLLs findable.** Before importing, add the pip-installed NVIDIA `bin` folders to the DLL search path:
   ```python
   import os, sys, pathlib
   sp = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
   for d in sp.glob("*/bin"):
       os.add_dll_directory(str(d))
   from llama_cpp import Llama
   ```
5. **Download models with Python**, not the `huggingface-cli` shim:
   ```
   python -m pip install huggingface_hub
   python -c "from huggingface_hub import hf_hub_download as d; print(d('unsloth/Qwen3.5-9B-GGUF','Qwen3.5-9B-Q4_K_M.gguf'))"
   ```
6. **Run.** Fully-on-GPU model:
   ```python
   llm = Llama(model_path="Qwen3.5-9B-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=16384,
               flash_attn=True, type_k=8, type_v=8)   # 8 = q8_0 KV cache
   ```
   MoE model using the 64 GB of RAM (Tier 3):
   ```python
   llm = Llama(model_path="Qwen3.6-35B-A3B-Q4_K_S.gguf", n_gpu_layers=-1, n_cpu_moe=99,
               n_ctx=32768, flash_attn=True, type_k=8, type_v=8, n_threads=8)
   ```
   Lower `n_cpu_moe` (35, 30, 25) until VRAM fills; each step is faster.
7. **Serve it to other tools** with the bundled OpenAI-compatible server, launched as a module:
   ```
   python -m pip install "llama-cpp-python[server]" --no-deps fastapi uvicorn sse-starlette starlette-context pydantic-settings
   python -m llama_cpp.server --model Qwen3.5-9B-Q4_K_M.gguf --n_gpu_layers -1 --n_ctx 16384 --port 8080
   ```
   Then point any OpenAI-compatible client (Continue, Cline, a Python script) at `http://localhost:8080/v1`. For a chat window without any executable: `python -m pip install gradio` and a 15-line `gr.ChatInterface` script, or `python -m jupyter lab`. Open WebUI's documented launcher is a console script (`open-webui serve`); it may work as `python -m open_webui serve` but that is `cannot_verify`.

### Plan B: PyTorch stack (works, slower)

If a llama.cpp wheel will not load, `python -m pip install torch --index-url https://download.pytorch.org/whl/cu128` gives an official sm_120 build (PyTorch 2.7+). With `transformers` + `bitsandbytes>=0.43` you can run any HF model in 4-bit NF4 from a Python script. Expect roughly half the tokens/s of llama.cpp on the same model, no MoE-to-RAM trick, and 2-4x the download size (safetensors, not GGUF). `onnxruntime-genai-cuda` with Microsoft's int4 ONNX builds (Phi-4-mini, Gemma, Qwen, Llama) sits between the two in speed. vLLM and SGLang do not run on native Windows; WSL2 would need its own approval.

### Sources for this section

- AppLocker default rules and DLL rule collection: https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/applocker/understanding-applocker-default-rules and https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/applocker/dll-rules-in-applocker
- LM Studio install/CLI: https://lmstudio.ai/docs/cli ; Ollama Windows: https://docs.ollama.com/windows ; GPT4All releases: https://github.com/nomic-ai/gpt4all/releases ; Jan on the Store: https://apps.microsoft.com/detail/xpdcnfn5cpzlqb
- JamePeng llama-cpp-python (cu128 Windows wheels, `n_cpu_moe` kwarg): https://github.com/JamePeng/llama-cpp-python/releases/tag/v0.4.2-cu128-win-20261003 and `llama_cpp/llama.py` in that repo
- dougeeai Blackwell wheels: https://github.com/dougeeai/llama-cpp-python-wheels
- CUDA runtime via pip: https://pypi.org/project/nvidia-cuda-runtime-cu12/ , https://pypi.org/project/nvidia-cublas-cu12/
- PyTorch cu128: https://pytorch.org/get-started/locally/ ; bitsandbytes: https://pypi.org/project/bitsandbytes/ ; onnxruntime-genai: https://github.com/microsoft/onnxruntime-genai

## Sources

1. RTX 4060 8 GB benchmarks, Ollama, Q4_K_M, March 2026: https://docs.bswen.com/blog/2026-03-27-rtx-4060-token-speed-benchmark-coding/
2. Qwen3.6-35B-A3B on RTX 2080 8 GB with `--n-cpu-moe`, llama-bench tables, July 2026: https://lemmus.org/post/24235317
3. RTX 3070 8 GB benchmarks (Qwen3.5-9B 57.9 t/s, Nemotron-Nano-12B 10.9 t/s): https://localllm.in/blog/best-local-llms-8gb-vram-2025
4. gpt-oss-20b CPU-MoE offload on RTX 4060: https://github.com/ollama/ollama/pull/16688 and https://github.com/ggml-org/llama.cpp/pull/15077
5. Model cards: https://huggingface.co/Qwen/Qwen3.5-4B, https://huggingface.co/Qwen/Qwen3.5-9B, https://huggingface.co/Qwen/Qwen3.6-35B-A3B, https://ai.google.dev/gemma/docs/core/model_card_4, https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf, https://huggingface.co/ibm-granite/granite-4.2-8b, https://huggingface.co/openai/gpt-oss-20b, https://mistral.ai/news/mistral-3/, https://huggingface.co/microsoft/Phi-4-mini-reasoning, https://huggingface.co/microsoft/Phi-4-multimodal-instruct, https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, https://huggingface.co/nvidia/Nemotron-Nano-9B-v2, https://huggingface.co/HuggingFaceTB/SmolLM3-3B, https://huggingface.co/openai/whisper-large-v3-turbo, https://github.com/hexgrad/kokoro
6. Ollama tag sizes: https://ollama.com/library/qwen3.5/tags, https://ollama.com/library/gpt-oss
7. GPU spec: https://www.nvidia.com/en-us/products/workstations/professional-laptops/compare/
8. Blackwell runtime: https://blogs.nvidia.com/blog/rtx-ai-garage-lmstudio-llamacpp-blackwell/, https://github.com/ggml-org/llama.cpp/discussions/15013
