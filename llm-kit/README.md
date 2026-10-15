# Offline local-LLM kit for Windows

Everything needed to run open-weight models on an NVIDIA RTX PRO 2000 Blackwell laptop (8 GB VRAM, 64 GB RAM) with plain Python. Nothing to install besides Python, and no internet needed after cloning.

Large files are stored as `*.partNNN` chunks (under 95 MB each) because GitHub rejects files over 100 MB. `kit.py setup` reassembles them.

## Start here

Everything is driven by one tool, `kit.py`. Open **Command Prompt** (Start, type `cmd`, Enter) and paste these lines one at a time.

**First time on this laptop:**
```
git clone https://github.com/Xephex/playground_stuff
cd playground_stuff\llm-kit
py -3.14 kit.py setup
```

**Already cloned it before (you had the Python 3.12 version):**
```
cd playground_stuff\llm-kit
git pull --no-rebase
py -3.14 kit.py setup
```

The first setup takes 10-30 minutes; it reassembles about 50 GB of model files. It ends with `GPU OK: NVIDIA RTX PRO 2000 Blackwell ...`. Running it again is safe: anything already done is checked and skipped.

**Every time after that:**
```
cd playground_stuff\llm-kit
py -3.14 kit.py
```
That opens this menu:
```
================ llm-kit ================
 Python 3.14.7   GPU: NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU
 Setup: ready   Models ready: 6/6
------------------------------------------
  1) Chat here in this window
  2) Chat in the browser
  3) Ask about an image (or audio)
  4) Ask questions about your documents
  5) Pi coding agent
  6) API server for editors (this laptop only)
  7) Share API with n8n / other machines (needs key)
  8) Tune speed for this GPU
  9) Setup / repair
 10) Update from GitHub, then setup
 11) Status details
  q) Quit
>
```
Type a number and press Enter; options 1-7 then ask which model (Enter = the default). To leave a chat, type `exit`; to stop the browser chat, API server or Pi, press `Ctrl+C`. Setup also puts an **llm-kit** shortcut on your desktop that opens this menu.

**Do this once after setup:** choose **8) Tune speed**. It tries settings for the two big models (`gpt-oss-20b`, `qwen3.6-35b-a3b`) on your GPU, keeps the fastest that fits, and every later run uses them. Takes about 10 minutes.

### What each option is for

- **1 Chat.** Type, get answers. Chats are saved as readable files in `chats\`. In a chat, `/image C:\path\photo.png` attaches a picture to your next message (`qwen3.5-4b`, `qwen3.5-9b`, `qwen3.6-35b-a3b`; `gemma4-e4b` also takes audio files), `/reset` starts over.
- **3 Image.** Pick a picture (drag it onto the window), then ask about it.
- **4 Documents.** Put PDF, Word (.docx), text or Markdown files in the `docs-inbox` folder (or point it at any folder). It reads and indexes them on this laptop, then answers questions with the file names it used. New or changed files are picked up the next time you open it; `/reload` re-reads without leaving.
- **5 Pi coding agent.** [Pi](https://github.com/badlogic/pi-mono) is a coding agent that reads, writes and edits files and runs commands in the folder you start it from, using the kit's model. It needs **Node.js 20.6 or newer** (`node --version`) and Git for Windows (for its shell). Open Command Prompt in your project folder and run `py -3.14 <path-to>\llm-kit\kit.py pi`. `qwen3.6-35b-a3b` is the best model for it; `granite-4.2-8b` is the faster option. Type `/quit` to leave Pi.
- **6 / 7 API.** OpenAI-compatible API for editors and tools (6) or for n8n on another machine (7, protected by a key). See "Use the models from n8n" below; ready-made workflows are in `n8n\`.

### Use the models from n8n

1. On the laptop: `py -3.14 kit.py`, choose **7**, and pick a model. `granite-4.2-8b` is the best one here for agents and tool calling; `qwen3.5-9b` is better for plain text.
2. The window prints an **API Key**, one or more **Base URL**s and the **model name**. The key is generated once and stays the same every time.
3. In n8n: **Credentials > Add credential > OpenAI**. Paste the API Key, and set **Base URL** to the one printed (`http://<laptop-ip>:8080/v1`). Save; n8n tests the connection.
4. In a workflow, add an **OpenAI Chat Model** node (or an **AI Agent** with that chat model), pick that credential, and type the model name from step 2 in **Model** (choose "By ID" if it shows a list).

Leave the window open while n8n uses it; `Ctrl+C` stops sharing. If Windows asks whether Python may use networks, choose **Private networks**. If n8n cannot connect, the machine running n8n cannot reach the laptop: same network, and no company firewall or VPN in between. Your laptop's IP can change; check the window each time.

The server runs one model at a time. To switch, stop it and choose 7 again. Two ready-made workflows (a chat agent and a summarize webhook) are in `n8n\`; see `n8n\README.md`.

### If something goes wrong

| What you see | What to do |
|---|---|
| `'py' is not recognized` or `No suitable Python runtime found` | Python 3.14 is not installed. In the Microsoft Store, search **Python 3.14** and click **Get** (no admin needed). Then rerun the command. |
| `'git' is not recognized` | Git is not installed. Download the repo as a ZIP from GitHub instead (green **Code** button, **Download ZIP**), unzip it, and run `py -3.14 kit.py setup` inside `llm-kit`. To update later, download the ZIP again. |
| `Not enough disk` | The first setup needs about 55 GB free. Free up space and run setup again; finished files are kept. |
| `GPU check failed` / `NO CUDA DEVICE` | Run `py -3.14 kit.py gpucheck`. It asks the NVIDIA driver directly and prints the reason: GPU switched off (laptop graphics mode set to integrated/eco, battery saver, or disabled in Device Manager), driver too old, or a missing DLL. |
| `DLL load failed` | Run `py -3.14 kit.py gpucheck` and send its output. |
| Out of memory with a big model | Use a smaller model, or close the browser and other GPU apps. |

Do **not** run `git checkout -- .`, `git stash` or `git reset` to "clean up". Setup deletes the downloaded chunk files after joining them, and those commands bring all of them back (about 40 GB).

### Direct commands (optional)

The menu does all of this. These are for scripts or shortcuts; all of them take `--model ID` to skip the model picker.

| What | Command |
|---|---|
| Chat in this window | `py -3.14 kit.py chat --model qwen3.5-4b` |
| Browser chat (http://127.0.0.1:7860) | `py -3.14 kit.py web` |
| API for editors (http://127.0.0.1:8080/v1) | `py -3.14 kit.py serve --model qwen3.5-9b` |
| API shared for n8n, with key | `py -3.14 kit.py share --model granite-4.2-8b` |
| Check what is installed and ready | `py -3.14 kit.py status` |
| Update from GitHub and install | `py -3.14 kit.py update` |
| One question, answer, exit | `py -3.14 kit.py ask --model qwen3.5-9b "your question"` |
| Ask about an image | `py -3.14 kit.py image --model qwen3.5-9b --image screenshot.png` |
| Questions about a folder of documents | `py -3.14 kit.py docs --folder C:\path\to\folder` |
| Pi coding agent in the current folder | `py -3.14 kit.py pi --model qwen3.6-35b-a3b` |
| Tune speed for this GPU | `py -3.14 kit.py tune` |

## Requirements

- Windows 11 x64 with NVIDIA driver 570 or newer (`nvidia-smi` shows the version).
- Python **3.14** x64; the bundled wheels are `cp314-win_amd64`, built and checked against 3.14.7. The Microsoft Store build is fine and needs no admin rights.
- About 55 GB free disk during the first setup.

## Models included

| id | File | Size | Fits | Use for |
|---|---|---|---|---|
| `qwen3.5-4b` | Qwen3.5-4B Q4_K_M + vision projector | 2.7 + 0.7 GB | GPU | daily driver; chat, code, images; 40+ tok/s |
| `qwen3.5-9b` | Qwen3.5-9B Q4_K_M + vision projector | 5.7 + 0.9 GB | GPU, tight | best quality that stays on the GPU; 22-28 tok/s; context capped at 12K |
| `gemma4-e4b` | Gemma 4 E4B QAT q4_0 + projector | 5.2 + 1.0 GB | GPU | text, images **and audio** input; 128K context |
| `granite-4.2-8b` | Granite 4.2 8B Q4_K_M | 5.4 GB | GPU | tool calling, agents, thinking mode |
| `gpt-oss-20b` | gpt-oss-20b MXFP4 | 12.1 GB | RAM offload | reasoning with adjustable effort; ~16-18 tok/s |
| `qwen3.6-35b-a3b` | Qwen3.6-35B-A3B UD-Q4_K_M + projector | 22.1 + 0.9 GB | RAM offload | strongest coding/agent model here; ~30 tok/s after tuning |
| (search) | Qwen3-Embedding-0.6B Q8_0 | 0.6 GB | GPU | used by option 4 to find passages in your documents |

Licenses: Qwen, Gemma 4, Granite, gpt-oss are all Apache 2.0. Each model folder keeps its upstream `LICENSE`/README where the repo provided one.

Speeds are from published runs on 8 GB RTX 4060 / 2080 cards; this GPU has more bandwidth, so expect similar or better on mains power. Full research notes with sources: `docs/local-models-rtx-pro-2000-blackwell.md`.

## How it works

- Everything runs as `python -m ...` or `python script.py`. pip normally also creates `.exe` launchers (`gradio.exe`, `uvicorn.exe`, ...) in Python's `Scripts` folder; the kit does not use them, so `kit.py setup` removes them (only launchers listed in the kit's own packages' install records).

- `llama_cpp_python-0.4.2+cu128-cp314-cp314-win_amd64.whl` is the [JamePeng fork](https://github.com/JamePeng/llama-cpp-python) built for CUDA 12.8; it contains `llama.dll` and `ggml-cuda.dll`, which `python.exe` loads as libraries.
- `nvidia_cublas_cu12` and `nvidia_cuda_runtime_cu12` wheels supply `cublas64_12.dll` and `cudart64_12.dll`; `run\_common.py` adds their folders to the DLL search path and PATH at import time, so no CUDA Toolkit install is needed.

The llama-cpp-python wheel is a community build from the JamePeng fork; its SHA-256 is in `wheels/SHA256SUMS`.

## Troubleshooting (details)

- GPU check says `NO CUDA DEVICE`: run `nvidia-smi`; driver must be >= 570. Then `py -3.14 -c "import nvidia.cublas, os; print(os.listdir(os.path.dirname(nvidia.cublas.__file__)+'/bin'))"` should list `cublas64_12.dll`.
- `ImportError: DLL load failed` on `import llama_cpp`: Python is not 3.14 x64, or setup did not finish; rerun setup.
- Out of memory loading `qwen3.5-9b`: close the browser, or add `--n-ctx 8192`.
- MoE model slow: run menu option 8 (Tune speed); it finds and saves the fastest setting.
- Adding another model: drop a `.gguf` under `models\<folder>\`, add a line to `REGISTRY` in `run\_common.py`.

## Layout

```
llm-kit/
  kit.py              the tool: menu and every action (setup, chat, docs, pi, tune, ...)
  setup_env.py        older one-shot setup script (kit.py setup replaces it)
  split_tool.py       split/join/verify (python split_tool.py --help)
  wheels/             win_amd64 cp314 wheels (+ .part chunks for the big one), SHA256SUMS
  models/<name>/      .gguf files as .partNNN chunks + .sha256
  run/                chat, webchat, docs, pi, tune, serve (+ toolcalls), check_gpu, _common
  pi/                 Pi coding agent bundle (unpacked to pi\app on first use)
  n8n/                ready-made n8n workflows
  docs-inbox/         put documents here for menu option 4
  chats/              saved conversations
  tools/              build_bundle.py: rebuild and verify the offline bundle
  tests/              python -m unittest discover -s tests
  docs/               research notes
```
