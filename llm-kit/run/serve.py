r"""OpenAI-compatible API on http://127.0.0.1:8080/v1 for editors and tools
(Continue, Cline, Pi, n8n, any `openai` client with base_url set).

    python run\serve.py --model qwen3.5-9b
    python run\serve.py --model qwen3.6-35b-a3b --n-cpu-moe 30

Runs llama-cpp-python's server, with the model loaded by run\_common.load()
so it uses exactly the same settings as chat (GPU layers, KV cache type,
MoE offload, saved tuning).
"""

import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import MODELS, REGISTRY, add_nvidia_dlls, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(REGISTRY))
    ap.add_argument("--n-cpu-moe", type=int)
    ap.add_argument("--n-ctx", type=int)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key", help="require this key (Authorization: Bearer KEY)")
    a = ap.parse_args()

    add_nvidia_dlls()
    import toolcalls

    import uvicorn
    from llama_cpp.server import app as server_app, model as server_model
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    toolcalls.fix_request_types()  # accept tool-call history (llama-cpp-python 0.4.2 bug)
    toolcalls.install()  # return tool calls as OpenAI tool_calls, not text

    overrides = {k: v for k, v in {"n_cpu_moe": a.n_cpu_moe, "n_ctx": a.n_ctx}.items() if v is not None}

    def load_with_kit_settings(_settings):
        return load(a.model, **overrides)

    server_model.LlamaProxy.load_llama_from_model_settings = staticmethod(load_with_kit_settings)

    rel = REGISTRY[a.model][0]
    app = server_app.create_app(
        server_settings=ServerSettings(host=a.host, port=a.port, api_key=a.api_key),
        model_settings=[ModelSettings(model=str(MODELS / rel), model_alias=a.model)],
    )
    toolcalls.relax_response_check(app)
    print(f"serving {a.model} on http://{a.host}:{a.port}/v1", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
