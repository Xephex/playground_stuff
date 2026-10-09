"""Browser chat UI (Gradio) with a model picker. Opens on http://127.0.0.1:7860

python run\webchat.py
python run\webchat.py --model qwen3.6-35b-a3b --n-cpu-moe 30
"""

import argparse, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import REGISTRY, load

_state = {"id": None, "llm": None}


def get(model_id: str, n_cpu_moe):
    if _state["id"] != model_id:
        _state["llm"] = None  # free VRAM before loading the next one
        kw = {"n_cpu_moe": n_cpu_moe} if n_cpu_moe is not None else {}
        _state["llm"] = load(model_id, **kw)
        _state["id"] = model_id
    return _state["llm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(REGISTRY))
    ap.add_argument("--n-cpu-moe", type=int)
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    import gradio as gr

    def reply(message, history, model_id, system):
        llm = get(model_id, a.n_cpu_moe)
        msgs = [{"role": "system", "content": system}]
        for h in history:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": message})
        t0 = time.perf_counter()
        n = 0
        text = ""
        for chunk in llm.create_chat_completion(
            messages=msgs, stream=True, temperature=0.7, max_tokens=4096
        ):
            piece = chunk["choices"][0]["delta"].get("content") or ""
            if piece:
                n += 1
                text += piece
                yield text
        yield (
            text + f"\n\n<sub>{n} tok, {n / (time.perf_counter() - t0):.1f} tok/s</sub>"
        )

    with gr.Blocks(title="local LLM") as demo:
        model = gr.Dropdown(
            list(REGISTRY),
            value=a.model,
            label="model (switching reloads; takes a few seconds)",
        )
        system = gr.Textbox("You are a helpful assistant.", label="system prompt")
        gr.ChatInterface(reply, additional_inputs=[model, system], type="messages")
    demo.launch(server_name="127.0.0.1", server_port=a.port, inbrowser=True)


if __name__ == "__main__":
    main()
