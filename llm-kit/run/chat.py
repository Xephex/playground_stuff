"""Terminal chat. Streams tokens and prints tokens/sec after each reply.

python run\chat.py --model qwen3.5-4b
python run\chat.py --model qwen3.6-35b-a3b --n-cpu-moe 30
python run\chat.py --model gemma4-e4b --image photo.jpg   (one image for the first turn)
"""

import argparse, base64, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import REGISTRY, load, mmproj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(REGISTRY))
    ap.add_argument(
        "--n-cpu-moe",
        type=int,
        help="MoE experts of first N layers on CPU (lower = more VRAM, faster)",
    )
    ap.add_argument("--n-ctx", type=int)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument(
        "--image",
        type=pathlib.Path,
        help="attach an image to the first message (vision models)",
    )
    a = ap.parse_args()

    kw = {
        k: v
        for k, v in {"n_cpu_moe": a.n_cpu_moe, "n_ctx": a.n_ctx}.items()
        if v is not None
    }
    handler = None
    proj = mmproj(a.model)
    if a.image:
        if not proj:
            sys.exit(f"{a.model} has no vision projector")
        from llama_cpp.llama_chat_format import GenericMtmdChatHandler

        handler = GenericMtmdChatHandler(clip_model_path=proj)
    llm = load(a.model, chat_handler=handler, **kw)

    msgs = [{"role": "system", "content": a.system}]
    first = True
    print("type a message; 'exit' to quit, '/reset' to clear history\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit"):
            break
        if user == "/reset":
            msgs = msgs[:1]
            first = True
            continue
        if not user:
            continue
        if first and a.image:
            b64 = base64.b64encode(a.image.read_bytes()).decode()
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": user},
            ]
        else:
            content = user
        first = False
        msgs.append({"role": "user", "content": content})
        t0 = time.perf_counter()
        n = 0
        out = []
        print("ai> ", end="", flush=True)
        for chunk in llm.create_chat_completion(
            messages=msgs, stream=True, temperature=0.7, max_tokens=4096
        ):
            piece = chunk["choices"][0]["delta"].get("content") or ""
            if piece:
                n += 1
                out.append(piece)
                print(piece, end="", flush=True)
        dt = time.perf_counter() - t0
        print(f"\n   [{n} tok, {n / dt:.1f} tok/s]\n")
        msgs.append({"role": "assistant", "content": "".join(out)})


if __name__ == "__main__":
    main()
