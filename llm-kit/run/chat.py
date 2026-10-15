r"""Terminal chat. Streams tokens and prints tokens/sec after each reply.

python run\chat.py --model qwen3.5-4b
python run\chat.py --model qwen3.6-35b-a3b --n-cpu-moe 30
python run\chat.py --model gemma4-e4b --image photo.jpg
python run\chat.py --model qwen3.5-4b --once "question"   (answer and exit)

In the chat:  /image FILE  attach a picture (or audio file, gemma4-e4b) to your next message
              /reset       clear the conversation      /save  write it to chats\ now
              exit         quit (the conversation is saved to chats\ automatically)
"""

import argparse, datetime, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import REGISTRY, ROOT, load, media_part, mmproj

CHATS = ROOT / "chats"


def strip_thinking(text):
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)


def save(msgs, model, path):
    """Write the conversation as Markdown (readable) and JSON (reloadable)."""
    if len(msgs) <= 1:
        return None
    CHATS.mkdir(exist_ok=True)
    lines = [f"# Chat with {model}", f"_{datetime.datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in msgs[1:]:
        c = m["content"]
        if isinstance(c, list):
            c = " ".join(p.get("text", "[attachment]") for p in c)
        lines += [f"**{'You' if m['role'] == 'user' else 'AI'}:** {c}", ""]
    md, js = path.parent / (path.name + ".md"), path.parent / (path.name + ".json")  # model ids contain dots
    md.write_text("\n".join(lines), encoding="utf-8")
    plain = [m if isinstance(m["content"], str) else
             dict(m, content=" ".join(p.get("text", "[attachment]") for p in m["content"])) for m in msgs]
    js.write_text(json.dumps({"model": model, "messages": plain}, indent=1), encoding="utf-8")
    return md


def reply(llm, msgs, show=True):
    t0, n, out = time.perf_counter(), 0, []
    for chunk in llm.create_chat_completion(messages=msgs, stream=True, temperature=0.7, max_tokens=4096):
        piece = chunk["choices"][0]["delta"].get("content") or ""
        if piece:
            n += 1
            out.append(piece)
            if show:
                print(piece, end="", flush=True)
    return "".join(out), n, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(REGISTRY))
    ap.add_argument("--n-cpu-moe", type=int, help="MoE experts of first N layers on CPU (lower = more VRAM, faster)")
    ap.add_argument("--n-ctx", type=int)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--image", type=pathlib.Path, help="attach an image or audio file to the first message")
    ap.add_argument("--once", metavar="TEXT", help="answer this one message and exit")
    ap.add_argument("--resume", type=pathlib.Path, help="continue a chat saved in chats\\ (.json)")
    a = ap.parse_args()

    kw = {k: v for k, v in {"n_cpu_moe": a.n_cpu_moe, "n_ctx": a.n_ctx}.items() if v is not None}
    llm = load(a.model, vision=bool(a.image), **kw)

    msgs = [{"role": "system", "content": a.system}]
    if a.resume:
        msgs = json.loads(a.resume.read_text(encoding="utf-8"))["messages"]
    pending = [a.image] if a.image else []

    if a.once:
        content = [*(media_part(p) for p in pending), {"type": "text", "text": a.once}] if pending else a.once
        text, _, _ = reply(llm, msgs + [{"role": "user", "content": content}], show=False)
        print(strip_thinking(text).strip())
        return

    path = CHATS / f"{datetime.datetime.now():%Y-%m-%d_%H%M%S}_{a.model}"
    print("type a message. /image FILE attaches a picture, /reset clears, exit quits.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit"):
            break
        if user == "/reset":
            msgs, pending = msgs[:1], []
            continue
        if user == "/save":
            print(f"   saved to {save(msgs, a.model, path)}")
            continue
        if user.startswith("/image"):
            f = pathlib.Path(user[6:].strip().strip('"'))
            if not mmproj(a.model):
                print(f"   {a.model} cannot read images; use qwen3.5-4b, qwen3.5-9b, gemma4-e4b or qwen3.6-35b-a3b")
            elif not f.is_file():
                print(f"   file not found: {f}")
            else:
                if llm.chat_handler is None:  # first image in this chat: add the vision part
                    from _common import media_handler
                    llm.chat_handler = media_handler(a.model)
                pending.append(f)
                print(f"   attached {f.name}; it goes with your next message")
            continue
        if not user:
            continue
        content = [*(media_part(p) for p in pending), {"type": "text", "text": user}] if pending else user
        pending = []
        msgs.append({"role": "user", "content": content})
        print("ai> ", end="", flush=True)
        text, n, dt = reply(llm, msgs)
        print(f"\n   [{n} tok, {n / dt:.1f} tok/s]\n")
        msgs.append({"role": "assistant", "content": text})
    saved = save(msgs, a.model, path)
    if saved:
        print(f"conversation saved to {saved}")


if __name__ == "__main__":
    main()
