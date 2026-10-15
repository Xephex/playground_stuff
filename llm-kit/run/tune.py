r"""Find the fastest settings that fit this GPU, and save them for every run.

python run\tune.py --model qwen3.6-35b-a3b
python run\tune.py --all

Mixture-of-experts models (gpt-oss-20b, qwen3.6-35b-a3b) keep some expert
layers in system RAM (n_cpu_moe). Fewer layers in RAM is faster but needs more
VRAM. This tries values from all-in-RAM downwards, measures tokens/second, stops
when a value no longer fits, and saves the fastest one. Other models are checked
at their default context; if that does not fit, the context is halved until it does.

Each attempt runs in a separate Python process, so a failed load cannot leave
GPU memory allocated.
"""

import argparse, json, pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import DEFAULTS, REGISTRY, STATE, MODELS

PROMPT = "Write three sentences about the history of the bicycle."
TOKENS = 96


def probe(model_id, kwargs, timeout=900):
    """Load with kwargs in a child process and return tokens/sec, or None if it failed."""
    code = f"""
import json, sys, time
sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})
from _common import load
llm = load({model_id!r}, **json.loads({json.dumps(json.dumps(kwargs))}))
llm.create_chat_completion(messages=[{{"role": "user", "content": "hi"}}], max_tokens=4)
t = time.perf_counter(); n = 0
for c in llm.create_chat_completion(messages=[{{"role": "user", "content": {PROMPT!r}}}],
                                    stream=True, max_tokens={TOKENS}, temperature=0):
    if c["choices"][0]["delta"].get("content"):
        n += 1
print("TPS", n / (time.perf_counter() - t))
"""
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("TPS "):
            return float(line.split()[1])
    return None


def save(model_id, kwargs, tps):
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}
    state.setdefault("tuned", {})[model_id] = {"kwargs": kwargs, "tok_per_s": round(tps, 1),
                                               "when": time.strftime("%Y-%m-%d %H:%M")}
    STATE.write_text(json.dumps(state, indent=1))


def tune(model_id):
    rel, _, extra = REGISTRY[model_id]
    if not (MODELS / rel).exists():
        print(f"{model_id}: not set up yet, skipped")
        return
    print(f"\n== {model_id}")
    if "n_cpu_moe" in extra:
        n_layers = layer_count(MODELS / rel) or 48
        best, best_tps, misses = None, 0.0, 0
        value = n_layers
        while value >= 0:
            tps = probe(model_id, {"n_cpu_moe": value})
            print(f"  n_cpu_moe={value:<3} " + (f"{tps:5.1f} tok/s" if tps else "does not fit"), flush=True)
            if tps is None:
                break
            if tps > best_tps:
                best, best_tps, misses = value, tps, 0
            else:
                misses += 1
                if misses >= 2:  # stopped getting faster
                    break
            value -= 4
        if best is None:
            print("  could not load at all; check `kit.py gpucheck`")
            return
        save(model_id, {"n_cpu_moe": best}, best_tps)
        print(f"  saved: n_cpu_moe={best} ({best_tps:.1f} tok/s)")
        return
    n_ctx = extra.get("n_ctx", DEFAULTS["n_ctx"])
    while n_ctx >= 2048:
        tps = probe(model_id, {"n_ctx": n_ctx})
        print(f"  n_ctx={n_ctx:<6} " + (f"{tps:5.1f} tok/s" if tps else "does not fit"), flush=True)
        if tps:
            save(model_id, {"n_ctx": n_ctx}, tps)
            print(f"  saved: n_ctx={n_ctx} ({tps:.1f} tok/s)")
            return
        n_ctx //= 2
    print("  could not load at all; check `kit.py gpucheck`")


def layer_count(path):
    """Read <arch>.block_count from the GGUF header without loading the model."""
    import struct

    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)
            _, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]
                key = f.read(klen).decode("utf-8", "replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                if key.endswith(".block_count") and vtype in (4, 5):
                    return struct.unpack("<I" if vtype == 4 else "<i", f.read(4))[0]
                skip_value(f, vtype)
    except (OSError, struct.error):
        return None
    return None


SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def skip_value(f, vtype):
    import struct

    if vtype in SIZES:
        f.read(SIZES[vtype])
    elif vtype == 8:  # string
        f.read(struct.unpack("<Q", f.read(8))[0])
    elif vtype == 9:  # array
        etype, n = struct.unpack("<IQ", f.read(12))
        if etype in SIZES:
            f.read(SIZES[etype] * n)
        else:
            for _ in range(n):
                skip_value(f, etype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(REGISTRY))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for mid in (REGISTRY if a.all else [a.model or "qwen3.6-35b-a3b"]):
        tune(mid)


if __name__ == "__main__":
    main()
