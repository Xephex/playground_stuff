r"""Ask questions about your own files, fully offline.

python run\docs.py                       (uses the docs\ folder next to kit.py)
python run\docs.py --folder C:\Users\me\Documents\policies --model qwen3.5-9b

Reads .pdf, .docx, .txt, .md, .csv, .html, .json and source files, splits them
into passages, and indexes them with a small embedding model. Each question
finds the most relevant passages and the chat model answers from them, citing
file names. The index is cached in <folder>\.llmkit-index.json and only changed
files are re-read.
"""

import argparse, hashlib, json, math, pathlib, re, sys, time, zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import MODELS, REGISTRY, ROOT, add_nvidia_dlls, load

EMBED = MODELS / "Qwen3-Embedding-0.6B-GGUF" / "Qwen3-Embedding-0.6B-Q8_0.gguf"
TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".xml", ".log", ".py", ".js", ".ts",
            ".ps1", ".bat", ".sql", ".yaml", ".yml", ".ini", ".cfg", ".rst"}
QUERY = "Instruct: Given a question, retrieve passages from the user's documents that answer it\nQuery: "
CHUNK, OVERLAP, TOP_K = 1200, 200, 6


def read_text(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    if ext in TEXT_EXT:
        text = path.read_text(encoding="utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", text) if ext in (".html", ".htm") else text
    return ""


def chunks(text: str):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    i = 0
    while i < len(text):
        end = min(len(text), i + CHUNK)
        if end < len(text):  # prefer to break at a paragraph or sentence
            cut = max(text.rfind("\n\n", i, end), text.rfind(". ", i, end))
            if cut > i + CHUNK // 2:
                end = cut + 1
        yield text[i:end].strip()
        if end >= len(text):
            break
        i = max(end - OVERLAP, i + 1)


def fingerprint(path: pathlib.Path) -> str:
    st = path.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


class Index:
    def __init__(self, folder: pathlib.Path):
        self.folder = folder
        self.file = folder / ".llmkit-index.json"
        try:
            self.data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {"files": {}}
        add_nvidia_dlls()
        from llama_cpp import Llama

        if not EMBED.exists():
            sys.exit(f"{EMBED} not found. Run: py -3.14 kit.py setup")
        self.embedder = Llama(model_path=str(EMBED), embeddings=True, pooling_type=3,
                              n_ctx=2048, n_gpu_layers=-1, verbose=False)

    def embed(self, text):
        return self.embedder.embed(text, normalize=True)

    def refresh(self):
        files = [p for p in sorted(self.folder.rglob("*"))
                 if p.is_file() and not p.name.startswith(".") and
                 (p.suffix.lower() in TEXT_EXT or p.suffix.lower() in (".pdf", ".docx"))]
        known = self.data["files"]
        changed = 0
        for p in files:
            rel = str(p.relative_to(self.folder))
            fp = fingerprint(p)
            if known.get(rel, {}).get("fp") == fp:
                continue
            try:
                parts = [c for c in chunks(read_text(p)) if len(c) > 40]
            except Exception as e:  # unreadable file: note it and move on
                print(f"  skipped {rel}: {e}")
                continue
            print(f"  indexing {rel} ({len(parts)} passages)", flush=True)
            known[rel] = {"fp": fp, "chunks": [{"text": c, "vec": self.embed(c)} for c in parts]}
            changed += 1
        for rel in [r for r in known if not (self.folder / r).exists()]:
            del known[rel]
            changed += 1
        if changed:
            self.file.write_text(json.dumps(self.data), encoding="utf-8")
        n = sum(len(f["chunks"]) for f in known.values())
        print(f"{len(known)} files, {n} passages indexed" + (" (updated)" if changed else ""))
        return n

    def search(self, question, k=TOP_K):
        q = self.embed(QUERY + question)
        scored = [(sum(a * b for a, b in zip(q, c["vec"])), rel, c["text"])
                  for rel, f in self.data["files"].items() for c in f["chunks"]]
        return sorted(scored, reverse=True)[:k]


def answer(llm, question, hits):
    sources = "\n\n".join(f"[{i + 1}] ({rel})\n{text}" for i, (_, rel, text) in enumerate(hits))
    msgs = [
        {"role": "system", "content": "Answer using only the numbered passages from the user's documents. "
                                      "Cite passages like [1]. If the passages do not contain the answer, say so."},
        {"role": "user", "content": f"Passages:\n{sources}\n\nQuestion: {question}"},
    ]
    t0, n = time.perf_counter(), 0
    for chunk in llm.create_chat_completion(messages=msgs, stream=True, temperature=0.2, max_tokens=1024):
        piece = chunk["choices"][0]["delta"].get("content") or ""
        if piece:
            n += 1
            print(piece, end="", flush=True)
    print(f"\n   [{n} tok, {n / (time.perf_counter() - t0):.1f} tok/s]")
    print("   sources: " + ", ".join(f"[{i + 1}] {rel}" for i, (_, rel, _) in enumerate(hits)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=pathlib.Path, default=ROOT / "docs-inbox")
    ap.add_argument("--model", default="qwen3.5-9b", choices=list(REGISTRY))
    ap.add_argument("--once", metavar="QUESTION", help="answer one question and exit")
    a = ap.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    print(f"documents folder: {a.folder}")
    idx = Index(a.folder)
    if not idx.refresh():
        print(f"Put PDF, Word or text files in {a.folder} and run this again.")
        return
    llm = load(a.model, n_ctx=16384)
    if a.once:
        answer(llm, a.once, idx.search(a.once))
        return
    print("ask a question about your documents. /reload re-reads the folder, exit quits.\n")
    while True:
        try:
            q = input("question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q == "/reload":
            idx.refresh()
            continue
        if q:
            answer(llm, q, idx.search(q))


if __name__ == "__main__":
    main()
