"""The kit's server returns tool calls as OpenAI tool_calls, streaming and not.

Runs llama-cpp-python's real server app with a stand-in model (no GGUF needed).
Skipped when llama-cpp-python is not installed (run it in the kit's environment).
"""

import json, pathlib, sys, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "run"))

try:
    import llama_cpp
    from fastapi.testclient import TestClient
    from llama_cpp.server import app as server_app, model as server_model
    from llama_cpp.server.settings import ModelSettings, ServerSettings
except ImportError:  # pragma: no cover - plain Python without the kit's packages
    llama_cpp = None

import toolcalls

QWEN_CALL = ("<tool_call>\n<function=write_file>\n<parameter=file_path>\nhello.py\n</parameter>\n"
             "<parameter=content>\nprint(2 + 3)\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
    "file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}}]


class FakeLlama:
    """Answers every chat request with the text tool call Qwen3.5 emits."""

    def create_chat_completion(self, **kw):
        if kw.get("stream"):
            def gen():
                yield {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                       "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                for i in range(0, len(QWEN_CALL), 7):
                    yield {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                           "choices": [{"index": 0, "delta": {"content": QWEN_CALL[i:i + 7]}, "finish_reason": None}]}
                yield {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            return gen()
        return {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": QWEN_CALL},
                             "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def close(self):
        pass


@unittest.skipIf(llama_cpp is None, "llama-cpp-python is not installed")
class ServeToolCallsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches = [
            mock.patch.object(llama_cpp.Llama, "create_chat_completion", FakeLlama.create_chat_completion),
            mock.patch.object(server_model.LlamaProxy, "load_llama_from_model_settings",
                              staticmethod(lambda settings: llama_cpp.Llama.__new__(llama_cpp.Llama))),
            mock.patch.object(llama_cpp.Llama, "close", lambda self: None, create=True),
        ]
        for p in cls.patches:
            p.start()
        toolcalls.fix_request_types()
        toolcalls.install()
        app = server_app.create_app(server_settings=ServerSettings(),
                                    model_settings=[ModelSettings(model="fake.gguf", model_alias="m")])
        toolcalls.relax_response_check(app)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):
            p.stop()

    def post(self, stream):
        return self.client.post("/v1/chat/completions", json={
            "model": "m", "stream": stream, "tools": TOOLS,
            "messages": [{"role": "user", "content": "write hello.py"}]})

    def test_non_streaming_reply_with_tool_call(self):
        r = self.post(False)
        self.assertEqual(r.status_code, 200, r.text[:400])
        choice = r.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "write_file")
        self.assertEqual(json.loads(call["arguments"]), {"file_path": "hello.py", "content": "print(2 + 3)"})

    def test_streaming_reply_with_tool_call(self):
        r = self.post(True)
        self.assertEqual(r.status_code, 200)
        calls, finish = [], None
        for line in r.text.splitlines():
            if line.startswith("data: {"):
                c = json.loads(line[6:])["choices"][0]
                calls += c["delta"].get("tool_calls") or []
                finish = c.get("finish_reason") or finish
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(calls[0]["function"]["name"], "write_file")


if __name__ == "__main__":
    unittest.main()
