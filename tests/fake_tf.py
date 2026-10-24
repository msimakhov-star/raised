"""A fake Token Factory (OpenAI-compatible POST /v1/chat/completions) for offline tests.

    with FakeTF({"TASK: CHART_SEMANTICS": [vote1, vote2, vote3]}) as tf:
        client = tf.client()          # a raised.llm.TokenFactory pointed at the fake, no retry sleeps
        ...
        tf.requests                   # every request body received, in order

Routes map a marker (a substring of the request's text parts) to the answer: a dict (sent as JSON content), a str
(sent verbatim), a list (one item per call, cycling) or a callable(body) -> dict | str. The defaults answer the three
Super VL prompts in raised/prompts.py. The fake is awkward on purpose, where the real service may be:
- reject_wrapper: the OpenAI json_schema wrapper gets 400, so the bare-schema fallback runs;
- honour_n=False: n is ignored and one choice comes back, so the separate-calls fallback runs;
- reject_params: request keys answered with 400 (e.g. "reasoning_effort");
- usage reports prompt_tokens_details.cached_tokens: 0 the first time a prompt is seen, `cached` after that;
- fail: HTTP statuses returned (and used up) before any normal answer, for retry tests;
- a canned dict that does not match the request's schema gets 500, so canned data cannot drift from the schemas.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from raised.llm import TokenFactory, conforms

SEMANTICS_VOTE = {
    "chart_type": "line", "title": "Exports and imports", "x_title": "Year", "y_title": "US$ bn",
    "x_scale": "linear", "y_scale": "linear", "x_ticks": ["2000", "2005", "2010"], "y_ticks": ["0", "5", "10"],
    "categories": [], "series": [{"name": "Exports", "color": "blue", "style": "solid", "marker": "o"},
                                 {"name": "Imports", "color": "red", "style": "dashed", "marker": ""}],
    "legend": ["Exports", "Imports"], "source": "Source: synthetic",
    "reading": {"x": ["2000", "2005", "2010"], "series": [{"name": "Exports", "values": [2.0, 5.5, 7.9]},
                                                          {"name": "Imports", "values": [3.0, 4.0, None]}]},
}
FIGURES_ANSWER = {"figures": [{"figure_number": "1.2", "caption": "Figure 1.2 Exports and imports, 2000 to 2010",
                               "is_data_chart": True, "box": {"x0": 100, "y0": 200, "x1": 900, "y1": 600}}]}
VERIFY_ANSWER = {"issues": [{"x": "2010", "series": "Exports", "issue": "redraw lower than original: about 7.1 against 7.9"}]}
DEFAULT_ROUTES = {"TASK: CHART_SEMANTICS": SEMANTICS_VOTE, "TASK: FIND_FIGURES": FIGURES_ANSWER,
                  "TASK: VERIFY_REDRAW": VERIFY_ANSWER}


class FakeTF:
    def __init__(self, routes: dict | None = None, *, reject_wrapper: bool = True, honour_n: bool = False,
                 reject_params: tuple = (), fail: list[int] | None = None, prompt_tokens: int = 1000,
                 completion_tokens: int = 200, cached: int = 800):
        self.routes = {**DEFAULT_ROUTES, **(routes or {})}
        self.reject_wrapper, self.honour_n, self.reject_params = reject_wrapper, honour_n, reject_params
        self.fail = list(fail or [])
        self.prompt_tokens, self.completion_tokens, self.cached = prompt_tokens, completion_tokens, cached
        self.requests: list[dict] = []
        self._calls: Counter = Counter()
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def __enter__(self) -> "FakeTF":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                status, payload = fake._answer(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("x-ratelimit-remaining-requests", "99")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def client(self, **kw) -> TokenFactory:
        tf = TokenFactory(api_key="test-key", base_url=self.url, **kw)
        tf.backoff = 0
        return tf

    def _answer(self, body: dict) -> tuple[int, dict]:
        with self._lock:
            self.requests.append(body)
            if self.fail:
                return self.fail.pop(0), {"error": {"message": "injected failure"}}
            js = (body.get("response_format") or {}).get("json_schema")
            wrapped = isinstance(js, dict) and {"name", "schema"} <= set(js)
            if wrapped and self.reject_wrapper:
                return 400, {"error": {"message": "response_format.json_schema: unexpected keys name, schema, strict"}}
            if bad := [p for p in self.reject_params if p in body]:
                return 400, {"error": {"message": f"unsupported parameter {bad[0]}"}}
            schema = js["schema"] if wrapped else js
            text = _text(body["messages"])
            marker = next((m for m in self.routes if m in text), None)
            if marker is None:
                return 404, {"error": {"message": "fake: no route matches this prompt"}}
            contents = []
            for _ in range(body.get("n", 1) if self.honour_n else 1):
                ans = self.routes[marker]
                if isinstance(ans, list):
                    ans = ans[self._calls[marker] % len(ans)]
                self._calls[marker] += 1
                if callable(ans):
                    ans = ans(body)
                if isinstance(ans, dict):
                    if schema and not conforms(ans, schema):
                        return 500, {"error": {"message": f"fake: canned answer for {marker!r} does not match the schema"}}
                    ans = json.dumps(ans)
                contents.append(ans)
            prompt = json.dumps(body["messages"], sort_keys=True)
            cached = self.cached if prompt in self._seen else 0
            self._seen.add(prompt)
        usage = {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens * len(contents),
                 "total_tokens": self.prompt_tokens + self.completion_tokens * len(contents),
                 "prompt_tokens_details": {"cached_tokens": cached}}
        return 200, {"id": "fake", "object": "chat.completion", "model": body["model"], "usage": usage,
                     "choices": [{"index": i, "finish_reason": "stop", "message": {"role": "assistant", "content": c}}
                                 for i, c in enumerate(contents)]}


def _text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        parts += [c] if isinstance(c, str) else [p.get("text", "") for p in c or [] if p.get("type") == "text"]
    return "\n".join(parts)


def images(body: dict) -> list[str]:
    """The data URLs of every image part in a request."""
    return [p["image_url"]["url"] for m in body["messages"] if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]
