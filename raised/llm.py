"""Thin client for Nebius Token Factory (OpenAI-compatible) with a per-run cost meter.

Copied from Homecoming's homecoming/llm.py (same author, MIT) and extended for Raised: cached-token pricing,
n > 1 choices with a separate-calls fallback, and a PNG-preserving shrink to 2000 px for charts.

Sources (all fetched 2026-10-24):
- Model ids and prices: https://tokenfactory.nebius.com/api/public/models_info (USD per 1M tokens; fields
  input_price_per_million_tokens, input_cached_price_per_million_tokens, output_price_per_million_tokens).
- Cached tokens: https://docs.tokenfactory.nebius.com/cached-tokens ("prompt_tokens ... is always the total number
  of input tokens, cached tokens included"; the cached count is usage.prompt_tokens_details.cached_tokens).
- `n` (1..128) and `reasoning_effort` (none|minimal|low|medium|high|xhigh|max) are request fields in
  https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion.md. Whether the engine
  honours n is measured by scripts/probe.py, and chat_json_n copes either way.
- The bare {"type": "json_schema", "json_schema": <schema>} form is the one shown in
  https://docs.tokenfactory.nebius.com/ai-models-inference/json.md.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")

MODELS = {
    "vl": "nvidia/Nemotron-3.5-Super-VL",          # finds figures, reads chart semantics, checks the redraw
    "reason": "nvidia/Nemotron-3-Ultra-550b-a55b",  # long description and caption audit
    "fast": "nvidia/Nemotron-3_5-Lightning",        # support guard, braille label shortening, alt text
    "minicpm": "openbmb/MiniCPM-V-4_5",             # non-NVIDIA baseline for the bench
}
PRICES = {  # (input, cached input, output) USD per 1M tokens; None = the catalogue lists no cached price
    "nvidia/Nemotron-3.5-Super-VL": (0.30, 0.06, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 0.12, 3.00),
    "nvidia/Nemotron-3_5-Lightning": (0.06, None, 0.24),
    "openbmb/MiniCPM-V-4_5": (0.658, None, 1.11),
}


class Meter:
    def __init__(self):
        self.calls: list[dict] = []

    def add(self, model: str, usage: dict | None, ms: int, step: str):
        """Cached tokens are billed at the cached price, or at the full input price when the model has none."""
        usage = usage or {}
        pi, pc, po = PRICES.get(model, (0, None, 0))
        prompt, out = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        cached = min(prompt, (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        cost = ((prompt - cached) * pi + cached * (pi if pc is None else pc) + out * po) / 1e6
        self.calls.append({"step": step, "model": model, "ms": ms, "in": prompt, "cached": cached, "out": out,
                           "usd": round(cost, 6)})

    @property
    def total_usd(self) -> float:
        return round(sum(c["usd"] for c in self.calls), 4)

    @property
    def cached_share(self) -> float:
        tokens = sum(c["in"] for c in self.calls)
        return round(sum(c["cached"] for c in self.calls) / tokens, 3) if tokens else 0.0

    def summary(self) -> dict:
        return {"calls": self.calls, "total_usd": self.total_usd, "cached_share": self.cached_share}


class TFError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, headers: dict | None = None):
        super().__init__(msg)
        self.status, self.headers = status, headers or {}


class TokenFactory:
    backoff = 1.5  # seconds times the attempt number before retrying 429, 5xx and network errors

    def __init__(self, api_key: str | None = None, meter: Meter | None = None, timeout: float = 120,
                 base_url: str | None = None):
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        self.meter = meter or Meter()
        self.timeout = timeout
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.last_headers: dict = {}
        self._bare: set[str] = set()       # models that rejected the OpenAI json_schema wrapper
        self._no_effort: set[str] = set()  # models that rejected reasoning_effort

    @property
    def live(self) -> bool:
        return bool(self.api_key)

    def post(self, body: dict) -> dict:
        """One raw POST to /chat/completions, retrying 429, 5xx and network errors twice. Raises TFError."""
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.last_headers = dict(r.headers)
                    return json.load(r)
            except urllib.error.HTTPError as e:
                self.last_headers = dict(e.headers or {})
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise TFError(f"{body['model']} HTTP {e.code}: {e.read()[:400]!r}", e.code, self.last_headers) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 2:
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise TFError(f"{body['model']} unreachable: {e}") from e

    def complete(self, model: str, messages: list[dict], *, step: str = "llm", schema: dict | None = None,
                 tools: list[dict] | None = None, temperature: float = 0.0, max_tokens: int = 4096,
                 reasoning_effort: str | None = None, n: int = 1) -> list[dict]:
        """Every choice's assistant message. Sends the OpenAI json_schema wrapper and, on 400/422, retries with the
        bare schema, then without reasoning_effort. A fallback that worked is remembered per model."""
        form = "bare" if model in self._bare else "openai"
        effort = None if model in self._no_effort else reasoning_effort
        t0 = time.time()
        while True:
            body: dict = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            if n > 1:
                body["n"] = n
            if schema:
                body["response_format"] = {"type": "json_schema", "json_schema": (
                    {"name": "out", "schema": schema, "strict": True} if form == "openai" else schema)}
            if tools:
                body["tools"] = tools
            if effort:
                body["reasoning_effort"] = effort
            try:
                data = self.post(body)
                break
            except TFError as e:
                if e.status in (400, 422) and schema and form == "openai":
                    form = "bare"
                elif e.status in (400, 422) and effort:
                    effort = None
                else:
                    raise
        if form == "bare":
            self._bare.add(model)
        if reasoning_effort and not effort:
            self._no_effort.add(model)
        self.meter.add(model, data.get("usage"), int((time.time() - t0) * 1000), step)
        return [c["message"] for c in data["choices"]]

    def chat(self, model: str, messages: list[dict], **kw) -> dict:
        """The first choice's message dict ({'content': ..., 'tool_calls': ...}). Raises TFError on failure."""
        return self.complete(model, messages, **kw)[0]

    def chat_json(self, model: str, messages: list[dict], schema: dict, **kw) -> dict:
        return parse_json(self.chat(model, messages, schema=schema, **kw))

    def chat_json_n(self, model: str, messages: list[dict], schema: dict, n: int, *, temperature: float = 0.7,
                    **kw) -> list[dict]:
        """Up to n parsed answers for voting. Asks for n choices in one request; if fewer come back (the server
        ignores n), the rest are separate calls at the same temperature, sent in parallel after the first so they can
        hit the prompt cache it warmed. Failed or unparseable answers are dropped; raises TFError if none is left."""
        try:
            msgs = self.complete(model, messages, schema=schema, n=n, temperature=temperature, **kw)
        except TFError as e:
            if e.status not in (400, 422) or n == 1:
                raise
            msgs = [self.chat(model, messages, schema=schema, temperature=temperature, **kw)]  # n itself rejected

        def one(_):
            try:
                return self.chat(model, messages, schema=schema, temperature=temperature, **kw)
            except TFError:
                return None

        if len(msgs) < n:
            with ThreadPoolExecutor(n - len(msgs)) as ex:
                msgs += [m for m in ex.map(one, range(n - len(msgs))) if m is not None]
        out = []
        for m in msgs:
            try:
                v = parse_json(m)
            except ValueError:
                continue
            if isinstance(v, dict):
                out.append(v)
        if not out:
            raise TFError(f"{model}: no parseable JSON answer out of {n}")
        return out


def parse_json(msg: dict | str):
    """The JSON object in an assistant message; reasoning models may wrap it in prose or <think> blocks."""
    content = msg if isinstance(msg, str) else (msg.get("content") or "")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start:end + 1] if start >= 0 else content)


def conforms(v, s: dict) -> bool:
    """Checks the JSON Schema subset Raised's schemas use (type, enum, anyOf, properties, required,
    additionalProperties: false, items). For the probe and the fake server; the real server enforces schemas itself."""
    if "anyOf" in s:
        return any(conforms(v, x) for x in s["anyOf"])
    if "enum" in s and v not in s["enum"]:
        return False
    t = s.get("type")
    if t == "object":
        props = s.get("properties", {})
        return (isinstance(v, dict) and all(k in v for k in s.get("required", []))
                and (s.get("additionalProperties", True) is not False or set(v) <= set(props))
                and all(conforms(v[k], p) for k, p in props.items() if k in v))
    if t == "array":
        return isinstance(v, list) and all(conforms(x, s.get("items", {})) for x in v)
    return {None: True, "string": isinstance(v, str), "boolean": isinstance(v, bool), "null": v is None,
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool)}[t]


def shrink(data: bytes, mime: str, max_side: int = 2000) -> tuple[bytes, str]:
    """Charts keep tick labels legible at 2000 px; larger inputs (page renders, phone photos) are downscaled, which
    cuts image tokens and upload time. Applies EXIF rotation. PNG stays PNG (thin lines survive), anything else
    becomes JPEG q88. Returns the input unchanged if it already fits upright, or if Pillow cannot read it."""
    try:
        im = Image.open(io.BytesIO(data))
        if max(im.size) <= max_side and mime in ("image/png", "image/jpeg") and im.getexif().get(274, 1) == 1:
            return data, mime
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        if mime == "image/png":
            im.save(buf, "PNG")
        else:
            im.convert("RGB").save(buf, "JPEG", quality=88)
            mime = "image/jpeg"
        return buf.getvalue(), mime
    except Exception:
        return data, mime


def image_part(data: bytes, mime: str | None = None, max_side: int = 2000) -> dict:
    """OpenAI image content part as a base64 data URL; the mime type is sniffed from the bytes when not given."""
    mime = mime or ("image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg")
    data, mime = shrink(data, mime, max_side)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}
