"""Go/no-go probe for Token Factory, to run the moment a real key exists (spec, Sun 25):

    NEBIUS_API_KEY=... .venv/bin/python scripts/probe.py [--charts 10] [--out probe_report.json]

Every check runs on its own (a failure is recorded and the rest still run) and lands in the JSON report:
  schema_forms   json_schema on Super VL with a synthetic chart: OpenAI wrapper vs bare schema
  n              whether n=3 returns 3 choices (if not, votes are separate calls)
  cached         prompt_tokens_details.cached_tokens on an identical repeated request
  image_sizes    status, prompt tokens and latency for the same chart at 512 to 4096 px long side
  multi_image    two images in one message (the verifier sends two)
  vl_reasoning   Super VL with reasoning_effort low: accepted, and completion tokens against the default
  ultra          Ultra with reasoning_effort medium and a schema
  lightning      Lightning with a schema
  minicpm        MiniCPM-V-4_5 semantics (the bench's non-NVIDIA baseline)
  find_figures   box IoU on a synthetic PDF page with a known chart position
  semantics      read_semantics (3 votes) on the first --charts probe charts, scored against the generator's truth
  rate_limits    x-ratelimit-* headers seen and any 429, per the header list in
                 https://docs.tokenfactory.nebius.com/ai-models-inference/rate-limits.md (fetched 2026-10-24)
  latency        median ms per step
It costs a few cents with the default 10 charts; the total is printed at the end.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from raised import prompts  # noqa: E402
from raised.figures import find_figures, render_pdf  # noqa: E402
from raised.llm import MODELS, TFError, TokenFactory, conforms, image_part, parse_json  # noqa: E402
from raised.model import Semantics, Table, parse_number  # noqa: E402
from raised.semantics import BARS, read_semantics  # noqa: E402

PROBE = ROOT / "bench" / "data" / "synth" / "probe"
VL, ULTRA, FAST, MINICPM = MODELS["vl"], MODELS["reason"], MODELS["fast"], MODELS["minicpm"]
OK = "TASK: PROBE_OK\nReply with the single word OK."
COUNT = "TASK: PROBE_COUNT\nHow many images are in this message? Reply with one digit."
MATH_SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"],
               "additionalProperties": False}
MATH = f"TASK: PROBE_MATH\nWhat is 17 * 23? Return only JSON matching this schema: {json.dumps(MATH_SCHEMA)}"
EXTRACT_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "year": {"type": "integer"}},
                  "required": ["city", "year"], "additionalProperties": False}
EXTRACT = ("TASK: PROBE_EXTRACT\nExtract the city and year from: 'The census in Leeds was taken in 2021.' "
           f"Return only JSON matching this schema: {json.dumps(EXTRACT_SCHEMA)}")
RATE_HEADERS = ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-requests",
                "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
                "x-ratelimit-over-limit", "retry-after")
seen_429: list[dict] = []


def user(*parts) -> list[dict]:
    return [{"role": "user", "content": [p if isinstance(p, dict) else {"type": "text", "text": p} for p in parts]}]


def rate(headers: dict) -> dict:
    low = {k.lower(): v for k, v in (headers or {}).items()}
    return {k: low[k] for k in RATE_HEADERS if k in low}


def response_format(schema: dict, form: str) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True} if form == "openai" else schema}


def raw(tf: TokenFactory, step: str, body: dict, schema: dict | None = None) -> dict:
    """One request; records status, latency, usage and the answer instead of raising."""
    body = {"temperature": 0, "max_tokens": 4096, **body}
    t0 = time.time()
    try:
        data = tf.post(body)
    except TFError as e:
        if e.status == 429:
            seen_429.append(rate(e.headers))
        return {"status": e.status or "unreachable", "ms": int((time.time() - t0) * 1000), "error": str(e)[:300]}
    ms = int((time.time() - t0) * 1000)
    tf.meter.add(body["model"], data.get("usage"), ms, step)
    choices = data.get("choices") or []
    msg = choices[0]["message"] if choices else {}
    out = {"status": 200, "ms": ms, "usage": data.get("usage"), "choices": len(choices), "message_keys": sorted(msg),
           "content": (msg.get("content") or "")[:1500]}
    if schema is not None:
        try:
            out["valid"] = conforms(parse_json(msg), schema)
        except ValueError:
            out["valid"] = False
    return out


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- checks

def check_schema_forms(tf, chart: bytes) -> dict:
    return {form: raw(tf, f"probe-schema-{form}", {"model": VL, "messages": user(image_part(chart), prompts.SEMANTICS),
                                                     "max_tokens": 8192,
                                                     "response_format": response_format(prompts.SEMANTICS_SCHEMA, form)},
                      prompts.SEMANTICS_SCHEMA)
            for form in ("openai", "bare")}


def check_n(tf, chart: bytes, form: str) -> dict:
    r = raw(tf, "probe-n", {"model": VL, "messages": user(image_part(chart), prompts.SEMANTICS), "n": 3, "temperature": 0.7,
                            "max_tokens": 8192, "response_format": response_format(prompts.SEMANTICS_SCHEMA, form)})
    return {**r, "honoured": r.get("choices") == 3}


def check_cached(tf, chart: bytes) -> dict:
    body = {"model": VL, "messages": user(image_part(chart), OK), "max_tokens": 16}
    first, second = raw(tf, "probe-cache", body), raw(tf, "probe-cache", body)
    cached = lambda r: ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"first": cached(first), "second": cached(second), "prompt_tokens": (second.get("usage") or {}).get("prompt_tokens"),
            "statuses": [first["status"], second["status"]]}


def check_image_sizes(tf, img: Image.Image) -> dict:
    out = {}
    for side in (512, 1024, 2000, 3000, 4096):
        scaled = img.resize((round(img.width * side / max(img.size)), round(img.height * side / max(img.size))))
        data = png(scaled)
        r = raw(tf, f"probe-image-{side}", {"model": VL, "messages": user(image_part(data, max_side=10 ** 6), OK), "max_tokens": 16})
        out[str(side)] = {"status": r["status"], "bytes": len(data), "ms": r["ms"],
                          "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "error": r.get("error")}
    return out


def check_multi_image(tf, chart: bytes) -> dict:
    r = raw(tf, "probe-multi", {"model": VL, "messages": user(image_part(chart), image_part(chart), COUNT), "max_tokens": 64})
    return {**r, "says_two": "2" in r.get("content", "")}


def check_vl_reasoning(tf, chart: bytes, form: str, default_out: int | None) -> dict:
    r = raw(tf, "probe-vl-low", {"model": VL, "messages": user(image_part(chart), prompts.SEMANTICS), "max_tokens": 8192,
                                 "reasoning_effort": "low", "response_format": response_format(prompts.SEMANTICS_SCHEMA, form)},
            prompts.SEMANTICS_SCHEMA)
    return {**r, "completion_tokens_low": (r.get("usage") or {}).get("completion_tokens"), "completion_tokens_default": default_out}


def check_ultra(tf, form: str) -> dict:
    r = raw(tf, "probe-ultra", {"model": ULTRA, "messages": user(MATH), "reasoning_effort": "medium",
                                "response_format": response_format(MATH_SCHEMA, form)}, MATH_SCHEMA)
    try:
        r["correct"] = parse_json(r.get("content", "")).get("answer") == 391
    except (ValueError, AttributeError):
        r["correct"] = False
    return r


def check_lightning(tf, form: str) -> dict:
    return raw(tf, "probe-lightning", {"model": FAST, "messages": user(EXTRACT), "response_format": response_format(EXTRACT_SCHEMA, form)},
               EXTRACT_SCHEMA)


def check_minicpm(tf, chart: bytes, form: str) -> dict:
    return raw(tf, "probe-minicpm", {"model": MINICPM, "messages": user(image_part(chart), prompts.SEMANTICS), "max_tokens": 8192,
                                     "response_format": response_format(prompts.SEMANTICS_SCHEMA, form)}, prompts.SEMANTICS_SCHEMA)


def check_find_figures(tf) -> dict:
    """A letter page with body text, one chart at a known place and its caption; IoU of the best box."""
    fig = Figure(figsize=(8.5, 11))
    fig.text(0.1, 0.92, "Exports grew quickly after 2000, as Figure 1.2 shows. Imports rose more slowly.", fontsize=10)
    ax = fig.add_axes((0.15, 0.5, 0.7, 0.3))
    ax.plot([2000, 2005, 2010], [2, 5.5, 7.9], marker="o", label="Exports")
    ax.plot([2000, 2005, 2010], [3, 4, 4.6], linestyle="--", label="Imports")
    ax.set(title="Exports and imports", xlabel="Year", ylabel="US$ bn")
    ax.legend()
    fig.text(0.1, 0.42, "Figure 1.2 Exports and imports, 2000 to 2010 (synthetic).", fontsize=10)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    tight = ax.get_tightbbox().transformed(fig.transFigure.inverted())
    truth = [round(tight.x0 * 1000), round((1 - tight.y1) * 1000), round(tight.x1 * 1000), round((1 - tight.y0) * 1000)]
    _, page, text = render_pdf(buf.getvalue(), dpi=150)[0]
    t0 = time.time()
    found = find_figures(tf, page, text)
    return {"truth_box": truth, "found": found, "ms": int((time.time() - t0) * 1000),
            "best_iou": max((round(iou(f["bbox"], truth), 3) for f in found), default=0.0)}


def iou(a, b) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union else 0.0


def check_semantics(tf, n: int) -> dict:
    rows = []
    for path in sorted(PROBE.glob("*.json"))[:n]:
        truth = json.loads(path.read_text())
        sem_t, tab_t = Semantics.from_dict(truth["semantics"]), Table.from_dict(truth["table"])
        usd0, t0 = tf.meter.total_usd, time.time()
        try:
            sem, reading, agree = read_semantics(tf, path.with_suffix(".jpg").read_bytes(), votes=3)
        except TFError as e:
            rows.append({"chart": path.stem, "error": str(e)[:300]})
            continue
        rows.append({"chart": path.stem, "type": sem_t.chart_type, "chart_type": sem.chart_type == sem_t.chart_type,
                     "x_ticks": sem.x_ticks == sem_t.x_ticks, "y_ticks": sem.y_ticks == sem_t.y_ticks,
                     "n_series": len(sem.series) == len(sem_t.series),
                     "log": (sem.y_scale == "log") == (sem_t.y_scale == "log"),
                     "errors": reading_errors(sem_t, tab_t, reading), "agreement": agree,
                     "usd": round(tf.meter.total_usd - usd0, 5), "s": round(time.time() - t0, 1)})
    ok = [r for r in rows if "error" not in r]
    share = lambda k: round(sum(r[k] for r in ok) / len(ok), 3) if ok else None
    errs = [e for r in ok for e in r["errors"]]
    return {"charts": len(rows), "failed": len(rows) - len(ok), "chart_type": share("chart_type"), "x_ticks": share("x_ticks"),
            "y_ticks": share("y_ticks"), "ticks_both": round(sum(r["x_ticks"] and r["y_ticks"] for r in ok) / len(ok), 3) if ok else None,
            "n_series": share("n_series"), "log": share("log"),
            "reading_a_median_err_pct_range": round(100 * statistics.median(errs), 2) if errs else None,
            "reading_a_within_2pct": round(sum(e <= 0.02 for e in errs) / len(errs), 3) if errs else None,
            "usd_per_chart": round(sum(r["usd"] for r in ok) / len(ok), 5) if ok else None, "rows": rows}


def reading_errors(sem_t: Semantics, tab_t: Table, reading: Table) -> list[float]:
    """|reading A - truth| as a share of the value-axis tick range (log10 units on log axes). Series pair by name,
    or as the only one on each side; x pairs by category or by the nearest x within half the truth spacing.
    Scatter is skipped: its points have no order to pair by."""
    if sem_t.chart_type == "scatter" or not reading.x:
        return []
    ticks = [v for t in (sem_t.x_ticks if sem_t.chart_type == "hbar" else sem_t.y_ticks) if (v := parse_number(t)) is not None]
    log = sem_t.y_scale == "log"
    f = (lambda v: math.log10(v) if v and v > 0 else None) if log else (lambda v: v)
    lo, hi = f(min(ticks, default=0)), f(max(ticks, default=0))
    if lo is None or hi is None or hi == lo:
        return []
    names = {k.casefold(): k for k in reading.series}
    pairs = [(t, names.get(t.casefold())) for t in tab_t.series]
    if len(tab_t.series) == 1 and len(reading.series) == 1:
        pairs = [(next(iter(tab_t.series)), next(iter(reading.series)))]
    nums = [x for x in tab_t.x if isinstance(x, (int, float))]
    gap = statistics.median(b - a for a, b in zip(nums, nums[1:])) / 2 if len(nums) > 1 else 0.5
    out = []
    for t_name, r_name in pairs:
        if r_name is None:
            continue
        for x, v in zip(tab_t.x, tab_t.series[t_name]):
            if sem_t.chart_type in BARS:
                j = next((i for i, rx in enumerate(reading.x) if str(rx).casefold() == str(x).casefold()), None)
            else:
                near = [(abs(rx - x), i) for i, rx in enumerate(reading.x) if isinstance(rx, (int, float))]
                j = min(near)[1] if near and min(near)[0] <= gap else None
            a = reading.series[r_name][j] if j is not None else None
            if a is not None and v is not None and f(a) is not None and f(v) is not None:
                out.append(abs(f(a) - f(v)) / (hi - lo))
    return out


# ---------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--charts", type=int, default=10, help="probe charts to read with 3 votes (50 exist)")
    ap.add_argument("--out", type=Path, default=ROOT / "probe_report.json")
    args = ap.parse_args()
    if not os.environ.get("NEBIUS_API_KEY"):
        sys.exit("NEBIUS_API_KEY is not set. Export the Token Factory key, then run scripts/probe.py again (nothing was sent).")
    tf = TokenFactory()
    first = sorted(PROBE.glob("*.jpg"))[0]
    chart, img = first.read_bytes(), Image.open(first).convert("RGB")
    report: dict = {"base_url": tf.base_url, "models": MODELS, "chart": first.name, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def run(name, fn, *a):
        print(f"{name} ...", flush=True)
        try:
            report[name] = fn(*a)
        except Exception as e:  # one broken check must not hide the others
            report[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
        return report[name]

    forms = run("schema_forms", check_schema_forms, tf, chart)
    form = next((f for f in ("openai", "bare") if forms.get(f, {}).get("valid")), "bare")
    report["schema_form_used"] = form
    run("n", check_n, tf, chart, form)
    run("cached", check_cached, tf, chart)
    run("image_sizes", check_image_sizes, tf, img)
    run("multi_image", check_multi_image, tf, chart)
    run("vl_reasoning", check_vl_reasoning, tf, chart, form, ((forms.get(form) or {}).get("usage") or {}).get("completion_tokens"))
    run("ultra", check_ultra, tf, form)
    run("lightning", check_lightning, tf, form)
    run("minicpm", check_minicpm, tf, chart, form)
    run("find_figures", check_find_figures, tf)
    sem = run("semantics", check_semantics, tf, args.charts)
    report["rate_limits"] = {"last_response": rate(tf.last_headers), "429s": seen_429}
    steps: dict[str, list[int]] = {}
    for c in tf.meter.calls:
        steps.setdefault(c["step"], []).append(c["ms"])
    report["latency_ms"] = {k: int(statistics.median(v)) for k, v in steps.items()}
    report["cost"] = {"total_usd": tf.meter.total_usd, "cached_share": tf.meter.cached_share, "calls": len(tf.meter.calls)}
    args.out.write_text(json.dumps(report, indent=2, default=str))

    g = _dig
    print(f"\njson_schema: openai wrapper {'OK' if g(forms, 'openai', 'valid') else 'FAIL'}, "
          f"bare {'OK' if g(forms, 'bare', 'valid') else 'FAIL'} -> using {form}")
    print(f"n=3 honoured: {g(report, 'n', 'honoured')} (choices {g(report, 'n', 'choices')})")
    print(f"cached tokens on repeat: {g(report, 'cached', 'second')} of {g(report, 'cached', 'prompt_tokens')}")
    sizes = report.get("image_sizes") or {}
    print("image sizes: " + ", ".join(f"{k}px {v.get('status')} {v.get('prompt_tokens')} tok" for k, v in sizes.items() if isinstance(v, dict)))
    print(f"two images: {g(report, 'multi_image', 'status')} says_two={g(report, 'multi_image', 'says_two')}")
    for k in ("vl_reasoning", "ultra", "lightning", "minicpm"):
        print(f"{k}: status {g(report, k, 'status')} valid={g(report, k, 'valid')} {g(report, k, 'ms')} ms")
    print(f"find_figures best IoU: {g(report, 'find_figures', 'best_iou')}")
    print(f"semantics on {g(sem, 'charts')} charts: type {g(sem, 'chart_type')}, both tick lists exact {g(sem, 'ticks_both')}, "
          f"series count {g(sem, 'n_series')}, reading A median error {g(sem, 'reading_a_median_err_pct_range')}% of range, "
          f"${g(sem, 'usd_per_chart')} per chart")
    print(f"rate limits: {report['rate_limits']['last_response']} 429s: {len(seen_429)}")
    print(f"cost ${tf.meter.total_usd} for {len(tf.meter.calls)} calls, cached share {tf.meter.cached_share}; report: {args.out}")
    go = bool(g(forms, form, "valid")) and (g(sem, "chart_type") or 0) >= 0.9 and (g(sem, "ticks_both") or 0) >= 0.85
    print("GO on Super VL semantics (json works, chart type >= 90%, both tick lists exact on >= 85%); the spec's criteria "
          "(a) and (b) also need the Tesseract match and the hybrid error" if go else
          "NO-GO on Super VL semantics as measured: read the report before building on them")


def _dig(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


if __name__ == "__main__":
    main()
