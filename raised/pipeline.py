"""Orchestration: document -> figures -> per figure (semantics x3, ticks, pixel trace, adjudication, redraw check,
facts, description with number guard, sound, tactile, BRF) -> result JSON + downloadable files.

Every module call goes through the small adapter functions in the middle of this file, so the modules stay
independent and the server only sees run_document(), accept(), recorded_events().
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .model import Point, Semantics, Table

MODELS = {"vl": "nvidia/Nemotron-3.5-Super-VL", "reason": "nvidia/Nemotron-3-Ultra-550b-a55b", "fast": "nvidia/Nemotron-3_5-Lightning"}
MAX_FIGURES = int(os.environ.get("MAX_FIGURES", "10"))
STEPS = ("pages", "find", "semantics", "measure", "adjudicate", "verify", "describe", "guard", "tactile")
REDRAW_RGB = (214, 0, 122)  # magenta: visible over any chart palette in the overlay


class Unreadable(Exception):
    """The upload is not a PDF or image we can read; shown to the user as is."""


def live() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY"))


def _noop(_ev):
    pass


# ---------------------------------------------------------------- adapters (one place to match module APIs)

def _client():
    from .llm import Meter, TokenFactory
    return TokenFactory(meter=Meter())


def _render_pdf(data: bytes):
    from .figures import render_pdf
    return render_pdf(data, dpi=200, max_pages=40)


def _find_figures(client, page_img, text):
    from .figures import find_figures
    return find_figures(client, page_img, text)


def _crop(page_img, bbox):
    from .figures import crop
    return crop(page_img, bbox, pad=0.04)


def _paragraph(text, number):
    from .figures import paragraph_citing
    return paragraph_citing(text, number) if number else ""


def _semantics(client, png: bytes):
    from .semantics import read_semantics
    return read_semantics(client, png, votes=3)


def _ticks(img, sem):
    from .ticks import locate_ticks
    return locate_ticks(img, sem)


def _trace(img, sem, tr):
    from .trace import trace
    out = trace(img, sem, tr)
    return out[0] if isinstance(out, tuple) else out


def _adjudicate(reading_a, reading_b, img, sem, tr):
    from .adjudicate import adjudicate
    return adjudicate(reading_a, reading_b, img, sem, tr)


def _trace_at(img, sem, tr, xs):
    from .trace import trace_at
    return trace_at(img, sem, tr, xs)


def _regular_line_table(img, sem: Semantics, tr: dict, table: Table) -> Table:
    """Without a model reading, trace() samples lines at their own vertices, which can mean hundreds of irregular x
    values. Re-measure at a grid a student can read: whole steps when the x ticks are integers (years), else the ticks."""
    if sem.chart_type != "line" or len(table.x) <= 40:
        return table
    ticks = [v for _, v, _ in (tr.get("x_ticks") or []) if v is not None]
    xs_line = [x for x in table.x if isinstance(x, (int, float))]
    if len(ticks) < 2 or not xs_line:
        return table
    lo, hi = min(xs_line), max(xs_line)
    if all(abs(t - round(t)) < 1e-6 for t in ticks) and hi - lo <= 120:
        xs = [float(x) for x in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    else:
        xs = [float(t) for t in ticks if lo <= t <= hi]
    if len(xs) < 2:
        return table
    vals = _trace_at(img, sem, tr, xs)
    return Table(x=xs, series={k: list(v) for k, v in vals.items()})


def _masks(table, sem, tr, size):
    from .render import rerender
    return rerender(table, sem, tr, size)


def _verify(client, original_png, redraw_png):
    from .verifier import verify
    return verify(client, original_png, redraw_png)


def _facts(table, sem):
    from .facts import facts
    return facts(table, sem)


def _describe(client, table, facts, sem, caption, context):
    from .describe import describe, fallback_description
    if client is None:
        return fallback_description(table, facts, sem)
    return describe(client, table, facts, sem, caption, context)


def _sound(table, sem):
    from .sonify import mapping, to_wav
    first = next(iter(table.series))
    return mapping(table, sem), to_wav(table, sem, first)


def _tactile(table, sem, title):
    from .tactile import compile as compile_tactile
    return compile_tactile(table, sem, title)


def _brf(title, key, description, table):
    from .brf import to_brf
    return to_brf(title, key, description, table)


# ---------------------------------------------------------------- helpers

def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _redraw_png(masks: dict, size: tuple[int, int]) -> bytes:
    """Transparent PNG with the redraw in magenta, for the overlay slider."""
    w, h = size
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for m in masks.values():
        m = np.asarray(m, dtype=bool)
        if m.shape != (h, w):
            continue
        rgba[m] = (*REDRAW_RGB, 230)
    return _png(Image.fromarray(rgba, "RGBA"))


def _csv(table: Table, sem: Semantics) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([sem.x_title or ("category" if sem.chart_type in ("bar", "grouped_bar", "hbar") else "x"), *table.series.keys()])
    for i, x in enumerate(table.x):
        w.writerow([x, *["" if v[i] is None else v[i] for v in table.series.values()]])
    return buf.getvalue().encode()


def _status(points: list[Point], iou: float | None, issues: list, sem: Semantics) -> tuple[str, str]:
    if sem.chart_type == "unsupported":
        return "unsupported", "Not a line, bar or scatter chart. Alt text only."
    disputed = [p for p in points if p.status == "disputed"]
    missing = [p for p in points if p.status == "missing"]
    reasons = []
    if disputed:
        reasons.append(f"{len(disputed)} value{'s' if len(disputed) > 1 else ''} where the two readings disagree")
    if missing:
        reasons.append(f"{len(missing)} value{'s' if len(missing) > 1 else ''} not readable")
    if issues:
        reasons.append(f"the visual check raised {len(issues)} issue{'s' if len(issues) > 1 else ''}")
    if iou is not None and iou < 0.5:
        reasons.append(f"the redraw overlaps the original only {iou:.0%}")
    return ("review", "; ".join(reasons).capitalize() + ".") if reasons else ("verified", "Both readings agree and the redraw matches the original.")


def _title(sem: Semantics, caption: str) -> str:
    return sem.title or (caption.split(".")[0] if caption else "") or "Untitled chart"


# ---------------------------------------------------------------- one figure

def process_figure(crop_img: Image.Image, *, fid: str, url: callable, client=None, caption: str = "", context: str = "",
                   page: int | None = None, number: str = "", semantics_override: Semantics | None = None,
                   reading_a_override: Table | None = None, progress=_noop, n: int = 1, of: int = 1) -> tuple[dict, dict]:
    """Returns (figure_json, files). url(name) maps a file name to the URL it will be served at."""
    files: dict[str, bytes] = {}
    crop_png = _png(crop_img)
    files[f"{fid}-chart.png"] = crop_png
    ev = lambda step, status, **kw: progress({"step": step, "status": status, "n": n, "of": of, "figure": fid, **kw,
                                               **({"usd": client.meter.total_usd} if client is not None else {})})
    # 1. meaning, three votes (or a fixed reading when the caller provides one: tests, recorded samples)
    ev("semantics", "running")
    if semantics_override is not None:
        sem, reading_a, agreement = semantics_override, reading_a_override, {}
    else:
        try:
            sem, reading_a, agreement = _semantics(client, crop_png)
        except Exception as e:  # one unreadable chart must not stop the document
            sem, reading_a, agreement = Semantics(chart_type="unsupported", title=caption[:120]), None, {"error": str(e)[:160]}
    sem.caption = sem.caption or caption
    ev("semantics", "done" if n == of else "running")
    fig = {"id": fid, "page": page, "figure_number": number, "caption": caption, "chart_type": sem.chart_type,
           "title": _title(sem, caption), "semantics": sem.to_dict(), "image": url(f"{fid}-chart.png"), "agreement": agreement}
    if sem.chart_type == "unsupported":
        desc = {"long_description": "", "exploration_guide": [], "audit": [], "alt_text": caption[:125]}
        fig.update(status="unsupported", reason="Not a line, bar or scatter chart. Alt text only.", table=None, points=[],
                   description=desc, _state={"sem": sem.to_dict(), "caption": caption, "context": context, "fid": fid})
        return fig, files
    # 2. pixels
    ev("measure", "running")
    tr = _ticks(crop_img, sem)
    reading_b = _trace(crop_img, sem, tr)
    if reading_a is None:
        reading_b = _regular_line_table(crop_img, sem, tr, reading_b)
    ev("measure", "done" if n == of else "running")
    # 3. settle disagreements by redrawing both candidates
    ev("adjudicate", "running")
    table, points = _adjudicate(reading_a, reading_b, crop_img, sem, tr)
    masks = _masks(table, sem, tr, crop_img.size)
    files[f"{fid}-redraw.png"] = _redraw_png(masks, crop_img.size)
    iou = _iou(masks, crop_img, sem, tr)
    ev("adjudicate", "done" if n == of else "running")
    # 4. second job for the vision model: compare the redraw with the original
    issues = []
    if client is not None:
        ev("verify", "running")
        try:
            issues = _verify(client, crop_png, _png(_side_by_side(crop_img, files[f"{fid}-redraw.png"])))
        except Exception:
            issues = []
        ev("verify", "done" if n == of else "running")
    status, reason = _status(points, iou, issues, sem)
    if not _readable(table):
        status, reason = "review", "No values could be measured. Check the chart type and axis labels, or enter the values."
    fig.update(status=status, reason=reason, iou=iou, issues=issues, rerender=url(f"{fid}-redraw.png"),
               points=[p.__dict__ for p in points], reading_b=reading_b.to_dict() if reading_b else None,
               reading_a=reading_a.to_dict() if reading_a else None)
    state = {"sem": sem.to_dict(), "table": table.to_dict(), "points": [p.__dict__ for p in points], "caption": caption,
             "context": context, "fid": fid, "fig": {k: v for k, v in fig.items() if k != "_state"}}
    out, more = derive(table, sem, caption, context, fid, url, client, ev, n, of)
    fig.update(out)
    files.update(more)
    fig["_state"] = state
    return fig, files


def _readable(table: Table) -> bool:
    return any(v is not None for col in table.series.values() for v in col)


def derive(table: Table, sem: Semantics, caption: str, context: str, fid: str, url, client, ev=lambda *a, **k: None,
           n: int = 1, of: int = 1) -> tuple[dict, dict]:
    """Everything computed from a checked table. Re-run when the reviewer accepts edited values.
    Each output is produced independently, so one failing step never takes the others down."""
    files = {}
    title = _title(sem, caption)
    if not _readable(table):
        ev("tactile", "done" if n == of else "running")
        return {"table": table.to_dict(), "facts": [], "description": {"long_description": "", "exploration_guide": [], "audit": [],
                "alt_text": title[:125]}, "sound": None, "tactile": None}, files
    f = _facts(table, sem)
    ev("describe", "running")
    try:
        desc = _describe(client, table, f, sem, caption, context)
    except Exception:
        desc = _describe(None, table, f, sem, caption, context)
    ev("describe", "done" if n == of else "running")
    ev("guard", "done" if n == of else "running")
    ev("tactile", "running")
    out = {"table": table.to_dict(), "facts": f, "description": desc, "csv": url(f"{fid}.csv")}
    files[f"{fid}.csv"] = _csv(table, sem)
    try:
        tac = _tactile(table, sem, title)
        files[f"{fid}-tactile.pdf"] = tac["pdf"]
        files[f"{fid}-tactile.svg"] = tac["svg"].encode()
        out["tactile"] = {"preview_svg": tac.get("preview_svg") or tac["svg"], "key": tac.get("key", []), "pages": tac.get("pages", 1),
                          "pdf": url(f"{fid}-tactile.pdf"), "svg": url(f"{fid}-tactile.svg")}
        files[f"{fid}.brf"] = _brf(title, tac.get("key", []), desc.get("long_description", ""), table).encode("ascii", "replace")
        out["brf"] = url(f"{fid}.brf")
    except Exception as e:
        out["tactile"] = {"error": f"tactile graphic not produced ({type(e).__name__}: {str(e)[:80]})"}
    try:
        mapping, wav = _sound(table, sem)
        files[f"{fid}.wav"] = wav
        out.update(sound=mapping, wav=url(f"{fid}.wav"))
    except Exception:
        out["sound"] = None
    ev("tactile", "done" if n == of else "running")
    return out, files


def _iou(masks: dict, img: Image.Image, sem: Semantics, tr) -> float | None:
    try:
        from .adjudicate import chart_iou
        return float(chart_iou(masks, img, sem, tr))
    except Exception:
        return None


def _side_by_side(original: Image.Image, redraw_png: bytes) -> Image.Image:
    red = Image.open(io.BytesIO(redraw_png)).convert("RGBA")
    base = Image.new("RGB", red.size, "white")
    base.paste(red, (0, 0), red)
    w, h = original.size
    out = Image.new("RGB", (w * 2 + 20, h), "white")
    out.paste(original.convert("RGB"), (0, 0))
    out.paste(base, (w + 20, 0))
    return out


# ---------------------------------------------------------------- whole document

def run_document(data: bytes, filename: str, *, job: str, progress=_noop, budget_left: float = 1.0):
    client = _client()
    url = lambda name: f"/api/job/{job}/{name}"
    t0 = time.time()
    progress({"step": "pages", "status": "running"})
    if data[:4] == b"%PDF":
        try:
            pages = _render_pdf(data)
        except Exception as e:
            raise Unreadable(f"could not read that PDF ({type(e).__name__})")
        progress({"step": "pages", "status": "done", "pages": len(pages)})
        progress({"step": "find", "status": "running"})
        found = []
        for k, (pi, img, text) in enumerate(pages):
            for f in _find_figures(client, img, text):
                if f.get("is_data_chart"):
                    found.append((pi, img, text, f))
            progress({"step": "find", "status": "running", "n": k + 1, "of": len(pages), "usd": client.meter.total_usd})
            if len(found) >= MAX_FIGURES:
                break
        progress({"step": "find", "status": "done", "figures": len(found)})
        jobs = [(_crop(img, f["bbox"]), pi + 1, f.get("caption", ""), f.get("figure_number", ""), _paragraph(text, f.get("figure_number")))
                for pi, img, text, f in found[:MAX_FIGURES]]
    else:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise Unreadable("that file is not a PDF, PNG or JPEG")
        progress({"step": "pages", "status": "done", "pages": 1})
        progress({"step": "find", "status": "done", "figures": 1})
        jobs = [(img, None, "", "", "")]
    figures, files, states = [], {}, {}
    for i, (crop_img, page, caption, number, context) in enumerate(jobs):
        fid = f"f{i + 1}"
        if client.meter.total_usd > budget_left:
            figures.append({"id": fid, "page": page, "figure_number": number, "caption": caption, "status": "unsupported",
                            "reason": "Skipped: the public demo's daily budget is used up.", "title": caption or "Chart", "image": None})
            continue
        fig, more = process_figure(crop_img, fid=fid, url=url, client=client, caption=caption, context=context, page=page,
                                   number=number, progress=progress, n=i + 1, of=len(jobs))
        states[fid] = fig.pop("_state")
        figures.append(fig)
        files.update(more)
    files["raised-export.zip"] = _zip(figures, files)
    result = {"job": job, "recorded": False, "document": {"name": filename}, "figures": figures,
              "export": url("raised-export.zip"), "cost": client.meter.summary(), "seconds": round(time.time() - t0, 1),
              "models": MODELS}
    return result, files, states


def _zip(figures: list[dict], files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if not name.endswith(".zip"):
                z.writestr(name, data)
        z.writestr("figures.json", json.dumps([{k: v for k, v in f.items() if k not in ("_state",)} for f in figures], indent=1, default=str))
        z.writestr("README.txt", "Raised export. Every file is a draft for transcriber review, not a certified transcription.\n"
                                 "Per figure: -chart.png original crop, .csv checked data, -tactile.pdf/.svg tactile graphic,\n"
                                 ".brf braille text, .wav sound mapping, figures.json descriptions and provenance.\n")
    return buf.getvalue()


# ---------------------------------------------------------------- review: accept edited values

def figure_state_from_result(fig: dict) -> dict:
    return {"sem": fig["semantics"], "table": fig["table"], "points": fig.get("points", []), "caption": fig.get("caption", ""),
            "context": "", "fid": fig["id"], "fig": fig}


def accept(st: dict, values: list[dict], job: str) -> tuple[dict, dict]:
    sem = Semantics.from_dict(st["sem"])
    table = Table.from_dict(st["table"])
    points = [dict(p) for p in st.get("points", [])]
    for v in values:
        name, x, val = v.get("series"), v.get("x"), v.get("value")
        if name not in table.series or not isinstance(val, (int, float)):
            continue
        idx = next((i for i, xx in enumerate(table.x) if str(xx) == str(x)), None)
        if idx is None:
            continue
        table.series[name][idx] = float(val)
        for p in points:
            if p["series"] == name and str(p["x"]) == str(x):
                p.update(value=float(val), status="agreed", note="accepted by reviewer")
    fid = st["fid"]
    url = lambda name: f"/api/job/{job}/{name}"
    client = _client() if live() else None
    out, files = derive(table, sem, st.get("caption", ""), st.get("context", ""), fid, url, client)
    fig = dict(st["fig"])
    fig.update(out, points=points)
    remaining = [p for p in points if p["status"] == "disputed"]
    if not remaining and fig.get("status") == "review":
        fig["status"], fig["reason"] = "verified", "Reviewed and accepted."
    fig["_state"] = {**st, "table": table.to_dict(), "points": points, "fig": {k: v for k, v in fig.items() if k != "_state"}}
    return fig, files


def retarget_urls(fig: dict, old_job: str, new_job: str) -> dict:
    s = json.dumps(fig).replace(f"/api/job/{old_job}/", f"/api/job/{new_job}/")
    return json.loads(s)


# ---------------------------------------------------------------- recorded samples

def recorded_events(sid: str, samples_dir: Path):
    p = samples_dir / sid / "events.json"
    if p.exists():
        for ev in json.loads(p.read_text()):
            yield {**ev, "recorded": True}
        return
    for step in STEPS:
        yield {"step": step, "status": "done", "recorded": True}
