"""Record demo samples by running the real pipeline, so the public demo works when the models are unavailable.

    python -m scripts.build_samples --live          # needs NEBIUS_API_KEY: runs the OpenStax excerpt through every step
    python -m scripts.build_samples --synthetic     # no key: 3 synthetic charts; meaning taken from the generator,
                                                    # pixels measured and checked by the real code, template descriptions

Each sample goes to data/samples/<id>/ (result.json, events.json with real step timings, every file the UI links to)
and is listed in data/samples/index.json with a note the UI shows in its banner.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from raised import pipeline
from raised.model import Semantics, Table

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
EXCERPT = ROOT / "data" / "demo" / "openstax-economics-2e-excerpt.pdf"
SYNTH = ROOT / "bench" / "data" / "synth" / "probe"


def _write(sid: str, result: dict, files: dict, events: list, entry: dict):
    out = SAMPLES / sid
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    s = json.dumps(result).replace(f"/api/job/sample-{sid}/", f"/samples/{sid}/")
    (out / "result.json").write_text(json.dumps(json.loads(s), indent=1, default=str))
    (out / "events.json").write_text(json.dumps(events))
    idx = SAMPLES / "index.json"
    entries = [e for e in (json.loads(idx.read_text()) if idx.exists() else []) if e["id"] != sid]
    entries.insert(0, entry)
    idx.write_text(json.dumps(entries, indent=1))
    print(f"wrote sample {sid}: {len(result['figures'])} figures, {len(files)} files")


def _recorder():
    events, t0 = [], time.time()
    def emit(ev):
        events.append({**ev, "t": round(time.time() - t0, 2)})
    return events, emit


def live():
    if not pipeline.live():
        raise SystemExit("Set NEBIUS_API_KEY first")
    sid = "openstax"
    events, emit = _recorder()
    result, files, _ = pipeline.run_document(EXCERPT.read_bytes(), EXCERPT.name, job=f"sample-{sid}", progress=emit, budget_left=5.0)
    result["job"] = f"sample-{sid}"
    first = next((f for f in result["figures"] if f.get("image")), None)
    thumb_name = first["image"].rsplit("/", 1)[-1] if first else None
    _write(sid, result, files, events, {
        "id": sid, "title": "Economics textbook, 7 pages", "subtitle": "OpenStax Principles of Economics 2e, CC BY 4.0",
        "thumb": f"/samples/{sid}/{thumb_name}" if thumb_name else "",
        "note": f"this chapter excerpt was read on {time.strftime('%d %b %Y')} by the full pipeline on Nebius Token Factory"})


def synthetic(ids=("0015", "0019", "0011")):
    sid = "synthetic"
    events, emit = _recorder()
    figures, files = [], {}
    url = lambda name: f"/api/job/sample-{sid}/{name}"
    for k, cid in enumerate(ids):
        truth = json.loads((SYNTH / f"{cid}.json").read_text())
        img = Image.open(SYNTH / f"{cid}.jpg").convert("RGB")
        sem = Semantics.from_dict(truth["semantics"])
        fig, more = pipeline.process_figure(img, fid=f"f{k + 1}", url=url, client=None, caption=sem.title, page=None,
                                            number="", semantics_override=sem, reading_a_override=None,
                                            progress=emit, n=k + 1, of=len(ids))
        fig.pop("_state", None)
        fig["truth"] = Table.from_dict(truth["table"]).to_dict()
        figures.append(fig)
        files.update(more)
    files["raised-export.zip"] = pipeline._zip(figures, files)
    result = {"job": f"sample-{sid}", "recorded": True, "document": {"name": "synthetic charts"}, "figures": figures,
              "export": url("raised-export.zip"), "cost": {"total_usd": 0.0, "calls": []}}
    _write(sid, result, files, events, {
        "id": sid, "title": "Three synthetic charts", "subtitle": "Known values, measured by the real code",
        "thumb": f"/samples/{sid}/f1-chart.png",
        "note": "synthetic charts whose meaning comes from the generator; pixels were measured and checked by the real code; descriptions come from the template writer, not the language model"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        synthetic()
    if a.live:
        live()
