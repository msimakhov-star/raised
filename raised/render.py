"""Redraw a table into the original chart's pixel frame (spec step 6), one bool mask per series, so it can be laid over
the original and scored by IoU against the original's colour masks.

Same frame as locate_ticks (deskewed crop, continuous coordinates), same calibration. Drawn with PIL rather than
matplotlib: the frame is a pixel frame already and PIL is fast enough to redraw one candidate value at a time. PIL
truncates coordinates to whole pixels (measured: a 2 px line at y = 10.0 to 10.7 always fills rows 10-11), so masks
are drawn at SS x scale and box-downsampled: a pixel is set when at least half covered, the same rule as the tracer's
colour masks, to within 1/(2 SS) px.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .model import Semantics, Table


SS = 4  # supersampling factor


class _Canvas:
    """Draw in continuous coordinates of the target frame; .mask() is the half-covered pixels."""

    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.im = Image.new("L", (size[0] * SS, size[1] * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def line(self, pts, width: float):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=255, width=max(1, int(round(width * SS))), joint="curve")

    def ellipse(self, x0, y0, x1, y1):
        self.d.ellipse([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=255)

    def rectangle(self, x0, y0, x1, y1):
        self.d.rectangle([x0 * SS, y0 * SS, x1 * SS - 1, y1 * SS - 1], fill=255)

    def mask(self) -> np.ndarray:
        return np.asarray(self.im.resize(self.size, Image.BOX)) >= 128


def rerender(table: Table, sem: Semantics, ticks_result: dict, size: tuple[int, int], style: dict | None = None
             ) -> dict[str, np.ndarray]:
    """size is (width, height) like PIL's Image.size. style (from trace.geometry) sets line_width, marker_radius and
    bar slots measured on the original; without it, sizes close to the synthetic charts (2 px lines, 3.5 px marker
    radius, bars 0.8 of a category step split between series at 0.9 of their slot, as bench/make_synth.py draws)."""
    style = style or {}
    xc, yc = ticks_result.get("x_cal"), ticks_result.get("y_cal")
    out = {}
    if sem.chart_type in ("bar", "grouped_bar", "hbar"):
        horizontal = sem.chart_type == "hbar"
        cal = xc if horizontal else yc
        cats = dict(ticks_result.get("categories") or [])
        x0, y0, x1, y1 = ticks_result["plot_box"]
        base = cal.px(0.0) if cal is not None and cal.kind == "linear" else None
        base = (max(base, x0) if horizontal else min(base, y1)) if base is not None else (x0 if horizontal else y1)
        names = list(table.series)
        pos = [cats.get(c) for c in table.x]
        known = [p for p in pos if p is not None]
        step = float(np.median(np.diff(sorted(known)))) if len(known) > 1 else 40.0
        thick = 0.8 * step / max(len(names), 1) * 0.9  # bench/make_synth.py's bar layout
        for k, name in enumerate(names):
            cv = _Canvas(size)
            slots = style.get("bars", {}).get(name) or [None] * len(table.x)
            for i, v in enumerate(table.series[name]):
                if v is None or cal is None or (slots[i] is None and pos[i] is None):
                    continue
                c, t = slots[i] if slots[i] is not None else (pos[i] + (k - (len(names) - 1) / 2) * thick / 0.9, thick)
                e = cal.px(v)
                if horizontal:
                    cv.rectangle(min(base, e), c - t / 2, max(base, e), c + t / 2)
                else:
                    cv.rectangle(c - t / 2, min(base, e), c + t / 2, max(base, e))
            out[name] = cv.mask()
        return out
    if xc is None or yc is None:
        return {n: np.zeros(size[::-1], bool) for n in table.series}
    lw = style.get("line_width", 2.0)
    r = style.get("marker_radius", 3.5)
    markers = {s.name: s.marker for s in sem.series}
    for name, vals in table.series.items():
        cv = _Canvas(size)
        pts = [(xc.px(x), yc.px(v)) if v is not None and (yc.kind != "log" or v > 0) else None for x, v in zip(table.x, vals)]
        if sem.chart_type == "scatter" or markers.get(name):
            for p in pts:
                if p:
                    cv.ellipse(p[0] - r, p[1] - r, p[0] + r, p[1] + r)
        if sem.chart_type == "line":
            run = []
            for p in pts + [None]:  # a missing value breaks the line
                if p:
                    run.append(p)
                    continue
                if len(run) > 1:
                    cv.line(run, lw)
                run = []
        out[name] = cv.mask()
    return out
