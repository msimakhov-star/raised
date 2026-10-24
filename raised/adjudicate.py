"""Adjudication (spec step 6): settle the vision model's reading (A) against the pixel reading (B), point by point.

Agree within 2% of the value axis range: take B (pixels are the measurement). Disagree: redraw the series with each
candidate value in the original's pixel frame and keep the one whose mask overlaps the original's colour mask better
near that point (a model-free tiebreak); the point is marked disputed either way, for the reviewer.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .model import Point, Semantics, Table
from .render import rerender
from .ticks import number, value_scale
from .trace import geometry, scene, trace_at

AGREE = 0.02  # share of the value axis range


def iou(a: np.ndarray, b: np.ndarray, tolerance: int = 1) -> float | None:
    """IoU after growing both masks by `tolerance` px, so a one-pixel offset of a thin line still counts as overlap.
    None when both are empty."""
    if tolerance:
        a, b = ndimage.binary_dilation(a, iterations=tolerance), ndimage.binary_dilation(b, iterations=tolerance)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else None


def value_range(sem: Semantics, tr: dict, table: Table) -> tuple[float, float, bool]:
    """(lo, hi, log) of the value axis from the outermost tick values (log10 units on a log axis); the data extent
    when ticks are missing."""
    log = value_scale(sem) == "log"
    ticks = sem.x_ticks if sem.value_axis == "x" else sem.y_ticks
    vals = [v for v in map(number, ticks) if v is not None and (v > 0 or not log)]
    if len(vals) < 2:
        vals = ([v for col in table.series.values() for v in col if v is not None and (v > 0 or not log)]
                or ([1.0, 10.0] if log else [0.0, 1.0]))  # nothing readable: a neutral one-decade / unit range
    lo, hi = min(vals), max(vals)
    if log:
        lo, hi = math.log10(lo), math.log10(hi)
    return lo, (hi if hi > lo else lo + 1.0), log


def _align(a: Table | None, xs: list) -> dict[str, list]:
    """A's values at B's x: categories by name (case-insensitive), numbers by nearest within half a step."""
    if a is None:
        return {}
    if all(isinstance(x, str) for x in xs):
        idx = {str(x).strip().lower(): i for i, x in enumerate(a.x)}
        pick = [idx.get(str(x).strip().lower()) for x in xs]
    else:
        ax = [float(x) if isinstance(x, (int, float)) else number(str(x)) for x in a.x]
        steps = np.diff(sorted(v for v in xs if v is not None))
        tol = 0.5 * float(np.median(steps)) if len(steps) else math.inf
        pick = []
        for x in xs:
            d = [(abs(v - x), i) for i, v in enumerate(ax) if v is not None]
            best = min(d, default=(math.inf, None))
            pick.append(best[1] if best[0] <= tol else None)
    return {n: [col[i] if i is not None and i < len(col) else None for i in pick] for n, col in a.series.items()}


def _window(sem: Semantics, tr: dict, table: Table, i: int, shape) -> tuple[slice, slice]:
    """Pixels around point i: its slot along the category/x axis, the whole plot box across."""
    h, w = shape
    x0, y0, x1, y1 = (int(v) for v in tr["plot_box"])
    horizontal = sem.chart_type == "hbar"
    if sem.chart_type in ("bar", "grouped_bar", "hbar"):
        cats = dict(tr.get("categories") or [])
        pos = [cats.get(c) for c in table.x]
    else:
        pos = [tr["x_cal"].px(x) for x in table.x]
    known = sorted(p for p in pos if p is not None)
    half = 0.5 * float(np.median(np.diff(known))) if len(known) > 1 else 20.0
    c = pos[i] if pos[i] is not None else (x0 + x1) / 2
    lo, hi = max(int(c - half), 0), int(c + half) + 1
    return (slice(max(lo, 0), min(hi, h)), slice(x0, x1)) if horizontal else (slice(y0, y1), slice(lo, min(hi, w)))


def adjudicate(reading_a: Table | None, reading_b: Table | None, img, sem: Semantics, ticks_result: dict
               ) -> tuple[Table, list[Point]]:
    """Final table and one Point per value. On a line or scatter chart the pixels are re-measured at the model's x
    values (B's own sample positions are a guess at where the data points are; A names them). Every series of either
    reading is kept."""
    tr = ticks_result
    numeric_a = reading_a is not None and reading_a.x and all(isinstance(x, (int, float)) for x in reading_a.x)
    if numeric_a and sem.chart_type in ("line", "scatter") and tr.get("x_cal") and tr.get("y_cal"):
        reading_b = Table(list(reading_a.x), trace_at(img, sem, tr, list(reading_a.x)))
    if reading_b is None or not reading_b.x:  # nothing measured: the model's reading, every point unconfirmed
        xs = list(reading_a.x) if reading_a else []
        reading_b = Table(xs, {n: [None] * len(xs) for n in (reading_a.series if reading_a else [])})
    for n in reading_a.series if reading_a else []:
        reading_b.series.setdefault(n, [None] * len(reading_b.x))
    lo, hi, log = value_range(sem, tr, reading_b)
    a_vals = _align(reading_a, reading_b.x)
    sc = scene(img, sem, tr)
    style = geometry(sc, sem, tr)
    size = sc.rgb.shape[1], sc.rgb.shape[0]
    scale = (lambda v: math.log10(v) if v and v > 0 else math.nan) if log else (lambda v: v)
    final, points = {}, []
    for name, col in reading_b.series.items():
        out = list(col)
        for i, (x, b) in enumerate(zip(reading_b.x, col)):
            a = (a_vals.get(name) or [None] * len(col))[i]
            if a is None and b is None:
                points.append(Point(name, x, None, a, b, "missing", "neither reading has this point"))
            elif a is None:
                points.append(Point(name, x, b, a, b, "pixel"))
            elif b is None:
                out[i] = a
                points.append(Point(name, x, a, a, b, "model", "the pixels could not be read here"))
            elif abs(scale(a) - scale(b)) <= AGREE * (hi - lo):
                points.append(Point(name, x, b, a, b, "agreed"))
            else:
                win = _window(sem, tr, reading_b, i, sc.rgb.shape[:2])
                orig = sc.masks.get(name, np.zeros(sc.rgb.shape[:2], bool))
                score = {}
                for cand in (a, b):
                    t = Table(reading_b.x, {name: [cand if j == i else v for j, v in enumerate(out)]})
                    score[cand] = iou(rerender(t, sem, tr, size, style)[name][win], orig[win]) or 0.0
                pick = max((a, b), key=lambda v: score[v])
                out[i] = pick
                points.append(Point(name, x, pick, a, b, "disputed",
                                    f"two readings disagree on {x}: {a:.4g} vs {b:.4g}; pixel match prefers {pick:.4g} "
                                    f"(IoU {score[pick]:.2f} vs {score[a if pick == b else b]:.2f})"))
        final[name] = out
    return Table(list(reading_b.x), final), points


def chart_iou(masks: dict[str, np.ndarray], img, sem: Semantics, ticks_result: dict) -> float:
    """Whole-chart score: mean over series of the IoU between a redraw's masks and the original's colour masks inside
    the plot box. Both masks grow by half the measured line width first (at least 1 px), so a redraw drawn thinner
    than the original still scores as overlap; a line is redrawn solid, so only the columns where the original shows
    ink count (a dash gap is not a disagreement)."""
    sc = scene(img, sem, ticks_result)
    x0, y0, x1, y1 = sc.box
    tol = 1
    if sem.chart_type in ("line", "scatter"):
        g = geometry(sc, sem, ticks_result)
        tol = max(1, int(round((g["marker_radius"] if sem.chart_type == "scatter" else g["line_width"]) / 2)))
    scores = []
    for name, m in masks.items():
        orig = sc.masks.get(name)
        if orig is None or np.asarray(m).shape != orig.shape:
            continue
        m, o = np.asarray(m, bool)[y0:y1, x0:x1], orig[y0:y1, x0:x1]
        if sem.chart_type == "line":
            m = m & ndimage.binary_dilation(o.any(0), iterations=2)[None, :]
        v = iou(m, o, tol)
        if v is not None:
            scores.append(v)
    return float(np.mean(scores)) if scores else 0.0
