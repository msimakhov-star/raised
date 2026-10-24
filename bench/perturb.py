"""Planted faults for the verifier catch-rate metric (spec section 3, "Perturbation set").

Each variant is a correct Table with exactly one fault and a label saying which:
  shift        one value moved by 5 to 20% of the value-axis range (of its log10 range on a log axis)
  swap_series  the values of two series exchanged (names stay)
  drop_point   one x position removed from every series
  flip_scale   every value re-read as if the axis had the other scale (linear read as log, log read as linear),
               computed from the pixel positions of the axis ticks, so it is what a wrong calibration produces
A variant counts only if it changes some value by at least MIN_SHIFT of the range, so "fault planted" is checkable.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy

import numpy as np

from bench.metrics import axis_range, is_log, value_ticks
from raised.model import Semantics, Table, parse_number

KINDS = ("shift", "swap_series", "drop_point", "flip_scale")
SHIFT = (0.05, 0.20)
MIN_SHIFT = 0.05


class NotApplicable(ValueError):
    """This fault cannot be planted in this chart (one series, too few ticks, ...)."""


def _cells(t: Table):
    cells = [(s, i) for s, col in t.series.items() for i, v in enumerate(col) if v is not None]
    if not cells:
        raise NotApplicable("no values")
    return cells


def _gap(x: float, y: float, log: bool) -> float:
    """Distance between two values in axis units (log10 on a log axis, where a non-positive value is infinitely far)."""
    if log:
        return abs(math.log10(x) - math.log10(y)) if x > 0 and y > 0 else math.inf
    return abs(x - y)


def _fit(ticks: list[tuple[float, float]], log: bool):
    """(a, b) with value = a*px + b (log10(value) for log), from (value, px) tick pairs."""
    pts = [(math.log10(v) if log else v, px) for v, px in ticks if v > 0 or not log]
    if len(pts) < 2 or len({px for _, px in pts}) < 2:
        raise NotApplicable("fewer than two usable ticks")
    a, b = np.polyfit([px for _, px in pts], [v for v, _ in pts], 1)
    if a == 0:
        raise NotApplicable("flat calibration")
    return float(a), float(b)


def plant(table: Table, sem: Semantics, kind: str, rng: random.Random, tick_px: list | None = None) -> tuple[Table, dict]:
    """(faulty copy, label). tick_px: the value axis' [[label, pixel]] pairs from the generator's oracle, needed for
    flip_scale. Raises NotApplicable. The input is not modified."""
    t = deepcopy(table)
    log = is_log(sem)
    lo, hi = axis_range(table, sem.chart_type, value_ticks(sem), log)
    span = hi - lo  # log10 units on a log axis
    if kind == "shift":
        s, i = rng.choice(_cells(t))
        v = t.series[s][i]
        delta = rng.uniform(*SHIFT) * span * rng.choice((-1, 1))
        move = lambda d: 10 ** (math.log10(v) + d) if log else v + d
        if not log and v >= 0 and move(delta) < 0:
            delta = -delta
        t.series[s][i] = move(delta)
        return t, {"kind": kind, "series": s, "x": t.x[i], "index": i, "from": v, "to": t.series[s][i]}
    if kind == "swap_series":
        pairs = [(a, b) for a in t.series for b in t.series if a < b and
                 any(x is not None and y is not None and _gap(x, y, log) >= MIN_SHIFT * span
                     for x, y in zip(t.series[a], t.series[b]))]
        if not pairs:
            raise NotApplicable("needs two series that differ")
        a, b = rng.choice(pairs)
        t.series[a], t.series[b] = t.series[b], t.series[a]
        return t, {"kind": kind, "series": [a, b]}
    if kind == "drop_point":
        if len(t.x) < 3:
            raise NotApplicable("too few points")
        i = rng.randrange(len(t.x))
        x = t.x.pop(i)
        for col in t.series.values():
            col.pop(i)
        return t, {"kind": kind, "x": x, "index": i}
    if kind == "flip_scale":
        ticks = [(v, float(px)) for lab, px in (tick_px or []) if (v := parse_number(lab)) is not None]
        a, b = _fit(ticks, log)
        a2, b2 = _fit(ticks, not log)
        flipped = {}
        for s, col in t.series.items():
            out = []
            for v in col:
                if v is None or (log and v <= 0):
                    out.append(v)
                    continue
                px = ((math.log10(v) if log else v) - b) / a
                w = a2 * px + b2
                out.append(w if log else 10 ** w)
            flipped[s] = out
        if not any(x is not None and y is not None and _gap(x, y, log) >= MIN_SHIFT * span
                   for s in t.series for x, y in zip(t.series[s], flipped[s])):
            raise NotApplicable("the flip changes nothing visible")
        t.series = flipped
        return t, {"kind": kind, "from": "log" if log else "linear", "to": "linear" if log else "log"}
    raise ValueError(f"unknown fault kind {kind!r}")


def _try(c, kind: str, rng: random.Random):
    axis = "x" if c.semantics.value_axis == "x" else "y"
    try:
        return plant(c.truth, c.semantics, kind, rng, (c.oracle or {}).get(f"{axis}_ticks_px"))
    except NotApplicable:
        return None


def variants(cases, n: int = 100, seed: int = 0) -> list[dict]:
    """Up to n faulty variants, kinds in rotation, each case used once (a kind that fits no remaining case is skipped).
    cases: objects with .id, .truth (Table), .semantics (Semantics) and .oracle (generator oracle dict or None).
    Returns [{"case": id, "fault": label, "table": Table}]."""
    rng = random.Random(seed)
    pool = sorted(cases, key=lambda c: c.id)
    rng.shuffle(pool)
    out: list[dict] = []
    for k in range(n):
        order = KINDS[k % len(KINDS):] + KINDS[:k % len(KINDS)]
        hit = next(((c, r) for kind in order for c in pool if (r := _try(c, kind, rng))), None)
        if hit is None:
            break
        (c, (table, fault)) = hit
        pool.remove(c)
        out.append({"case": c.id, "fault": fault, "table": table})
    return out
