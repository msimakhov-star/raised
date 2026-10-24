"""Facts computed by code only: the numbers the description model may cite, each with a stable id.

    facts(table, sem) -> [{"id": "S1.max", "kind": "max", "text": "...", "values": {...}}, ...]

Ids: S<i>.<kind> per series (i counts table.series in order), G.<kind> for cross-series facts.
`values` holds the raw numbers behind `text` (also read by numguard); `text` shows them the way the chart labels
its axis (see format_like). Line charts, and bar charts whose categories are all numbers (years), are "ordered" and
get first/last/change/runs/step; bar charts get a ranking; scatter plots get max/min only.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .model import Semantics, Table, parse_number

_BARS = ("bar", "hbar", "grouped_bar")
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12, "mn": 1e6}
_TICK = re.compile(r"^-?(?P<cur>[$£€])?(?P<num>[\d,]*\.?\d+)(?P<suf>%|bn|mn|[kKmMbBtT])?$")


def value_ticks(sem: Semantics) -> list[str]:
    """Tick labels of the axis that carries the measured values."""
    return sem.x_ticks if sem.value_axis == "x" else sem.y_ticks


def value_scale(sem: Semantics) -> str:
    """'linear' or 'log' for the value axis. The synthetic generator labels hbar scales the other way round
    (x_scale 'category', y_scale 'linear'), so a 'category' answer falls back to the other axis."""
    a, b = (sem.x_scale, sem.y_scale) if sem.value_axis == "x" else (sem.y_scale, sem.x_scale)
    return a if a != "category" else (b if b != "category" else "linear")


def _style(sem: Semantics) -> dict:
    """Currency prefix, suffix (%, k, M, ...), thousands separator and decimals, read off the value-axis ticks."""
    ms = [m for t in value_ticks(sem) if (m := _TICK.match(t.strip().replace("−", "-")))]
    cur = Counter(m["cur"] or "" for m in ms).most_common(1)
    suf = Counter(m["suf"] or "" for m in ms if m["suf"] or float(m["num"].replace(",", "")) != 0).most_common(1)
    nums = [m["num"] for m in ms]
    return {
        "cur": cur[0][0] if cur else "",
        "suf": suf[0][0] if suf else "",
        "dec": max((len(n.split(".")[1]) for n in nums if "." in n), default=0),
        # thousands separators unless the ticks print 4+ digit numbers without them
        "comma": any("," in n for n in nums) or not any(len(n.split(".")[0]) >= 4 for n in nums),
    }


def format_like(value: float, sem: Semantics) -> str:
    """Format a value the way the chart labels its value axis: 3200 -> '$3,200' / '3.2k' / '3,200', 12.34 -> '12.3%'.
    Keeps the tick decimals and adds up to three for small numbers so that about three significant digits show."""
    st = _style(sem)
    v = abs(value) / _SCALE.get(st["suf"].lower(), 1.0)
    dec = st["dec"]
    if v:
        dec = max(dec, min(3, max(0, 2 - math.floor(math.log10(v)))))
    s = f"{v:,.{dec}f}" if st["comma"] else f"{v:.{dec}f}"
    while dec > st["dec"] and s.endswith("0"):
        s, dec = s[:-1], dec - 1
    s = s.rstrip(".")
    return f"{'-' if value < 0 and float(s.replace(',', '')) else ''}{st['cur']}{s}{st['suf']}"


def _fx(x) -> str:
    return x if isinstance(x, str) else (str(int(x)) if float(x).is_integer() else f"{x:g}")


def _when(x) -> str:
    n = parse_number(x) if isinstance(x, str) else x
    if n is None:
        return f"for {x}"
    return f"in {_fx(x)}" if float(n).is_integer() and 1000 <= n <= 2200 else f"at {_fx(x)}"


def _pct(p: float) -> str:
    return f"{p:.1f}".rstrip("0").rstrip(".") + "%"


def _amount(v: float, sem: Semantics) -> str:
    """An absolute difference: percentage points on a percent axis."""
    s = format_like(abs(v), sem)
    return s[:-1] + " percentage points" if s.endswith("%") else s


def _fact(id: str, kind: str, text: str, **values) -> dict:
    return {"id": id, "kind": kind, "text": text, "values": values}


def _runs(pts: list, max_runs: int = 4) -> list[list]:
    """Maximal runs of consecutive rises or falls spanning at least 3 points; the max_runs biggest, in x order."""
    runs, cur, sign = [], [pts[0]], 0
    for a, b in zip(pts, pts[1:]):
        s = (b[1] > a[1]) - (b[1] < a[1])
        if s and s == sign:
            cur.append(b)
            continue
        if len(cur) >= 3:
            runs.append(cur)
        cur, sign = ([a, b] if s else [b]), s
    if len(cur) >= 3:
        runs.append(cur)
    keep = sorted(runs, key=lambda r: -abs(r[-1][1] - r[0][1]))[:max_runs]
    return [r for r in runs if r in keep]


def _series_facts(i: int, name: str, pts: list, ordered: bool, bar: bool, sem: Semantics) -> list[dict]:
    p, fv = f"S{i}", lambda v: format_like(v, sem)
    hi, lo = max(pts, key=lambda t: t[1]), min(pts, key=lambda t: t[1])
    out = [
        _fact(f"{p}.max", "max", f"{name} peaks at {fv(hi[1])} {_when(hi[0])}.", series=name, x=hi[0], value=hi[1]),
        _fact(f"{p}.min", "min", f"{name} is lowest at {fv(lo[1])} {_when(lo[0])}.", series=name, x=lo[0], value=lo[1]),
    ]
    if ordered:
        (x0, v0), (x1, v1) = pts[0], pts[-1]
        d = v1 - v0
        pct = d / v0 * 100 if v0 > 0 else None
        verb = "rose" if d > 0 else "fell" if d < 0 else "was unchanged"
        by = f" by {_amount(d, sem)}" + (f" ({_pct(abs(pct))})" if pct is not None else "") if d else ""
        out += [
            _fact(f"{p}.first", "first", f"{name} starts at {fv(v0)} {_when(x0)}.", series=name, x=x0, value=v0),
            _fact(f"{p}.last", "last", f"{name} ends at {fv(v1)} {_when(x1)}.", series=name, x=x1, value=v1),
            _fact(f"{p}.change", "change", f"{name} {verb}{by} between {_fx(x0)} and {_fx(x1)}.", series=name,
                  x_from=x0, x_to=x1, start=v0, end=v1, change=abs(d), pct=None if pct is None else abs(pct),
                  direction="up" if d > 0 else "down" if d < 0 else "flat"),
        ]
        for k, r in enumerate(_runs(pts), 1):
            up = r[-1][1] > r[0][1]
            out.append(_fact(f"{p}.run{k}", "run",
                             f"{name} {'rises' if up else 'falls'} steadily from {fv(r[0][1])} {_when(r[0][0])} "
                             f"to {fv(r[-1][1])} {_when(r[-1][0])}.", series=name, direction="up" if up else "down",
                             x_from=r[0][0], x_to=r[-1][0], start=r[0][1], end=r[-1][1], points=len(r)))
        a, b = max(zip(pts, pts[1:]), key=lambda ab: abs(ab[1][1] - ab[0][1]))
        d = b[1] - a[1]
        out.append(_fact(f"{p}.step", "step",
                         f"The largest single step for {name} is a {'rise' if d > 0 else 'fall'} of {_amount(d, sem)} "
                         f"between {_fx(a[0])} and {_fx(b[0])}.", series=name, x_from=a[0], x_to=b[0],
                         start=a[1], end=b[1], change=abs(d), direction="up" if d > 0 else "down"))
    if bar:
        order = sorted(pts, key=lambda t: -t[1])
        top = ", then ".join(f"{_fx(x)} ({fv(v)})" for x, v in order[:3])
        tail = f", and the lowest is {_fx(order[-1][0])} ({fv(order[-1][1])})" if len(order) > 3 else ""
        out.append(_fact(f"{p}.rank", "rank", f"For {name}, the highest is {top}{tail}.", series=name,
                         order=[{"x": x, "value": v} for x, v in order]))
    return out


def facts(table: Table, sem: Semantics) -> list[dict]:
    if sem.chart_type == "unsupported":
        return []
    bar = sem.chart_type in _BARS
    ordered = sem.chart_type == "line" or (bar and bool(table.x) and all(parse_number(str(x)) is not None for x in table.x))
    fv = lambda v: format_like(v, sem)
    series = {n: [(x, v) for x, v in zip(table.x, vs) if v is not None] for n, vs in table.series.items()}
    out = []
    for i, (name, pts) in enumerate(series.items(), 1):
        if len(pts) >= 2:
            out += _series_facts(i, name, pts, ordered, bar, sem)
    if len(series) < 2 or sem.chart_type == "scatter":
        return out
    cols = list(zip(*table.series.values()))  # one tuple per x
    if bar and not ordered:
        won = Counter(max((v, -j) for j, v in enumerate(c) if v is not None)[1] * -1
                      for c in cols if any(v is not None for v in c))
        names = list(table.series)
        if sum(won.values()) >= 2:
            parts = ", ".join(f"{names[j]} in {n}" for j, n in won.most_common())
            out.append(_fact("G.leaders", "leaders", f"Across {sum(won.values())} categories, the highest series is {parts}.",
                             total=sum(won.values()), counts={names[j]: n for j, n in won.items()}))
        return out
    k = next((k for k in range(len(cols) - 1, -1, -1) if sum(v is not None for v in cols[k]) >= 2), None)
    if k is not None:
        order = sorted(((n, v) for n, v in zip(table.series, cols[k]) if v is not None), key=lambda t: -t[1])
        x = table.x[k]
        text = f"{_when(x).capitalize()}, {order[0][0]} is highest at {fv(order[0][1])}, ahead of {order[1][0]} at {fv(order[1][1])}"
        if len(order) > 2:
            text += f"; {order[-1][0]} is lowest at {fv(order[-1][1])}"
        out.append(_fact("G.last_compare", "compare_last", text + ".", x=x, gap=order[0][1] - order[1][1],
                         order=[{"series": n, "value": v} for n, v in order]))
    return out
