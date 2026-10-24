"""Pixel tracer. Unit checks on charts drawn here, then the oracle-semantics benchmark on 40 SynthCharts from the bench
split (true semantics in, values out, scored against the generator's exact table)."""
import io
import json
import math
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from bench.metrics import axis_range, point_errors  # noqa: E402
from raised.model import Semantics, Series, Table  # noqa: E402
from raised.ticks import locate_ticks, value_scale  # noqa: E402
from raised.trace import _coverage, reference_rgb, trace, trace_at  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs the tesseract binary")
BENCH = Path(__file__).resolve().parents[1] / "bench" / "data" / "synth" / "bench"


def draw(kind: str, style: str = "default", blur: float = 0.0):
    """(image, semantics, truth) for a small chart of each type, drawn the way the generator does."""
    plt.style.use("default")
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    xs = list(range(2000, 2013))
    cats = ["Food", "Energy", "Housing", "Health", "Leisure"]
    if kind == "line":
        a = [20 + 10 * math.sin(i / 2) for i in range(13)]
        b = [45 - 1.5 * i for i in range(13)]
        ax.plot(xs, a, color="tab:red", lw=2, label="UK")
        ax.plot(xs, b, color="tab:blue", lw=2, ls="--", marker="o", markersize=5, label="Japan")
        ax.legend(loc="upper right", fontsize=8)
        series, truth = [Series("UK", "red"), Series("Japan", "blue", "dashed", "o")], Table(xs, {"UK": a, "Japan": b})
    elif kind == "scatter":
        ys = [30 + 7 * math.cos(i) for i in range(13)]
        ax.scatter(xs, ys, color="tab:green", s=28)
        series, truth = [Series("Rate", "green", "solid", "o")], Table(xs, {"Rate": ys})
    else:
        a, b = [120, 340, 210, 400, 260], [90, 150, 330, 280, 310]
        if kind == "grouped_bar":
            for k, (vals, col) in enumerate(((a, "tab:orange"), (b, "tab:purple"))):
                ax.bar(np.arange(5) + (k - 0.5) * 0.4, vals, width=0.36, color=col, label=["UK", "Japan"][k])
            ax.legend(loc="upper left", fontsize=8)
            series, truth = [Series("UK", "orange"), Series("Japan", "purple")], Table(cats, {"UK": a, "Japan": b})
        elif kind == "hbar":
            ax.barh(range(5), a, height=0.72, color="tab:brown")
            ax.set_yticks(range(5), cats)
            ax.invert_yaxis()
            series, truth = [Series("Spend", "brown")], Table(cats, {"Spend": a})
        else:
            ax.bar(range(5), a, width=0.72, color="tab:grey")
            series, truth = [Series("Spend", "grey")], Table(cats, {"Spend": a})
        if kind != "hbar":
            ax.set_xticks(range(5), cats)
    ax.set_title("Chart")
    fig.canvas.draw()
    labels = lambda axis: [t.get_text() for v, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                           if t.get_text() and min(axis.get_view_interval()) - 1e-9 <= v <= max(axis.get_view_interval()) + 1e-9]
    xt, yt = labels(ax.xaxis), labels(ax.yaxis)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    plt.style.use("default")
    im = Image.open(buf).convert("RGB")
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    bar = kind in ("bar", "grouped_bar", "hbar")
    sem = Semantics(chart_type=kind, title="Chart", x_ticks=xt, y_ticks=yt, x_scale="category" if bar else "linear",
                    categories=cats if bar else [], series=series, legend=[s.name for s in series] if len(series) > 1 else [])
    return im, sem, truth


def rel_errors(sem: Semantics, truth: Table, pred: dict) -> list[float | None]:
    """|pred - true| / value-axis range, the benchmark's own metric (bench/metrics.py: outermost tick labels, else the
    data extent; log10 units on a log axis)."""
    log = value_scale(sem) == "log"
    ticks = sem.x_ticks if sem.value_axis == "x" else sem.y_ticks
    return point_errors(truth, Table(truth.x, pred), axis_range(truth, sem.chart_type, ticks, log), log)


def test_reference_colours():
    assert np.allclose(reference_rgb("blue"), [31, 119, 180], atol=1)  # matplotlib tab:blue
    assert np.allclose(reference_rgb("#ff0000"), [255, 0, 0])
    assert reference_rgb("dark blue") is not None and reference_rgb("no such colour") is None


def test_coverage_model():
    bg, c = np.array([255.0, 255, 255]), np.array([31.0, 119, 180])
    a, d = _coverage(np.array([bg + 0.5 * (c - bg), [0.0, 0, 0]]), bg, c)
    assert abs(a[0] - 0.5) < 1e-9 and d[0] < 1e-9 and d[1] > 50


@pytest.mark.parametrize("kind,style,blur", [("line", "default", 0.0), ("line", "ggplot", 1.0), ("scatter", "bmh", 0.0),
                                             ("bar", "seaborn-v0_8-whitegrid", 0.8), ("grouped_bar", "default", 0.0),
                                             ("hbar", "fivethirtyeight", 0.0)])
def test_trace_at_true_x(kind, style, blur):
    im, sem, truth = draw(kind, style, blur)
    tr = locate_ticks(im, sem)
    errs = rel_errors(sem, truth, trace_at(im, sem, tr, truth.x))
    assert None not in errs and max(errs) < 0.01, errs


def test_trace_finds_the_x_grid():
    """Without the true x, a line is sampled on the regular grid its vertices sit on."""
    im, sem, truth = draw("line")
    t, conf = trace(im, sem, locate_ticks(im, sem))
    assert [round(x, 6) for x in t.x] == truth.x
    assert max(rel_errors(sem, truth, t.series)) < 0.01
    assert all(0 <= c <= 1 for col in conf.values() for c in col)


def _bench_set(n_per_type: dict) -> list[Path]:
    """The first charts of each type in the bench split (log line/scatter charts included)."""
    picked, count = [], {}
    for p in sorted(BENCH.glob("*.json")):
        s = json.loads(p.read_text())["semantics"]
        key = s["chart_type"] + ("/log" if s["y_scale"] == "log" else "")
        if count.get(key, 0) < n_per_type.get(key, 0):
            picked.append(p)
            count[key] = count.get(key, 0) + 1
    return picked


def _score(p: Path) -> dict:
    d = json.loads(p.read_text())
    sem, truth = Semantics.from_dict(d["semantics"]), Table.from_dict(d["table"])
    im = Image.open(p.with_suffix(".jpg"))
    tr = locate_ticks(im, sem)
    used = tr["diagnostics"][sem.value_axis].get("used", 0)
    return {"type": sem.chart_type, "ticks_ok": used >= 3, "errs": rel_errors(sem, truth, trace_at(im, sem, tr, truth.x))}


@pytest.mark.skipif(not BENCH.exists(),
                    reason="bench split not generated: python -m bench.make_synth --split bench --n 300 --seed 1")
def test_oracle_semantics_benchmark():
    """Targets (spec GO/NO-GO): ticks matched (>= 3 on the value axis) on >= 85% of charts; median error <= 2% of the
    axis range on line and bar charts."""
    files = _bench_set({"line": 8, "line/log": 3, "bar": 8, "grouped_bar": 7, "hbar": 7, "scatter": 5, "scatter/log": 2})
    assert len(files) == 40
    with ThreadPoolExecutor(8) as ex:  # Tesseract runs as a subprocess, so threads overlap the OCR
        res = list(ex.map(_score, files))
    rows, by = [], {}
    for r in res:
        by.setdefault(r["type"], []).append(r)
    for t, rs in sorted(by.items()):
        errs = [e for r in rs for e in r["errs"]]
        got = [e for e in errs if e is not None]
        rows.append((t, len(rs), len(errs), statistics.median(got) if got else float("nan"),
                     sum(e <= 0.02 for e in got) / len(errs), sum(not r["ticks_ok"] for r in rs)))
    print("\ntype          charts values  median  <=2%  tick fails")
    for t, n, nv, med, share, fails in rows:
        print(f"{t:13s} {n:6d} {nv:6d} {med:7.2%} {share:5.0%} {fails:6d}")
    assert sum(r["ticks_ok"] for r in res) / len(res) >= 0.85
    for kinds in (("line",), ("bar", "grouped_bar", "hbar")):
        got = [e for r in res if r["type"] in kinds for e in r["errs"] if e is not None]
        assert statistics.median(got) <= 0.02, kinds
