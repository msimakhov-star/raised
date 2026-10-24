"""SynthCharts: charts with exact ground truth for the Raised benchmark.

    python -m bench.make_synth --split bench --n 300 --seed 1
    python -m bench.make_synth --split probe --n 50 --seed 2

Writes bench/data/synth/<split>/NNNN.jpg and NNNN.json. The JSON holds:
  semantics  - raised.model.Semantics as dict (tick strings exactly as rendered)
  table      - raised.model.Table as dict (the true values)
  oracle     - pixel positions of every tick and the plot box BEFORE degradation, plus the degradation applied
Charts vary: type (line / bar / grouped_bar / hbar / scatter), 1 to 4 series, 15% log value axes, six matplotlib
styles, four fonts, tick formats (plain, %, $, k, M), legends, markers, then a scan-like degradation pass.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from raised.model import Semantics, Series, Table  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "synth"
STYLES = ["default", "ggplot", "seaborn-v0_8-whitegrid", "bmh", "fivethirtyeight", "classic"]
FONTS = ["DejaVu Sans", "Arial", "Times New Roman", "Courier New"]
# colours with the plain names a reader (or the vision model) would use
PALETTE = [("#1f77b4", "blue"), ("#d62728", "red"), ("#2ca02c", "green"), ("#ff7f0e", "orange"),
           ("#9467bd", "purple"), ("#8c564b", "brown"), ("#17becf", "teal"), ("#7f7f7f", "grey"), ("#000000", "black")]
COUNTRIES = ["UK", "France", "Germany", "Spain", "Italy", "Poland", "Japan", "Brazil", "India", "Kenya", "Canada", "Chile"]
SECTORS = ["Food", "Energy", "Housing", "Transport", "Health", "Education", "Leisure", "Clothing"]
METRICS = [("GDP per capita", "$"), ("Unemployment rate", "%"), ("Population", "M"), ("Exports", "$bn"),
           ("Emissions", "Mt"), ("Price index", ""), ("Enrolment", "k"), ("Share of households", "%")]


def fmt_factory(kind: str):
    """Tick formatters whose output our parser understands; returns (formatter, unit-for-title)."""
    if kind == "pct":
        return ticker.FuncFormatter(lambda v, _: f"{v:g}%"), ""
    if kind == "usd":
        return ticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if abs(v) >= 1 else f"${v:g}"), ""
    if kind == "k":
        return ticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}k" if v else "0"), ""
    if kind == "M":
        return ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"), ""
    if kind == "comma":
        return ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1 else f"{v:g}"), ""
    return ticker.FuncFormatter(lambda v, _: f"{v:g}"), ""


def log_formatter():
    def f(v, _):
        if v <= 0:
            return ""
        e = np.log10(v)
        if abs(e - round(e)) > 1e-9:
            return ""
        return f"{v:,.0f}" if v >= 1 else f"{v:g}"
    return ticker.FuncFormatter(f)


def make_values(rng, n, kind, scale):
    if scale == "log":
        base = rng.uniform(1, 2.5)
        return list(10 ** (base + np.cumsum(rng.normal(0.12, 0.12, n))))
    start = {"pct": rng.uniform(2, 60), "usd": rng.uniform(500, 50000), "k": rng.uniform(5e3, 9e4),
             "M": rng.uniform(2e6, 9e7), "comma": rng.uniform(100, 9000)}.get(kind, rng.uniform(5, 300))
    shape = rng.choice(["walk", "trend", "seasonal", "step"])
    t = np.arange(n)
    if shape == "walk":
        v = start + np.cumsum(rng.normal(0, start * 0.06, n))
    elif shape == "trend":
        v = start * (1 + rng.uniform(-0.04, 0.08) * t) + rng.normal(0, start * 0.02, n)
    elif shape == "seasonal":
        v = start * (1 + 0.15 * np.sin(t * 2 * np.pi / rng.uniform(4, 8))) + rng.normal(0, start * 0.02, n)
    else:
        v = start * np.where(t < n // 2, 1.0, rng.uniform(0.6, 1.5)) + rng.normal(0, start * 0.015, n)
    if kind == "pct":
        v = np.clip(v, 0.5, 99)
    return list(np.maximum(v, start * 0.05))


def chart(rng: np.random.Generator, i: int):
    ctype = str(rng.choice(["line", "line", "bar", "grouped_bar", "hbar", "scatter"]))
    style = STYLES[i % len(STYLES)]
    font = FONTS[(i // len(STYLES)) % len(FONTS)]
    metric, unit = METRICS[rng.integers(len(METRICS))]
    kind = {"$": "usd", "%": "pct", "M": "M", "k": "k", "$bn": "comma"}.get(unit, rng.choice(["plain", "comma"]))
    log = ctype in ("line", "scatter") and rng.random() < 0.15
    if log:
        kind = "plain"
    nser = 1 if ctype in ("bar", "hbar", "scatter") else int(rng.integers(1, 5)) if ctype == "line" else int(rng.integers(2, 4))
    cols = rng.permutation(len(PALETTE))[:nser]
    names = [str(c) for c in rng.choice(COUNTRIES, nser, replace=False)] if nser > 1 else [metric]

    plt.style.use(style)
    plt.rcParams["font.family"] = font
    w, h = rng.uniform(5.5, 8.5), rng.uniform(3.6, 5.2)
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    sem = Semantics(chart_type=ctype, title=f"{metric}{' by country' if nser > 1 else ''}",
                    y_scale="log" if log else "linear", x_scale="linear")
    series_vals = {}
    if ctype in ("line", "scatter"):
        n = int(rng.integers(6, 26))
        x0 = int(rng.integers(1960, 2006))
        step = int(rng.choice([1, 1, 2, 5]))
        xs = [x0 + step * k for k in range(n)]
        sem.x_title = "Year"
        sem.y_title = metric + (f" ({unit})" if unit and unit not in "$%" else "")
        styles = ["solid", "dashed", "dotted", "solid"]
        markers = ["", "o", "s", "^"]
        for k, name in enumerate(names):
            vals = make_values(rng, n, kind, "log" if log else "linear")
            hexc, cname = PALETTE[cols[k]]
            if ctype == "line":
                ls = styles[k] if rng.random() < 0.5 else "solid"
                mk = markers[k] if rng.random() < 0.4 else ""
                ax.plot(xs, vals, color=hexc, lw=rng.uniform(1.6, 2.6), linestyle={"solid": "-", "dashed": "--", "dotted": ":"}[ls],
                        marker=mk or None, markersize=5, label=name)
                sem.series.append(Series(name, cname, ls, mk))
            else:
                xs_j = list(np.array(xs) + rng.uniform(-0.3, 0.3, n) * step)
                xs = xs_j
                ax.scatter(xs, vals, color=hexc, s=28, label=name)
                sem.series.append(Series(name, cname, "solid", "o"))
            series_vals[name] = vals
        x_axis_vals = xs
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(log_formatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.yaxis.set_major_formatter(fmt_factory(kind)[0])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ncat = int(rng.integers(3, 9))
        cats = [str(c) for c in rng.choice(SECTORS if rng.random() < 0.5 else COUNTRIES, ncat, replace=False)]
        sem.categories = cats
        sem.x_scale = "category"
        width = 0.8 / nser
        for k, name in enumerate(names):
            vals = list(np.abs(make_values(rng, ncat, kind, "linear")))
            hexc, cname = PALETTE[cols[k]]
            pos = np.arange(ncat) + (k - (nser - 1) / 2) * width
            if ctype == "hbar":
                ax.barh(pos, vals, height=width * 0.9, color=hexc, label=name)
            else:
                ax.bar(pos, vals, width=width * 0.9, color=hexc, label=name)
            sem.series.append(Series(name, cname, "solid", ""))
            series_vals[name] = vals
        if ctype == "hbar":
            ax.set_yticks(range(ncat), cats)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(fmt_factory(kind)[0])
            sem.x_title, sem.y_title = metric, ""
        else:
            ax.set_xticks(range(ncat), cats, rotation=30 if ncat > 6 else 0)
            ax.yaxis.set_major_formatter(fmt_factory(kind)[0])
            sem.x_title, sem.y_title = "", metric
        x_axis_vals = cats
    if nser > 1 and rng.random() < 0.85:
        ax.legend(loc=rng.choice(["best", "upper left", "lower right", "upper right"]), fontsize=8)
        sem.legend = names
    ax.set_title(sem.title)
    if sem.x_title:
        ax.set_xlabel(sem.x_title)
    if sem.y_title:
        ax.set_ylabel(sem.y_title)
    src = rng.choice(["Source: World Bank (synthetic)", "Source: national statistics office (synthetic)", ""])
    if src:
        fig.text(0.01, 0.01, src, fontsize=7)
        sem.source = src
    fig.tight_layout()
    fig.canvas.draw()

    # tick strings and pixel positions exactly as rendered (crop pixel coordinates, origin top-left)
    H = fig.canvas.get_width_height()[1]
    def ticks_of(axis):
        out = []
        locs = axis.get_majorticklocs()
        labels = [t.get_text() for t in axis.get_majorticklabels()]
        lo, hi = sorted(axis.get_view_interval())
        for v, s in zip(locs, labels):
            if s and lo - 1e-9 <= v <= hi + 1e-9:
                if axis is ax.xaxis:
                    px = ax.transData.transform((v, ax.get_ylim()[0]))[0]
                    out.append((s, float(px)))
                else:
                    py = H - ax.transData.transform((ax.get_xlim()[0], v))[1]
                    out.append((s, float(py)))
        return out
    xt, yt = ticks_of(ax.xaxis), ticks_of(ax.yaxis)
    if ctype == "hbar":
        yt = list(reversed(yt))  # reading order for y is bottom to top
    else:
        yt = sorted(yt, key=lambda t: -t[1])
    sem.x_ticks = [s for s, _ in xt]
    sem.y_ticks = [s for s, _ in yt]
    bbox = ax.get_window_extent()
    plot_box = [float(bbox.x0), float(H - bbox.y1), float(bbox.x1), float(H - bbox.y0)]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    plt.style.use("default")
    im = Image.open(buf).convert("RGB")
    table = Table(x=[float(v) if not isinstance(v, str) else v for v in x_axis_vals],
                  series={k: [float(v) for v in vs] for k, vs in series_vals.items()})
    oracle = {"x_ticks_px": xt, "y_ticks_px": yt, "plot_box": plot_box, "style": style, "font": font}
    return im, sem, table, oracle


def degrade(im: Image.Image, rng) -> tuple[Image.Image, dict]:
    d = {"rotate": 0.0, "blur": 0.0, "jpeg": 92, "noise": 0.0}
    if rng.random() < 0.3:
        d["rotate"] = float(rng.uniform(-0.5, 0.5))
        im = im.rotate(d["rotate"], resample=Image.BICUBIC, expand=False, fillcolor="white")
    if rng.random() < 0.6:
        d["blur"] = float(rng.uniform(0.2, 1.2))
        im = im.filter(ImageFilter.GaussianBlur(d["blur"]))
    if rng.random() < 0.4:
        d["noise"] = float(rng.uniform(2, 8))
        a = np.asarray(im).astype(float) + rng.normal(0, d["noise"], (im.height, im.width, 1))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d["jpeg"] = int(rng.integers(40, 91)) if rng.random() < 0.7 else 92
    return im, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bench")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clean", action="store_true", help="skip the degradation pass")
    a = ap.parse_args()
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    counts = {}
    for i in range(a.n):
        im, sem, table, oracle = chart(rng, i)
        deg = {"rotate": 0.0, "blur": 0.0, "jpeg": 95, "noise": 0.0}
        if not a.clean:
            im, deg = degrade(im, rng)
        oracle["degradation"] = deg
        im.save(out / f"{i:04d}.jpg", quality=deg["jpeg"])
        (out / f"{i:04d}.json").write_text(json.dumps({"semantics": sem.to_dict(), "table": table.to_dict(), "oracle": oracle}))
        counts[sem.chart_type] = counts.get(sem.chart_type, 0) + 1
    print(f"wrote {a.n} charts to {out}: {counts}")


if __name__ == "__main__":
    main()
