"""Tick location and axis calibration. Offline: Tesseract is a local binary; charts are drawn here with matplotlib,
so the true pixel position of every tick is known exactly."""
import io
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from raised import ticks as T  # noqa: E402
from raised.model import Semantics, Series  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs the tesseract binary")


def draw(kind="line", style="default", log=False, labels=True, rotate=0.0):
    """A chart plus its semantics and the true pixel position of each tick (continuous coords, origin top-left)."""
    plt.style.use("default")
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    cats = ["Food", "Energy", "Housing", "Health"]
    if kind == "bar":
        ax.bar(range(4), [120, 340, 210, 400], color="tab:green")
        ax.set_xticks(range(4), cats)
    else:
        ys = [20, 400, 3000, 9000] if log else [12.5, 30.0, 22.0, 41.0]
        ax.plot([2000, 2004, 2008, 2012], ys, color="tab:blue")
        if log:  # plain labels, as a textbook prints them (matplotlib's default is mathtext 10^n)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if not labels:
        ax.set_yticklabels([])
    ax.set_title("Exports")
    fig.canvas.draw()
    h = fig.canvas.get_width_height()[1]

    def ticks(axis, horizontal):
        lo, hi = sorted(axis.get_view_interval())
        out = []
        for v, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels()):
            if lo - 1e-9 <= v <= hi + 1e-9 and (t.get_text() or not labels):
                p = (ax.transData.transform((v, ax.get_ylim()[0]))[0] if horizontal
                     else h - ax.transData.transform((ax.get_xlim()[0], v))[1])
                out.append((t.get_text() or f"{v:g}", float(p)))
        return out

    xt, yt = ticks(ax.xaxis, True), sorted(ticks(ax.yaxis, False), key=lambda t: -t[1])
    bb = ax.get_window_extent()
    box = (bb.x0, h - bb.y1, bb.x1, h - bb.y0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    plt.style.use("default")
    im = Image.open(buf).convert("RGB")
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC, fillcolor="white")
    sem = Semantics(chart_type=kind, x_ticks=[s for s, _ in xt], y_ticks=[s for s, _ in yt], y_scale="log" if log else "linear",
                    x_scale="category" if kind == "bar" else "linear", categories=cats if kind == "bar" else [],
                    series=[Series("Exports", "green" if kind == "bar" else "blue")], title="Exports")
    return im, sem, xt, yt, box


def test_number_tolerates_ocr_junk():
    assert T.number("$1,200") == 1200 and T.number("40%") == 40
    assert T.number("100-") is None  # parse_number raises on this one


def test_label_score():
    assert T.label_score("2010", "2010", True) == 1.0
    assert T.label_score("2010", "2O1O", True) == 0.9      # O read for 0
    assert T.label_score("$20,000", "S20,000", True) == 0.9  # '$' read as 'S'
    assert T.label_score("1,000", "1.000", True) == 0.6      # same digits, ',' read as '.'
    assert T.label_score("2010", "2011", True) == 0.0
    assert T.label_score("Germany", "Germamy", False) >= 0.75
    assert T.label_score("Germany", "Spain", False) == 0.0


def test_calibrate_drops_an_outlier_and_picks_log():
    ticks = [("0", 0.0, 400.0), ("10", 10.0, 300.0), ("20", 20.0, 200.0), ("30", 30.0, 140.0), ("40", 40.0, 0.0)]
    cal, used = T.calibrate(ticks, "linear")
    assert len(used) == 4 and cal.kind == "linear" and abs(cal.px(25) - 150) < 1e-6
    logt = [("10", 10.0, 300.0), ("100", 100.0, 200.0), ("1,000", 1000.0, 100.0)]
    cal, _ = T.calibrate(logt, "linear")  # semantics said linear; log fits exactly
    assert cal.kind == "log" and abs(cal.value(150.0) - 10 ** 2.5) < 1e-6


def test_rotated_ocr_maps_back():
    fig = plt.figure(figsize=(4, 2.5), dpi=110)
    fig.text(0.3, 0.5, "Transport", rotation=30, ha="center", va="center", fontsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    rgb = np.asarray(Image.open(buf).convert("RGB"))
    w = [t for t in T.ocr_words(rgb, 11, rotate=30) if t.text == "Transport"][0]
    assert abs(w.cx - 0.3 * rgb.shape[1]) < 3 and abs(w.cy - 0.5 * rgb.shape[0]) < 3


@pytest.mark.parametrize("style", ["default", "ggplot", "fivethirtyeight", "classic"])
def test_plot_box(style):
    im, sem, xt, yt, box = draw(style=style)
    rgb = np.asarray(im)
    found = T.find_plot_box(rgb, T.ocr_words(rgb))[0]
    assert np.abs(np.array(found) - np.array(box)).max() <= 3, (found, box)


@pytest.mark.parametrize("log", [False, True])
def test_locate_ticks_line(log):
    im, sem, xt, yt, box = draw(log=log)
    r = T.locate_ticks(im, sem)
    assert r["x_cal"].kind == "linear" and r["y_cal"].kind == ("log" if log else "linear")
    for cal, truth in ((r["x_cal"], xt), (r["y_cal"], yt)):
        err = max(abs(cal.px(T.number(s)) - p) for s, p in truth)
        assert err < 1.0, (truth, cal)
    assert r["diagnostics"]["y"]["used"] >= 3 and not r["diagnostics"]["y"]["low_confidence"]


def test_locate_ticks_bar_categories():
    im, sem, xt, yt, box = draw(kind="bar")
    r = T.locate_ticks(im, sem)
    assert r["x_cal"] is None and r["diagnostics"]["x"]["method"] == "labels"
    for (lab, p), (tl, tp) in zip(r["categories"], xt):
        assert lab == tl and abs(p - tp) < 2


def test_deskew():
    im, sem, xt, yt, box = draw(rotate=0.6)
    a = T.estimate_skew(np.asarray(im))
    assert abs(a + 0.6) <= 0.06
    r = T.locate_ticks(im, sem)
    assert max(abs(r["y_cal"].px(T.number(s)) - p) for s, p in yt) < 1.5


def test_tick_marks_fallback_without_labels():
    im, sem, xt, yt, box = draw(labels=False)
    r = T.locate_ticks(im, sem)
    assert r["diagnostics"]["y"]["method"] == "marks"
    assert r["y_cal"] is not None and r["y_cal"].n == len(sem.y_ticks)
