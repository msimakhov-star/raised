"""Redraw and adjudication: plant a wrong value in one reading and check the redraw picks the other."""
import copy
import io
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from raised.adjudicate import _align, adjudicate, chart_iou, iou  # noqa: E402
from raised.model import Semantics, Series, Table  # noqa: E402
from raised.render import rerender  # noqa: E402
from raised.ticks import locate_ticks  # noqa: E402
from raised.trace import trace_at  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs the tesseract binary")


def chart(kind: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    if kind == "line":
        xs, ys = list(range(2000, 2011)), [12, 18, 15, 22, 30, 27, 33, 31, 38, 36, 40]
        ax.plot(xs, ys, color="tab:blue", lw=2)
    else:
        xs, ys = ["Food", "Energy", "Housing", "Health"], [120, 340, 210, 400]
        ax.bar(range(4), ys, color="tab:green")
        ax.set_xticks(range(4), xs)
    fig.canvas.draw()
    lab = lambda axis: [t.get_text() for v, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                        if t.get_text() and min(axis.get_view_interval()) <= v <= max(axis.get_view_interval())]
    sem = Semantics(chart_type=kind, x_ticks=lab(ax.xaxis), y_ticks=lab(ax.yaxis),
                    x_scale="category" if kind == "bar" else "linear",
                    categories=xs if kind == "bar" else [], series=[Series("S", "blue" if kind == "line" else "green")])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    return im, sem, Table(xs, {"S": [float(v) for v in ys]})


def test_iou():
    a = np.zeros((10, 10), bool)
    a[2:5, 2:5] = True
    assert iou(a, a, 0) == 1.0 and iou(a, np.zeros_like(a), 0) == 0.0 and iou(np.zeros_like(a), np.zeros_like(a)) is None
    b = np.roll(a, 1, axis=1)
    assert iou(a, b, 0) == 0.5 and iou(a, b, 1) > 0.5  # the tolerance forgives a 1 px shift


def test_align_numbers_and_categories():
    a = Table([2000.1, 2001.0, 2005.0], {"S": [1.0, 2.0, 3.0]})
    assert _align(a, [2000, 2001, 2002, 2003])["S"] == [1.0, 2.0, None, None]
    c = Table(["food", "Energy"], {"S": [5.0, 6.0]})
    assert _align(c, ["Energy", "Food", "Health"])["S"] == [6.0, 5.0, None]
    assert _align(None, [1, 2]) == {}


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_redraw_overlaps_and_a_wrong_value_scores_lower(kind):
    im, sem, truth = chart(kind)
    tr = locate_ticks(im, sem)
    good = chart_iou(rerender(truth, sem, tr, im.size), im, sem, tr)
    bad_t = copy.deepcopy(truth)
    bad_t.series["S"][2] *= 1.4
    bad = chart_iou(rerender(bad_t, sem, tr, im.size), im, sem, tr)
    assert good > 0.7 and bad < good


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_adjudicate_picks_the_reading_that_redraws_better(kind):
    im, sem, truth = chart(kind)
    tr = locate_ticks(im, sem)
    b = Table(truth.x, trace_at(im, sem, tr, truth.x))
    wrong = copy.deepcopy(b)
    wrong.series["S"][3] = b.series["S"][3] + 9 if kind == "line" else b.series["S"][3] * 0.7
    # the model misread one point: keep the pixels
    t, pts = adjudicate(wrong, b, im, sem, tr)
    assert t.series["S"][3] == b.series["S"][3] and pts[3].status == "disputed" and "prefers" in pts[3].note
    assert {p.status for i, p in enumerate(pts) if i != 3} == {"agreed"}
    # the pixels misread one point: keep the model's reading (a line is re-measured at the model's x, so it agrees)
    t, pts = adjudicate(b, wrong, im, sem, tr)
    assert t.series["S"][3] == b.series["S"][3] and pts[3].status == ("agreed" if kind == "line" else "disputed")


def test_adjudicate_without_reading_a_and_with_gaps():
    im, sem, truth = chart("bar")
    tr = locate_ticks(im, sem)
    b = Table(truth.x, {"S": [truth.series["S"][0], None, *truth.series["S"][2:]]})
    t, pts = adjudicate(None, b, im, sem, tr)
    assert [p.status for p in pts] == ["pixel", "missing", "pixel", "pixel"]
    a = Table(truth.x, {"S": truth.series["S"]})
    t, pts = adjudicate(a, b, im, sem, tr)
    assert pts[1].status == "model" and t.series["S"][1] == truth.series["S"][1]
