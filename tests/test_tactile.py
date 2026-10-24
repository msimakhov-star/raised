"""Tactile compiler and BRF writer on synthetic charts with known tables (bench/data/synth)."""
import io
import json
import os
import pathlib
import re
import xml.etree.ElementTree as ET

import cairosvg
import pypdfium2 as pdfium
import pytest
from PIL import Image

from raised import braille, brf, tactile
from raised import tactile_rules as R
from raised.model import Semantics, Series, Table

pytestmark = pytest.mark.skipif(not os.path.exists(braille.LOU), reason="liblouis lou_translate not installed")

SYNTH = pathlib.Path(__file__).resolve().parents[1] / "bench/data/synth/bench"


def _synth(kind: str, n_series: int):
    for f in sorted(SYNTH.glob("*.json")):
        d = json.loads(f.read_text())
        if d["semantics"]["chart_type"] == kind and len(d["table"]["series"]) == n_series:
            return Table.from_dict(d["table"]), Semantics.from_dict(d["semantics"])
    pytest.skip(f"no synthetic {kind} chart with {n_series} series")


CASES = {"line3": ("line", 3), "grouped_bar": ("grouped_bar", 2), "hbar": ("hbar", 1)}


@pytest.fixture(scope="module", params=list(CASES))
def chart(request):
    table, sem = _synth(*CASES[request.param])
    return request.param, table, sem, tactile.compile(table, sem, sem.title or "Chart")


def test_labels_never_overlap_and_stay_clear_of_marks(chart):
    name, table, sem, _ = chart
    pages, _ = tactile.layout(table, sem, sem.title)
    for p in pages:
        labs = p.labels
        for i, a in enumerate(labs):
            x0, y0, x1, y1 = a.geom.bounds
            assert tactile.X0 - 1e-9 <= x0 and x1 <= tactile.X1 + 1e-9
            assert tactile.Y0 - 1e-9 <= y0 and y1 <= tactile.Y1 + 1e-9
            for m in p.marks:  # TG 3.4.3.12: 1/8 in between a label and any other component
                assert a.geom.distance(m) >= R.CLEAR - 1e-9, (name, a.text, a.kind)
            for b in labs[i + 1:]:
                assert a.geom.distance(b.geom) >= R.CLEAR - 1e-9, (name, a.text, b.text)


def test_graphic_rules(chart):
    name, table, sem, out = chart
    pages, key = tactile.layout(table, sem, sem.title)
    graphic = [p for p in pages if any(lab.kind in ("ytick", "xtick") for lab in p.labels)]
    for p in graphic:
        for kind in ("xtick", "ytick"):
            assert sum(lab.kind == kind for lab in p.labels) <= R.MAX_TICK_LABELS
        svg = "".join(p.els)
        assert svg.count("<polyline") <= R.MAX_LINES_PER_PAGE
        widths = [float(w) for w, h in re.findall(r'<rect [^>]*width="([\d.]+)" height="([\d.]+)"', svg)]
        heights = [float(h) for w, h in re.findall(r'<rect [^>]*width="([\d.]+)" height="([\d.]+)"', svg)]
        thick = heights if name == "hbar" else widths
        assert all(R.BAR_MIN - 1e-6 <= t <= R.BAR_MAX + 1e-6 for t in thick)  # TG 6.6.6.7
    if name == "line3":  # one texture per series, every series labelled on the graphic
        assert [k.split(":")[0] for k in key[:3]] == ["solid line", "dashed line", "dotted line"]
        placed = {lab.text for p in pages for lab in p.labels if lab.kind.startswith("series")}
        assert placed == set(table.series)
    if name == "grouped_bar":  # texture key before the letter keys (TG 5.7.1)
        assert key[0].startswith("plain bar") and key[1].startswith("diagonal lines")
        assert all(len(k.split(":")[0]) == 2 for k in key[2:])
    assert out["pages"] == len(pages) and out["key"] == key


def test_svg_parses_and_pdf_has_the_pages(chart, tmp_path):
    name, _, _, out = chart
    root = ET.fromstring(out["svg"])
    assert root.tag == "{http://www.w3.org/2000/svg}svg" and root.get("width") == f"{R.PAGE_W}in"
    preview = ET.fromstring(out["preview_svg"])
    assert preview.find(".//{http://www.w3.org/2000/svg}feDiffuseLighting") is not None
    assert out["pdf"][:5] == b"%PDF-"
    doc = pdfium.PdfDocument(out["pdf"])
    assert len(doc) == out["pages"]
    assert doc[0].get_size() == pytest.approx((R.PAGE_W * 72, R.PAGE_H * 72), abs=1)
    png = cairosvg.svg2png(bytestring=out["svg"].encode(), output_width=400)
    assert Image.open(io.BytesIO(png)).convert("L").getextrema()[0] == 0  # something black was drawn


def test_too_many_series_split_into_parts():
    x = list(range(2000, 2010))
    table = Table(x, {f"Region {c}": [float(i * k + k) for i in range(10)] for k, c in enumerate("ABCDE", 1)})
    sem = Semantics("line", title="Sales", x_ticks=[str(v) for v in x[::2]], y_ticks=["0", "20", "40"],
                    series=[Series(n) for n in table.series])
    pages, key = tactile.layout(table, sem, "Sales")
    graphic = [p for p in pages if p.labels[0].text.startswith("Sales (part")]
    assert len(graphic) == 2 and len(pages) == 3  # key page + 3 lines + 2 lines
    assert sum("(part 2)" in k for k in key) == 2
    bars = Semantics("grouped_bar", categories=["a"], series=[Series(f"s{i}") for i in range(6)])
    pages, key = tactile.layout(Table(["Q1", "Q2"], {f"s{i}": [1.0, 2.0] for i in range(6)}), bars, "Bars")
    assert sum(p.labels[0].text.startswith("Bars (part") for p in pages) == 2  # 4 + 2 bar series


def test_key_letters_follow_bana():
    k = tactile.Keyer()
    keys = [k.letters(n) for n in ["Chile", "Canada", "Cameroon", "Ecuador", "Abba", "2019"]]
    assert len(set(keys)) == len(keys) and k.letters("Chile") == keys[0]  # unique and stable
    for key in keys:
        assert len(key) == 2 and key not in R.KEY_AVOID
        assert any({"3", "6"} & set(d) for d in braille.ascii_to_dots(key))  # TG 5.8.1.2
    assert keys[0][0] == "c" and keys[3][0] == "e"  # first letter of the item (TG 5.8.1.1)


def test_rule_constants_are_consistent():
    assert all(s["w"] > R.AXIS_W > R.TICK_W for s in R.LINE_STYLES)  # plotted lines strongest (TG 6.6.2.2)
    assert 0.25 <= R.LINE_STYLES[1]["dash"][0] <= 0.375  # TG 3.4.3.3
    assert tactile.BOX_W == pytest.approx(9.8) and tactile.BOX_H == pytest.approx(10.0)
    assert tactile.X0 > 0 and tactile.Y0 > 0


def test_unsupported_chart_type_is_refused():
    with pytest.raises(ValueError):
        tactile.layout(Table([1], {"a": [1.0]}), Semantics("unsupported"), "x")


def test_brf_pages_lines_and_rows():
    table, sem = _synth("line", 3)
    table.series[next(iter(table.series))][1] = None
    desc = " ".join(["The three lines rise and fall together over the period, with one clear peak."] * 12)
    out = brf.to_brf(sem.title, ["solid line: A", "dashed line: B", "ab: Abba"], desc + "\n\n" + desc, table)
    assert set(out) <= set(braille.ASCII_DOTS) | set(map(chr, range(0x60, 0x7F))) | {"\r", "\n", "\f"}
    pages = out.split("\f")
    assert pages[-1] == "" and len(pages) >= 3
    for page in pages[:-1]:
        lines = page.split("\r\n")
        assert lines[-1] == "" and len(lines) - 1 <= R.LINES_PER_PAGE
        assert all(len(line) <= R.CELLS_PER_LINE for line in lines)
    assert R.BLANK_ENTRY in out and R.TN_OPEN + braille.translate("Key:") in out
    rows = [line for line in out.replace("\f", "").split("\r\n") if re.match(r"#[a-j]+3 ", line)]
    assert len(rows) == len(table.x)  # every row is one line here, none split across pages
    assert brf._num(2752.4234) == "2,752" and brf._num(0.0123456) == "0.01235" and brf._num(12.0) == "12"
