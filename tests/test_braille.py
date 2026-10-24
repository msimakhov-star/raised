"""raised.braille against liblouis itself (expected strings were produced by lou_translate 3.39, en-ueb-g1/g2)."""
import os
import subprocess
import xml.etree.ElementTree as ET

import pytest

from raised import braille as B
from raised import tactile_rules as R

pytestmark = pytest.mark.skipif(not os.path.exists(B.LOU), reason="liblouis lou_translate not installed")

WORDS = ["temperature", "GDP per capita (US$)", "1,200.5%", "The quick brown fox jumps over the lazy dog!",
         "x² + y = 3; 'quoted' [brackets] {braces} <angle> #hash @at", "Emissions per 1,000 people, 2019?",
         "Revenue ($bn) & costs: -5 to 10", "Key: CO2 / H2O \\ ~ | `"]


def test_known_translations():
    assert B.translate("temperature") == "temp}ature"     # 'er' contraction
    assert B.translate("the") == "!"                       # whole-word 'the'
    assert B.translate("2020") == "#bjbj"                  # numeric indicator, digits a-j
    assert B.translate("12.5") == "#ab4e"                  # decimal point dots 256
    assert B.translate("1,200") == "#a1bjj"                # comma inside a number: dot 2
    assert B.translate("Temperature (°C)") == ',temp}ature "<~j,c">'
    assert B.translate("2020", grade=1) == B.translate("2020", grade=2)  # numbers agree across grades
    assert B.translate("cat;")[-1] == R.SEMICOLON


def test_batch_matches_single_and_drops_unknown_characters():
    many = B.translate_many(WORDS)
    assert many == [B.translate(w) for w in WORDS]
    assert "\\x" not in B.translate("CO₂ level") and B.translate("CO₂ level") == B.translate("CO2 level")
    assert B.translate("") == "" and B.cell_count("temperature") == 10


def _unicode_dots(text: str) -> list[str]:
    out = subprocess.run([B.LOU, "unicode.dis," + B.TABLES[2]], input=text + "\n", capture_output=True, text=True,
                         check=True).stdout.rstrip("\n")
    return ["".join(str(d + 1) for d in range(6) if (ord(ch) - 0x2800) >> d & 1) for ch in out]


def test_ascii_table_matches_liblouis_unicode_cells():
    seen = set()
    for w in WORDS:
        ascii = B.translate(w)
        assert B.ascii_to_dots(ascii) == _unicode_dots(w), w
        seen |= set(ascii.upper())
    assert len(seen) >= 35  # the words exercise most of the 64-cell table


def test_ascii_table_is_a_bijection_onto_the_64_cells():
    assert len(B.ASCII_DOTS) == 64 and len({frozenset(d) for d in B.ASCII_DOTS.values()}) == 64
    assert B.ascii_to_dots("a{") == B.ascii_to_dots("A[")  # lower case folds onto upper case
    # UEB 3.27 transcriber's note indicators, read as Unicode braille from the ICEB rulebook: ⠈⠨⠣ and ⠈⠨⠜
    assert B.ascii_to_dots(R.TN_OPEN) == ["4", "46", "126"] and B.ascii_to_dots(R.TN_CLOSE) == ["4", "46", "345"]
    assert B.ascii_to_dots(R.GUIDE_DOT) == ["5"]


def test_svg_cells_draws_one_circle_per_dot():
    ascii = B.translate("year 2020")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg">{B.svg_cells(ascii, 1.0, 2.0)}</svg>'
    circles = ET.fromstring(svg).findall("{http://www.w3.org/2000/svg}circle")
    assert len(circles) == sum(map(len, B.ascii_to_dots(ascii)))
    xs = sorted({float(c.get("cx")) for c in circles})
    ys = sorted({float(c.get("cy")) for c in circles})
    r = B.DIMS.dot / 2
    assert xs[0] == pytest.approx(1.0 + r) and ys[0] == pytest.approx(2.0 + r)
    assert ys[-1] - ys[0] == pytest.approx(2 * R.DOT_SPACING)  # rows 0.092 in apart
    w, h = B.ink_size(len(ascii))
    assert xs[-1] + r <= 1.0 + w + 1e-9 and h == pytest.approx(2 * R.DOT_SPACING + B.DIMS.dot)


def test_wrap_respects_width_and_margins():
    text = B.translate("Gross domestic product per capita rose in every country between 1990 and 2020 " * 3)
    lines = B.wrap(text, 40, 3, 1)
    assert all(len(line) <= 40 for line in lines) and lines[0].startswith("  ") and not lines[1].startswith(" ")
    assert " ".join(line.strip() for line in lines) == text
