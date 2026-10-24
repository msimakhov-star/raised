"""UEB braille through the liblouis CLI, Braille ASCII to dot patterns, and braille drawn as dots for swell paper.

    translate("temperature") -> "temp}ature"     (lou_translate en-ueb-g2.ctb; lower-case Braille ASCII out)
    ascii_to_dots("#bj") -> ["3456", "12", "245"]
    svg_cells(ascii, x, y) -> '<circle .../>...'  (x, y = top-left of the dots' ink box, inches by default)

Numbers: UEB writes digits with the numeric indicator (dots 3456) and letters a-j, and grade 1 and grade 2 agree on
them (liblouis 3.39: "2020" -> "#bjbj" in both en-ueb-g1.ctb and en-ueb-g2.ctb), so axis numbers go through the same
call as words. BANA TG 6.6.4.1 / 6.6.6.2 ask for the numeric indicator on line and bar graph values.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
import unicodedata
from typing import NamedTuple

from . import tactile_rules as R

TABLES = {1: "en-ueb-g1.ctb", 2: "en-ueb-g2.ctb"}
LOU = shutil.which("lou_translate") or "/opt/homebrew/bin/lou_translate"

# North American Braille ASCII: characters 0x20..0x5F in code order, each with its dots (1-6).
# Source: https://en.wikipedia.org/wiki/Braille_ASCII (fetched 2026-10-24), table of the 64 characters; the same page
# says lower-case letters read as their upper-case equivalents, which liblouis emits (0x60..0x7E fold onto 0x40..0x5E).
# Cross-checked cell by cell against liblouis' unicode.dis output in tests/test_braille.py.
_DOTS = ("", "2346", "5", "3456", "1246", "146", "12346", "3", "12356", "23456", "16", "346", "6", "36", "46", "34",
         "356", "2", "23", "25", "256", "26", "235", "2356", "236", "35", "156", "56", "126", "123456", "345", "1456",
         "4", "1", "12", "14", "145", "15", "124", "1245", "125", "24", "245", "13", "123", "134", "1345", "135",
         "1234", "12345", "1235", "234", "2345", "136", "1236", "2456", "1346", "13456", "1356", "246", "1256", "12456",
         "45", "456")
ASCII_DOTS = {chr(0x20 + i): d for i, d in enumerate(_DOTS)}
_FOLD = {c: c - 32 for c in range(0x60, 0x7F)}
_ESCAPE = re.compile(r"'?\\x[0-9a-fA-F]{4}'?")  # liblouis writes characters its tables lack as '\x2082'


class Dims(NamedTuple):
    dot: float      # printed dot diameter
    spacing: float  # dot to dot inside a cell
    cell: float     # cell to cell
    line: float     # line to line


DIMS = Dims(R.DOT_DIAMETER * R.SWELL_DOT_SCALE, R.DOT_SPACING, R.CELL_SPACING, R.LINE_SPACING)

_cache: dict[tuple[str, int], str] = {}


def _lou(lines: list[str], grade: int) -> list[str]:
    out = subprocess.run([LOU, TABLES[grade]], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True, timeout=30).stdout
    return out.split("\n")[:len(lines)]


def translate_many(texts: list[str], grade: int = 2) -> list[str]:
    """One lou_translate call for many strings (each ~7 ms as separate processes)."""
    texts = [" ".join(str(t).split()) for t in texts]
    todo = sorted({t for t in texts if t and (t, grade) not in _cache})
    if todo:
        for t, b in zip(todo, _lou(todo, grade)):
            if "\\x" in b:  # subscripts, IPA...: NFKC folds "CO₂" to "CO2"; whatever liblouis still lacks is dropped
                b = _ESCAPE.sub("", _lou([unicodedata.normalize("NFKC", t)], grade)[0])
            if bad := set(b.translate(_FOLD)) - ASCII_DOTS.keys():
                raise ValueError(f"lou_translate gave non-braille characters {bad!r} for {t!r}")
            _cache[t, grade] = b
    return [_cache[t, grade] if t else "" for t in texts]


def translate(text: str, grade: int = 2) -> str:
    """UEB contracted (grade 2) or uncontracted (grade 1) braille as Braille ASCII."""
    return translate_many([text], grade)[0]


def cell_count(text: str, grade: int = 2) -> int:
    return len(translate(text, grade))


def ascii_to_dots(ascii: str) -> list[str]:
    """Dot numbers per cell: '#b' -> ['3456', '12']; space -> ''."""
    return [ASCII_DOTS[c] for c in ascii.translate(_FOLD)]


def ink_size(cells: int, dims: Dims = DIMS) -> tuple[float, float]:
    """Width and height of the dots' ink box for a run of cells (full cells, as if every dot were raised)."""
    return (max(cells, 1) - 1) * dims.cell + dims.spacing + dims.dot, 2 * dims.spacing + dims.dot


def svg_cells(ascii: str, x: float, y: float, dims: Dims = DIMS) -> str:
    """Each raised dot as an SVG circle. (x, y) is the top-left of the ink box of the full cell grid."""
    r = dims.dot / 2
    out = []
    for i, dots in enumerate(ascii_to_dots(ascii)):
        for d in dots:
            col, row = divmod(int(d) - 1, 3)
            out.append(f'<circle cx="{x + r + i * dims.cell + col * dims.spacing:.4f}" '
                       f'cy="{y + r + row * dims.spacing:.4f}" r="{r:.4f}"/>')
    return "".join(out)


def wrap(cells: str, width: int = R.CELLS_PER_LINE, first: int = 1, runover: int = 1) -> list[str]:
    """Word-wrap Braille ASCII to `width` cells; first/runover are 1-based start cells (BANA margins such as 3-1).
    Braille ASCII '-' is a hyphen or minus cell, never a break point."""
    return textwrap.wrap(cells, width, initial_indent=" " * (first - 1), subsequent_indent=" " * (runover - 1),
                         break_on_hyphens=False)
