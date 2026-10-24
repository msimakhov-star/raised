"""BRF: the figure's title, key, long description and data table as UEB braille text pages for an embosser.

    to_brf(title, key, description, table) -> str

Pages are 40 cells x 25 lines (BANA Braille Formats 2016 1.7.1, constants in tactile_rules.py), lines end CR LF and
every page ends with a form feed: "BRF files contain plain Braille ASCII plus spaces, Carriage Return, Line Feed, and
Form Feed" (https://en.wikipedia.org/wiki/Braille_ASCII, fetched 2026-10-24). No braille page numbers yet.
"""
from __future__ import annotations

import math

from . import tactile_rules as R
from .braille import translate, translate_many, wrap
from .facts import fx
from .model import Table

W, H = R.CELLS_PER_LINE, R.LINES_PER_PAGE


def _num(v: float) -> str:
    """Four significant digits, thousands separators: 2752.42 -> '2,752', 0.012345 -> '0.01235'."""
    if v == 0:
        return "0"
    dec = max(0, 3 - math.floor(math.log10(abs(v))))
    s = f"{v:,.{dec}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _cell(s) -> str:
    """A table entry: the linear format may not carry colons or semicolons (FMT 11.17.1), so they become commas."""
    return fx(s).replace(":", ",").replace(";", ",")


def _rows(table: Table, x_title: str) -> list[list[str]]:
    """FMT 11.17 linear table: a transcriber's note with the column order ('first: second; third'), then one row per
    print row, 'x: v1; v2', 1-3 margins, three dot 5s for a value that could not be read."""
    names = list(table.series)
    head = x_title or ("Category" if any(isinstance(x, str) for x in table.x) else "x")
    note = ("Table in linear format: columns in the order below, separated by a colon then semicolons. "
            "Three dot 5s mark a value that could not be read.")
    order = f"{_cell(head)}: " + "; ".join(_cell(n) for n in names)
    t_note, t_order = translate_many([note, order])
    out = [wrap(R.TN_OPEN + t_note, W, R.KEY_HEAD_CELL, R.TN_RUNOVER) + [""],  # FMT 11.17.1 b; TN margins 7-5
           wrap(t_order + R.TN_CLOSE, W, R.TABLE_INDENT, R.TABLE_RUNOVER) + [""]]   # FMT 11.17.1 c, d, e
    for i, x in enumerate(table.x):
        vals = [table.series[n][i] for n in names]
        last = len(vals) - 1
        texts = [f"{_cell(x)}:"] + [f"{_num(v)}{';' * (j < last)}" for j, v in enumerate(vals) if v is not None]
        cells = iter(translate_many(texts))
        parts = [next(cells)] + [next(cells) if v is not None else R.BLANK_ENTRY + R.SEMICOLON * (j < last)
                                 for j, v in enumerate(vals)]
        out.append(wrap(" ".join(parts), W, R.TABLE_INDENT, R.TABLE_RUNOVER))  # one block: never split (FMT 11.17.1 g)
    return out


def to_brf(title: str, key: list[str], description: str, table: Table, x_title: str = "") -> str:
    centred = [" " * ((W - len(line)) // 2) + line for line in wrap(translate(title or "Graph"))]
    blocks = [centred + [""]]  # centered heading and a blank line (FMT, TG 5.3.1)
    if key:  # TG 5.8.4: transcriber's note symbol and "Key:" in cell 7, a blank line, entries, closing symbol
        blocks.append([" " * (R.KEY_HEAD_CELL - 1) + R.TN_OPEN + translate("Key:"), ""])
        tk = translate_many(key)
        tk[-1] += R.TN_CLOSE
        blocks += [wrap(t, W, 1, R.KEY_LETTER_RUNOVER) for t in tk]
        blocks.append([""])
    for para in filter(None, (p.strip() for p in description.split("\n"))):
        blocks.append(wrap(translate(para), W, R.PARA_INDENT, R.PARA_RUNOVER))  # FMT 1.9.2 indented paragraphs 3-1
    if table.x:
        blocks.append([""])
        blocks += _rows(table, x_title)

    pages, cur = [], []
    for block in blocks:
        if cur and len(cur) + len(block) > H and len(block) <= H:  # keep a block on one page when it can be
            pages.append(cur)
            cur = []
        for line in block:
            if len(cur) == H:
                pages.append(cur)
                cur = []
            if cur or line:  # no blank line at the top of a page
                cur.append(line)
    pages.append(cur)
    return "".join("\r\n".join(p) + "\r\n\f" for p in pages if p)
