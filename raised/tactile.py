"""Tactile graphic compiler: redraws a chart FROM ITS TABLE (never traces the image) for swell paper, with UEB braille.

    compile(table, sem, title) -> {"svg", "preview_svg", "pdf", "key", "pages"}
    layout(table, sem, title) -> (pages, key)    # geometry only, for tests and other renderers

Rules follow BANA "Guidelines and Standards for Tactile Graphics, 2010" (TG); every constant and its section is in
tactile_rules.py. Output order: key page(s) first, then the graphic page(s) (TG 5.8.3.2 facing-page key). No gridlines,
at most 6 labelled ticks per axis, at most 3 lines or 4 bar series per graphic page (more are split into parts, TG 3.9),
braille labels at least 1/8 in from every other component (TG 3.4.3.12), key letters when a label does not fit (TG 5.8).
Coordinates are inches, origin at the page's top-left. A draft for transcriber review, not a finished transcription.
"""
from __future__ import annotations

import io
import math
import zlib
from dataclasses import dataclass, field
from string import ascii_lowercase
from xml.sax.saxutils import escape

import cairosvg
import pypdfium2 as pdfium
from matplotlib.ticker import LogLocator, MaxNLocator
from shapely.geometry import LineString, Polygon, box
from shapely.geometry import Point as Pt
from shapely.ops import unary_union

from . import tactile_rules as R
from .braille import ASCII_DOTS, DIMS, ink_size, svg_cells, translate, translate_many, wrap
from .facts import format_like, fx, value_scale, value_ticks
from .model import Semantics, Table, parse_number

CELL, LINE = DIMS.cell, DIMS.line
INK_H = ink_size(1)[1]
INK_W1 = ink_size(1)[0]
MID = DIMS.spacing + DIMS.dot / 2  # ink edge to the centre of dots 2-5 (rows) or 4-6 (columns)
BOX_W, BOX_H = R.CELLS_PER_LINE * CELL, R.LINES_PER_PAGE * LINE
X0, Y0 = (R.PAGE_W - BOX_W) / 2, (R.PAGE_H - BOX_H) / 2
X1, Y1 = X0 + BOX_W, Y0 + BOX_H
PAD = R.TICK_HALF + R.CLEAR  # keeps plotted marks clear of the other axis' ticks
BARS = ("bar", "grouped_bar", "hbar")
DOT36 = {c for c in ascii_lowercase if {"3", "6"} & set(ASCII_DOTS[c.upper()])}  # TG 5.8.1.2
TEXTURE_NAMES = {"none": "plain bar, outline only", "diagonal": "diagonal lines", "dots": "dots", "grid": "grid"}
_DIRS = ((1, 0), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1))  # E NE SE N S W NW SW
_KIND_ORDER = {"area": 0, "line": 1, "point": 2, "letters": 3}  # TG 5.7.1 key order


@dataclass
class Label:
    text: str   # print text
    cells: str  # Braille ASCII
    x: float    # top-left of the ink box
    y: float
    kind: str

    @property
    def geom(self):
        w, h = ink_size(len(self.cells))
        return box(self.x, self.y, self.x + w, self.y + h)


@dataclass
class Page:
    els: list[str] = field(default_factory=list)   # SVG drawing elements, in paint order
    holes: list[tuple] = field(default_factory=list)  # (cx, cy, r) blank rings around point symbols
    syms: list[str] = field(default_factory=list)   # point symbols, painted last
    marks: list = field(default_factory=list)      # shapely geometry of every inked component except labels
    labels: list[Label] = field(default_factory=list)

    def text(self, text: str, cells: str, x: float, y: float, kind: str) -> Label:
        lab = Label(text, cells, x, y, kind)
        self.labels.append(lab)
        self.els.append(f"<g><title>{escape(text)}</title>{svg_cells(cells, x, y)}</g>")
        return lab

    def seg(self, x0, y0, x1, y1, w) -> None:
        self.els.append(f'<line x1="{x0:.4f}" y1="{y0:.4f}" x2="{x1:.4f}" y2="{y1:.4f}" stroke="#000" '
                        f'stroke-width="{w:.4f}"/>')
        self.marks.append(LineString([(x0, y0), (x1, y1)]).buffer(w / 2, cap_style="flat"))

    def polyline(self, pts, style) -> object:
        dash, w = style["dash"], style["w"]
        attrs = f'stroke-dasharray="{dash[0]:.3f} {dash[1]:.3f}" ' if dash else ""
        cap = "round" if dash and dash[0] == 0 else "butt"
        self.els.append(f'<polyline points="{" ".join(f"{x:.4f},{y:.4f}" for x, y in pts)}" fill="none" stroke="#000" '
                        f'stroke-width="{w:.4f}" {attrs}stroke-linecap="{cap}" stroke-linejoin="round"/>')
        g = LineString(pts).buffer(w / 2)
        self.marks.append(g)
        return g

    def symbol(self, kind: str, cx: float, cy: float, size: float, halo: bool) -> object:
        """A point symbol; with `halo`, a blank ring of TG 6.6.2.2's 1/8 in cut out of the lines under it."""
        if halo:
            self.holes.append((cx, cy, size / 2 + R.POINT_CLEAR))
        s = size / 2
        if kind == "circle":
            g = Pt(cx, cy).buffer(s)
            self.syms.append(f'<circle cx="{cx:.4f}" cy="{cy:.4f}" r="{s:.4f}" fill="#000"/>')
        else:
            pts = ([(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)] if kind == "square" else
                   [(cx, cy - s), (cx + s * 1.15, cy + s), (cx - s * 1.15, cy + s)])
            g = Polygon(pts)
            self.syms.append(f'<polygon points="{" ".join(f"{x:.4f},{y:.4f}" for x, y in pts)}" fill="#000"/>')
        self.marks.append(g)
        return g

    def bar(self, x, y, w, h, texture: str) -> None:
        self.els.append(f'<rect x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" fill="none" stroke="#000" '
                        f'stroke-width="{R.BAR_OUTLINE:.4f}"/>{_texture(texture, x, y, w, h)}')
        self.marks.append(box(x, y, x + w, y + h).buffer(R.BAR_OUTLINE / 2, join_style="mitre"))


def _circle(x: float, y: float, r: float) -> str:
    """A circle as path data, so many of them fit in one <path>."""
    return f"M{x - r:.4f},{y:.4f}a{r:.4f},{r:.4f} 0 1,0 {2 * r:.4f},0a{r:.4f},{r:.4f} 0 1,0 {-2 * r:.4f},0"


def _steps(start: float, length: float, pitch: float) -> list[float]:
    n = int(length / pitch)
    return [start + (length - (n - 1) * pitch) / 2 + k * pitch for k in range(n)]


def _texture(kind: str, x: float, y: float, w: float, h: float) -> str:
    """Area texture drawn as plain geometry clipped to the bar (cairosvg rasterises <pattern> tiles at user-unit
    resolution, which loses them at inch units). Same drawing for bars and key samples (TG 6.6.6.7)."""
    p, r, area = R.TEXTURE_PITCH, R.TEXTURE_DOT / 2, box(x, y, x + w, y + h)
    if kind == "none":
        return ""
    if kind == "dots":
        d = "".join(_circle(cx, cy, r) for cx in _steps(x, w, p) for cy in _steps(y, h, p))
        return f'<path d="{d}" fill="#000"/>'
    if kind == "diagonal":  # 45 degree hatch, lines `p` apart
        s, n = p * math.sqrt(2), math.ceil((w + h) / (p * math.sqrt(2)))
        lines = [LineString([(x - h + k * s, y + h), (x + k * s, y)]) for k in range(1, n + 1)]
    else:  # grid
        g = 1.5 * p
        lines = ([LineString([(x, cy), (x + w, cy)]) for cy in _steps(y, h, g)] +
                 [LineString([(cx, y), (cx, y + h)]) for cx in _steps(x, w, g)])
    segs = [c.coords for c in (ln.intersection(area) for ln in lines) if c.geom_type == "LineString" and not c.is_empty]
    d = "".join(f"M{a[0]:.4f},{a[1]:.4f}L{b[0]:.4f},{b[1]:.4f}" for a, b in ((c[0], c[-1]) for c in segs))
    return f'<path d="{d}" fill="none" stroke="#000" stroke-width="{R.TICK_W:.4f}"/>'


@dataclass
class Axis:
    lo: float
    hi: float
    log: bool
    ticks: list[tuple[float, str]]  # (value, label as printed)

    def frac(self, v: float) -> float:
        f = math.log10 if self.log else float
        return (f(v) - f(self.lo)) / (f(self.hi) - f(self.lo))


@dataclass
class KeyEntry:
    kind: str   # area | line | point | letters
    name: str   # what it stands for (print)
    style: object  # texture name, line style index, symbol name, or the key letters
    symbol: str = ""
    part: int | str = ""  # graphic part(s) it applies to when the graph is split

    @property
    def print(self) -> str:
        part = f" (part {self.part})" if self.part else ""
        what = {"area": lambda: TEXTURE_NAMES[self.style],
                "line": lambda: R.LINE_STYLES[self.style]["name"] + (f" with {self.symbol}s" if self.symbol else ""),
                "point": lambda: self.style, "letters": lambda: self.style}[self.kind]()
        return f"{what}{part}: {self.name}"


class Keyer:
    """Key letters (TG 5.8.1): two cells, the item's first letter first, one letter with dot 3 or 6, the same item
    always gets the same key (TG 2.18, 5.8.2.6). Entries are kept in TG 5.7.1 order, each listed once."""

    def __init__(self):
        self.keys: dict[str, str] = {}
        self.entries: list[KeyEntry] = []

    def letters(self, name: str) -> str:
        if name not in self.keys:
            ls = [c for c in name.lower() if c in ascii_lowercase]
            first = ls[0] if ls else "k"
            used = set(self.keys.values())
            self.keys[name] = next(first + c for c in ls[1:] + list(ascii_lowercase)
                                   if (first in DOT36 or c in DOT36) and first + c not in R.KEY_AVOID | used)
        return self.keys[name]

    def add(self, e: KeyEntry) -> None:
        if all(e.print != o.print for o in self.entries):
            self.entries.append(e)

    def sorted(self) -> list[KeyEntry]:
        return sorted(self.entries, key=lambda e: _KIND_ORDER[e.kind])


def _axis(labels: list[str], values: list, log: bool, zero: bool, fmt) -> Axis:
    """Axis over the data, ending on the nearest tick outside it. Print tick labels are kept as printed (TG 6.6.4.4
    'If tick marks are shown in print, they should be shown in braille'); fewer than 3 usable: round values."""
    ok = lambda v: v is not None and math.isfinite(v) and (v > 0 or not log)  # noqa: E731
    vals = [float(v) for v in values if ok(v)] or [1.0]
    lo, hi = min(vals), max(vals)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)

    def enclose(ticks):
        below, above = [v for v, _ in ticks if v <= lo], [v for v, _ in ticks if v >= hi]
        a, b = (max(below) if below else -math.inf), (min(above) if above else math.inf)
        return [(v, s) for v, s in ticks if a <= v <= b]

    ticks = enclose(sorted({(v, s) for s in labels if ok(v := parse_number(s))}))
    if len(ticks) < 3:
        if hi <= lo:
            lo, hi = (lo / 10, hi * 10) if log else (lo - 1, hi + 1)
        loc = LogLocator() if log else MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        ticks = enclose([(float(v), fmt(float(v))) for v in loc.tick_values(lo, hi) if ok(v)])
    lo, hi = min([lo] + [v for v, _ in ticks]), max([hi] + [v for v, _ in ticks])
    if hi <= lo:
        hi = lo * 10 if log else lo + 1
    return Axis(lo, hi, log, ticks)


def _thin(spans: list[tuple[float, float]], values: list[float]) -> list[int]:
    """Every k-th tick label, smallest k leaving at most MAX_TICK_LABELS that do not crowd (TG 6.6.1.3). Counting
    starts from the zero tick when there is one, so the kept labels stay round."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    s = next((j for j, i in enumerate(order) if values[i] == 0), 0)
    for k in range(1, len(order) + 1):
        pick = [order[j] for j in range(len(order)) if (j - s) % k == 0]
        if len(pick) <= R.MAX_TICK_LABELS and all(spans[b][0] - spans[a][1] >= R.CLEAR for a, b in zip(pick, pick[1:])):
            return pick
    return order[:1]


def _ticks(page: Page, ax: Axis, pos, at: float, vertical: bool) -> None:
    """Tick marks straddling the axis line and braille values: on a vertical axis right-aligned with dots 25 on the
    tick, on a horizontal axis dots 456 of the first cell on the tick, 1/8 in from the tick (TG 6.6.4.5)."""
    cells = translate_many([s for _, s in ax.ticks])
    p = [pos(v) for v, _ in ax.ticks]
    spans = [(q - MID, q - MID + (INK_H if vertical else ink_size(len(c))[0])) for q, c in zip(p, cells)]
    keep = _thin(spans, [v for v, _ in ax.ticks])
    q = sorted(p)
    marks = range(len(p)) if all(b - a >= R.TICK_MARK_MIN_SPACING for a, b in zip(q, q[1:])) else keep
    for i in marks:
        if vertical:
            page.seg(at - R.TICK_HALF, p[i], at + R.TICK_HALF, p[i], R.TICK_W)
        else:
            page.seg(p[i], at - R.TICK_HALF, p[i], at + R.TICK_HALF, R.TICK_W)
    for i in keep:
        if vertical:
            x = at - R.TICK_HALF - R.CLEAR - ink_size(len(cells[i]))[0]
            page.text(ax.ticks[i][1], cells[i], x, spans[i][0], "ytick")
        else:
            page.text(ax.ticks[i][1], cells[i], spans[i][0], at + R.TICK_HALF + R.CLEAR, "xtick")


def _heading(page: Page, title: str) -> float:
    """Centered heading, then a blank line (TG 5.3.1). Returns the next free line's top."""
    y = Y0
    for line in wrap(translate(title)):
        page.text(title, line, X0 + (R.CELLS_PER_LINE - len(line)) // 2 * CELL, y, "title")
        y += LINE
    return y + LINE


def _block(page: Page, text: str, x: float, y: float, cells: int, kind: str) -> float:
    """A label wrapped at `cells`, runovers left-aligned (TG 5.10.7, 6.6.4.5). Returns the next line's top."""
    for line in wrap(translate(text), max(cells, 1)) if text.strip() else []:
        page.text(text, line, x, y, kind)
        y += LINE
    return y


def _fits(n: int) -> int:
    return max(1, int((X1 - X0 - INK_W1) / CELL) + 1 - n)


def _place_series_label(page: Page, name: str, own, others, anchors, keyer: Keyer, rsym: float) -> Label | None:
    """Greedy 8-position placement: the first box (anchors in order x 2 offsets x E NE SE N S W NW SW) that is within
    1/4 in of its own line (TG 5.10.5), 1/8 in clear of every mark and label (TG 3.4.3.12), farther from every other
    series than that, and inside the 40 x 25 cell box. The full name first, then its key letters (TG 5.8)."""
    blocked = unary_union(page.marks + [lab.geom for lab in page.labels])
    other = unary_union(others) if others else None
    for keyed in (False, True):
        cells = keyer.letters(name) if keyed else translate(name)
        w, h = ink_size(len(cells))
        for ax, ay in anchors:
            for g in (rsym + R.CLEAR + 0.03, rsym + R.LABEL_GAP_MAX - 0.03):
                for dx, dy in _DIRS:
                    gx = gy = g * (0.7 if dx and dy else 1)
                    x = ax + gx if dx > 0 else ax - gx - w if dx < 0 else ax - w / 2
                    y = ay + gy if dy > 0 else ay - gy - h if dy < 0 else ay - h / 2
                    if not (X0 <= x and x + w <= X1 and Y0 <= y and y + h <= Y1):
                        continue
                    b = box(x, y, x + w, y + h)
                    if (b.distance(own) <= R.LABEL_GAP_MAX + 1e-9 and b.distance(blocked) >= R.CLEAR - 1e-9
                            and (other is None or b.distance(other) >= R.LABEL_GAP_MAX + R.CLEAR)):
                        return page.text(name, cells, x, y, "series-key" if keyed else "series")
    return None


def _xy_page(table: Table, sem: Semantics, title: str, names: list[str], xi: list[int], va: Axis, xa: Axis | None,
             keyer: Keyer, part: int) -> Page | None:
    """Line, scatter and vertical bar graphs. names: series on this page, xi: table rows on this page, xa: numeric
    x axis or None for categories. Returns None when the bars would be narrower than TG 6.6.6.7 allows."""
    kind, page = sem.chart_type, Page()
    bars = kind in BARS
    y = _heading(page, title)
    vcells = translate_many([s for _, s in va.ticks])
    axis_x = X0 + max(ink_size(len(c))[0] for c in vcells) + R.CLEAR + R.TICK_HALF
    y = _block(page, sem.y_title, X0, y, R.AXIS_TITLE_MAX_CELLS, "axis-title")  # aligned with the values (TG 6.6.4.5)
    top = y + MID
    left = axis_x + PAD
    xt = wrap(translate(sem.x_title), _fits(round((left - MID - X0) / CELL))) if sem.x_title.strip() else []
    row = Y1 - LINE * (1 + len(xt))  # top of the value/category label row under the x axis
    axis_y = row - R.CLEAR - R.TICK_HALF
    if xa is None:
        right = X1
    else:
        xcells = translate_many([s for _, s in xa.ticks])
        right = X1 - max(ink_size(len(c))[0] for c in xcells) + MID
        if kind == "line" and len(table.series) > 1:
            right = min(right, X1 - R.SERIES_LABEL_RESERVE)
    if xa is None:
        slot = (right - left) / len(xi)
        centers = {i: left + (j + 0.5) * slot for j, i in enumerate(xi)}
        X = centers.__getitem__
    else:
        X = lambda i: left + xa.frac(_num(table.x[i])) * (right - left)  # noqa: E731
    base = axis_y - (0 if bars else PAD)
    Y = lambda v: base - va.frac(v) * (base - top)  # noqa: E731

    if bars:
        group = min(slot - R.BAR_GAP, len(names) * R.BAR_MAX)
        bw = group / len(names)
        if bw < R.BAR_MIN - 1e-9 or slot < ink_size(2)[0] + R.CLEAR:
            return None
        b0 = Y(0.0 if not va.log and va.lo <= 0 <= va.hi else va.lo)
        for i in xi:
            for k, name in enumerate(names):
                v = table.series[name][i]
                if v is not None and (v > 0 or not va.log):
                    page.bar(X(i) - group / 2 + k * bw, min(Y(v), b0), bw, abs(b0 - Y(v)), R.BAR_TEXTURES[k])
    page.seg(axis_x, axis_y + R.AXIS_W / 2, axis_x, top, R.AXIS_W)
    page.seg(axis_x, axis_y, right, axis_y, R.AXIS_W)
    _ticks(page, va, Y, axis_x, vertical=True)
    if xa is None:  # category labels centred under their bar set or point (TG 6.6.6.6); keyed when too wide
        for i, c in zip(xi, translate_many([str(table.x[i]) for i in xi])):
            name = str(table.x[i])
            if ink_size(len(c))[0] > slot - R.CLEAR:
                c = keyer.letters(name)
                keyer.add(KeyEntry("letters", name, c))
            page.text(name, c, X(i) - ink_size(len(c))[0] / 2, row, "category")
    else:
        _ticks(page, xa, lambda v: left + xa.frac(v) * (right - left), axis_y, vertical=False)
    first = min((lab.x for lab in page.labels if lab.kind in ("xtick", "category")), default=left - MID)
    if first + ink_size(max(map(len, xt), default=0))[0] > X1:
        first = left - MID
    for k, line in enumerate(xt):  # left-justified with the first horizontal value (TG 6.6.4.5)
        page.text(sem.x_title, line, first, row + LINE * (k + 1), "axis-title")

    if kind in ("line", "scatter"):
        _plot_series(page, table, sem, names, xi, X, Y, keyer, part)
    return page


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) else parse_number(x)


def _plot_series(page, table, sem, names, xi, X, Y, keyer, part) -> None:
    """Plotted lines (solid / dashed / dotted, TG-F) or scatter points, point symbols with a 1/8 in blank ring when the
    points are far enough apart (TG 3.4.3.11, 6.6.2.2), then direct braille labels for multi-series graphs."""
    log = value_scale(sem) == "log"
    multi = len(table.series) > 1
    pts = {n: [(i, X(i), Y(v)) if v is not None and (v > 0 or not log) else None
               for i, v in ((i, table.series[n][i]) for i in xi)] for n in names}
    ink, sym = {}, {}
    for k, n in enumerate(names):
        p = [q for q in pts[n] if q]
        if sem.chart_type == "scatter":
            sym[n] = (R.POINT_SYMBOLS[k], R.POINT_SIZE) if multi else ("circle", R.POINT_MIN)  # TG 6.6.5.7
            ink[n] = []
            continue
        runs, cur = [], []
        for q in pts[n] + [None]:  # a missing value breaks the line
            if q:
                cur.append(q[1:])
            elif cur:
                runs.append(cur)
                cur = []
        ink[n] = [page.polyline(r, R.LINE_STYLES[k]) for r in runs if len(r) > 1]
        # symbols only where the line between two blank rings stays a readable line (TG 3.4.3.2 1/2 in minimum)
        far = all(math.dist(a[1:], b[1:]) >= R.POINT_SIZE + 2 * R.POINT_CLEAR + R.MIN_LINE for a, b in zip(p, p[1:]))
        sym[n] = (R.POINT_SYMBOLS[k], R.POINT_SIZE) if far or len(p) == 1 else None
        if multi:
            keyer.add(KeyEntry("line", n, k, sym[n][0] if sym[n] else "", part))
    for n in names:
        if sym[n]:
            for _, x, y in filter(None, pts[n]):
                ink[n].append(page.symbol(sym[n][0], x, y, sym[n][1], halo=sem.chart_type == "line"))
        if sem.chart_type == "scatter" and multi:
            keyer.add(KeyEntry("point", n, sym[n][0], part=part))
    if sem.chart_type != "line" or not multi:
        return
    for n in names:
        p = [q[1:] for q in pts[n] if q]
        if not p:
            continue
        mids = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(p, p[1:])]
        anchors = [p[-1], p[0]] + p[-2:0:-1] + mids[::-1]
        rsym = sym[n][1] / 2 if sym[n] else R.LINE_STYLES[names.index(n)]["w"] / 2
        lab = _place_series_label(page, n, unary_union(ink[n]), [g for m in names if m != n for g in ink[m]],
                                  anchors, keyer, rsym)
        if lab is not None and lab.kind == "series-key":
            keyer.add(KeyEntry("letters", n, lab.cells))


def _hbar_page(table: Table, sem: Semantics, title: str, names: list[str], xi: list[int], va: Axis,
               keyer: Keyer) -> Page | None:
    """Horizontal bars: category labels left-aligned in a column with guide dots to the axis, centred on their bar set
    (TG 6.6.6.6); values on the horizontal axis below."""
    page = Page()
    y = _heading(page, title)
    cats = [str(table.x[i]) for i in xi]
    cells = translate_many(cats)
    for j, (n, c) in enumerate(zip(cats, cells)):
        if len(c) > R.HBAR_LABEL_MAX_CELLS:  # TG 6.6.6.6 "If values are lengthy, they may be keyed"
            cells[j] = keyer.letters(n)
            keyer.add(KeyEntry("letters", n, cells[j]))
    width = max(len(c) for c in cells)
    axis_x = X0 + (width + R.MIN_GUIDE_DOTS) * CELL + INK_W1 + R.CLEAR + max(R.AXIS_W, R.BAR_OUTLINE) / 2
    top = _block(page, sem.y_title, X0, y, R.AXIS_TITLE_MAX_CELLS, "axis-title")
    xt = wrap(translate(sem.x_title), _fits(round((axis_x - MID - X0) / CELL))) if sem.x_title.strip() else []
    row = Y1 - LINE * (1 + len(xt))
    axis_y = row - R.CLEAR - R.TICK_HALF
    bottom = axis_y - R.TICK_HALF - R.CLEAR
    right = X1 - max(ink_size(len(c))[0] for c in translate_many([s for _, s in va.ticks])) + MID
    slot = (bottom - top) / len(xi)
    group = min(slot - R.BAR_GAP, len(names) * R.BAR_MAX)
    bh = group / len(names)
    if bh < R.BAR_MIN - 1e-9:
        return None
    X = lambda v: axis_x + va.frac(v) * (right - axis_x)  # noqa: E731
    b0 = X(0.0 if not va.log and va.lo <= 0 <= va.hi else va.lo)
    for j, (i, name, c) in enumerate(zip(xi, cats, cells)):
        cy = top + (j + 0.5) * slot
        guide = R.GUIDE_DOT * (width + R.MIN_GUIDE_DOTS - len(c))  # one blank cell, then >= 3 guide dots
        page.text(name, f"{c} {guide}", X0, cy - MID, "category")
        for k, n in enumerate(names):
            v = table.series[n][i]
            if v is not None and (v > 0 or not va.log):
                page.bar(min(X(v), b0), cy - group / 2 + k * bh, abs(X(v) - b0), bh, R.BAR_TEXTURES[k])
    page.seg(axis_x, top, axis_x, axis_y + R.AXIS_W / 2, R.AXIS_W)
    page.seg(axis_x, axis_y, right, axis_y, R.AXIS_W)
    _ticks(page, va, X, axis_y, vertical=False)
    for k, line in enumerate(xt):
        page.text(sem.x_title, line, axis_x - MID, row + LINE * (k + 1), "axis-title")
    return page


def _chunks(seq: list, n: int) -> list[list]:
    """Split into n nearly equal consecutive parts."""
    q, r = divmod(len(seq), n)
    out, i = [], 0
    for k in range(n):
        out.append(seq[i:i + q + (k < r)])
        i += q + (k < r)
    return [c for c in out if c]


def _key_pages(title: str, entries: list[KeyEntry]) -> list[Page]:
    """The key on its own page(s) before the graphic, after the title (TG 5.8.3.2): 'Key:' after the transcriber's note
    symbol in cell 7 and a blank line (5.8.4.1); samples in cells 1-4 with explanations from cell 6, runovers cell 8
    (5.8.4.3); key letters in cell 1, explanations from cell 4, runovers cell 3 (5.8.4.5); closing symbol (5.8.4.6)."""
    pages = [Page()]
    y = _heading(pages[0], title)
    pages[0].text("Key:", R.TN_OPEN + translate("Key:"), X0 + (R.KEY_HEAD_CELL - 1) * CELL, y, "key")
    y += 2 * LINE
    texts = translate_many([e.name for e in entries])
    for j, (e, t) in enumerate(zip(entries, texts)):
        t += R.TN_CLOSE if j == len(entries) - 1 else ""
        lines = (wrap(t, R.CELLS_PER_LINE, R.KEY_LETTER_TEXT_CELL, R.KEY_LETTER_RUNOVER) if e.kind == "letters" else
                 wrap(t, R.CELLS_PER_LINE, R.KEY_TEXT_CELL, R.KEY_TEXT_RUNOVER))
        need = max(len(lines), 2 if e.kind == "area" else 1)  # an area sample is two braille lines tall (TG 5.9.1.2)
        if y + (need - 1) * LINE + INK_H > Y1:
            pages.append(Page())
            y = Y0
        page = pages[-1]
        for k, line in enumerate(lines):
            body = line.lstrip(" ")
            page.text(e.print, body, X0 + (len(line) - len(body)) * CELL, y + k * LINE, "key")
        if e.kind == "letters":
            page.text(e.print, e.style, X0, y, "key")
        elif e.kind == "line":  # 1 in sample on the line of dots 25 (TG 5.8.4.4, 5.9.2)
            page.polyline([(X0, y + MID), (X0 + R.KEY_SAMPLE_LEN, y + MID)], R.LINE_STYLES[e.style])
            if e.symbol:
                page.symbol(e.symbol, X0 + R.KEY_SAMPLE_LEN / 2, y + MID, R.POINT_SIZE, halo=True)
        elif e.kind == "area":  # 1 x 1/2 in sample, top on dots 14 (TG 5.8.4.4, 5.9.1)
            page.bar(X0, y, *R.KEY_AREA, e.style)
        else:  # point symbol centred in cells 1-4, top on dots 14 (TG 5.9.3.1)
            page.symbol(e.style, X0 + (3 * CELL + INK_W1) / 2, y + R.POINT_SIZE / 2, R.POINT_SIZE, halo=False)
        y += need * LINE
    return pages


def layout(table: Table, sem: Semantics, title: str) -> tuple[list[Page], list[str]]:
    """Pages (key pages first) and the key as print strings, e.g. 'dashed line with squares: Germany', 'ch: Chile'."""
    kind = sem.chart_type
    if kind not in ("line", "scatter", *BARS):
        raise ValueError(f"no tactile layout for chart type {kind!r}")
    title = title or sem.title or "Graph"
    names = [n for n in table.series if any(v is not None for v in table.series[n])]
    if not names or not table.x:
        raise ValueError("empty table")
    per = R.MAX_BAR_SERIES_PER_PAGE if kind in BARS else R.MAX_LINES_PER_PAGE
    groups = _chunks(names, math.ceil(len(names) / per))
    log = value_scale(sem) == "log"
    # one value scale per group of series: parts split by series are scaled to their own lines (TG 6.6.4.6 allows two
    # parts "as long as the concept of the graph remains clear"); category splits of a bar graph share their scale
    vas = [_axis(value_ticks(sem), [v for n in g for v in table.series[n]], log, kind in BARS and not log,
                 lambda v: format_like(v, sem)) for g in groups]
    xs = [_num(x) for x in table.x]
    xa = None if kind in BARS or None in xs else _axis(sem.x_ticks, xs, sem.x_scale == "log", False, fx)
    for n_cat in range(1, len(table.x) + 1):  # split categories over more pages until bars reach TG's minimum width
        parts = [(g, va, xi) for g, va in zip(groups, vas) for xi in _chunks(list(range(len(table.x))), n_cat)]
        keyer, pages = Keyer(), []
        for p, (g, va, xi) in enumerate(parts, 1):
            t, part = (f"{title} (part {p} of {len(parts)})", p) if len(parts) > 1 else (title, 0)
            pages.append(_hbar_page(table, sem, t, g, xi, va, keyer) if kind == "hbar" else
                         _xy_page(table, sem, t, g, xi, va, xa, keyer, part))
        if all(pages):
            break
    if kind in BARS and len(names) > 1:  # bar textures key (TG 6.6.6.4), in left-to-right order
        nc = len(parts) // len(groups)  # pages per series group
        for gi, g in enumerate(groups):
            part = "" if len(groups) == 1 else str(gi + 1) if nc == 1 else f"{gi * nc + 1}-{(gi + 1) * nc}"
            for k, n in enumerate(g):
                keyer.add(KeyEntry("area", n, R.BAR_TEXTURES[k], part=part))
    entries = keyer.sorted()
    return (_key_pages(title, entries) if entries else []) + pages, [e.print for e in entries]


_PAPER, _INK = "#f6f1e6", "#5b5852"


def _defs() -> str:
    """Raised-paper look for the preview: ink alpha blurred into a height map, lit from the top left, multiplied onto
    the paper colour."""
    return (f'<defs><filter id="raised-swell" filterUnits="userSpaceOnUse" x="0" y="0" width="{R.PAGE_W}" '
            f'height="{R.PAGE_H}" color-interpolation-filters="sRGB">'
            '<feGaussianBlur in="SourceAlpha" stdDeviation="0.012" result="h"/>'
            '<feDiffuseLighting in="h" surfaceScale="3" diffuseConstant="1.25" lighting-color="#fff" result="lit">'
            '<feDistantLight azimuth="225" elevation="50"/></feDiffuseLighting>'
            f'<feFlood flood-color="{_PAPER}" result="paper"/>'
            '<feComposite in="SourceGraphic" in2="paper" operator="over" result="printed"/>'
            '<feBlend in="lit" in2="printed" mode="multiply"/></filter></defs>')


def _svg(pages: list[Page], preview: bool = False, title: str = "") -> str:
    """Pages stacked top to bottom in one SVG. The production SVG is black ink on white for swell paper (only black
    swells, TG 4.1.2); the preview adds the lighting filter and paper colours for the web UI."""
    gap = 0.3
    h = len(pages) * R.PAGE_H + (len(pages) - 1) * gap
    size = 'width="100%"' if preview else f'width="{R.PAGE_W}in" height="{h:.2f}in"'
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" {size} viewBox="0 0 {R.PAGE_W} {h:.2f}" role="img" '
            f'aria-label="{escape(title or "Tactile graphic", {chr(34): "&quot;"})}, tactile graphic '
            f'{"preview" if preview else "for swell paper"}, {len(pages)} page{"s" * (len(pages) > 1)}">')
    body = []
    for i, p in enumerate(pages):
        els, syms = "".join(p.els), "".join(p.syms)
        at = f'<g transform="translate(0 {i * (R.PAGE_H + gap):.2f})">'
        if preview:  # rings cut by an even-odd clip, not white paint, so they stay flat under the lighting filter
            cid = f"rings-{zlib.crc32(els.encode()):x}-{i}"  # unique when several previews share one DOM
            rings = "".join(_circle(x, y, r) + "Z" for x, y, r in p.holes)
            clip = (f'<clipPath id="{cid}"><path clip-rule="evenodd" d="M0,0H{R.PAGE_W}V{R.PAGE_H}H0Z{rings}"/>'
                    "</clipPath>")
            ink = f'<g clip-path="url(#{cid})">{els}</g>{syms}'.replace('"#000"', f'"{_INK}"')
            body.append(f'{at}{clip}<rect width="{R.PAGE_W}" height="{R.PAGE_H}" fill="{_PAPER}" stroke="#bbb" '
                        f'stroke-width="0.02"/><g filter="url(#raised-swell)" fill="{_INK}">{ink}</g></g>')
        else:
            holes = "".join(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{r:.4f}"/>' for x, y, r in p.holes)
            body.append(f'{at}<rect width="{R.PAGE_W}" height="{R.PAGE_H}" fill="#fff"/>'
                        f'<g fill="#000">{els}<g fill="#fff">{holes}</g>{syms}</g></g>')
    return head + (_defs() if preview else "") + "".join(body) + "</svg>"


def _pdf(pages: list[Page]) -> bytes:
    """One PDF page per graphic or key page, 11.5 x 11 in, via cairosvg; pages joined with pdfium."""
    doc = pdfium.PdfDocument.new()
    for p in pages:
        doc.import_pages(pdfium.PdfDocument(cairosvg.svg2pdf(bytestring=_svg([p]).encode())))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def compile(table: Table, sem: Semantics, title: str) -> dict:  # noqa: A001 (name fixed by the pipeline contract)
    pages, key = layout(table, sem, title)
    return {"svg": _svg(pages, title=title), "preview_svg": _svg(pages, preview=True, title=title), "pdf": _pdf(pages),
            "key": key, "pages": len(pages)}
