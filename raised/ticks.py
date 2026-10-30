"""Axis calibration: find the tick labels the vision model read, in pixels, and fit value <-> pixel maps.

Steps: deskew (scans are rarely square), OCR word boxes with Tesseract on a 2x upscale (spec step 3), find the plot box
from long axis-parallel lines (spines, grid lines, face), match the semantics' tick strings to words beside the axes
(strip OCR retries for blurred, slanted or crowded labels), then a consensus least-squares fit, linear or log10 (spec
step 4). Tick marks and grid lines are detected as a cross-check, as the fallback when OCR finds fewer than two labels,
and on a log axis with a single label, where the minor ticks give the decade length.

Pixel frame: continuous coordinates (pixel i spans [i, i+1)) in the DESKEWED crop; `deskewed(img, result)` rebuilds
it, and trace/render/adjudicate work in that frame. The skew is at most MAX_SKEW degrees, so overlays on the original
crop are off by a pixel or two at the corners at worst.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image

from .model import Calibration, Semantics, parse_number

MAX_SKEW = 1.0       # degrees searched by the deskew step (a crop from a scanned page; a nearly flat data line
                     # can fake a larger skew)
OCR_SCALE = 2        # Tesseract reads small chart text far better upscaled (spec step 3)
# Page segmentation modes used (from `tesseract --help-psm`, tesseract 5.5.3): 11 'sparse text, find as much text as
# possible in no particular order' (tick labels scattered round a plot); 6 'a single uniform block of text' (a retry).


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    rot: float = 0.0  # OCR'd from a rotated strip: the box is only a centre and a size

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


# ---------------------------------------------------------------- image helpers

def as_rgb(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a[..., :3].astype(np.uint8)


def page_colour(rgb: np.ndarray) -> np.ndarray:
    """Most common colour on the image border (the figure background)."""
    border = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                             rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
    q = (border // 8).astype(np.int32)
    keys, inv, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_inverse=True, return_counts=True)
    return np.median(border[inv.ravel() == counts.argmax()], axis=0)


def _rotate(rgb: np.ndarray, angle: float, fill) -> np.ndarray:
    im = Image.fromarray(rgb).rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=tuple(int(v) for v in fill))
    return np.asarray(im)


def deskewed(img, ticks_result: dict) -> np.ndarray:
    """The image in the pixel frame of a locate_ticks result."""
    rgb = as_rgb(img)
    angle = ticks_result.get("diagnostics", {}).get("angle", 0.0)
    return _rotate(rgb, angle, page_colour(rgb)) if angle else rgb


def ink(rgb: np.ndarray, bg) -> np.ndarray:
    """Per-pixel contrast against a background colour (max channel difference), lightly smoothed against scan noise."""
    from scipy.ndimage import uniform_filter
    d = np.abs(rgb.astype(np.float32) - np.asarray(bg, np.float32)).max(-1)
    return uniform_filter(d, 3)


def estimate_skew(rgb: np.ndarray) -> float:
    """Angle (degrees, PIL convention) that makes axis-parallel structure straight: maximise the energy of row and column
    projections of the grey ink (spines, grid lines, text; coloured data lines can run at any angle), measured on the
    central part so the corners the rotation fills do not count, coarse then fine."""
    a = rgb.astype(np.int16)
    grey = (a.max(-1) - a.min(-1)) < 40
    mask = ((ink(rgb, page_colour(rgb)) > 25) & grey).astype(np.uint8) * 255
    im = Image.fromarray(mask)
    h, w = mask.shape
    mid = np.s_[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)]

    def score(a):
        m = np.asarray(im.rotate(a, resample=Image.NEAREST))[mid] > 0
        return float((m.sum(0).astype(np.float64) ** 2).sum() + (m.sum(1).astype(np.float64) ** 2).sum())

    coarse = np.arange(-MAX_SKEW, MAX_SKEW + 1e-9, 0.1)
    best = coarse[np.argmax([score(a) for a in coarse])]
    fine = np.arange(best - 0.1, best + 0.1 + 1e-9, 0.02)
    s = [score(a) for a in fine]
    a = float(fine[int(np.argmax(s))])
    return 0.0 if abs(a) < 0.05 or score(a) < score(0.0) * 1.002 else round(a, 3)


# ---------------------------------------------------------------- plot box

def _runs(row: np.ndarray, gap: int = 2):
    """Longest run of True in a 1-D bool array, bridging gaps up to `gap`: (length, start, end)."""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return 0, 0, 0
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    k = int(np.argmax(ends - starts))
    return int(ends[k] - starts[k] + 1), int(starts[k]), int(ends[k])


def _edge(values, low: bool, support: int = 3) -> float:
    """Outermost value that at least `support` long runs agree on to within 3 px (one stray run cannot move an edge)."""
    v = np.sort(np.asarray(values, float))
    if not low:
        v = -v[::-1]
    x = next((x for x in v if (np.abs(v - x) <= 3).sum() >= min(support, len(v))), v[0])
    return float(x if low else -x)


def _long_runs(mask: np.ndarray, axis: int):
    """Longest run along each row (axis=1) or column (axis=0), and the indices whose run is long (60% of the longest)."""
    m = mask if axis == 1 else mask.T
    runs = np.array([_runs(r) for r in m])
    return runs, np.flatnonzero(runs[:, 0] >= max(0.25 * m.shape[1], 0.6 * runs[:, 0].max()))


def _lines(idx: np.ndarray) -> np.ndarray:
    """Long runs that can be an axes edge: thin groups (spines, grid lines) and wide ones (a tinted face), not the
    5 to 40 px bands that long text makes (a big title, a vertical axis title). All of them when nothing is left."""
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 4) + 1)
    keep = [g for g in groups if g[-1] - g[0] <= 6 or g[-1] - g[0] >= 40]
    return np.concatenate(keep) if keep else idx


def _thin(idx: np.ndarray) -> np.ndarray:
    """The members of thin groups (up to 7 adjacent rows or columns: a line, blurred) of a sorted index list; all of
    it when there is no thin group."""
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    thin = [g for g in groups if g[-1] - g[0] <= 6]
    return np.concatenate(thin) if thin else idx


def _extend(profile: np.ndarray, lo: int, hi: int, level: float) -> tuple[int, int]:
    """Grow [lo, hi] outwards while a mean ink profile stays above `level`: grid lines run to the edge of the axes,
    while the pad before a tick label is a gap. A spine stops it at once (nothing continues past it)."""
    while lo > 0 and profile[lo - 1] >= level:
        lo -= 1
    while hi < len(profile) and profile[hi] >= level:
        hi += 1
    return lo, hi


def find_plot_box(rgb: np.ndarray, words: list[Word] = (), limit=None) -> tuple[tuple, tuple, str]:
    """Axes rectangle from long axis-parallel lines: spines, grid lines or the edges of a tinted face.

    OCR'd words are blanked and the ink mask is smoothed along each direction (15 px) so faint, noisy grid lines still
    form runs. Seeds: the outermost long column on each side and long row top and bottom (spines or face edges; on a
    chart without spines, the outermost grid lines). Each edge then grows while the mean raw ink across the long lines
    stays up, so grid lines that run past the outermost perpendicular grid line carry the edge to the true axes edge
    (fivethirtyeight). `limit` (x0, y0, x1, y1) ignores everything outside it."""
    from scipy.ndimage import uniform_filter1d
    h, w = rgb.shape[:2]
    diff = np.abs(rgb.astype(np.float32) - page_colour(rgb)).max(-1)
    for t in words:
        if t.conf >= 50 and len(t.text) >= 2:
            diff[int(t.y0):int(np.ceil(t.y1)), int(t.x0):int(np.ceil(t.x1))] = 0  # no pad: a title can sit 2 px above a spine
    if limit is not None:
        lx0, ly0, lx1, ly1 = (int(round(v)) for v in limit)
        diff[:, :max(lx0, 0)] = diff[:, max(lx1, 0):] = 0
        diff[:max(ly0, 0)] = diff[max(ly1, 0):] = 0
    rows, lr = _long_runs(uniform_filter1d(diff, 15, axis=1) > 10, 1)
    cols, lc = _long_runs(uniform_filter1d(diff, 15, axis=0) > 10, 0)
    if len(lr) == 0 or len(lc) == 0:
        box = (0.05 * w, 0.05 * h, 0.95 * w, 0.9 * h)
        return box, box, "fallback"
    sc, sr = _lines(lc), _lines(lr)
    seed = [_edge(sc, True, 1), _edge(sr, True, 1), _edge(sc, False, 1) + 1, _edge(sr, False, 1) + 1]
    tr_, tc = _thin(lr), _thin(lc)  # grid lines carry the edge; bars and faces would drown the profile
    px, py = diff[tr_].mean(0), diff[:, tc].mean(1)
    gx = px[int(np.median(rows[tr_, 1])):int(np.median(rows[tr_, 2])) + 1]
    gy = py[int(np.median(cols[tc, 1])):int(np.median(cols[tc, 2])) + 1]
    x0, x1 = _extend(px, int(seed[0]), int(seed[2]), 0.5 * np.percentile(gx, 25) if gx.size else np.inf)
    y0, y1 = _extend(py, int(seed[1]), int(seed[3]), 0.5 * np.percentile(gy, 25) if gy.size else np.inf)
    grid = seed[1] - y0 > 8 or x1 - seed[2] > 8
    if grid:  # no spines: bars can reach past the last grid line
        inside = lc[(lc > x0 + 2) & (lc < x1 - 2)]
        y0 = min(y0, int(cols[inside, 1].min())) if len(inside) else y0
        inside = lr[(lr > y0 + 2) & (lr < y1 - 2)]
        x1 = max(x1, int(rows[inside, 2].max()) + 1) if len(inside) else x1
    return (float(x0), float(y0), float(x1), float(y1)), tuple(float(v) for v in seed), "grid" if grid else "lines"


# ---------------------------------------------------------------- OCR

def number(s: str) -> float | None:
    """parse_number that tolerates OCR junk ('100-' raises inside parse_number's power-of-ten branch)."""
    try:
        return parse_number(s)
    except (ValueError, OverflowError):
        return None


def ocr_words(rgb: np.ndarray, psm: int = 11, region=None, rotate: float = 0.0, prep: str = "plain") -> list[Word]:
    """Tesseract word boxes in image pixels. `region` (x0, y0, x1, y1) OCRs a strip only (less clutter: a spine next
    to the labels makes Tesseract drop them); `rotate` (degrees, counterclockwise text) straightens slanted labels;
    `prep` 'sharp' (unsharp mask, for blur) or 'clean' (median then unsharp, for scan noise)."""
    from PIL import ImageFilter
    h, w = rgb.shape[:2]
    rx0, ry0, rx1, ry1 = (0, 0, w, h) if region is None else (
        int(max(region[0], 0)), int(max(region[1], 0)), int(min(region[2], w)), int(min(region[3], h)))
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return []
    g = Image.fromarray(rgb[ry0:ry1, rx0:rx1]).convert("L")
    if prep == "clean":
        g = g.filter(ImageFilter.MedianFilter(3))
    big = g.resize((g.width * OCR_SCALE, g.height * OCR_SCALE), Image.LANCZOS)
    if prep != "plain":
        big = big.filter(ImageFilter.UnsharpMask(2, 200, 0))
    if rotate:
        big = big.rotate(-rotate, resample=Image.BICUBIC, expand=True, fillcolor=255)
    d = pytesseract.image_to_data(big, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    th = np.radians(rotate)  # back to the original: a counterclockwise turn (y down)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        x, y, ww, hh = (d[k][i] for k in ("left", "top", "width", "height"))
        if not t or hh > 0.25 * big.height + 40 or ww > 0.7 * big.width:  # Tesseract boxes whole bars as 'words'
            continue
        if rotate:  # map the box centre back through the rotation (PIL rotates about the centre, y down)
            dx, dy = x + ww / 2 - big.width / 2, y + hh / 2 - big.height / 2
            cx = np.cos(th) * dx + np.sin(th) * dy + g.width * OCR_SCALE / 2
            cy = -np.sin(th) * dx + np.cos(th) * dy + g.height * OCR_SCALE / 2
            x, y = cx - ww / 2, cy - hh / 2
        x, y, ww, hh = (v / OCR_SCALE for v in (x, y, ww, hh))
        out.append(Word(t, x + rx0, y + ry0, x + rx0 + ww, y + ry0 + hh, float(d["conf"][i]), rotate))
    return out


def tokens(words: list[Word]) -> list[Word]:
    """Single words plus runs of 2-3 neighbours on one line (Tesseract splits '1990' into '1' '990', 'New York' in two)."""
    out = list(words)
    ws = sorted(words, key=lambda w: (round(w.cy / 4), w.x0))
    for i, a in enumerate(ws):
        cur = a
        for b in ws[i + 1:i + 3]:
            hgt = max(cur.y1 - cur.y0, b.y1 - b.y0, 1)
            if abs(b.cy - cur.cy) > 0.4 * hgt or not 0 <= b.x0 - cur.x1 < 0.8 * hgt:
                break
            sep = " " if b.x0 - cur.x1 > 0.25 * hgt else ""
            cur = Word(cur.text + sep + b.text, cur.x0, min(cur.y0, b.y0), b.x1, max(cur.y1, b.y1), min(cur.conf, b.conf), a.rot)
            out.append(cur)
    return out


# Tesseract's usual digit confusions. 'S' is ambiguous: a 5, or a '$' (both variants are tried).
_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "−": "-", "–": "-", "—": "-"})


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).replace("−", "-").replace("–", "-").lower()


def label_score(label: str, text: str, numeric: bool) -> float:
    """How well an OCR token matches a tick label string (0 = no match)."""
    a, b = _norm(label), _norm(text)
    if a == b:
        return 1.0
    if numeric:
        if not re.search(r"\d", text):
            return 0.0
        va, da = number(label), re.sub(r"\D", "", label)
        fixed = text.translate(_FIX)
        variants = (fixed.replace("S", "5").replace("s", "5"), fixed.replace("S", "$").replace("s", "$"))
        if va is not None and any((vb := number(v)) is not None and abs(va - vb) <= 1e-9 * max(1.0, abs(va)) for v in variants):
            return 0.9
        if len(da) >= 2 and any(re.sub(r"\D", "", v) == da for v in variants):
            return 0.6  # same digits, mangled symbols ('%' dropped, ',' read as '.')
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.75 else 0.0


# ---------------------------------------------------------------- matching and calibration

def ink_mid_y(rgb: np.ndarray, t: Word, bg) -> float:
    """Vertical centre of the text ink inside a word box: Tesseract box heights are loose (a tick mark or the minor
    ticks of a log axis get swallowed), so take the ink-weighted centre of the main band of inked rows (row mean ink
    above a third of its peak). Box widths are reliable as they are."""
    if t.rot:
        return t.cy
    x0, y0 = max(int(t.x0), 0), max(int(t.y0), 0)
    c = ink(rgb[y0:int(np.ceil(t.y1)) + 1, x0:int(np.ceil(t.x1)) + 1], bg)
    if c.size == 0 or c.max() <= 0:
        return t.cy
    prof = c.mean(1)
    _, r0, r1 = _runs(prof >= prof.max() / 3, gap=1)
    w = prof[r0:r1 + 1]
    return y0 + float((w * (np.arange(r0, r1 + 1) + 0.5)).sum() / w.sum())


def _lis(pairs, key):
    """Longest subsequence of pairs (sorted by label index) whose pixel positions follow `key` order."""
    n = len(pairs)
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if key(pairs[j], pairs[i]) and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    if not n:
        return []
    i = int(np.argmax(best))
    out = []
    while i >= 0:
        out.append(pairs[i])
        i = prev[i]
    return out[::-1]


def match_axis(labels: list[str], toks: list[Word], axis: str, box, numeric: bool) -> list[tuple[int, Word]]:
    """Pick one OCR token per label: beside the right axis, aligned with the others, in reading order.
    axis 'x' = labels under the plot (left to right); 'y' = labels left of the plot (bottom to top)."""
    x0, y0, x1, y1 = box
    pad = 0.08 * (y1 - y0) if axis == "y" else 0.05 * (x1 - x0)
    cands = []
    for i, lab in enumerate(labels):
        for t in toks:
            if axis == "y" and not (t.x1 <= x0 + 6 and y0 - pad <= t.cy <= y1 + pad):
                continue
            if axis == "x" and not (t.y0 >= y1 - 6 and x0 - pad <= t.cx <= x1 + pad):
                continue
            s = label_score(lab, t.text, numeric)
            if s:
                cands.append((i, t, s))
    if not cands:
        return []
    # alignment: y labels share a right edge, x labels share a baseline row
    anchor = (lambda t: t.x1) if axis == "y" else (lambda t: t.cy)
    vals = np.array([anchor(t) for _, t, _ in cands])
    weights = np.array([s for *_, s in cands])
    centre = max(vals, key=lambda v: weights[np.abs(vals - v) <= 4].sum())
    tol = 6 if axis == "y" else 0.6 * np.median([t.y1 - t.y0 for _, t, _ in cands]) + 3
    best = {}
    for i, t, s in cands:
        if abs(anchor(t) - centre) <= tol and (i not in best or s > best[i][1]):
            best[i] = (t, s)
    pairs = sorted((i, t) for i, (t, _) in best.items())
    if axis == "x":
        order = lambda a, b: a[1].cx < b[1].cx
    else:  # value labels read bottom to top, category labels top to bottom
        order = (lambda a, b: a[1].cy > b[1].cy) if numeric else (lambda a, b: a[1].cy < b[1].cy)
    return _lis(pairs, order)


def fit(pos: list[float], vals: list[float], kind: str) -> Calibration | None:
    p, v = np.asarray(pos, float), np.asarray(vals, float)
    if kind == "log":
        if (v <= 0).any():
            return None
        v = np.log10(v)
    if len(p) < 2 or np.ptp(p) < 1:
        return None
    a, b = np.polyfit(p, v, 1)
    res = float(np.sqrt(np.mean((a * p + b - v) ** 2))) if len(p) > 2 else 0.0
    return Calibration(kind, float(a), float(b), res, len(p))


def calibrate(ticks: list[tuple[str, float, float]], scale: str):
    """Least squares on (label, value, px) after a consensus step: every pair of ticks proposes a line, the one most
    ticks agree with (within 3 px) wins, so a mismatched label or two cannot bend the fit. Log is tried too unless the
    semantics say log; it wins on more agreeing ticks, or on the same count with a much better fit. Returns
    (calibration or None, ticks used)."""
    ticks = [t for t in ticks if t[1] is not None]
    best = None
    for kind in (["log"] if scale == "log" else ["linear", "log"]):
        for i in range(len(ticks)):
            for j in range(i + 1, len(ticks)):
                c = fit([ticks[i][2], ticks[j][2]], [ticks[i][1], ticks[j][1]], kind)
                if c is None:
                    continue
                inl = [t for t in ticks if (t[1] > 0 or kind == "linear") and abs(c.px(t[1]) - t[2]) <= 3]
                full = fit([t[2] for t in inl], [t[1] for t in inl], kind)
                if full is None:
                    continue
                span = np.ptp([np.log10(t[1]) if kind == "log" else t[1] for t in inl]) or 1.0
                key = (len(inl), -full.residual / span * (1 if kind == "linear" or scale == "log" else 3))
                if best is None or key > best[0]:
                    best = (key, full, inl)
    return (best[1], best[2]) if best else (None, ticks)


# ---------------------------------------------------------------- tick marks and grid lines

def _profile(rgb: np.ndarray, bg, axis: str, band: tuple[int, int]) -> np.ndarray:
    """Contrast profile along an axis, averaged over a band of the perpendicular coordinate."""
    lo, hi = max(int(band[0]), 0), int(band[1])
    if axis == "y":
        hi = min(hi, rgb.shape[1])
        sl = rgb[:, lo:hi]
        return ink(sl, bg).mean(1) if hi > lo else np.zeros(rgb.shape[0])
    hi = min(hi, rgb.shape[0])
    sl = rgb[lo:hi]
    return ink(sl, bg).mean(0) if hi > lo else np.zeros(rgb.shape[1])


def _peaks(prof: np.ndarray, min_height: float) -> list[float]:
    """Local maxima above a height, with sub-pixel centroid, in continuous coordinates (pixel i spans [i, i+1))."""
    out = []
    for i in range(1, len(prof) - 1):
        if prof[i] >= min_height and prof[i] >= prof[i - 1] and prof[i] > prof[i + 1]:
            w = prof[i - 1:i + 2] - prof[i - 1:i + 2].min()
            out.append(i - 0.5 + float((w * np.arange(3)).sum() / max(w.sum(), 1e-9)))
    return out


def tick_marks(rgb: np.ndarray, box, axis: str, log: bool = False) -> list[float]:
    """Positions of the major tick marks along an axis: 'y' gives rows along the left spine, 'x' columns along the
    bottom one. Bands tried in order: just outside the spine (ticks out), just inside (ticks in), across the middle
    of the plot (grid lines). A band counts when its peaks are evenly spaced (a data line crossing the middle band
    is not); on a log axis minor ticks break the spacing, so only the grid band can serve, unchecked."""
    x0, y0, x1, y1 = box
    bg = page_colour(rgb)
    if axis == "y":
        bands = [(x0 - 7, x0 - 1), (x0 + 2, x0 + 7), (x0 + 0.3 * (x1 - x0), x0 + 0.7 * (x1 - x0))]
        span = (y0 - 2, y1 + 2)
    else:
        bands = [(y1 + 1, y1 + 7), (y1 - 7, y1 - 2), (y0 + 0.3 * (y1 - y0), y0 + 0.7 * (y1 - y0))]
        span = (x0 - 2, x1 + 2)
    for band in bands:
        prof = _profile(rgb, bg, axis, band)
        pk = [p for p in _peaks(prof - np.median(prof), 6.0) if span[0] <= p <= span[1]]
        gaps = np.diff(pk)
        if len(pk) >= 3 and (np.std(gaps) <= 0.05 * np.mean(gaps) + 0.5 or log):
            return pk
    return []


def log_decade(tick, marks: list[float], axis: str, span: tuple[float, float]) -> Calibration | None:
    """A log axis with one labelled tick (matplotlib labels decades only, so a chart spanning under two decades shows
    one label): the decade length from the minor tick marks around it, which sit at log10(k), k = 2..9, of a decade.
    Every mark proposes a decade length for every k; the length most marks agree with (within 1.5 px) wins, if at
    least 4 marks agree and marks are found at 60% of the tick positions it predicts inside the axis span."""
    _, v0, p0 = tick
    if not v0 or v0 <= 0:
        return None
    sign = -1.0 if axis == "y" else 1.0  # values grow upwards on y
    m = np.asarray([x for x in marks if abs(x - p0) > 2])
    steps = np.array([np.log10(k) + j for j in (-2, -1, 0, 1) for k in range(1, 10)])
    steps = steps[steps != 0]
    best = None
    for x in m:
        for u in steps:
            s = sign * (x - p0) / u
            if not 20 <= s <= 5000:
                continue
            pred = p0 + sign * s * steps
            pred = pred[(pred >= span[0]) & (pred <= span[1])]
            hit = int(sum(np.min(np.abs(pred - y)) <= 1.5 for y in m))
            if hit >= 4 and hit >= 0.6 * len(pred) and (best is None or hit > best[0]):
                best = (hit, s)
    if best is None:
        return None
    s = best[1]
    return Calibration("log", sign / s, float(np.log10(v0) - sign * p0 / s), 0.0, 1)


def marks_agree(ticks, marks: list[float], max_dist: float = 4.0) -> bool | None:
    """Cross-check: do the label positions sit at a constant offset from detected tick marks or grid lines? (matplotlib
    centres x labels on their ticks and aligns y labels 'center_baseline': rcParams xtick/ytick.alignment, read from
    the installed matplotlib 3.11.2.) Snapping the labels to the marks measured no better than the labels alone on the
    probe split (median 0.52 vs 0.47 px), so this only flags; None when there is nothing to compare."""
    if not marks or len(ticks) < 2:
        return None
    m = np.asarray(marks)
    off = np.array([m[np.argmin(np.abs(m - t[2]))] - t[2] for t in ticks])
    ok = np.abs(off) <= max_dist
    return bool(ok.sum() >= max(2, 0.6 * len(ticks)) and np.std(off[ok]) <= 1.5)


# ---------------------------------------------------------------- public

def value_scale(sem: Semantics) -> str:
    s = getattr(sem, f"{sem.value_axis}_scale")
    if s == "category":
        s = sem.x_scale if sem.value_axis == "y" else sem.y_scale
    return "log" if s == "log" else "linear"


def axis_specs(sem: Semantics) -> dict:
    """What each pixel axis carries: ('value', labels, scale) or ('category', labels, None)."""
    bar = sem.chart_type in ("bar", "grouped_bar", "hbar")
    va = sem.value_axis
    ca = "y" if va == "x" else "x"
    vt = sem.x_ticks if va == "x" else sem.y_ticks
    spec = {va: ("value", list(vt), value_scale(sem))}
    if bar:
        cats = list(sem.categories) or list(sem.y_ticks if ca == "y" else sem.x_ticks)
        if ca == "y" and not sem.categories:
            cats = cats[::-1]  # y tick strings are bottom to top, categories top to bottom
        spec[ca] = ("category", cats, None)
    else:
        ot = sem.x_ticks if ca == "x" else sem.y_ticks
        numeric = sum(parse_number(t) is not None for t in ot) >= max(2, 0.8 * len(ot))
        scale = getattr(sem, f"{ca}_scale")
        if numeric and scale != "category":
            spec[ca] = ("value", list(ot), "log" if scale == "log" else "linear")
        else:
            spec[ca] = ("category", list(ot), None)
    return spec


def _category_centres(labels, matched, axis, box):
    """Pixel centre of every category: a straight line through the matched labels, else equal spacing in the box."""
    n = len(labels)
    if len(matched) >= 2:
        idx, pos = np.array([m[1] for m in matched]), np.array([m[2] for m in matched])
        while True:  # categories are evenly spaced: drop the worst label while it sits a quarter step off the line
            a, b = np.polyfit(idx, pos, 1)
            err = np.abs(a * idx + b - pos)
            if len(idx) <= 2 or err.max() < 0.25 * abs(a):
                break
            idx, pos = np.delete(idx, err.argmax()), np.delete(pos, err.argmax())
        return [(lab, float(a * i + b)) for i, lab in enumerate(labels)], "labels"
    x0, y0, x1, y1 = box
    lo, hi = (x0, x1) if axis == "x" else (y0, y1)
    return [(lab, float(lo + (i + 0.5) * (hi - lo) / max(n, 1))) for i, lab in enumerate(labels)], "equal"


def locate_ticks(img, sem: Semantics) -> dict:
    """Calibrate both axes of a chart crop. Returns
    {x_cal, y_cal: Calibration | None (category axis), x_ticks, y_ticks: [(label, value, px)] matched ticks (category axes:
    value = category index), categories: [(label, px)] centres on the category axis, plot_box: (x0, y0, x1, y1),
    words: OCR words [(text, x0, y0, x1, y1)], diagnostics: {angle, box_method, per-axis method/n/low_confidence}}."""
    rgb0 = as_rgb(img)
    angle = estimate_skew(rgb0)
    rgb = _rotate(rgb0, angle, page_colour(rgb0)) if angle else rgb0
    words = ocr_words(rgb, 11)
    h, w = rgb.shape[:2]
    bg = page_colour(rgb)
    vertical = ocr_words(rgb, 11, (0, 0, 0.15 * w, h), rotate=90)  # the y axis title: long vertical text fakes a line
    box, inner, box_method = find_plot_box(rgb, words + vertical)
    specs = axis_specs(sem)
    out = {"x_cal": None, "y_cal": None, "x_ticks": [], "y_ticks": [], "categories": [],
           "diagnostics": {"angle": angle, "box_method": box_method, "ocr": ["page"]}}
    # labels sit left of / below the outermost lines even where grid lines carry the box further out; along the axis
    # they span the full box (one grid line alone gives no extent)
    side = (inner[0], min(inner[1], box[1]), max(inner[2], box[2]), inner[3])
    x0, y0, x1, y1 = side
    strips = {"y": (0, y0 - 0.1 * h, x0 - 2, y1 + 0.1 * h), "x": (x0 - 0.1 * w, y1 + 2, x1 + 0.1 * w, h)}

    def run_axis(axis, ws):
        kind, labels, scale = specs[axis]
        m = match_axis(labels, tokens(ws), axis, side, numeric=kind == "value")
        return [(labels[i], float(i) if kind == "category" else number(labels[i]), t,
                 t.cx if axis == "x" else ink_mid_y(rgb, t, bg))
                for i, t in m if kind == "category" or number(labels[i]) is not None]

    results, pool = {}, {}
    for a in ("x", "y"):  # strip OCR variants until the axis is (nearly) fully matched
        n = len(specs[a][1])
        tries = [(11, 0.0, p) for p in ("plain", "sharp", "clean")] + [(6, 0.0, p) for p in ("plain", "sharp")]
        if a == "x" and specs[a][0] == "category":
            tries += [(11, r, p) for r in (30, 45, 90) for p in ("plain", "sharp")]
        pool[a], results[a] = list(words), run_axis(a, words)
        for psm, rot, prep in tries:
            if len(results[a]) == n or len(results[a]) >= max(4, n - 1):
                break
            more = ocr_words(rgb, psm, strips[a], rotate=rot, prep=prep)
            out["diagnostics"]["ocr"].append(f"{a}:psm{psm}:{prep}" + (f":rot{rot:g}" if rot else ""))
            alt = run_axis(a, pool[a] + more)
            if len(alt) >= len(results[a]):
                results[a], pool[a] = alt, pool[a] + more
    # labels sit outside the axes: never let the box reach past them (a label touching a grid line extends its run)
    ylab = [t for *_, t, _ in results["y"]]
    xlab = [t for *_, t, _ in results["x"]]
    lim = (max([t.x1 for t in ylab], default=-1) + 1, 0, w, min([t.y0 for t in xlab], default=h + 1) - 1)
    if lim != (0, 0, w, h):
        limited = find_plot_box(rgb, words + vertical, lim)[0]
        # a stray label match (rotated year labels read high up) can squeeze the limit to nothing: keep the
        # unlimited box then, and failing that use the span of the matched ticks themselves
        if limited[2] - limited[0] >= 20 and limited[3] - limited[1] >= 20:
            box = limited
    if box[2] - box[0] < 20 or box[3] - box[1] < 20:
        xs_ = [p for *_, p in results["x"]] or [0, w]
        ys_ = [p for *_, p in results["y"]] or [0, h]
        box = (min(xs_), min(ys_), max(xs_), max(ys_))
        out["diagnostics"]["box_method"] = "tick span (fallback)"
    out["plot_box"] = box
    results = {a: [(lab, v, p) for lab, v, _, p in r] for a, r in results.items()}
    for axis in ("x", "y"):
        kind, labels, scale = specs[axis]
        ticks = results[axis]
        d = {"kind": kind, "labels": len(labels), "matched": len(ticks)}
        if kind == "category":
            cats, how = _category_centres(labels, ticks, axis, box)
            out["categories"], d["method"] = cats, how
            out[f"{axis}_ticks"] = ticks
        else:
            marks = tick_marks(rgb, box, axis, scale == "log")
            method = "ocr"
            if len(ticks) < 2 and len(marks) == len(labels) and len(labels) >= 2:  # fallback: marks in order
                vals = [number(s) for s in labels]
                order = sorted(marks) if axis == "x" else sorted(marks, reverse=True)
                if None not in vals:
                    ticks, method = [(s, v, p) for s, v, p in zip(labels, vals, order)], "marks"
            cal, ticks = calibrate(ticks, scale) if len(ticks) >= 2 else (None, ticks)
            if cal is None and scale == "log" and len(ticks) == 1:
                cal = log_decade(ticks[0], marks, axis, (box[1], box[3]) if axis == "y" else (box[0], box[2]))
                method = "label+minor ticks" if cal else method
            out[f"{axis}_cal"], out[f"{axis}_ticks"] = cal, ticks
            d.update(method=method, marks=len(marks), marks_agree=marks_agree(ticks, marks), used=len(ticks),
                     low_confidence=len(ticks) < 3, residual=cal.residual if cal else None)
        out["diagnostics"][axis] = d
    out["words"] = [(t.text, t.x0, t.y0, t.x1, t.y1) for t in words if t.conf >= 30]
    return out
