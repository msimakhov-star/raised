"""Axis calibration: find the tick labels the vision model read, in pixels, and fit value <-> pixel maps.

Steps: deskew (scans are rarely square), find the plot box from long axis-parallel runs (spines, grid lines, face),
OCR word boxes with Tesseract on a 2x upscale, match the semantics' tick strings to words beside the axes, snap each
label to its tick mark or grid line when one is visible, then least squares (linear or log10).

Pixel frame: every pixel coordinate in the result refers to the DESKEWED crop; `deskewed(img, result)` rebuilds it, and
trace/render/adjudicate work in that frame. The skew angle is at most a degree or two, so overlays on the original crop
are off by under a pixel or two at the corners.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image

from .model import Calibration, Semantics, parse_number

MAX_SKEW = 2.0       # degrees searched by the deskew step
OCR_SCALE = 2        # Tesseract reads small chart text far better upscaled (spec step 3)


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float

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
    border = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3), rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
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
    projections of the ink mask, coarse then fine."""
    mask = (ink(rgb, page_colour(rgb)) > 25).astype(np.uint8) * 255
    im = Image.fromarray(mask)

    def score(a):
        m = np.asarray(im.rotate(a, resample=Image.NEAREST)) > 0
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


def _edge(values, low: bool, support: int = 3) -> float | None:
    """Outermost value that at least `support` long runs agree on to within 3 px (one stray run cannot move an edge)."""
    v = np.sort(np.asarray(values, float))
    if not low:
        v = -v[::-1]
    for x in v:
        if (np.abs(v - x) <= 3).sum() >= min(support, len(v)):
            return float(x if low else -x)
    return None


def find_plot_box(rgb: np.ndarray, words: list[Word] = ()) -> tuple[tuple[float, float, float, float], str]:
    """Axes rectangle from long axis-parallel runs of non-background pixels (spines, grid lines or a tinted face), with
    text blanked out first so labels cannot extend a grid line. The x extent comes from long rows and the y extent from
    long columns, so grid lines alone are enough (fivethirtyeight has no spines)."""
    h, w = rgb.shape[:2]
    nb = ink(rgb, page_colour(rgb)) > 12
    for t in words:
        if t.conf >= 30:
            nb[max(int(t.y0) - 1, 0):int(t.y1) + 2, max(int(t.x0) - 1, 0):int(t.x1) + 2] = False
    rows = np.array([_runs(nb[y]) for y in range(h)])
    cols = np.array([_runs(nb[:, x]) for x in range(w)])
    longr = rows[rows[:, 0] >= max(0.4 * w, 0.8 * rows[:, 0].max())]
    longc = cols[cols[:, 0] >= max(0.3 * h, 0.8 * cols[:, 0].max())]
    if len(longr) == 0 or len(longc) == 0:
        return (0.05 * w, 0.05 * h, 0.95 * w, 0.9 * h), "fallback"
    return (_edge(longr[:, 1], True), _edge(longc[:, 1], True), _edge(longr[:, 2], False) + 1, _edge(longc[:, 2], False) + 1), "runs"


# ---------------------------------------------------------------- OCR

def number(s: str) -> float | None:
    """parse_number that tolerates OCR junk ('100-' raises inside parse_number's power-of-ten branch)."""
    try:
        return parse_number(s)
    except (ValueError, OverflowError):
        return None


def ocr_words(rgb: np.ndarray, psm: int = 11) -> list[Word]:
    g = Image.fromarray(rgb).convert("L")
    big = g.resize((g.width * OCR_SCALE, g.height * OCR_SCALE), Image.LANCZOS)
    d = pytesseract.image_to_data(big, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t:
            x, y, ww, hh = (d[k][i] / OCR_SCALE for k in ("left", "top", "width", "height"))
            out.append(Word(t, x, y, x + ww, y + hh, float(d["conf"][i])))
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
            cur = Word(cur.text + sep + b.text, cur.x0, min(cur.y0, b.y0), b.x1, max(cur.y1, b.y1), min(cur.conf, b.conf))
            out.append(cur)
    return out


_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "−": "-", "–": "-", "—": "-"})


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).replace("−", "-").replace("–", "-").lower()


def label_score(label: str, text: str, numeric: bool) -> float:
    """How well an OCR token matches a tick label string (0 = no match)."""
    a, b = _norm(label), _norm(text)
    if a == b:
        return 1.0
    if numeric:
        va = number(label)
        fixed = text.translate(_FIX) if re.search(r"\d", text) else text
        vb = number(fixed)
        if va is not None and vb is not None and abs(va - vb) <= 1e-9 * max(1.0, abs(va)):
            return 0.9
        da, db = re.sub(r"\D", "", label), re.sub(r"\D", "", fixed)
        if da and da == db and len(da) >= 2:
            return 0.6  # same digits, mangled symbols ('$' read as 'S', '%' dropped)
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.75 else 0.0


# ---------------------------------------------------------------- matching and calibration

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
    """Least squares on (label, value, px). Drops one gross outlier at a time while 4+ ticks remain.
    Chooses log over linear when the semantics say so or when log fits far better."""
    ticks = list(ticks)
    while True:
        kinds = ["log"] if scale == "log" else ["linear", "log"]
        fits = [c for k in kinds if (c := fit([t[2] for t in ticks], [t[1] for t in ticks], k))]
        if not fits:
            return None, ticks

        def norm_res(c):
            span = np.ptp([np.log10(t[1]) if c.kind == "log" else t[1] for t in ticks]) or 1.0
            return c.residual / span
        cal = fits[0]
        if len(fits) == 2 and norm_res(fits[1]) < 0.3 * norm_res(fits[0]):
            cal = fits[1]
        if len(ticks) < 4:
            return cal, ticks
        err = [abs(cal.px(t[1]) - t[2]) for t in ticks]
        k = int(np.argmax(err))
        if err[k] < 3 or err[k] < 3 * np.median(err):
            return cal, ticks
        ticks.pop(k)


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
    """Local maxima above a height, with sub-pixel centroid."""
    out = []
    for i in range(1, len(prof) - 1):
        if prof[i] >= min_height and prof[i] >= prof[i - 1] and prof[i] > prof[i + 1]:
            w = prof[i - 1:i + 2] - prof[i - 1:i + 2].min()
            out.append(i - 1 + float((w * np.arange(3)).sum() / max(w.sum(), 1e-9)))
    return out


def tick_marks(rgb: np.ndarray, box, axis: str) -> list[float]:
    """Positions of tick marks just outside (or inside) the spine, or of grid lines across the plot.
    axis 'y' gives row positions along the left spine, 'x' column positions along the bottom spine."""
    x0, y0, x1, y1 = box
    bg = page_colour(rgb)
    best = []
    if axis == "y":
        bands = [(x0 - 7, x0 - 1), (x0 + 2, x0 + 7), (x0 + 0.3 * (x1 - x0), x0 + 0.7 * (x1 - x0))]
        span = (y0 - 2, y1 + 2)
    else:
        bands = [(y1 + 1, y1 + 7), (y1 - 7, y1 - 2), (y0 + 0.3 * (y1 - y0), y0 + 0.7 * (y1 - y0))]
        span = (x0 - 2, x1 + 2)
    for band in bands:
        prof = _profile(rgb, bg, axis, band)
        base = np.median(prof)
        pk = [p for p in _peaks(prof - base, 6.0) if span[0] <= p <= span[1]]
        if len(pk) > len(best) and len(pk) <= 40:
            best = pk
    return best


def snap(ticks, marks: list[float], max_dist: float = 4.0):
    """Move each label position to the nearest tick mark when the label-to-mark offsets agree (they do for any one
    axis: labels are centred on their ticks, up to a constant baseline offset on y)."""
    if not marks or len(ticks) < 2:
        return ticks, False
    m = np.asarray(marks)
    near = [m[np.argmin(np.abs(m - t[2]))] for t in ticks]
    off = np.array([n - t[2] for n, t in zip(near, ticks)])
    ok = np.abs(off) <= max_dist
    if ok.sum() < max(2, 0.6 * len(ticks)) or np.std(off[ok]) > 1.5:
        return ticks, False
    shift = float(np.median(off[ok]))
    return [(lab, v, float(n) if o else px + shift) for (lab, v, px), n, o in zip(ticks, near, ok)], True


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
        spec[ca] = ("value", list(ot), "log" if scale == "log" else "linear") if numeric and scale != "category" else ("category", list(ot), None)
    return spec


def _category_centres(labels, matched, axis, box):
    """Pixel centre of every category: a straight line through the matched labels, else equal spacing in the box."""
    n = len(labels)
    if len(matched) >= 2:
        idx = [m[1] for m in matched]
        a, b = np.polyfit(idx, [m[2] for m in matched], 1)
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
    box, box_method = find_plot_box(rgb, words)
    specs = axis_specs(sem)
    out = {"x_cal": None, "y_cal": None, "x_ticks": [], "y_ticks": [], "categories": [], "plot_box": box,
           "diagnostics": {"angle": angle, "box_method": box_method, "psm": [11]}}
    toks = tokens(words)

    def run_axis(axis, toks):
        kind, labels, scale = specs[axis]
        m = match_axis(labels, toks, axis, box, numeric=kind == "value")
        if kind == "category":
            return [(labels[i], float(i), t.cx if axis == "x" else t.cy) for i, t in m], None
        ticks = [(labels[i], parse_number(labels[i]), t.cx if axis == "x" else t.cy) for i, t in m]
        return [t for t in ticks if t[1] is not None], scale

    results = {a: run_axis(a, toks) for a in ("x", "y")}
    if any(len(results[a][0]) < 3 and len(specs[a][1]) >= 3 for a in ("x", "y")):
        more = ocr_words(rgb, 6)  # second Tesseract pass: block mode catches labels sparse mode drops
        out["diagnostics"]["psm"].append(6)
        words = words + more
        toks2 = tokens(words)
        for a in ("x", "y"):
            alt = run_axis(a, toks2)
            if len(alt[0]) > len(results[a][0]):
                results[a] = alt
    for axis in ("x", "y"):
        kind, labels, scale = specs[axis]
        ticks, _ = results[axis]
        d = {"kind": kind, "labels": len(labels), "matched": len(ticks)}
        if kind == "category":
            cats, how = _category_centres(labels, ticks, axis, box)
            out["categories"], d["method"] = cats, how
            out[f"{axis}_ticks"] = ticks
        else:
            marks = tick_marks(rgb, box, axis)
            ticks, snapped = snap(ticks, marks)
            method = "ocr+marks" if snapped else "ocr"
            if len(ticks) < 2 and len(marks) == len(labels) and len(labels) >= 2:
                vals = [parse_number(s) for s in labels]
                order = sorted(marks) if axis == "x" else sorted(marks, reverse=True)
                if None not in vals:
                    ticks, method = [(s, v, p) for s, v, p in zip(labels, vals, order)], "marks"
            cal, ticks = calibrate(ticks, scale) if len(ticks) >= 2 else (None, ticks)
            out[f"{axis}_cal"], out[f"{axis}_ticks"] = cal, ticks
            d.update(method=method, marks=len(marks), used=len(ticks), low_confidence=len(ticks) < 3,
                     residual=cal.residual if cal else None)
        out["diagnostics"][axis] = d
    out["words"] = [(w.text, w.x0, w.y0, w.x1, w.y1) for w in words if w.conf >= 30]
    return out
