"""Pixel tracer (reading B, spec step 5): series colour masks inside the plot box, then values through the calibration.

Colour model: blur and antialiasing mix a series colour c into the face colour bg, so a pixel is p = bg + a (c - bg)
with coverage a. A pixel belongs to the series whose segment bg..c_k explains it best, and enters that series' mask at
a >= 0.5 for bars (the 50% point of a blurred edge is the true edge) or at half the line's own typical peak for thin
lines. (The spec says Lab clustering; the mixing segment in RGB is what blur and antialiasing actually do, and it
separates black from grey by coverage.) Series colours: the chart's own colour modes, matched to the legend swatch
when the legend label is OCR'd, else to the colour name, by direction from the face (Hungarian assignment, so 'blue'
and 'teal' cannot both take one mode). Grid lines on tick rows, the baseline spine under bars, and the fringe of a
black line in a grey series are removed from the masks.

Lines: per column, the centre of the masked run nearest the neighbours (continuity). Values at given x come from
straight-line fits to each stretch between neighbouring x values, away from the vertices: markers, joins and a
dashed pattern cannot bias them, and a vertex is where its two stretches meet. `trace` does not know the data x
values, so it samples where the line has vertices: on the coarsest regular grid that fits the vertices it finds
(Douglas-Peucker), else at the vertices themselves. The benchmark compares at the true x, through `trace_at`.
Bars: connected components per colour standing on the baseline; the top (hbar: right) edge to sub-pixel by summing
coverage; components to categories by nearest centre. Scatter: component centroids, merged blobs split by area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .model import Semantics, Table
from .ticks import deskewed, label_score, number, page_colour

SHRINK = 5  # px inside the plot box: spines and inward tick marks stay out of the colour masks

# Plain colour names as a reader (or the vision model) says them -> matplotlib's default 'tab' palette, the most common
# chart palette; anything else goes to matplotlib's CSS4 / xkcd tables. Only a starting point: the colour actually used
# is the nearest peak in the image.
_TAB = {"blue": "tab:blue", "orange": "tab:orange", "green": "tab:green", "red": "tab:red", "purple": "tab:purple",
        "brown": "tab:brown", "pink": "tab:pink", "grey": "tab:gray", "gray": "tab:gray", "olive": "tab:olive",
        "cyan": "tab:cyan", "teal": "tab:cyan", "black": "black"}


def reference_rgb(name: str) -> np.ndarray | None:
    from matplotlib import colors
    n = (name or "").strip().lower()
    for cand in (_TAB.get(n), n, f"xkcd:{n}", n.replace(" ", "")):
        if cand:
            try:
                return np.array(colors.to_rgb(cand)) * 255
            except ValueError:
                pass
    words = n.split()
    return reference_rgb(words[-1]) if len(words) > 1 else None  # 'dark blue' -> 'blue' when xkcd has no entry


@dataclass
class Scene:
    rgb: np.ndarray                 # deskewed crop
    box: tuple[int, int, int, int]  # region traced: plot box shrunk by SHRINK
    bg: np.ndarray                  # face colour
    colours: dict[str, np.ndarray]  # series -> colour
    alpha: dict[str, np.ndarray]    # series -> coverage map (full image, 0 outside region / where another series wins)
    masks: dict[str, np.ndarray]    # series -> bool mask
    legend: tuple | None            # legend box masked out, if found
    region: np.ndarray              # pixels traced (box minus known text and the legend)


def _mode_colour(px: np.ndarray) -> np.ndarray:
    q = (px // 8).astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return np.median(px[inv.ravel() == counts.argmax()], axis=0)


def _palette(px: np.ndarray, bg: np.ndarray, starts: list[np.ndarray]) -> list[np.ndarray]:
    """Candidate series colours among non-face pixels: mean-shift modes (radius 40) seeded at each expected colour and
    at the most common 16-level colour bins, plus, per expected colour, the strong end of the pixels that lie in its
    direction from the face (a thin blurred dashed line after JPEG never reaches its nominal colour)."""
    far = px[np.abs(px - bg).max(1) > 30]
    if len(far) < 10:
        return []
    q = np.clip(far // 16, 0, 15).astype(np.int32)
    keys, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
    top = [np.array([k // 256, k // 16 % 16, k % 16]) * 16 + 8.0 for k in keys[np.argsort(-counts)][:8]]
    out = []

    def add(c):
        if all(np.linalg.norm(c - o) > 20 for o in out):
            out.append(c)
    for c in list(starts) + top:
        c = np.asarray(c, float)
        for _ in range(20):
            sel = np.linalg.norm(far - c, axis=1) < 40
            if sel.sum() < 8:
                break
            nc = far[sel].mean(0)
            done = np.linalg.norm(nc - c) < 0.5
            c = nc
            if done:
                break
        if sel.sum() >= 8:
            add(c)
    d = far - bg
    norm = np.linalg.norm(d, axis=1)
    for r in starts:
        u = (r - bg) / max(np.linalg.norm(r - bg), 1.0)
        along = d @ u
        sel = along >= 0.97 * norm  # within about 14 degrees of the expected direction
        if sel.sum() >= 15:
            strong = far[sel][along[sel] >= np.percentile(along[sel], 70)]
            add(np.median(strong, axis=0))
    return out


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), -1, 1))))


def _legend(rgb, sem: Semantics, words, bg, box) -> tuple[tuple | None, dict[str, np.ndarray]]:
    """Legend box (to mask) and swatch colours, from legend labels OCR'd inside the plot box. The swatch sits left
    of its label: matplotlib's handle is legend.handlelength 2.0 em plus handletextpad 0.8 em before the text, inside
    borderpad 0.4 em (rcParams defaults of the installed matplotlib 3.11.2). OCR word boxes of 8 pt legend labels at
    110 dpi measured 9.5 to 12 px against a 12.2 px em, so the box reaches 4.5 word heights left of the labels and the
    swatch is sampled 0.6 to 3.8 word heights left."""
    names = [s.name for s in sem.series]
    labels = list(sem.legend) or (names if len(names) > 1 else [])
    x0, y0, x1, y1 = box
    inside = [w for w in words if x0 <= (w[1] + w[3]) / 2 <= x1 and y0 <= (w[2] + w[4]) / 2 <= y1]
    found = {}
    for lab in labels:
        best = max(((label_score(lab, w[0], False), w) for w in inside), default=(0, None), key=lambda t: t[0])
        if best[0] >= 0.8:
            found[lab] = best[1]
    if len(found) < max(1, len(labels) // 2):
        return None, {}
    hh = float(np.median([w[4] - w[2] for w in found.values()]))
    x0 = min(w[1] for w in found.values()) - 4.5 * hh
    box = (x0, min(w[2] for w in found.values()) - 0.8 * hh, max(w[3] for w in found.values()) + 0.8 * hh,
           max(w[4] for w in found.values()) + 0.8 * hh)
    named = {s.name: reference_rgb(s.color) for s in sem.series}
    swatch = {}
    for lab, w in found.items():
        ys = slice(int(w[2]), int(np.ceil(w[4])) + 1)
        xs = slice(max(int(w[1] - 3.8 * hh), 0), max(int(w[1] - 0.6 * hh), 0))
        px = rgb[ys, xs].reshape(-1, 3).astype(float)
        px = px[np.abs(px - bg).max(1) > 40] if len(px) else px
        if len(px) < 4:
            continue
        # the swatch's colour clusters (16 levels), commonest first; a marker or patch outline is often black, so
        # take the cluster nearest the colour name when there is one
        q = (px // 16).astype(np.int32)
        _, inv, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_inverse=True, return_counts=True)
        cands = [px[inv.ravel() == k].mean(0) for k in np.argsort(-counts) if counts[k] >= 3] or [px.mean(0)]
        name = names[labels.index(lab)] if len(names) == len(labels) else lab
        ref = named.get(name)
        swatch[name] = cands[0] if ref is None else min(cands, key=lambda c: _cost(ref, c, bg))
    return box, swatch


def _cost(ref: np.ndarray, c: np.ndarray, bg: np.ndarray) -> float:
    """How unlike an expected colour a found one is, seen from the face: direction (blur and JPEG shorten a colour's
    distance from the face, they barely turn it) plus length (black against grey)."""
    dr, dc = ref - bg, c - bg
    return _angle(dr, dc) + 30 * abs(np.log(max(np.linalg.norm(dc), 1.0) / max(np.linalg.norm(dr), 1.0)))


def _coverage(px: np.ndarray, bg: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage a along bg->c and distance from the segment bg..c, per pixel."""
    d = c - bg
    a = (px - bg) @ d / max(d @ d, 1.0)
    proj = bg + np.clip(a, 0, 1)[:, None] * d
    return a, np.linalg.norm(px - proj, axis=1)


def scene(img, sem: Semantics, tr: dict) -> Scene:
    rgb = deskewed(img, tr)
    h, w = rgb.shape[:2]
    bx0, by0, bx1, by1 = tr["plot_box"]
    sx0, sy1 = SHRINK, SHRINK
    if sem.chart_type in ("bar", "grouped_bar"):  # keep short bars: come close to the baseline, clear the spine below
        sy1 = 1
    elif sem.chart_type == "hbar":
        sx0 = 1
    box = (max(int(bx0) + sx0, 0), max(int(by0) + SHRINK, 0), min(int(np.ceil(bx1)) - SHRINK, w), min(int(np.ceil(by1)) - sy1, h))
    x0, y0, x1, y1 = box
    region = np.zeros((h, w), bool)
    region[y0:y1, x0:x1] = True
    words = tr.get("words", [])
    # chart text that can sit inside the plot; not every OCR word: Tesseract reads markers and noise as 'words'
    known = [t for t in (sem.title, sem.x_title, sem.y_title, sem.source, *sem.legend, *(s.name for s in sem.series)) if t]
    for t, a, b, c, d in words:
        if len(t) >= 3 and any(label_score(k, t, False) for k in known for k in [k] + k.split()):
            region[max(int(b) - 1, 0):int(d) + 2, max(int(a) - 1, 0):int(c) + 2] = False
    frame = region.copy()  # the face shows along the inside of the box frame (bars can fill most of the middle)
    frame[y0 + 8:y1 - 8, x0 + 8:x1 - 8] = False
    if sem.chart_type in ("bar", "grouped_bar"):  # bars stand on the bottom edge and may fill the sides
        frame[y0 + 8:] = False
    elif sem.chart_type == "hbar":
        frame[:, :x1 - 8] = False
    bg = _mode_colour(rgb[frame].astype(float)) if frame.any() else page_colour(rgb)
    # Axis spines and gridlines: straight runs of non-face pixels across most of the plot. Some house styles
    # (OpenStax) draw axes in the series colour, which otherwise traces as a flat line at the axis value.
    ink = np.abs(rgb.astype(float) - bg).max(2) > 40
    inner = ink[y0:y1, x0:x1]
    if inner.size and sem.chart_type in ("line", "scatter"):
        for r in np.where(inner.mean(1) > 0.6)[0]:
            region[max(y0 + r - 2, 0):y0 + r + 3, :] = False
        for c in np.where(inner.mean(0) > 0.6)[0]:
            region[:, max(x0 + c - 2, 0):x0 + c + 3] = False
    legend_box, swatch = _legend(rgb, sem, words, bg, box)
    if legend_box:
        lx0, ly0, lx1, ly1 = legend_box
        region[max(int(ly0), 0):int(ly1) + 1, max(int(lx0), 0):int(lx1) + 1] = False
    px = rgb[region].astype(float)
    refs = []
    for s in sem.series:
        r = swatch.get(s.name)
        if r is None:
            r = reference_rgb(s.color)
        refs.append(r if r is not None else np.zeros(3))
    palette = _palette(px, bg, refs)
    colours = {}
    if palette and refs:
        cm = np.array([[_cost(r, p, bg) for p in palette] for r in refs])
        rows, cols = linear_sum_assignment(cm)
        for r, c in zip(rows, cols):
            colours[sem.series[r].name] = palette[c] if cm[r, c] < 35 else refs[r]
    for s, r in zip(sem.series, refs):
        colours.setdefault(s.name, r)
    names = list(colours)
    # each pixel to the series whose mixing segment explains it best, slightly preferring fuller coverage (a grey
    # pixel is a full grey series pixel before it is a half-covered black one)
    best = np.full(len(px), np.inf)
    owner = np.full(len(px), -1)
    alphas = []
    for k, n in enumerate(names):
        c = colours[n]
        span = np.linalg.norm(c - bg)
        a, dist = _coverage(px, bg, c)
        score = dist + 0.1 * span * np.clip(1 - a, 0, 1)
        better = (a >= 0.25) & (dist <= 20 + 0.3 * span) & (score < best)
        best[better], owner[better] = score[better], k
        alphas.append(a)
    alpha, masks = {}, {}
    for k, n in enumerate(names):
        full = np.zeros((h, w), np.float32)
        full[region] = np.where(owner == k, np.clip(alphas[k], 0, 1.5), 0)
        alpha[n] = full
    bars = sem.chart_type in ("bar", "grouped_bar", "hbar")
    grid_rows = [int(p) for _, _, p in tr.get("y_ticks", []) if y0 <= p < y1] if not bars else []
    grid_cols = [int(p) for _, _, p in tr.get("x_ticks", []) if x0 <= p < x1] if not bars else []
    for k, n in enumerate(names):
        thr = 0.5  # the 50% point is the true edge of a bar
        if not bars:  # a thin blurred line never reaches full colour: half of its own typical peak
            owned = alpha[n][alpha[n] >= 0.25]
            thr = 0.5 * float(np.clip(np.percentile(owned, 90), 0.6, 1.0)) if owned.size else 0.5
        m = alpha[n] >= thr
        for r in grid_rows:  # a grid line in a light series colour: long runs on a tick row
            for rr in range(max(r - 1, y0), min(r + 2, y1)):
                if m[rr, x0:x1].mean() >= 0.5:
                    m[rr] = False
        for c in grid_cols:
            for cc in range(max(c - 1, x0), min(c + 2, x1)):
                if m[y0:y1, cc].mean() >= 0.5:
                    m[:, cc] = False
        if bars:  # an axis line in the series colour along the baseline (a black spine under black bars)
            if sem.chart_type == "hbar":
                cut = [c for c in range(x0, min(x0 + 5, x1)) if m[y0:y1, c].mean() >= 0.6]
                m[:, cut] = False
            else:
                cut = [r for r in range(max(y1 - 5, y0), y1) if m[r, x0:x1].mean() >= 0.6]
                m[cut, :] = False
        for j in names:  # the soft edge of a black line looks like a grey line: drop grey right next to black cores
            a, dist = _coverage(colours[n][None], bg, colours[j])
            if j != n and dist[0] < 25 and 0.2 < a[0] < 0.85:
                m &= ~ndimage.binary_dilation(alpha[j] >= 0.9, iterations=2)
        masks[n] = m
    return Scene(rgb, box, bg, colours, alpha, masks, legend_box, region)


def series_masks(img, sem: Semantics, tr: dict) -> dict[str, np.ndarray]:
    """Bool mask per series in the deskewed frame: what the original chart draws in each series colour."""
    return scene(img, sem, tr).masks


def geometry(sc: Scene, sem: Semantics, tr: dict) -> dict:
    """Drawing sizes measured on the original so a redraw can match it: line width and marker radius (px), and for
    bars the slot (centre, thickness) of every bar read."""
    if sem.chart_type in ("bar", "grouped_bar", "hbar"):
        return {"bars": _bars(sc, sem, tr)[2]}
    x0, y0, x1, y1 = sc.box
    widths, radii = [], []
    for m in sc.masks.values():
        runs = [r[0][1] - r[0][0] + 1 for c in range(x0, x1) if len(r := _runs_1d(m[y0:y1, c])) == 1]
        if runs:  # the thinner columns: where the line runs flattest
            widths.append(np.percentile(runs, 25))
        if sem.chart_type == "scatter":
            lab, n = ndimage.label(m)
            if n:
                radii.append(np.sqrt(np.median(ndimage.sum(m, lab, range(1, n + 1))) / np.pi))
    return {"line_width": float(np.median(widths)) if widths else 2.0,
            "marker_radius": float(np.median(radii)) if radii else 3.5}


# ---------------------------------------------------------------- lines

def _runs_1d(col: np.ndarray):
    idx = np.flatnonzero(col)
    if not idx.size:
        return []
    br = np.flatnonzero(np.diff(idx) > 2)
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]))


def _despeckle(ys: np.ndarray, tol: float = 10.0) -> None:
    """Drop centres more than tol px from the running median of their 11 valid neighbours (in place): stray pixels in
    a dash gap, a crossing line's fringe. A steep straight stretch keeps its median, so only a sharp apex can lose a
    column or two, and the stretch fits never use the apex."""
    good = np.flatnonzero(~np.isnan(ys))
    if good.size >= 5:
        ref = ndimage.median_filter(ys[good], size=min(11, good.size - (good.size + 1) % 2), mode="nearest")
        ys[good[np.abs(ys[good] - ref) > tol]] = np.nan


def centreline(mask: np.ndarray, box) -> np.ndarray:
    """Row centre (continuous coords) of the line per column, NaN where absent. Where a column holds several runs
    (a crossing series' leftovers, a legend swatch, a marker of a neighbour), keep the run nearest the neighbours."""
    x0, y0, x1, y1 = box
    ys = np.full(mask.shape[1], np.nan)
    multi = {}
    for c in range(x0, x1):
        runs = _runs_1d(mask[y0:y1, c])
        if len(runs) == 1:
            ys[c] = y0 + (runs[0][0] + runs[0][1] + 1) / 2
        elif runs:
            multi[c] = [y0 + (a + b + 1) / 2 for a, b in runs]
    _despeckle(ys)
    single = np.flatnonzero(~np.isnan(ys))
    for c, cs in multi.items():
        if single.size:
            ref = np.interp(c, single, ys[single])
            ys[c] = min(cs, key=lambda v: abs(v - ref))
    _despeckle(ys)
    return ys


def _fit(cols: np.ndarray, ys: np.ndarray):
    """Straight line through (column centre, row), refitted once without points more than 2 px off."""
    x = cols + 0.5
    if len(x) < 2:
        return None
    a, b = np.polyfit(x, ys, 1)
    ok = np.abs(a * x + b - ys) <= 2
    if 2 <= ok.sum() < len(x):
        a, b = np.polyfit(x[ok], ys[ok], 1)
    return a, b


def _line_values(ys: np.ndarray, pxs: list[float], margin: float) -> tuple[list[float | None], list[float]]:
    """Row at each vertex column pxs (sorted), from straight fits to the stretches between them."""
    good = np.flatnonzero(~np.isnan(ys))
    if not good.size:
        return [None] * len(pxs), [0.0] * len(pxs)
    lo, hi = good[0], good[-1] + 1
    fits = []
    for a, b in zip(pxs, pxs[1:]):
        m = min(margin, 0.3 * (b - a))
        sel = good[(good + 0.5 >= a + m) & (good + 0.5 <= b - m)]
        if len(sel) < 2:
            sel = good[(good + 0.5 >= a) & (good + 0.5 <= b)]
        fits.append((_fit(sel, ys[sel]), len(sel) / max(b - a - 2 * m, 1)))
    out, conf = [], []
    for i, p in enumerate(pxs):
        if p < lo - SHRINK - 4 or p > hi + SHRINK + 4:  # a line may run into the box edge (no x margin)
            out.append(None)
            conf.append(0.0)
            continue
        est = [f[0] * p + f[1] for f, _ in (fits[i - 1:i] if i else []) + fits[i:i + 1] if f is not None]
        cov = [c for f, c in (fits[i - 1:i] if i else []) + fits[i:i + 1] if f is not None]
        if not est:
            out.append(float(np.interp(p, good + 0.5, ys[good])))
            conf.append(0.3)
            continue
        out.append(float(np.mean(est)))
        spread = abs(est[0] - est[-1])
        conf.append(float(min(1.0, np.mean(cov)) * (1.0 if spread < 2 else 2 / spread)))
    return out, conf


def _douglas_peucker(xs: np.ndarray, ys: np.ndarray, tol: float) -> list[int]:
    keep = {0, len(xs) - 1}
    stack = [(0, len(xs) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        x, y = xs[i:j + 1], ys[i:j + 1]
        dx, dy = xs[j] - xs[i], ys[j] - ys[i]
        d = np.abs(dy * (x - xs[i]) - dx * (y - ys[i])) / max(np.hypot(dx, dy), 1e-9)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep.add(i + k)
            stack += [(i, i + k), (i + k, j)]
    return sorted(keep)


def _sample_xs(ys: np.ndarray, x_cal, sem: Semantics) -> list[float]:
    """Data x values to sample a traced line at, when the true ones are unknown: the coarsest regular grid (the tick
    step divided by 1, 2, 2.5, 4, 5, 10, ...) through the line's ends that also lands on three quarters of its
    Douglas-Peucker vertices; else the vertices themselves."""
    good = np.flatnonzero(~np.isnan(ys))
    if good.size < 2:
        return []
    cols = good + 0.5
    vx = np.array([x_cal.value(cols[v]) for v in _douglas_peucker(cols, ys[good], 2.0)])
    xa, xb, inner = vx[0], vx[-1], vx[1:-1]
    tv = sorted(v for v in (number(t) for t in sem.x_ticks) if v is not None)
    step0 = float(np.median(np.diff(tv))) if len(tv) >= 2 else (xb - xa) / 10
    for k in (1, 2, 2.5, 4, 5, 10, 20, 25, 50, 100):
        st = step0 / k
        if st <= 0 or (xb - xa) / st > 300:
            break
        start = round(xa / st) * st
        grid = start + st * np.arange(int(round((xb - start) / st)) + 1)
        if abs(start - xa) > 0.35 * st or abs(grid[-1] - xb) > 0.35 * st:
            continue
        if not len(inner) or np.mean([np.min(np.abs(grid - v)) <= 0.25 * st for v in inner]) >= 0.75:
            return [float(g) for g in grid]
    return [float(v) for v in vx]


# ---------------------------------------------------------------- bars

def _bar_edge(alpha: np.ndarray, comp: tuple[slice, slice], horizontal: bool) -> float:
    """Sub-pixel value edge of one bar, over its middle half: in a window running from 6 px inside the bar to past
    the edge, coverage sums to the covered length, so edge = window start + sum (hbar) or window end - sum."""
    rs, cs = comp
    if horizontal:
        n = rs.stop - rs.start
        mid = slice(rs.start + n // 4, rs.start + max(3 * n // 4, n // 4 + 1))
        s = max(cs.stop - 7, 0)
        return float(s + np.median(np.clip(alpha[mid, s:cs.stop + 6], 0, 1).sum(1)))
    n = cs.stop - cs.start
    mid = slice(cs.start + n // 4, cs.start + max(3 * n // 4, n // 4 + 1))
    e = rs.start + 7
    return float(e - np.median(np.clip(alpha[max(rs.start - 6, 0):e, mid], 0, 1).sum(0)))


def _bars(sc: Scene, sem: Semantics, tr: dict) -> tuple[Table, dict, dict]:
    """(table, confidence, slots): slots[series][category] = (centre px, thickness px) of the bar read, or None."""
    horizontal = sem.chart_type == "hbar"
    cal = tr["x_cal"] if horizontal else tr["y_cal"]
    cats = tr.get("categories") or [(c, None) for c in sem.categories]
    labels = [c for c, _ in cats]
    x0, y0, x1, y1 = sc.box
    base = None
    if cal is not None and cal.kind == "linear":
        z = cal.px(0.0)
        base = max(z, x0) if horizontal else min(z, y1)
    thickness = lambda s: (s[0].stop - s[0].start) if horizontal else (s[1].stop - s[1].start)
    comps = {}
    for name, m in sc.masks.items():
        m = ndimage.binary_opening(m, np.ones((3, 3)))  # thin bridges: a soft spine or edge between neighbours
        m = ndimage.binary_closing(m, np.ones((1, 5) if horizontal else (5, 1)))  # grid lines drawn over the bars
        lab, n = ndimage.label(m)
        found = []
        for sl in ndimage.find_objects(lab):
            rs, cs = sl
            hgt, wid = rs.stop - rs.start, cs.stop - cs.start
            thick, length = (hgt, wid) if horizontal else (wid, hgt)
            if thick < 3 or length < 2 or (lab[sl] > 0).mean() < 0.6:  # hollow: another bar's outline
                continue
            foot = cs.start if horizontal else rs.stop
            if base is not None and abs(foot - base) > 8 and abs(foot - (x0 if horizontal else y1)) > 3:
                continue  # not standing on the baseline: a legend swatch or a fragment
            found.append(sl)
        if found:  # bars of one series share a thickness
            th = np.median([thickness(s) for s in found])
            found = [s for s in found if thickness(s) >= 0.5 * th]
        comps[name] = found
    centre = lambda s: (s[0].start + s[0].stop) / 2 if horizontal else (s[1].start + s[1].stop) / 2
    pos = [p for _, p in cats]
    allc = np.sort([centre(s) for f in comps.values() for s in f])
    method = tr.get("diagnostics", {}).get("y" if horizontal else "x", {}).get("method")
    if method != "labels" and len(allc) >= len(labels) > 1:  # no category labels read: the bars group themselves
        cut = np.sort(np.argsort(np.diff(allc))[-(len(labels) - 1):])
        pos = [float(g.mean()) for g in np.split(allc, cut + 1)]
    values, conf, slots = {}, {}, {}
    for name, found in comps.items():
        vals, area, slot, hidden = [None] * len(labels), [0] * len(labels), [None] * len(labels), [False] * len(labels)
        for s in found:
            if not pos or pos[0] is None:
                break
            k = int(np.argmin([abs(centre(s) - p) for p in pos]))
            a = (s[0].stop - s[0].start) * (s[1].stop - s[1].start)
            if a > area[k]:  # two pieces in one slot: the bigger one is the bar
                vals[k] = cal.value(_bar_edge(sc.alpha[name], s, horizontal)) if cal else None
                area[k], slot[k] = a, (centre(s), thickness(s))
                mid = int(centre(s))  # the end touches the legend or the box edge: maybe cut off, not ended
                end = (mid, s[1].stop) if horizontal else (s[0].start - 2, mid)
                hidden[k] = not (0 <= end[0] < sc.region.shape[0] and 0 <= end[1] < sc.region.shape[1]) or not sc.region[end]
        values[name] = [None if v is None else float(v) for v in vals]
        conf[name] = [0.3 if h else 1.0 if a else 0.0 for a, h in zip(area, hidden)]
        slots[name] = slot
    return Table(labels, values), conf, slots


# ---------------------------------------------------------------- scatter

def _blobs(mask: np.ndarray) -> list[tuple[float, float, float]]:
    """(cx, cy, weight) per marker; blobs that are several markers merged are split by k-means."""
    from scipy.cluster.vq import kmeans2
    lab, n = ndimage.label(mask)
    if not n:
        return []
    areas = ndimage.sum(mask, lab, range(1, n + 1))
    keep = [i for i, a in enumerate(areas, 1) if a >= 4]
    if not keep:
        return []
    unit = float(np.median([areas[i - 1] for i in keep]))
    out = []
    for i in keep:
        yy, xx = np.nonzero(lab == i)
        k = int(round(areas[i - 1] / unit))
        if k >= 2:
            cents, _ = kmeans2(np.c_[xx, yy].astype(float), k, seed=0, minit="++")
            out += [(float(cx) + 0.5, float(cy) + 0.5, 0.5) for cx, cy in cents]
        else:
            out.append((float(xx.mean()) + 0.5, float(yy.mean()) + 0.5, 1.0))
    return out


# ---------------------------------------------------------------- public

def trace(img, sem: Semantics, ticks_result: dict) -> tuple[Table, dict[str, list[float]]]:
    """Reading B. Lines and scatter: x in data units; bars: x = category labels. Returns (table, per-point confidence
    0..1 per series aligned with table.x)."""
    sc = scene(img, sem, ticks_result)
    xc, yc = ticks_result.get("x_cal"), ticks_result.get("y_cal")
    if sem.chart_type in ("bar", "grouped_bar", "hbar"):
        return _bars(sc, sem, ticks_result)[:2]
    if xc is None or yc is None:
        return Table([], {s.name: [] for s in sem.series}), {s.name: [] for s in sem.series}
    if sem.chart_type == "scatter":
        pts = {n: [(xc.value(cx), yc.value(cy), w) for cx, cy, w in _blobs(m)] for n, m in sc.masks.items()}
        xs = sorted({round(p[0], 6) for v in pts.values() for p in v})
        series = {n: [next((p[1] for p in v if round(p[0], 6) == x), None) for x in xs] for n, v in pts.items()}
        conf = {n: [next((p[2] for p in v if round(p[0], 6) == x), 0.0) for x in xs] for n, v in pts.items()}
        return Table(xs, series), conf
    lines = {n: centreline(m, sc.box) for n, m in sc.masks.items()}
    xs = sorted({round(x, 9) for ys in lines.values() for x in _sample_xs(ys, xc, sem)})
    vals, conf = _at(lines, sem, xc, yc, xs)
    return Table(xs, vals), conf


def _at(lines, sem, xc, yc, xs):
    pxs = [xc.px(x) for x in xs]
    order = np.argsort(pxs)
    marker = {s.name: bool(s.marker) for s in sem.series}
    vals, conf = {}, {}
    for n, ys in lines.items():
        rows, cf = _line_values(ys, [pxs[i] for i in order], 6.0 if marker.get(n) else 3.0)
        back = np.empty(len(xs), int)
        back[order] = np.arange(len(xs))
        vals[n] = [None if rows[j] is None else float(yc.value(rows[j])) for j in back]
        conf[n] = [cf[j] for j in back]
    return vals, conf


def trace_at(img, sem: Semantics, ticks_result: dict, xs: list) -> dict[str, list[float | None]]:
    """Values of every series at given x: data values (line, scatter) or category labels (bars)."""
    if sem.chart_type in ("bar", "grouped_bar", "hbar"):
        t, _ = trace(img, sem, ticks_result)
        idx = {str(c).lower(): i for i, c in enumerate(t.x)}
        return {n: [v[idx[str(x).lower()]] if str(x).lower() in idx else None for x in xs] for n, v in t.series.items()}
    xc, yc = ticks_result.get("x_cal"), ticks_result.get("y_cal")
    if xc is None or yc is None:
        return {s.name: [None] * len(xs) for s in sem.series}
    sc = scene(img, sem, ticks_result)
    if sem.chart_type == "scatter":
        tol = 0.5 * float(np.median(np.diff(sorted(xs)))) if len(xs) > 1 else np.inf  # nearest marker within half a step
        out = {}
        for n, m in sc.masks.items():
            pts = [(xc.value(cx), yc.value(cy)) for cx, cy, _ in _blobs(m)]
            near = [min(((abs(px - x), py) for px, py in pts), default=(np.inf, None)) for x in xs]
            out[n] = [py if d <= tol else None for d, py in near]
        return out
    lines = {n: centreline(m, sc.box) for n, m in sc.masks.items()}
    return _at(lines, sem, xc, yc, [float(x) for x in xs])[0]
