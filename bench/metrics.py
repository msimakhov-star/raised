"""Table metrics for the benchmark.

1. RMS-F1 (Relative Mapping Similarity), the table metric of DePlot (Liu et al. 2022, arXiv 2212.10505, section 3.1:
   a table is an unordered set of (row header, column header, value) triples; keys are matched by Hungarian
   assignment on normalised Levenshtein distance (tau = 0.5), values are compared by relative distance
   (theta = 0.1); precision and recall come from the summed similarity of matched pairs; transposed predictions
   are tried and the best F1 wins).

   PORTED from Apache-2.0 code, algorithm unchanged:
     google-research/google-research  deplot/metrics.py       (commit a6b3465, Copyright The Google Research Authors)
     google-research/pix2struct       pix2struct/metrics.py   (commit 6fe25c1, Copyright The pix2struct Authors;
                                                               only anls_metric is needed)
   Deviations from the originals: (a) Levenshtein is a small pure-Python function (the original calls the C
   package `editdistance`, which we do not install); (b) anls_metric of two empty strings is 1 (the original
   divides by zero); (c) the `Ellipsis` type annotations in the upstream file are written `...`.
   Note: the DePlot paper reports RMS-F1 for plot-to-table only on PlotQA (Table 4, DePlot 94.2), not on ChartQA.

2. Numeric metrics on aligned tables: error as a share of the value-axis range, share within 2%, missing rate.
3. Converters between raised.model.Table and the DePlot linearised text ("header | s1 | s2" then "x | v | v" lines).
4. Verifier catch statistics and semantics accuracy for the perturbation set and the synthetic split.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import statistics
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import optimize

from raised.model import Semantics, Table, parse_number

# ---------------------------------------------------------------- pix2struct string similarity


@lru_cache(maxsize=200_000)
def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls_metric(target: str, prediction: str, theta: float = 0.5) -> float:
    """Average normalised Levenshtein similarity with a cut-off (pix2struct metrics.anls_metric)."""
    if target == prediction:
        return 1.0
    longest = max(len(target), len(prediction))
    if abs(len(target) - len(prediction)) / longest >= theta:  # the distance is at least the length gap
        return 0.0
    nld = _levenshtein(target, prediction) / longest
    return 1 - nld if nld < theta else 0.0


# ---------------------------------------------------------------- deplot/metrics.py


def _to_float(text):
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0  # percentages become fractions
        return float(text)
    except ValueError:
        return None


def _get_relative_distance(target, prediction, theta=1.0):
    """Returns min(1, |target-prediction|/|target|), set to 1 above theta."""
    if not target:
        return int(not prediction)
    distance = min(abs((target - prediction) / target), 1)
    return distance if distance < theta else 1


def _table_numbers_match(target, prediction):
    """Matching similarity between the numbers of two tables, following ChartQA (the RNSS of the paper)."""
    target_numbers = _get_table_numbers(target)
    prediction_numbers = _get_table_numbers(prediction)
    if not target_numbers and not prediction_numbers:
        return 1
    if not target_numbers or not prediction_numbers:
        return 0
    max_len = max(len(target_numbers), len(prediction_numbers))
    cost_matrix = np.array([[_get_relative_distance(t, p) for p in prediction_numbers] for t in target_numbers])
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    return 1 - cost_matrix[row_ind, col_ind].sum() / max_len


def _get_table_numbers(text):
    numbers = []
    for line in text.splitlines():
        for part in line.split(" | "):
            if part.strip():
                try:
                    numbers.append(float(part))
                except ValueError:
                    pass
    return numbers


def _permute(values, indexes):
    return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class DTable:
    """The content of a linearised table (upstream name: Table)."""

    title: Optional[str] = None
    headers: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    rows: tuple[tuple[str, ...], ...] = dataclasses.field(default_factory=tuple)

    def permuted(self, indexes):
        return DTable(self.title, _permute(self.headers, indexes), tuple(_permute(r, indexes) for r in self.rows))

    def aligned(self, headers, text_theta=0.5):
        """Column permutation that puts this table's headers in the order of `headers`."""
        if len(headers) != len(self.headers):
            raise ValueError(f"Header length {headers} must match {self.headers}.")
        cost_matrix = np.array([[1 - anls_metric(h1, h2, text_theta) for h1 in headers] for h2 in self.headers])
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
        score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
        return self.permuted(permutation), score


def _parse_table(text, transposed=False):
    """Builds a table from its linearised representation."""
    lines = text.lower().splitlines()
    if not lines:
        return DTable()
    if lines[0].startswith("title |"):
        title = lines[0][len("title |"):].strip()
        offset = 1
    else:
        title = None
        offset = 0
    if len(lines) < offset + 1:
        return DTable(title=title)
    rows = [tuple(v.strip() for v in line.split(" | ")) for line in lines[offset:]]
    if transposed:
        rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]
    return DTable(title=title, headers=rows[0], rows=tuple(rows[1:]))


def _get_table_datapoints(table):
    """{"row header column header": value} for every cell, plus the title."""
    datapoints = {}
    if table.title is not None:
        datapoints["title"] = table.title
    if not table.rows or len(table.headers) <= 1:
        return datapoints
    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            datapoints[f"{row[0]} {header}"] = cell
    return datapoints


def _get_datapoint_metric(target, prediction, text_theta=0.5, number_theta=0.1):
    """Similarity of two (key, value) pairs: key similarity times (1 - relative error)."""
    key_metric = anls_metric(target[0], prediction[0], text_theta)
    pred_float = _to_float(prediction[1])
    target_float = _to_float(target[1])
    if pred_float is not None and target_float:
        return key_metric * (1 - _get_relative_distance(target_float, pred_float, number_theta))
    if target[1] == prediction[1]:
        return key_metric
    return key_metric * anls_metric(target[1], prediction[1], text_theta)


def _table_datapoints_precision_recall_f1(target_table, prediction_table, text_theta=0.5, number_theta=0.1):
    target_datapoints = list(_get_table_datapoints(target_table).items())
    prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
    if not target_datapoints and not prediction_datapoints:
        return 1, 1, 1
    if not target_datapoints:
        return 0, 1, 0
    if not prediction_datapoints:
        return 1, 0, 0
    cost_matrix = np.array([[1 - anls_metric(t, p, text_theta) for p, _ in prediction_datapoints]
                            for t, _ in target_datapoints])
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    score = 0
    for r, c in zip(row_ind, col_ind):
        score += _get_datapoint_metric(target_datapoints[r], prediction_datapoints[c], text_theta, number_theta)
    if score == 0:
        return 0, 0, 0
    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    return precision, recall, 2 * precision * recall / (precision + recall)


def table_datapoints_precision_recall_per_point(targets, predictions, text_theta=0.5, number_theta=0.1):
    """RMS precision, recall and F1 per example. targets: list of list of strings (several accepted targets),
    predictions: list of strings. Returns {"precision": [...], "recall": [...], "f1": [...]} in [0, 1]."""
    assert len(targets) == len(predictions)
    per_point = {"precision": [], "recall": [], "f1": []}
    for pred, target in zip(predictions, targets):
        all_metrics = []
        for transposed in [True, False]:
            pred_table = _parse_table(pred, transposed=transposed)
            all_metrics.extend(_table_datapoints_precision_recall_f1(_parse_table(t), pred_table, text_theta,
                                                                     number_theta) for t in target)
        p, r, f = max(all_metrics, key=lambda x: x[-1])
        per_point["precision"].append(p)
        per_point["recall"].append(r)
        per_point["f1"].append(f)
    return per_point


def table_datapoints_precision_recall(targets, predictions, text_theta=0.5, number_theta=0.1):
    """Aggregated RMS precision, recall and F1, in percent."""
    s = table_datapoints_precision_recall_per_point(targets, predictions, text_theta, number_theta)
    n = len(targets)
    return {"table_datapoints_precision": 100.0 * sum(s["precision"]) / n,
            "table_datapoints_recall": 100.0 * sum(s["recall"]) / n,
            "table_datapoints_f1": 100.0 * sum(s["f1"]) / n}


def _get_row_datapoints(table):
    if table.title is None:
        return table.rows
    return table.rows + (("title", table.title),)


def _get_row_metric(target_parts, prediction_parts, text_theta=0.5, number_theta=0.1):
    if len(target_parts) != len(prediction_parts) or not target_parts:
        return 0.0
    result = []
    for target, prediction in zip(target_parts, prediction_parts):
        pred_float = _to_float(prediction)
        target_float = _to_float(target)
        if target == prediction:
            result.append(1.0)
        elif pred_float is not None and target_float:
            result.append(1 - _get_relative_distance(target_float, pred_float, number_theta))
        elif target_float is not None:
            result.append(0.0)
        else:
            result.append(anls_metric(target, prediction, text_theta))
    return np.prod(result)


def _row_datapoints_precision_recall_f1(target, prediction, text_theta=0.5, number_theta=0.1):
    target_datapoints = _get_row_datapoints(target)
    aligned_prediction, aligned_score = prediction.aligned(target.headers, text_theta)
    prediction_datapoints = _get_row_datapoints(aligned_prediction)
    if not target_datapoints and not prediction_datapoints:
        return 1, 1, 1
    if not target_datapoints:
        return 0, 1, 0
    if not prediction_datapoints or not aligned_score:
        return 1, 0, 0
    metrics_matrix = np.array([[aligned_score * _get_row_metric(t, p, text_theta, number_theta)
                                for p in prediction_datapoints] for t in target_datapoints])
    row_ind, col_ind = optimize.linear_sum_assignment(1 - metrics_matrix)
    score = metrics_matrix[row_ind, col_ind].sum()
    if score == 0:
        return 0, 0, 0
    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    return precision, recall, 2 * precision * recall / (precision + recall)


def row_datapoints_precision_recall(targets, predictions, text_theta=0.5, number_theta=0.1):
    """Row-wise variant of the metric (kept for fidelity with upstream; the benchmark headline is RMS-F1 above)."""
    if len(targets) != len(predictions):
        raise ValueError(f"Targets has length {len(targets)} and predictions has length {len(predictions)}.")
    precision, recall, f1 = 0, 0, 0
    for pred, target in zip(predictions, targets):
        all_metrics = []
        prediction_tables = [_parse_table(pred, transposed=t) for t in [True, False]]
        for t in target:
            for target_transposed in [True, False]:
                target_table = _parse_table(t, transposed=target_transposed)
                for prediction_table in prediction_tables:
                    if len(target_table.headers) != len(prediction_table.headers):
                        continue
                    all_metrics.append(_row_datapoints_precision_recall_f1(target_table, prediction_table,
                                                                           text_theta, number_theta))
        p, r, f = max(all_metrics, key=lambda x: x[-1], default=(0, 0, 0))
        precision += p
        recall += r
        f1 += f
    return {"row_datapoints_precision": 100.0 * precision / len(targets),
            "row_datapoints_recall": 100.0 * recall / len(targets),
            "row_datapoints_f1": 100.0 * f1 / len(targets)}


# ---------------------------------------------------------------- Table <-> DePlot text


def _fmt(v) -> str:
    """Canonical text of a cell: None -> '', 2017.0 -> '2017', 6.12 -> '6.12', strings unchanged."""
    if v is None:
        return ""
    if isinstance(v, str):
        return " ".join(v.split())
    f = float(v)
    if math.isnan(f):
        return ""
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:.6g}"


def to_deplot(table: Table, x_header: str = "", title: str | None = None) -> str:
    """'x_header | s1 | s2' then one 'x | v | v' line per row; None cells are empty. Title line only if given."""
    lines = [f"title | {title}"] if title else []
    lines.append(" | ".join([x_header or "x", *map(_fmt, table.series)]))
    cols = list(table.series.values())
    for i, x in enumerate(table.x):
        lines.append(" | ".join([_fmt(x), *(_fmt(c[i]) if i < len(c) else "" for c in cols)]))
    return "\n".join(lines)


def from_deplot(text: str) -> Table:
    """Inverse of to_deplot (title line skipped). Cells go through raised.model.parse_number, so '40%' is 40
    (RMS-F1 itself reads 40% as 0.4; both sides pass through the same Table, so scores stay comparable)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("title |"):
        lines = lines[1:]
    if not lines:
        return Table([], {})
    names = [c.strip() for c in lines[0].split(" | ")][1:]
    rows = [[c.strip() for c in ln.split(" | ")] for ln in lines[1:]]
    series = {n: [parse_number(r[j + 1]) if j + 1 < len(r) else None for r in rows] for j, n in enumerate(names)}
    return Table([r[0] for r in rows], series)


def rms_f1(truth: Table, pred: Table | None, x_header: str = "") -> tuple[float, float, float]:
    """(precision, recall, F1) in [0, 1] between two Tables; a failed reading (None) scores 0."""
    if pred is None:
        return 0.0, 0.0, 0.0
    scores = table_datapoints_precision_recall_per_point([[to_deplot(truth, x_header)]], [to_deplot(pred)])
    return scores["precision"][0], scores["recall"][0], scores["f1"][0]


# ---------------------------------------------------------------- numeric metrics on aligned tables


def _key(x):
    """Alignment key of an x entry: its number if it has one, else the lower-cased text."""
    n = parse_number(x) if isinstance(x, str) else x
    return float(n) if isinstance(n, (int, float)) else str(x).strip().lower()


def _norm(name) -> str:
    return " ".join(str(name).lower().split())


def align(truth: Table, pred: Table) -> dict[str, list[float | None]]:
    """Predicted values laid out on the truth grid: {truth series name: [value or None per truth row]}.
    Series match by normalised name, leftovers by order; rows by x key, else the nearest numeric x within half the
    median truth spacing, else by position when both tables have the same length."""
    pnames = {_norm(n): n for n in pred.series}
    tnames = list(truth.series)
    pairs = {t: pnames.pop(_norm(t)) for t in tnames if _norm(t) in pnames}
    spare = iter(pnames.values())
    for t in tnames:
        if t not in pairs and (p := next(spare, None)) is not None:
            pairs[t] = p
    pk = [_key(x) for x in pred.x]
    tk = [_key(x) for x in truth.x]
    nums = sorted(k for k in tk if isinstance(k, float))
    tol = statistics.median(b - a for a, b in zip(nums, nums[1:])) / 2 if len(nums) > 1 else 0.0
    index: list[int | None] = []
    for i, k in enumerate(tk):
        if k in pk:
            index.append(pk.index(k))
        elif isinstance(k, float) and (near := [j for j, q in enumerate(pk) if isinstance(q, float) and abs(q - k) <= tol]):
            index.append(min(near, key=lambda j: abs(pk[j] - k)))
        else:
            index.append(i if len(pk) == len(tk) else None)
    out = {}
    for t in tnames:
        col = pred.series.get(pairs.get(t, ""), [])
        out[t] = [col[j] if j is not None and j < len(col) else None for j in index]
    return out


def is_log(sem: Semantics) -> bool:
    """Log value axis? Both axes are checked because the generator files horizontal bars' scale on the category axis."""
    return "log" in (sem.x_scale, sem.y_scale)


def value_ticks(sem: Semantics) -> list[str]:
    return sem.x_ticks if sem.value_axis == "x" else sem.y_ticks


def axis_range(table: Table, chart_type: str = "line", ticks: list[str] | None = None, log: bool = False):
    """(lo, hi) of the value axis: the outermost numeric tick labels when known, else the data extent (bars start at
    zero). Log axes return log10 bounds."""
    nums = [n for n in map(parse_number, ticks or []) if n is not None and (n > 0 or not log)]
    vals = [v for col in table.series.values() for v in col if v is not None]
    if len(nums) >= 2:
        lo, hi = min(nums), max(nums)
    elif vals:
        lo, hi = min(vals), max(vals)
        if chart_type in ("bar", "grouped_bar", "hbar"):
            lo = min(lo, 0.0)
    else:
        lo, hi = 0.0, 1.0
    if log:
        lo, hi = math.log10(max(lo, 1e-12)), math.log10(max(hi, 1e-12))
    return lo, (hi if hi > lo else lo + 1.0)


def point_errors(truth: Table, pred: Table | None, rng: tuple[float, float], log: bool = False) -> list[float | None]:
    """|pred - truth| / axis range for every truth value; None where the prediction is missing."""
    lo, hi = rng
    aligned = align(truth, pred) if pred else {t: [None] * len(truth.x) for t in truth.series}
    errs: list[float | None] = []
    for name, col in truth.series.items():
        for t, p in zip(col, aligned[name]):
            if t is None:
                continue
            if p is None or (log and (p <= 0 or t <= 0)):
                errs.append(None)
            else:
                errs.append(abs((math.log10(p) - math.log10(t)) if log else (p - t)) / (hi - lo))
    return errs


def summarize_errors(errs: list[float | None]) -> dict:
    """Median error (share of axis range, over values that were read), share within 2% (missing counts as a miss),
    missing rate."""
    got = [e for e in errs if e is not None]
    n = len(errs)
    return {"n_values": n,
            "median_err": statistics.median(got) if got else None,
            "within_2pct": sum(e <= 0.02 for e in got) / n if n else None,
            "missing": (n - len(got)) / n if n else None}


# ---------------------------------------------------------------- verifier and semantics


def catch_stats(records: list[dict]) -> dict:
    """records: {"fault": kind or None (clean), "flagged": bool}. Catch rate per fault kind and overall, and the
    false-alarm rate on clean extractions."""
    by = defaultdict(list)
    for r in records:
        by[r["fault"]].append(bool(r["flagged"]))
    faulty = [f for k, v in by.items() if k for f in v]
    return {"catch_rate": sum(faulty) / len(faulty) if faulty else None,
            "catch_by_kind": {k: sum(v) / len(v) for k, v in by.items() if k},
            "false_alarm_rate": sum(by[None]) / len(by[None]) if by.get(None) else None,
            "n_faulty": len(faulty), "n_clean": len(by.get(None, []))}


def semantics_accuracy(truth: Semantics, pred: Semantics | None) -> dict[str, bool]:
    """Chart type, series count and log-or-not of the value axis against the generator's truth."""
    if pred is None:
        return {"chart_type": False, "n_series": False, "scale": False}
    return {"chart_type": pred.chart_type == truth.chart_type,
            "n_series": len(pred.series) == len(truth.series),
            "scale": is_log(pred) == is_log(truth)}
