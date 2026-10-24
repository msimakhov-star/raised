"""Metric tests. The first block is google-research/google-research deplot/metrics_test.py (Apache-2.0, same
cases and expected values) run against the port in bench/metrics.py; the rest covers our additions."""
import pytest

from bench import metrics
from raised.model import Semantics, Table

_TARGET = """title | my table
year | argentina | brazil
1999 | 200 | 158
"""

_TRANSPOSED_TARGET = """title | my table
country | 1999
argentina | 200
brazil | 158
"""

_PREDICTION = """title | my table
year | argentina | brazil
1999 | 202 | 0
"""

_MISALIGNED_PREDICTION = """title | my table
argentina | brazil | time
200 | 158 | 1999
"""

_TARGET_SCATTER = """x | y
10 | 20
30 | 40
50 | 60
"""

_PREDICTION_SCATTER = """y | x
40 | 30
"""


# ---- upstream cases ----

def test_get_table_datapoints():
    table = metrics._parse_table(_TARGET)
    assert metrics._get_table_datapoints(table) == {"title": "my table", "1999 argentina": "200", "1999 brazil": "158"}


def test_align_table():
    table, score = metrics._parse_table(_MISALIGNED_PREDICTION).aligned(("year", "argentina", "brazil"))
    assert table.headers == ("time", "argentina", "brazil")
    assert score == 1


def test_transposed_table():
    transposed = metrics._parse_table(_TRANSPOSED_TARGET, transposed=True)
    assert transposed.rows == metrics._parse_table(_TARGET).rows


def test_row_datapoints_precision_recall():
    # 1 comes from the matched title, 99 from the argentina entry; both tables transposed and read by columns
    result = metrics.row_datapoints_precision_recall([[_TARGET]], [_PREDICTION])
    assert result == {"row_datapoints_precision": 100 * 1.99 / 3, "row_datapoints_recall": 100 * 1.99 / 3,
                      "row_datapoints_f1": 100 * 1.99 / 3}


def test_row_datapoints_scatter():
    result = metrics.row_datapoints_precision_recall([[_TARGET_SCATTER]], [_PREDICTION_SCATTER])
    assert result["row_datapoints_precision"] == pytest.approx(100)
    assert result["row_datapoints_recall"] == pytest.approx(100.0 / 3)
    assert result["row_datapoints_f1"] == pytest.approx(100 / 2.0)


def test_get_datapoint_metric():
    assert metrics._get_datapoint_metric(("1999 argentina", "200"), ("1999 argentina", "202")) == 0.99


def test_table_datapoints_precision_recall():
    result = metrics.table_datapoints_precision_recall([[_TARGET]], [_PREDICTION])
    transposed = metrics.table_datapoints_precision_recall([[_TRANSPOSED_TARGET]], [_PREDICTION])
    assert result == {"table_datapoints_precision": 100 * 1.99 / 3, "table_datapoints_recall": 100 * 1.99 / 3,
                      "table_datapoints_f1": 100 * 1.99 / 3}
    assert transposed == result


def test_table_numbers_match():
    assert metrics._table_numbers_match("0.5 | foo | 0.6", "0.5 | foo | 0.6") == 1
    # 1 - relative_distance(0.4, 0.5) / 2; the 0.6 is ignored
    assert metrics._table_numbers_match("baz | foo | 0.4", "0.5 | foo | 0.6") == 0.875


# ---- our additions ----

def test_levenshtein_and_anls():
    assert metrics._levenshtein("kitten", "sitting") == 3
    assert metrics.anls_metric("", "") == 1.0                    # upstream would divide by zero
    assert metrics.anls_metric("brazil", "brasil") == pytest.approx(1 - 1 / 6)
    assert metrics.anls_metric("abc", "xyz") == 0.0               # distance 1.0 >= theta
    assert metrics.anls_metric("a", "abcdefgh") == 0.0            # length gap alone exceeds theta


T = Table(["2018", "2019", "2020"], {"Brazil": [10.0, 20.0, 30.0], "Chile": [1.0, None, 3.0]})


def test_deplot_roundtrip():
    text = metrics.to_deplot(T, x_header="year")
    assert text.splitlines()[:2] == ["year | Brazil | Chile", "2018 | 10 | 1"]
    assert text.splitlines()[2] == "2019 | 20 | "
    back = metrics.from_deplot(text)
    assert back.x == T.x and back.series == T.series
    assert metrics.to_deplot(Table([2018.0], {"s": [6.12]})) == "x | s\n2018 | 6.12"
    assert metrics.to_deplot(T, title="t").startswith("title | t\n")


def test_rms_f1_perfect_and_degraded():
    assert metrics.rms_f1(T, T)[2] == 1.0
    assert metrics.rms_f1(T, None) == (0.0, 0.0, 0.0)
    off = Table(T.x, {"Brazil": [10.0, 20.0, 31.0], "Chile": [1.0, None, 3.0]})   # 3% off stays partial credit
    assert 0.9 < metrics.rms_f1(T, off)[2] < 1.0
    far = Table(T.x, {"Brazil": [10.0, 20.0, 60.0], "Chile": [1.0, None, 3.0]})   # 100% off scores 0 on that cell
    assert metrics.rms_f1(T, far)[2] < metrics.rms_f1(T, off)[2]
    assert metrics.rms_f1(T, Table(T.x, {"Other": [10.0, 20.0, 30.0]}))[2] < 0.7  # wrong series name is penalised


def test_rms_f1_transposed_prediction():
    transposed = Table(["Brazil", "Chile"], {"2018": [10.0, 1.0], "2019": [20.0, None], "2020": [30.0, 3.0]})
    assert metrics.rms_f1(T, transposed)[2] == pytest.approx(1.0)


def test_align_by_name_x_and_position():
    pred = Table([2020.0, 2018.0, 2019.0], {"chile": [3.0, 1.0, None], "BRAZIL": [30.0, 10.0, 20.0]})
    got = metrics.align(T, pred)
    assert got == {"Brazil": [10.0, 20.0, 30.0], "Chile": [1.0, None, 3.0]}
    renamed = Table(["a", "b", "c"], {"s1": [10.0, 20.0, 30.0], "s2": [1.0, None, 3.0]})  # order, then position
    assert metrics.align(T, renamed) == {"Brazil": [10.0, 20.0, 30.0], "Chile": [1.0, None, 3.0]}
    near = Table([2018.4, 2019.1], {"Brazil": [11.0, 22.0]})                    # nearest x within half spacing
    assert metrics.align(T, near)["Brazil"] == [11.0, 22.0, None]


def test_numeric_metrics():
    rng = metrics.axis_range(T, "line", ["0", "10", "20", "30", "40"])
    assert rng == (0.0, 40.0)
    pred = Table(T.x, {"Brazil": [10.0, 21.0, None], "Chile": [1.0, None, 3.0]})  # one 2.5% miss, one missing
    errs = metrics.point_errors(T, pred, rng)
    assert errs == [0.0, 1 / 40, None, 0.0, 0.0] or sorted(e for e in errs if e is not None) == [0.0, 0.0, 0.0, 1 / 40]
    s = metrics.summarize_errors(errs)
    assert s["n_values"] == 5 and s["missing"] == pytest.approx(0.2)
    assert s["within_2pct"] == pytest.approx(3 / 5) and s["median_err"] == pytest.approx(0.0)
    assert metrics.summarize_errors(metrics.point_errors(T, None, rng))["missing"] == 1.0


def test_axis_range_defaults_and_log():
    bars = Table(["a", "b"], {"v": [5.0, 9.0]})
    assert metrics.axis_range(bars, "bar") == (0.0, 9.0)           # bars start at zero
    assert metrics.axis_range(bars, "line") == (5.0, 9.0)
    lo, hi = metrics.axis_range(bars, "line", ["1", "10", "100"], log=True)
    assert (lo, hi) == (0.0, 2.0)
    errs = metrics.point_errors(Table(["a"], {"v": [10.0]}), Table(["a"], {"v": [100.0]}), (lo, hi), log=True)
    assert errs == [0.5]                                           # one decade of two


def test_catch_stats():
    recs = [{"fault": "shift", "flagged": True}, {"fault": "shift", "flagged": False},
            {"fault": "drop_point", "flagged": True}, {"fault": None, "flagged": False},
            {"fault": None, "flagged": True}]
    s = metrics.catch_stats(recs)
    assert s["catch_rate"] == pytest.approx(2 / 3) and s["catch_by_kind"] == {"shift": 0.5, "drop_point": 1.0}
    assert s["false_alarm_rate"] == 0.5 and (s["n_faulty"], s["n_clean"]) == (3, 2)


def test_semantics_accuracy():
    truth = Semantics.from_dict({"chart_type": "hbar", "x_scale": "log", "series": [{"name": "a"}]})
    good = Semantics.from_dict({"chart_type": "hbar", "x_scale": "log", "series": [{"name": "z"}]})
    assert metrics.semantics_accuracy(truth, good) == {"chart_type": True, "n_series": True, "scale": True}
    bad = Semantics.from_dict({"chart_type": "bar", "x_scale": "linear", "series": []})
    assert not any(metrics.semantics_accuracy(truth, bad).values())
    assert not any(metrics.semantics_accuracy(truth, None).values())
    quirk = Semantics.from_dict({"chart_type": "hbar", "x_scale": "category", "y_scale": "linear"})  # generator's hbar labels
    assert metrics.semantics_accuracy(quirk, Semantics.from_dict({"chart_type": "hbar", "x_scale": "linear", "y_scale": "category"}))["scale"]
