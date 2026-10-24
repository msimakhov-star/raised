import copy

from fake_tf import SEMANTICS_VOTE, FakeTF, images

from raised import prompts
from raised.llm import conforms
from raised.semantics import merge_votes, read_semantics

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16  # never decoded: shrink returns unreadable bytes unchanged


def _dissent():
    v = copy.deepcopy(SEMANTICS_VOTE)
    v["title"] = "Exports & imports"
    v["y_ticks"] = ["0", "5", "10", "15"]
    v["series"][0]["color"] = "dark blue"
    v["series"].append({"name": "Balance", "color": "green", "style": "dotted", "marker": ""})
    v["reading"]["series"][0]["values"] = [2.4, 9.0, 8.1]
    v["reading"]["series"][1]["values"] = [3.2, 4.4, 5.0]
    return v


def _agreeing(values):
    v = copy.deepcopy(SEMANTICS_VOTE)
    v["series"][0]["name"] = "exports"  # names match case-insensitively
    v["reading"]["series"][0]["name"] = "EXPORTS"
    v["reading"]["series"][0]["values"] = values
    return v


def test_three_votes_one_dissenting():
    votes = [SEMANTICS_VOTE, _agreeing([2.2, 5.0, 7.5]), _dissent()]
    assert all(conforms(v, prompts.SEMANTICS_SCHEMA) for v in votes)
    with FakeTF({"TASK: CHART_SEMANTICS": votes}) as tf:
        sem, reading, agree = read_semantics(tf.client(), PNG, votes=3)
    assert sem.chart_type == "line" and sem.title == "Exports and imports" and sem.y_ticks == ["0", "5", "10"]
    assert agree["chart_type"] == 1 and agree["title"] == agree["y_ticks"] == agree["series"] == 2 / 3
    assert [s.name for s in sem.series] == ["Exports", "Imports"]  # "Balance" is in one vote of three
    assert sem.series[0].color == "blue" and sem.series[1].style == "dashed"
    assert reading.x == [2000.0, 2005.0, 2010.0]
    assert reading.series["Exports"] == [2.2, 5.5, 7.9]  # per-point medians
    assert reading.series["Imports"] == [3.0, 4.0, 5.0]  # null readings are skipped, not counted
    body = tf.requests[-1]
    assert len(images(body)) == 1 and "TASK: CHART_SEMANTICS" in body["messages"][0]["content"][1]["text"]


def test_single_vote_is_one_call_at_temperature_zero():
    with FakeTF() as tf:
        sem, reading, agree = read_semantics(tf.client(), PNG, votes=1)
    assert [r["temperature"] for r in tf.requests] == [0.0, 0.0]  # wrapper rejected, then the bare schema
    assert agree["x_ticks"] == 1 and reading.series["Imports"] == [3.0, 4.0, None]


def test_bar_votes_align_by_category_and_hbar_keeps_scales():
    def bar(values, order):
        return {**SEMANTICS_VOTE, "chart_type": "hbar", "x_scale": "category", "y_scale": "linear",
                "x_ticks": ["0", "50", "100"], "y_ticks": ["UK", "France"], "categories": ["France", "UK"],
                "series": [{"name": "Share", "color": "grey", "style": "solid", "marker": ""}], "legend": [],
                "reading": {"x": order, "series": [{"name": "Share", "values": values}]}}
    sem, reading, _ = merge_votes([bar([40, 60], ["France", "UK"]), bar([62, 41], ["uk", "france"]), bar([44, 58], ["France", "UK"])])
    assert sem.value_axis == "x" and sem.x_scale == "category"
    assert reading.x == ["France", "UK"] and reading.series == {"Share": [41, 60]}


def test_unsupported_chart_has_no_reading_and_odd_values_are_normalised():
    pie = {**SEMANTICS_VOTE, "chart_type": "pie", "x_scale": "radial"}
    sem, reading, _ = merge_votes([pie])
    assert sem.chart_type == "unsupported" and sem.x_scale == "linear" and reading.series == {}


def test_reading_without_series_names_the_semantics_after_it():
    v = {**SEMANTICS_VOTE, "series": [], "reading": {"x": ["1", "2"], "series": [{"name": "value", "values": ["40%", 50]}]}}
    sem, reading, _ = merge_votes([v])
    assert [s.name for s in sem.series] == ["value"] and reading.series == {"value": [40.0, 50.0]}
