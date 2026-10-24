"""facts(): known answers on a hand-made table, number formatting in the chart's own style, and a sweep over the
synthetic charts."""
import json
from pathlib import Path

import pytest

from raised.facts import facts, format_like
from raised.model import Semantics, Table

SYNTH = Path(__file__).resolve().parent.parent / "bench" / "data" / "synth"

X = list(range(2000, 2008))
TABLE = Table(X, {"Alpha": [10, 12, 15, 14, 13, 20, 22, 25], "Beta": [30, 28, 26, 27, 24, 22, 21, 17]})
LINE = Semantics("line", title="Output", y_title="Output", x_ticks=["2000", "2004", "2008"], y_ticks=["0", "10", "20", "30"],
                 series=[])


def by_id(fs):
    return {f["id"]: f for f in fs}


def test_line_known_answers():
    f = by_id(facts(TABLE, LINE))
    assert f["S1.max"]["values"] == {"series": "Alpha", "x": 2007, "value": 25}
    assert f["S1.min"]["values"]["x"] == 2000 and f["S1.min"]["values"]["value"] == 10
    assert f["S1.first"]["values"]["value"] == 10 and f["S1.last"]["values"]["value"] == 25
    ch = f["S1.change"]["values"]
    assert (ch["change"], ch["pct"], ch["direction"]) == (15, 150, "up")
    assert f["S1.change"]["text"] == "Alpha rose by 15 (150%) between 2000 and 2007."
    assert f["S2.change"]["values"]["direction"] == "down"
    assert f["S2.change"]["values"]["pct"] == pytest.approx(43.333, abs=1e-3)
    assert f["S2.max"]["values"]["x"] == 2000 and f["S2.min"]["values"]["x"] == 2007
    assert f["S1.step"]["values"] == {"series": "Alpha", "x_from": 2004, "x_to": 2005, "start": 13, "end": 20, "change": 7, "direction": "up"}
    assert f["S2.step"]["values"]["change"] == 4 and f["S2.step"]["values"]["x_to"] == 2007


def test_runs_are_maximal_and_at_least_three_points():
    f = by_id(facts(TABLE, LINE))
    runs = [(r["values"]["direction"], r["values"]["x_from"], r["values"]["x_to"]) for k, r in f.items() if k.startswith("S1.run")]
    assert runs == [("up", 2000, 2002), ("down", 2002, 2004), ("up", 2004, 2007)]
    # a run is a stretch of consecutive rises or falls: no run may contain a reversal
    for k, r in f.items():
        if ".run" in k:
            v = r["values"]
            assert (v["end"] > v["start"]) == (v["direction"] == "up")


def test_cross_series_comparison_at_last_x():
    g = by_id(facts(TABLE, LINE))["G.last_compare"]
    assert [o["series"] for o in g["values"]["order"]] == ["Alpha", "Beta"]
    assert g["values"]["gap"] == 8 and g["values"]["x"] == 2007
    assert g["text"] == "In 2007, Alpha is highest at 25, ahead of Beta at 17."


def test_ids_are_unique_and_stable():
    ids = [f["id"] for f in facts(TABLE, LINE)]
    assert len(ids) == len(set(ids)) and ids == [f["id"] for f in facts(TABLE, LINE)]
    assert all(f["kind"] and f["text"] and isinstance(f["values"], dict) for f in facts(TABLE, LINE))


def test_missing_values_are_skipped():
    t = Table([1, 2, 3, 4], {"A": [1, None, 3, 9]})
    f = by_id(facts(t, Semantics("line", y_ticks=["0", "10"])))
    assert f["S1.max"]["values"]["value"] == 9 and f["S1.first"]["values"]["x"] == 1


def test_nan_is_treated_as_unread():
    t = Table([1, 2, 3, 4], {"A": [1, float("nan"), 3, 9]})
    f = by_id(facts(t, Semantics("line", y_ticks=["0", "10"])))
    assert f["S1.max"]["values"]["value"] == 9 and f["S1.min"]["values"]["value"] == 1


def test_bar_charts_rank_and_have_no_trend_facts():
    t = Table(["Health", "Transport", "Housing", "Energy"], {"Emissions": [2752.4, 2330.2, 1943.0, 1976.3]})
    sem = Semantics("bar", y_ticks=["0", "1,000", "2,000"], categories=t.x)
    f = by_id(facts(t, sem))
    assert [o["x"] for o in f["S1.rank"]["values"]["order"]] == ["Health", "Transport", "Energy", "Housing"]
    assert f["S1.rank"]["text"] == "For Emissions, the highest is Health (2,752), then Transport (2,330), then Energy (1,976), and the lowest is Housing (1,943)."
    assert f["S1.max"]["text"] == "Emissions peaks at 2,752 for Health."
    assert not any(k.endswith(("first", "last", "change", "step")) for k in f)


def test_grouped_bars_count_who_leads():
    t = Table(["A", "B", "C"], {"Red": [5, 7, 2], "Blue": [4, 3, 9]})
    g = by_id(facts(t, Semantics("grouped_bar", y_ticks=["0", "5"])))["G.leaders"]
    assert g["values"]["counts"] == {"Red": 2, "Blue": 1} and g["values"]["total"] == 3
    assert g["text"] == "Across 3 categories, the highest series is Red in 2 and Blue in 1."


def test_year_categories_make_a_bar_chart_ordered():
    t = Table(["2019", "2020", "2021"], {"Sales": [10, 20, 30]})
    f = by_id(facts(t, Semantics("bar", y_ticks=["0", "10"])))
    assert f["S1.change"]["text"] == "Sales rose by 20 (200%) between 2019 and 2021."


def test_scatter_gets_extremes_only_and_unsupported_gets_nothing():
    t = Table([1.5, 2.5, 3.5], {"S": [3, 1, 2]})
    assert sorted(by_id(facts(t, Semantics("scatter", y_ticks=["0", "5"])))) == ["S1.max", "S1.min"]
    assert facts(t, Semantics("unsupported")) == []


@pytest.mark.parametrize("ticks,value,expected", [
    (["$5,000", "$10,000"], 3200, "$3,200"),
    (["50k", "55k", "60k"], 52300, "52.3k"),
    (["0", "2.5k", "5k", "7.5k"], 4000, "4.0k"),  # keeps the tick decimals
    (["20M", "30M", "40M"], 19417579, "19.4M"),
    (["0%", "10%", "20%"], 12.34, "12.3%"),
    (["47.5%", "50%", "52.5%"], 49.04, "49.0%"),
    (["0", "500", "1,000", "1,500"], 1234.6, "1,235"),
    (["1000", "2000", "3000"], 2400, "2400"),
    (["0", "5", "10"], -5.2, "-5.2"),
    (["0", "5", "10"], 0.04567, "0.046"),  # three decimals at most
    (["100", "1,000", "10,000"], 18.389, "18.4"),
])
def test_format_like_follows_the_axis_style(ticks, value, expected):
    assert format_like(value, Semantics("line", y_ticks=ticks)) == expected


def test_horizontal_bars_take_their_style_from_the_x_axis():
    sem = Semantics("hbar", x_ticks=["$0", "$2,000"], y_ticks=["a", "b"])
    assert format_like(1500, sem) == "$1,500"


@pytest.mark.skipif(not SYNTH.exists(), reason="synthetic charts not generated")
def test_every_synthetic_chart_yields_unique_ids_and_matching_series_counts():
    files = sorted(SYNTH.glob("*/*.json"))
    assert len(files) >= 300
    for p in files:
        d = json.loads(p.read_text())
        sem, table = Semantics.from_dict(d["semantics"]), Table.from_dict(d["table"])
        fs = facts(table, sem)
        ids = [f["id"] for f in fs]
        assert len(ids) == len(set(ids)), p
        assert {i.split(".")[0] for i in ids if i.startswith("S")} == {f"S{k}" for k in range(1, len(table.series) + 1)}, p
        assert all("—" not in f["text"] and "–" not in f["text"] for f in fs)
