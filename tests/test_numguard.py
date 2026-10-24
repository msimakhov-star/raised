"""numguard: planted wrong numbers are caught, correctly formatted ones (separators, %, $, k, M, words, years,
ordinals) are accepted, and whole sentences are stripped."""
import json
import random
from pathlib import Path

import pytest

from raised.facts import facts, format_like
from raised.model import Semantics, Table
from raised.numguard import check, mentions, sentences, strip_unsupported

SYNTH = Path(__file__).resolve().parent.parent / "bench" / "data" / "synth"

TABLE = Table(list(range(2000, 2008)), {"Alpha": [10, 12, 15, 14, 13, 20, 22, 25], "Beta": [30, 28, 26, 27, 24, 22, 21, 17]})
SEM = Semantics("line", y_ticks=["0", "10", "20", "30"], x_ticks=["2000", "2004", "2008"], x_title="Year", y_title="Output")
FACTS = facts(TABLE, SEM)


def bad(text, table=TABLE, fs=FACTS, sem=SEM, **kw):
    return [o["text"] for o in check(text, table, fs, sem, **kw)]


def test_correct_numbers_pass():
    assert bad("Alpha rose from 10 in 2000 to 25 in 2007, a rise of 15, or 150%.") == []
    assert bad("Beta fell by 43.3% from 30 to 17, its largest step being 4 between 2006 and 2007.") == []
    assert bad("In 2007 Alpha is 8 ahead of Beta.") == []          # the gap in the comparison fact
    assert bad("Alpha's peak of 25 is 15 above its low of 10.") == []  # difference between two facts of one series


def test_planted_wrong_numbers_are_caught_with_exact_spans():
    text = "Alpha rose from 10 to 31 in 2007, up 80%."
    got = check(text, TABLE, FACTS, SEM)
    assert [o["text"] for o in got] == ["31", "80%"]
    for o in got:
        assert text[o["start"]:o["end"]] == o["text"] and o["sentence"] == text
    assert [o["unit"] for o in got] == ["plain", "pct"]


def test_a_year_must_be_an_x_value_or_tick():
    assert bad("It peaked in 2007.") == [] and bad("From 2000 to 2007.") == [] and bad("Ticks run to 2008.") == []
    assert bad("It peaked in 2012.") == ["2012"] and bad("Back in 1999 it was lower.") == ["1999"]


def test_a_percentage_is_not_satisfied_by_a_plain_value():
    assert bad("Alpha reaches 25.") == []
    assert bad("Alpha reaches 25%.") == ["25%"]  # 25 is a table value, 25% is not a fact (150% and 43.3% are)
    assert bad("Alpha reaches 25 percent.") == ["25 percent"]


def test_percent_axes_accept_their_own_values():
    t = Table([1990, 2000], {"Rate": [12.5, 30.0]})
    sem = Semantics("line", y_ticks=["0%", "10%", "20%", "30%"])
    assert bad("The rate was 12.5% in 1990 and 30% in 2000, up 17.5 percentage points.", t, facts(t, sem), sem) == []
    assert bad("The rate was 14% in 1990.", t, facts(t, sem), sem) == ["14%"]


@pytest.mark.parametrize("text", [
    "It reached $3.4k.", "It reached $3,400.", "It reached 3,400.", "It reached 3400.", "It reached 3.4 thousand.",
    "It reached about 3,420.", "It reached 3.4k.", "It reached $3.4 thousand.", "It reached roughly $3.41k.",
])
def test_formatted_variants_of_one_value_pass(text):
    t = Table([2000, 2001, 2002], {"Sales": [3400, 2900, 3100]})
    sem = Semantics("line", y_ticks=["$2k", "$3k", "$4k"])
    assert bad(text, t, facts(t, sem), sem) == []


@pytest.mark.parametrize("text", ["It reached $3.9k.", "It reached 3,700.", "It reached 3.45k.", "It reached 3,420."])
def test_unhedged_numbers_must_match_to_the_precision_they_print(text):
    t = Table([2000, 2001, 2002], {"Sales": [3400, 2900, 3100]})
    sem = Semantics("line", y_ticks=["$2k", "$3k", "$4k"])
    assert bad(text, t, facts(t, sem), sem) != []


def test_millions_words_and_axis_units():
    t = Table([2000, 2001, 2002], {"Pop": [19_417_579, 22_737_009, 20_774_666]})
    sem = Semantics("line", y_ticks=["20M", "25M"], y_title="Population (M)")
    fs = facts(t, sem)
    assert bad("Population peaked at 22.7 million in 2001.", t, fs, sem) == []
    assert bad("It peaked at 22.7M, from 19.4M.", t, fs, sem) == []
    assert bad("It peaked at 22,737,009.", t, fs, sem) == []
    assert bad("It peaked at 22.7 in the axis unit.", t, fs, sem) == []
    assert bad("It peaked at 24.7 million.", t, fs, sem) == ["24.7 million"]
    assert bad("It peaked at forty million.", t, fs, sem) == ["forty million"]


def test_a_table_kept_in_axis_units_still_matches_scaled_text():
    t = Table([2000, 2001], {"Pop": [20.0, 30.0]})
    sem = Semantics("line", y_ticks=["10", "20", "30"])
    assert bad("It reached 30 million.", t, facts(t, sem), sem) == []


def test_hedged_numbers_get_the_relative_tolerance():
    t = Table([2000, 2001, 2002], {"Sales": [4464, 3000, 3100]})
    sem = Semantics("line", y_ticks=["0", "5,000"])
    fs = facts(t, sem)
    assert bad("About 4,470 at the start.", t, fs, sem) == []
    assert bad("About 4,700 at the start.", t, fs, sem) == ["4,700"]
    assert bad("It was 4,470 at the start.", t, fs, sem) == ["4,470"]
    assert bad("It was 4,470 at the start.", t, fs, sem, tolerance=0.0) == ["4,470"]
    assert bad("Roughly 4,500 at the start.", t, fs, sem) == []


def test_ratios_spans_and_counts():
    assert bad("Alpha ends 2.5 times its starting value.") == []      # 25 / 10
    assert bad("Alpha ends 4 times its starting value.") == ["4 times"]
    assert bad("Over the 7 years from 2000.") == []                   # 2007 - 2000
    assert bad("Over the 12 years from 2000.") == ["12 years"]
    assert bad("There are 2 series and 8 points.") == []              # structural counts
    assert bad("There are 40 series.") == ["40"]


def test_references_ordinals_list_markers_and_labels_are_not_numbers():
    assert bad("As Figure 4.2 and Table 3 show, Alpha is the 3rd highest in the 21st century.") == []
    assert bad("1. Press Home.\n2) Press End.") == []
    assert bad("Series Q3 and COVID-19 are labels.") == []
    sem = Semantics("line", y_ticks=["0", "30"], series=[], legend=["Group 2"], title="Output in 2020 USD")
    assert bad("Group 2 rises, as the chart Output in 2020 USD shows.", TABLE, FACTS, sem) == []
    assert bad("Group 2 rises in 2020.", TABLE, FACTS, sem) == ["2020"]
    assert bad("Alpha has three lines and one peak.") == []           # bare number words are not claims


def test_signs_and_ranges_use_the_magnitude():
    assert bad("Beta changed by -13, from 30 to 17.") == []
    assert bad("The 2000-2007 span.") == []
    assert [m["text"] for m in mentions("Between 2000–2007.")] == ["2000", "2007"]


def test_strip_unsupported_drops_whole_sentences_and_keeps_the_rest():
    text = ("Alpha rises from 10 to 25. It then jumped to 99 in 2005, which is unusual. Beta falls.\n\n"
            "Pi is about 3.14 here. Beta ends at 17, e.g. below Alpha.")
    got = check(text, TABLE, FACTS, SEM)
    assert [o["text"] for o in got] == ["99", "3.14"]
    assert strip_unsupported(text, got) == "Alpha rises from 10 to 25. Beta falls.\n\nBeta ends at 17, e.g. below Alpha."
    assert strip_unsupported("Fine text 25.", []) == "Fine text 25."
    assert strip_unsupported("Only 99 here.", check("Only 99 here.", TABLE, FACTS, SEM)) == ""


def test_sentences_do_not_split_on_decimals_or_abbreviations():
    for text, want in [("It is 4.5 million, i.e. large. Next one.", ["It is 4.5 million, i.e. large.", "Next one."]),
                       ("U.S. data vs. UK data hold. Done", ["U.S. data vs. UK data hold.", "Done"]),
                       ("A 4.5 rise. B 3.2 fall", ["A 4.5 rise.", "B 3.2 fall"])]:
        assert [text[a:b] for a, b, _ in sentences(text)] == want


def test_no_numbers_no_work():
    assert check("Alpha rises steadily.", TABLE, FACTS, SEM) == []
    assert strip_unsupported("", []) == ""


@pytest.mark.skipif(not SYNTH.exists(), reason="synthetic charts not generated")
def test_fact_sentences_never_trip_the_guard_and_planted_values_mostly_do():
    files = sorted(SYNTH.glob("*/*.json"))
    rng = random.Random(7)
    planted = caught = 0
    for p in files:
        d = json.loads(p.read_text())
        sem, table = Semantics.from_dict(d["semantics"]), Table.from_dict(d["table"])
        fs = facts(table, sem)
        assert check(" ".join(f["text"] for f in fs), table, fs, sem) == [], p
        vals = [v for vs in table.series.values() for v in vs]
        if p.parent.name == "bench":  # a value 25% off one the table holds, written the way the chart writes it
            planted += 1
            caught += bool(check(f"It reaches {format_like(rng.choice(vals) * 1.25, sem)}.", table, fs, sem))
    assert caught / planted >= 0.85, (caught, planted)
