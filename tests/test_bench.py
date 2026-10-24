"""Harness tests: runner with stub readers, planted faults, the ChartQA fetcher against an in-memory zip.
Everything is offline; the one network test runs only with LIVE=1."""
import io
import json
import os
import random
import sys
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import raised
from bench import fetch_chartqa as fc
from bench import perturb, run
from raised.model import Semantics, Series, Table

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- fixtures


def make_cases(tmp_path, n=8):
    img = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), "white").save(img)
    cases = []
    for k in range(n):
        ctype = ["line", "grouped_bar", "hbar", "bar"][k % 4]
        names = ["A", "B", "C"][: 1 if ctype == "bar" else 2 + k % 2]
        rng = random.Random(k)
        sem = Semantics(chart_type=ctype, x_ticks=["0", "50", "100"] if ctype == "hbar" else [], y_ticks=[] if ctype == "hbar" else ["0", "50", "100"],
                        series=[Series(n) for n in names])
        table = Table([2000.0 + i for i in range(6)] if ctype == "line" else [f"c{i}" for i in range(6)],
                      {n: [rng.uniform(10, 90) for _ in range(6)] for n in names})
        axis = "x" if ctype == "hbar" else "y"
        oracle = {f"{axis}_ticks_px": [["0", 400.0], ["50", 300.0], ["100", 200.0]]}
        cases.append(run.Case(f"{k:04d}", img, table, ctype, "synth-bench", "x", sem, oracle))
    return cases


def perfect(case, img):
    return run.Reading(case.truth, case.semantics, usd=0.002)


def noisy(case, img):
    rng = random.Random(case.id)
    cols = {n: [None if rng.random() < 0.1 else v + rng.gauss(0, 6) for v in col] for n, col in case.truth.series.items()}
    return run.Reading(Table(list(case.truth.x), cols), Semantics(chart_type="line", series=case.semantics.series[:1]))


# ---------------------------------------------------------------- runner


def test_perfect_reader_scores_perfectly(tmp_path):
    cases = make_cases(tmp_path)
    s = run.summarize(run.run_cases(cases, perfect, workers=2))
    assert s["n"] == 8 and s["failed"] == 0
    assert s["rms_f1"] == pytest.approx(100) and s["rms_precision"] == pytest.approx(100)
    assert s["median_err_pct"] == 0 and s["within_2pct"] == 1 and s["missing"] == 0
    assert s["semantics"] == {"chart_type": 1, "n_series": 1, "scale": 1}
    assert s["usd_per_figure"] == pytest.approx(0.002) and s["usd_total"] == pytest.approx(0.016)
    assert set(s["by_chart_type"]) == {"line", "grouped_bar", "hbar", "bar"}


def test_noisy_reader_scores_lower(tmp_path):
    cases = make_cases(tmp_path)
    good = run.summarize(run.run_cases(cases, perfect))
    bad = run.summarize(run.run_cases(cases, noisy))
    assert bad["rms_f1"] < good["rms_f1"] - 20
    assert bad["median_err_pct"] > 2 and bad["within_2pct"] < 0.5
    assert 0.03 < bad["missing"] < 0.3
    assert bad["semantics"]["chart_type"] < 1          # the noisy reader calls everything a line chart
    assert bad["semantics"]["n_series"] < 1


def test_failing_reader_counts_as_failed_not_skipped(tmp_path):
    def boom(case, img):
        raise RuntimeError("model said no")

    recs = run.run_cases(make_cases(tmp_path, 4), boom)
    s = run.summarize(recs)
    assert s["failed"] == 4 and s["rms_f1"] == 0 and s["missing"] == 1 and s["within_2pct"] == 0
    assert "model said no" in recs[0]["error"]


def test_no_table_scores_zero_precision_too(tmp_path):
    s = run.summarize(run.run_cases(make_cases(tmp_path, 4), lambda c, i: run.Reading(None)))
    assert s["rms_precision"] == 0 and s["failed"] == 4


def test_numeric_metrics_skipped_without_oracle(tmp_path):
    case = make_cases(tmp_path, 1)[0]
    case.semantics = case.oracle = None
    rec = run.evaluate(case, perfect)
    assert "errs" not in rec and rec["f1"] == pytest.approx(1)
    assert "median_err_pct" not in run.summarize([rec])


def test_log_axis_error_is_measured_in_decades(tmp_path):
    case = make_cases(tmp_path, 1)[0]
    case.semantics = Semantics(chart_type="line", y_scale="log", y_ticks=["1", "10", "100", "1,000"])
    case.truth = Table([1.0, 2.0], {"A": [10.0, 100.0]})
    off = lambda c, i: run.Reading(Table([1.0, 2.0], {"A": [10.0 * 10 ** 0.3, 100.0]}))   # 0.3 decades of 3
    assert run.evaluate(case, off)["errs"] == [0.1, 0.0]


def test_pixel_only_plumbing(tmp_path, monkeypatch):
    """The reader passes oracle semantics and truth x to ticks/trace and builds a Table from the per-series output."""
    calls = {}
    ticks = types.SimpleNamespace(locate_ticks=lambda img, sem: calls.setdefault("ticks", (img.size, sem.chart_type)) and {"ok": 1})
    trace = types.SimpleNamespace(trace_at=lambda img, sem, t, xs: calls.update(xs=xs, tr=t) or {"A": [1.0] * len(xs), "B": [2.0] * len(xs)})
    for name, mod in (("ticks", ticks), ("trace", trace)):
        monkeypatch.setattr(raised, name, mod, raising=False)
        monkeypatch.setitem(sys.modules, f"raised.{name}", mod)
    case = make_cases(tmp_path, 2)[0]            # line chart with two series A, B
    r = run.get_reader("pixel_only")(case, case.load_image())
    assert calls["ticks"] == ((8, 8), "line") and calls["tr"] == {"ok": 1} and calls["xs"] == case.truth.x
    assert r.table.series == {"A": [1.0] * 6, "B": [2.0] * 6} and r.table.x == case.truth.x
    trace.trace_at = lambda img, sem, t, xs: [[3.0] * len(xs), [4.0] * len(xs)]       # a list per series, in the order of the semantics
    assert run.get_reader("pixel_only")(case, case.load_image()).table.series == {"A": [3.0] * 6, "B": [4.0] * 6}


def test_missing_code_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setitem(run.READERS, "ghost", lambda: __import__("raised.no_such_module"))
    with pytest.raises(SystemExit, match="cannot run yet"):
        run.get_reader("ghost")
    if not hasattr(__import__("raised.pipeline", fromlist=["x"]), "check_table"):
        with pytest.raises(SystemExit, match="check_table"):
            run.get_checker("hybrid")


def test_main_writes_results_and_summary(tmp_path, monkeypatch):
    cases = make_cases(tmp_path)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "load_cases", lambda split, limit=None: cases[:limit])
    monkeypatch.setitem(run.READERS, "stub_good", lambda: perfect)
    monkeypatch.setitem(run.READERS, "stub_bad", lambda: noisy)
    monkeypatch.setattr(run, "get_checker", lambda config: lambda case, table: table.to_dict() != case.truth.to_dict())
    run.main(["--config", "stub_good", "--split", "synth-bench", "--limit", "6", "--perturb", "4"])
    run.main(["--config", "stub_bad", "--split", "synth-bench", "--limit", "6"])
    res = json.loads((tmp_path / "results" / "stub_good-synth-bench.json").read_text())
    assert res["summary"]["n"] == 6 and len(res["cases"]) == 6
    assert res["verifier"]["catch_rate"] == 1 and res["verifier"]["false_alarm_rate"] == 0 and res["verifier"]["n_faulty"] == 4
    md = (tmp_path / "results" / "SUMMARY.md").read_text()
    assert "## synth-bench" in md and "| stub_good |" in md and "| stub_bad |" in md and "By chart type" in md
    assert "Verifier on planted faults" in md and "100.0%" in md
    merged = json.loads((tmp_path / "results" / "results.json").read_text())
    assert set(merged) == {"stub_good-synth-bench", "stub_bad-synth-bench"}
    with pytest.raises(SystemExit):
        run.main(["--config", "pixel_only", "--split", "chartqa"])


def test_load_synth_cases():
    if not (ROOT / "bench/data/synth/bench/0000.json").exists():
        pytest.skip("run python -m bench.make_synth first")
    cases = run.load_cases("synth-bench", 5)
    assert len(cases) == 5 and cases[0].image.exists() and cases[0].semantics.chart_type == cases[0].chart_type
    assert "plot_box" in cases[0].oracle


# ---------------------------------------------------------------- planted faults


def _plant(case, kind, seed=0):
    axis = "x" if case.semantics.value_axis == "x" else "y"
    return perturb.plant(case.truth, case.semantics, kind, random.Random(seed), case.oracle[f"{axis}_ticks_px"])


def test_shift_moves_exactly_one_value_by_5_to_20_percent(tmp_path):
    for case in make_cases(tmp_path):
        for seed in range(10):
            t, fault = _plant(case, "shift", seed)
            diff = [(s, i) for s in t.series for i, (a, b) in enumerate(zip(case.truth.series[s], t.series[s])) if a != b]
            assert diff == [(fault["series"], fault["index"])]
            assert 0.05 - 1e-9 <= abs(fault["to"] - fault["from"]) / 100 <= 0.20 + 1e-9    # axis 0..100
            assert t.x == case.truth.x


def test_bars_never_shift_below_zero():
    sem = Semantics(chart_type="bar", y_ticks=["0", "10", "20"], series=[Series("v")])
    tbl = Table(["a", "b"], {"v": [0.0, 0.5]})
    for seed in range(30):
        t, _ = perturb.plant(tbl, sem, "shift", random.Random(seed))
        assert min(t.series["v"]) >= 0


def test_swap_drop_and_input_untouched(tmp_path):
    case = next(c for c in make_cases(tmp_path) if len(c.truth.series) >= 2)
    before = json.dumps(case.truth.to_dict())
    t, fault = _plant(case, "swap_series")
    a, b = fault["series"]
    assert t.series[a] == case.truth.series[b] and t.series[b] == case.truth.series[a]
    t, fault = _plant(case, "drop_point")
    assert len(t.x) == 5 and all(len(c) == 5 for c in t.series.values()) and case.truth.x[fault["index"]] not in t.x
    assert json.dumps(case.truth.to_dict()) == before
    single = next(c for c in make_cases(tmp_path) if len(c.truth.series) == 1)
    with pytest.raises(perturb.NotApplicable):
        _plant(single, "swap_series")


def test_flip_scale_matches_hand_calculation():
    sem = Semantics(chart_type="line", y_ticks=["0", "50", "100"], series=[Series("v")])
    ticks = [["0", 100.0], ["50", 200.0], ["100", 300.0]]
    t, fault = perturb.plant(Table([1, 2], {"v": [25.0, 75.0]}), sem, "flip_scale", random.Random(0), ticks)
    # 75 sits at pixel 250, halfway between the 50 and 100 ticks; read on a log axis that is sqrt(50 * 100). 25 is at pixel 150.
    assert t.series["v"] == pytest.approx([50 / 2 ** 0.5, 50 * 2 ** 0.5]) and fault["to"] == "log"
    logsem = Semantics(chart_type="line", y_scale="log", y_ticks=["1", "10", "100"], series=[Series("v")])
    t, fault = perturb.plant(Table([1], {"v": [10.0]}), logsem, "flip_scale", random.Random(0), [["1", 100.0], ["10", 200.0], ["100", 300.0]])
    assert t.series["v"] == pytest.approx([37.0]) and fault["to"] == "linear"    # pixel 200 on the least-squares line through 1, 10, 100
    with pytest.raises(perturb.NotApplicable):
        perturb.plant(Table([1], {"v": [10.0]}), sem, "flip_scale", random.Random(0), None)


def test_variants_rotate_kinds_and_use_each_case_once(tmp_path):
    cases = make_cases(tmp_path, 16)
    vs = perturb.variants(cases, 12, seed=3)
    assert len(vs) == 12 and len({v["case"] for v in vs}) == 12
    assert [v["fault"]["kind"] for v in vs[:4]] == list(perturb.KINDS)
    assert perturb.variants(cases, 12, seed=3)[0]["fault"] == vs[0]["fault"]      # seeded
    by = {c.id: c for c in cases}
    assert all(v["table"].to_dict() != by[v["case"]].truth.to_dict() for v in vs)  # every variant really differs
    assert len(perturb.variants(cases[:3], 10)) <= 3


def test_variants_on_generated_charts():
    if not (ROOT / "bench/data/synth/bench/0000.json").exists():
        pytest.skip("run python -m bench.make_synth first")
    cases = run.load_cases("synth-bench")
    vs = perturb.variants(cases, 100, seed=0)
    kinds = [v["fault"]["kind"] for v in vs]
    assert len(vs) == 100 and all(kinds.count(k) >= 20 for k in perturb.KINDS)


def test_verifier_eval_stub_checkers(tmp_path):
    cases = make_cases(tmp_path, 12)
    exact = lambda case, table: table.to_dict() != case.truth.to_dict()
    s = run.verifier_eval(cases, exact, 8)
    assert s["catch_rate"] == 1 and s["false_alarm_rate"] == 0 and (s["n_faulty"], s["n_clean"]) == (8, 8)
    s = run.verifier_eval(cases, lambda c, t: False, 8)
    assert s["catch_rate"] == 0 and s["false_alarm_rate"] == 0
    s = run.verifier_eval(cases, lambda c, t: True, 8)
    assert s["catch_rate"] == 1 and s["false_alarm_rate"] == 1
    s = run.verifier_eval(cases, lambda c, t: len(t.x) != len(c.truth.x), 8)      # sees dropped points only
    assert s["catch_by_kind"] == {"shift": 0, "swap_series": 0, "drop_point": 1, "flip_scale": 0} and s["catch_rate"] == 0.25


# ---------------------------------------------------------------- ChartQA fetcher (in-memory zip, no network)


def test_source_of():
    assert [fc.source_of(i) for i in ("two_col_100025", "multi_col_10", "00339007006077", "10099", "OECD_POVERTY_RATE_GBR_SVN_000002")] == \
           ["statista", "statista", "owid", "pew", "oecd"]


def test_csv_to_table():
    t, header = fc.csv_to_table('Characteristic,2017,2018\nWhite,"1,457",399\nBlack,223,nan\n')
    assert header == "Characteristic" and t.x == ["White", "Black"]
    assert t.series == {"2017": [1457.0, 223.0], "2018": [399.0, None]}
    t, _ = fc.csv_to_table("Year,a,a,words\n2000,1,2,x\n2001,3,4,y\n")
    assert list(t.series) == ["a", "a (2)"]                              # duplicate names kept apart, text column dropped
    assert fc.csv_to_table("Year,a\n2000,1\n") is None                   # one row is not a chart
    assert fc.csv_to_table("Year,a\n2000,x\n2001,y\n") is None           # nothing numeric


def test_quotas_and_allocation():
    assert fc.quotas({"statista": 1207, "owid": 143, "pew": 123, "oecd": 36}, 200) == {"oecd": 36, "pew": 54, "owid": 55, "statista": 55}
    assert fc.quotas({"a": 10, "b": 10}, 30) == {"a": 10, "b": 10}        # cannot exceed what exists
    assert fc.allocate({"bar": 60, "line": 30, "pie": 10}, 10) == {"bar": 6, "line": 3, "pie": 1}
    assert sum(fc.allocate({"a": 3, "b": 3, "c": 3}, 4).values()) == 4


def _cand(i, src, typ):
    return {"id": f"{src}{i}", "source": src, "chart_type": typ}


def test_select_is_stratified_seeded_and_interleaved():
    cands = [_cand(i, "statista", "bar" if i % 4 else "line") for i in range(100)] + [_cand(i, "pew", "hbar") for i in range(30)] + \
            [_cand(i, "oecd", "line") for i in range(6)]
    pick = fc.select(cands, 20, seed=1)
    assert len(pick) == len({c["id"] for c in pick}) == 20
    src = [c["source"] for c in pick]
    assert (src.count("oecd"), src.count("pew"), src.count("statista")) == (6, 7, 7)
    assert len(set(src[:3])) == 3                                        # any prefix stays balanced
    assert [c["id"] for c in fc.select(cands, 20, seed=1)] == [c["id"] for c in pick] != [c["id"] for c in fc.select(cands, 20, seed=2)]
    stat = [c for c in pick if c["source"] == "statista"]
    assert {c["chart_type"] for c in stat} == {"bar", "line"} or len(stat) < 4


def _fake_zip():
    """A zip shaped like 'ChartQA Dataset.zip' with 12 test charts, plus a train member that must be ignored."""
    ids = {"two_col_1": "v_bar", "two_col_2": "h_bar", "multi_col_3": "line", "two_col_4": "v_bar",
           "00339007006077": "h_bar", "00339007006078": "h_bar", "00339007006079": "line",
           "10099": "h_bar", "10146": "line", "10150": "pie", "OECD_A_1": "v_bar", "OECD_B_2": "line"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ChartQA Dataset/train/png/t.png", b"train")
        for kind in ("annotations", "png", "tables"):
            for i, t in ids.items():
                body = {"annotations": json.dumps({"type": t, "models": []}).encode(),
                        "tables": f"Characteristic,v1,v2\na,{len(i)},2\nb,3,4.5\nc,1,nan\n".encode(),
                        "png": bytes(random.Random(i).randbytes(3000))}[kind]
                z.writestr(zipfile.ZipInfo(f"ChartQA Dataset/test/{kind}/{i}.{ {'annotations': 'json', 'png': 'png', 'tables': 'csv'}[kind] }"),
                           body, zipfile.ZIP_STORED if kind == "png" else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


class FakeRemote(fc.Remote):
    log: list = []

    def __init__(self, data):
        self.data, self.size, self.url = data, len(data), "fake"

    def get(self, start, end):
        FakeRemote.log.append((start, min(end, self.size)))
        return self.data[start:min(end, self.size)]


def test_fetch_members_coalesces_nearby_reads():
    data = _fake_zip()
    remote = FakeRemote(data)
    zf = zipfile.ZipFile(io.BytesIO(data))
    idx = {i.filename: [i.header_offset, i.compress_size, i.compress_type] for i in zf.infolist()}
    names = [f"ChartQA Dataset/test/tables/{i}.csv" for i in ("two_col_1", "two_col_2", "10099")]
    FakeRemote.log = []
    got = fc.fetch_members(remote, idx, names, gap=1 << 20, workers=1)
    assert len(FakeRemote.log) == 1                                     # all neighbours in one request
    assert all(got[n] == zf.read(n) for n in names)
    pngs = [f"ChartQA Dataset/test/png/{i}.png" for i in ("two_col_1", "OECD_B_2")]
    FakeRemote.log = []
    got = fc.fetch_members(remote, idx, pngs, gap=100, workers=2)
    assert len(FakeRemote.log) == 2 and all(got[n] == zf.read(n) for n in pngs)   # far apart: separate reads


def test_fetch_end_to_end_and_idempotent(tmp_path, monkeypatch):
    data = _fake_zip()
    monkeypatch.setattr(fc, "ROOT", tmp_path / "chartqa")
    monkeypatch.setattr(fc, "Remote", lambda: FakeRemote(data))
    sample = fc.fetch(n=8, seed=0)
    assert len(sample) == 8
    assert sorted(c["source"] for c in sample) == ["oecd"] * 2 + ["owid"] * 2 + ["pew"] * 2 + ["statista"] * 2
    first = sample[0]
    img = fc.ROOT / first["image"]
    assert img.read_bytes() == zipfile.ZipFile(io.BytesIO(data)).read(f"ChartQA Dataset/test/png/{first['id']}.png")
    assert first["table"]["series"]["v2"] == [2.0, 4.5, None] and first["x_header"] == "Characteristic"
    assert {c["chart_type"] for c in sample} <= {"bar", "hbar", "line", "pie"}
    cases = fc.json.loads((fc.ROOT / "sample.json").read_text())
    assert cases == sample
    class Offline(FakeRemote):
        def get(self, start, end):
            pytest.fail("second call must not touch the network")

    monkeypatch.setattr(fc, "Remote", lambda: Offline(data))
    assert fc.fetch(n=8, seed=0) == sample                              # idempotent
    monkeypatch.setattr(run, "CHARTQA", fc.ROOT)
    loaded = run.load_cases("chartqa", 3)
    assert [c.id for c in loaded] == [c["id"] for c in sample[:3]] and loaded[0].semantics is None and loaded[0].image.exists()


@pytest.mark.skipif(not os.environ.get("LIVE"), reason="network: set LIVE=1")
def test_live_remote_index_and_one_member(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "ROOT", tmp_path)
    remote = fc.Remote()
    idx = fc.load_index(remote)
    assert sum(k.startswith(fc.PREFIX + "tables/") for k in idx) == 1509
    name = fc.PREFIX + "tables/two_col_100025.csv"
    assert fc.fetch_members(remote, idx, [name], gap=0)[name].startswith(b"Characteristic,")
