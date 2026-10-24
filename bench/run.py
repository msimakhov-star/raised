"""Benchmark runner.

    python -m bench.run --config pixel_only --split synth-bench [--limit N] [--workers K] [--perturb N]
    python -m bench.run --summary                      # only rebuild bench/results/SUMMARY.md and results.json

Configs (spec section 4): pixel_only (tracer with the generator's oracle semantics, no key), vl_only, hybrid, minicpm.
Splits: synth-bench (300 charts), synth-probe (50), chartqa (200, run `python -m bench.fetch_chartqa` first).

A reader is a function (Case, PIL image) -> Reading. pixel_only is built here from raised.ticks and raised.trace.
The other configs call raised.pipeline.read_chart(img, config) -> {"table": Table | None, "semantics": Semantics | None,
"usd": float}; "usd" is what raised.llm.Meter spent on that chart. --perturb N additionally needs
raised.pipeline.check_table(img, table, config, semantics) -> bool (True = the verifier flags the table).
A reader that raises, or returns no table, scores zero on that chart and counts under "failed".

Writes bench/results/<config>-<split>.json (summary plus one record per chart) and rebuilds SUMMARY.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image

from bench import metrics, perturb
from raised.model import Semantics, Table

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SYNTH = HERE / "data" / "synth"
CHARTQA = HERE / "data" / "chartqa"
SPLITS = ("synth-bench", "synth-probe", "chartqa")


@dataclass
class Case:
    id: str
    image: Path
    truth: Table
    chart_type: str
    source: str
    x_header: str = ""
    semantics: Semantics | None = None   # generator's exact semantics (synthetic splits only)
    oracle: dict | None = None           # generator's tick pixels, plot box, degradation (synthetic only)

    def load_image(self) -> Image.Image:
        return Image.open(self.image).convert("RGB")


@dataclass
class Reading:
    table: Table | None
    semantics: Semantics | None = None
    usd: float = 0.0


Reader = Callable[[Case, Image.Image], Reading]
Checker = Callable[[Case, Table], bool]


def load_cases(split: str, limit: int | None = None) -> list[Case]:
    if split == "chartqa":
        p = CHARTQA / "sample.json"
        if not p.exists():
            raise SystemExit("no ChartQA sample: run `python -m bench.fetch_chartqa` first")
        cases = [Case(c["id"], CHARTQA / c["image"], Table.from_dict(c["table"]), c["chart_type"], c["source"], c["x_header"])
                 for c in json.loads(p.read_text())]
    else:
        files = sorted((SYNTH / split.removeprefix("synth-")).glob("*.json"))
        if not files:
            raise SystemExit(f"no synthetic charts in {SYNTH / split.removeprefix('synth-')}: run `python -m bench.make_synth`")
        cases = []
        for f in files:
            j = json.loads(f.read_text())
            sem = Semantics.from_dict(j["semantics"])
            cases.append(Case(f.stem, f.with_suffix(".jpg"), Table.from_dict(j["table"]), sem.chart_type, split,
                              sem.x_title or "x", sem, j["oracle"]))
    return cases[:limit]


# ---------------------------------------------------------------- readers


def _pixel_only() -> Reader:
    from raised import ticks, trace  # written by the pixel-core builder
    locate_ticks, trace_at = ticks.locate_ticks, trace.trace_at  # fail now, not on every chart

    def read(case: Case, img: Image.Image) -> Reading:
        if case.semantics is None:
            raise ValueError("pixel_only needs oracle semantics; use a synth split")
        found = locate_ticks(img, case.semantics)
        # sample where the truth is: x values for line and scatter, category labels for bars
        out = trace_at(img, case.semantics, found, list(case.truth.x))
        cols = list(out.values()) if isinstance(out, dict) else out
        names = [s.name for s in case.semantics.series]
        return Reading(Table(list(case.truth.x), {n: list(c) for n, c in zip(names, cols)}))

    return read


def _pipeline(config: str) -> Callable[[], Reader]:
    def factory() -> Reader:
        from raised import pipeline
        fn = pipeline.read_chart

        def read(case: Case, img: Image.Image) -> Reading:
            out = fn(img, config)
            t, s = out.get("table"), out.get("semantics")
            return Reading(Table.from_dict(t) if isinstance(t, dict) else t,
                           Semantics.from_dict(s) if isinstance(s, dict) else s, float(out.get("usd", 0.0)))

        return read

    return factory


READERS: dict[str, Callable[[], Reader]] = {"pixel_only": _pixel_only, **{c: _pipeline(c) for c in ("vl_only", "hybrid", "minicpm")}}


def _need(what: str, loader):
    try:
        return loader()
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"cannot run yet: {what} is missing ({type(e).__name__}: {e})")


def get_reader(config: str) -> Reader:
    return _need(f"the code behind config {config!r} (raised.ticks / raised.trace for pixel_only, "
                 "raised.pipeline.read_chart otherwise)", READERS[config])


def get_checker(config: str) -> Checker:
    def load():
        from raised import pipeline
        fn = pipeline.check_table
        return lambda case, table: bool(fn(case.load_image(), table, config, case.semantics))
    return _need("raised.pipeline.check_table", load)


# ---------------------------------------------------------------- running


def _pmap(fn, items: list, workers: int):
    done = 0

    def tick(x):
        nonlocal done
        r = fn(x)
        done += 1
        if done % max(1, len(items) // 10) == 0:
            print(f"  {done}/{len(items)}", file=sys.stderr, flush=True)
        return r

    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(tick, items))


def evaluate(case: Case, reader: Reader) -> dict:
    t0 = time.perf_counter()
    try:
        r, err = reader(case, case.load_image()), None
    except Exception as e:  # a reader crash is a failed chart, not a failed benchmark
        r, err = Reading(None), f"{type(e).__name__}: {e}"[:300]
    rec = {"id": case.id, "source": case.source, "chart_type": case.chart_type, "seconds": round(time.perf_counter() - t0, 3),
           "usd": r.usd, "error": err, "got_table": r.table is not None}
    rec["precision"], rec["recall"], rec["f1"] = metrics.rms_f1(case.truth, r.table, case.x_header)
    sem = case.semantics
    if sem is not None:  # numeric errors need the axis range, which only the synthetic splits know exactly
        rng = metrics.axis_range(case.truth, sem.chart_type, metrics.value_ticks(sem), metrics.is_log(sem))
        rec["errs"] = [None if e is None else round(e, 5)
                       for e in metrics.point_errors(case.truth, r.table, rng, metrics.is_log(sem))]
        if r.semantics is not None:
            rec["sem"] = metrics.semantics_accuracy(sem, r.semantics)
    return rec


def run_cases(cases: list[Case], reader: Reader, workers: int = 4) -> list[dict]:
    return _pmap(lambda c: evaluate(c, reader), cases, workers)


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def _agg(recs: list[dict]) -> dict:
    n = len(recs)
    out = {"n": n, "failed": sum(not r["got_table"] for r in recs),
           "rms_f1": 100 * _mean(r["f1"] for r in recs), "rms_precision": 100 * _mean(r["precision"] for r in recs),
           "rms_recall": 100 * _mean(r["recall"] for r in recs)}
    errs = [e for r in recs for e in r.get("errs", [])]
    if errs:
        s = metrics.summarize_errors(errs)
        out.update(median_err_pct=None if s["median_err"] is None else 100 * s["median_err"],
                   within_2pct=s["within_2pct"], missing=s["missing"], n_values=s["n_values"])
    sem = [r["sem"] for r in recs if "sem" in r]
    if sem:
        out["semantics"] = {k: _mean(x[k] for x in sem) for k in sem[0]}
    out.update(seconds_per_figure=_mean(r["seconds"] for r in recs), usd_total=sum(r["usd"] for r in recs),
               usd_per_figure=_mean(r["usd"] for r in recs))
    return out


def summarize(recs: list[dict]) -> dict:
    """Overall numbers plus the same split by chart type and by source."""
    by = lambda key: {k: _agg([r for r in recs if r[key] == k]) for k in sorted({r[key] for r in recs})}
    return {**_agg(recs), "by_chart_type": by("chart_type"), "by_source": by("source")} if recs else {"n": 0}


def verifier_eval(cases: list[Case], checker: Checker, n: int, seed: int = 0, workers: int = 4) -> dict:
    """Catch rate of the verifier on n planted faults, and its false-alarm rate on the correct tables of the same charts."""
    by_id = {c.id: c for c in cases}
    planted = perturb.variants(cases, n, seed)
    jobs = [(by_id[v["case"]], v["table"], v["fault"]["kind"]) for v in planted] + [(by_id[v["case"]], by_id[v["case"]].truth, None) for v in planted]
    flags = _pmap(lambda j: checker(j[0], j[1]), jobs, workers)
    return metrics.catch_stats([{"fault": j[2], "flagged": f} for j, f in zip(jobs, flags)])


# ---------------------------------------------------------------- reporting


def _fmt(v, nd=1, pct=False):
    return "-" if v is None else f"{100 * v:.{nd}f}%" if pct else f"{v:.{nd}f}"


COLS = [("n", lambda s: str(s["n"])), ("failed", lambda s: str(s.get("failed", "-"))),
        ("RMS-F1", lambda s: _fmt(s.get("rms_f1"))),
        ("median err (% of range)", lambda s: _fmt(s.get("median_err_pct"), 2)),
        ("within 2%", lambda s: _fmt(s.get("within_2pct"), pct=True)), ("missing", lambda s: _fmt(s.get("missing"), pct=True)),
        ("chart type", lambda s: _fmt(s.get("semantics", {}).get("chart_type"), pct=True)),
        ("series count", lambda s: _fmt(s.get("semantics", {}).get("n_series"), pct=True)),
        ("axis scale", lambda s: _fmt(s.get("semantics", {}).get("scale"), pct=True)),
        ("$ / figure", lambda s: _fmt(s.get("usd_per_figure"), 4)), ("s / figure", lambda s: _fmt(s.get("seconds_per_figure"), 2))]


def _md(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in rows)]) + "\n"


def write_summary(results_dir: Path = RESULTS) -> str:
    """Merge every <config>-<split>.json into SUMMARY.md and results.json (summaries only, for the /bench page)."""
    runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*-*.json"))]
    merged = {f"{r['config']}-{r['split']}": {"summary": r["summary"], "verifier": r.get("verifier"), "date": r["date"]} for r in runs}
    (results_dir / "results.json").write_text(json.dumps(merged, indent=1))
    md = ["# Raised benchmark\n",
          "RMS-F1: DePlot's table metric (arXiv 2212.10505) in percent, a failed chart scores 0. Median err: median "
          "|read - true| / value-axis range over every true value that was read (percent); `within 2%` and `missing` count all "
          "true values, so a missing value is a miss. Numeric metrics need the generator's exact axis, so only synthetic splits "
          "have them. Semantic accuracy is shown for configs that read semantics with a model.\n"]
    for split in SPLITS:
        rs = [r for r in runs if r["split"] == split]
        if not rs:
            continue
        md.append(f"\n## {split}\n")
        md.append(_md(["config", *(h for h, _ in COLS)], [[r["config"], *(f(r["summary"]) for _, f in COLS)] for r in rs]))
        for key, label in (("by_chart_type", "chart type"), ("by_source", "source")):
            groups = [(r["config"], g, s) for r in rs for g, s in r["summary"].get(key, {}).items()]
            if len({g for _, g, _ in groups}) > 1:
                md.append(f"\nBy {label}:\n")
                md.append(_md(["config", label, *(h for h, _ in COLS)], [[c, g, *(f(s) for _, f in COLS)] for c, g, s in sorted(groups, key=lambda t: (t[1], t[0]))]))
        ver = [(r["config"], r["verifier"]) for r in rs if r.get("verifier")]
        if ver:
            md.append("\nVerifier on planted faults:\n")
            md.append(_md(["config", "faults", "caught", "clean charts", "false alarms", "caught by kind"],
                          [[c, str(v["n_faulty"]), _fmt(v["catch_rate"], pct=True), str(v["n_clean"]), _fmt(v["false_alarm_rate"], pct=True),
                            ", ".join(f"{k} {_fmt(x, 0, pct=True)}" for k, x in v["catch_by_kind"].items())] for c, v in ver]))
    text = "".join(md)
    (results_dir / "SUMMARY.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", choices=list(READERS))
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--perturb", type=int, default=0, help="also plant this many faults and measure the verifier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", action="store_true", help="only rebuild SUMMARY.md")
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    if a.summary:
        print(write_summary())
        return
    if not (a.config and a.split):
        ap.error("--config and --split are required")
    if a.config == "pixel_only" and a.split == "chartqa":
        ap.error("pixel_only needs the generator's oracle semantics; ChartQA has none")
    reader = get_reader(a.config)
    checker = get_checker(a.config) if a.perturb else None
    cases = load_cases(a.split, a.limit)
    print(f"{a.config} on {len(cases)} charts of {a.split}", file=sys.stderr)
    recs = run_cases(cases, reader, a.workers)
    out = {"config": a.config, "split": a.split, "limit": a.limit, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "summary": summarize(recs), "cases": recs}
    if checker:
        out["verifier"] = verifier_eval(cases, checker, a.perturb, a.seed, a.workers)
    (RESULTS / f"{a.config}-{a.split}.json").write_text(json.dumps(out, indent=1))
    write_summary()
    s = out["summary"]
    print(f"RMS-F1 {s['rms_f1']:.1f}  failed {s['failed']}/{s['n']}  -> {RESULTS / (a.config + '-' + a.split + '.json')}")


if __name__ == "__main__":
    main()
