"""Step 2: chart semantics and reading A from Super VL, as a vote over several answers.

Fields merge by majority (ties go to the earliest answer); tick and label lists vote as whole exact lists; series
are matched by name (case-insensitive) and kept when at least half the votes have them; reading A is the per-point
median. The share of votes behind each winner is that field's agreement, used as its confidence.
"""
from __future__ import annotations

import statistics
from collections import Counter

from . import prompts
from .llm import MODELS, image_part
from .model import Semantics, Series, Table, parse_number

SCALARS = ("chart_type", "title", "x_title", "y_title", "x_scale", "y_scale", "source")
LISTS = ("x_ticks", "y_ticks", "categories", "legend")
BARS = ("bar", "grouped_bar", "hbar")


def read_semantics(client, crop_png: bytes, votes: int = 3, model: str = MODELS["vl"]) -> tuple[Semantics, Table, dict]:
    """-> (semantics, reading A, per-field agreement 0..1). votes=1 is a single call at temperature 0."""
    msgs = [{"role": "user", "content": [image_part(crop_png), {"type": "text", "text": prompts.SEMANTICS}]}]
    kw = {"step": "semantics", "max_tokens": 8192}
    answers = (client.chat_json_n(model, msgs, prompts.SEMANTICS_SCHEMA, votes, **kw) if votes > 1
               else [client.chat_json(model, msgs, prompts.SEMANTICS_SCHEMA, **kw)])
    return merge_votes(answers)


def merge_votes(answers: list[dict]) -> tuple[Semantics, Table, dict]:
    agreement, fields = {}, {}
    for f in SCALARS:
        fields[f], agreement[f] = _majority([str(a.get(f) or "").strip() for a in answers])
    for f in LISTS:
        won, agreement[f] = _majority([tuple(str(t).strip() for t in a.get(f) or []) for a in answers])
        fields[f] = list(won)
    if fields["chart_type"] not in prompts.CHART_TYPES:
        fields["chart_type"] = "unsupported"
    for axis in ("x_scale", "y_scale"):
        if fields[axis] not in ("linear", "log", "category"):
            fields[axis] = "linear"

    per_vote = [{_key(s.get("name")): s for s in a.get("series") or [] if isinstance(s, dict)} for a in answers]
    keys = _kept([list(v) for v in per_vote])
    agreement["series"] = sum(set(v) == set(keys) for v in per_vote) / len(answers)
    series = []
    for k in keys:
        seen = [v[k] for v in per_vote if k in v]
        series.append(Series(**{f: _majority([str(s.get(f) or "").strip() for s in seen])[0]
                                for f in ("name", "color", "style", "marker")}))
    sem = Semantics(**fields, series=series)
    if sem.chart_type == "unsupported":
        return sem, Table([], {}), agreement
    table = _merge_reading([a.get("reading") or {} for a in answers], sem)
    if not sem.series:  # keep table and semantics naming the same series
        sem.series = [Series(name) for name in table.series]
    return sem, table, agreement


def _merge_reading(readings: list[dict], sem: Semantics) -> Table:
    xs = [[str(x).strip() for x in r.get("x") or []] for r in readings]
    base = sem.categories if sem.chart_type in BARS and sem.categories else list(_majority([tuple(x) for x in xs])[0])
    per_vote = [{_key(s.get("name")): s.get("values") or [] for s in r.get("series") or [] if isinstance(s, dict)}
                for r in readings]
    names = {_key(s.name): s.name for s in sem.series}
    keys = list(names) or _kept([list(v) for v in per_vote])
    index = [{_key(x): j for j, x in enumerate(vx)} for vx in xs]
    series = {}
    for k in keys:
        col = []
        for i, bx in enumerate(base):
            vals = []
            for vx, at, vote in zip(xs, index, per_vote):
                j = at.get(_key(bx), i if len(vx) == len(base) else None)  # unmatched x, same length: by position
                v = _num(vote[k][j]) if k in vote and j is not None and j < len(vote[k]) else None
                if v is not None:
                    vals.append(v)
            col.append(statistics.median(vals) if vals else None)
        series[names.get(k) or _display(k, readings)] = col
    x = base if sem.chart_type in BARS else [v if (v := parse_number(s)) is not None else s for s in base]
    return Table(list(x), series)


def _display(k: str, readings: list[dict]) -> str:
    """First original spelling of a series name that only the readings have."""
    return next(str(s.get("name") or "").strip() for r in readings for s in r.get("series") or []
                if isinstance(s, dict) and _key(s.get("name")) == k)


def _majority(values: list):
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in values if counts[v] == best), best / len(values)


def _kept(key_lists: list[list[str]]) -> list[str]:
    """Keys present in at least half the votes, in order of first appearance."""
    counts = Counter(k for ks in key_lists for k in set(ks))
    order = dict.fromkeys(k for ks in key_lists for k in ks)
    return [k for k in order if counts[k] * 2 >= len(key_lists)]


def _key(name) -> str:
    return " ".join(str(name or "").split()).casefold()


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return parse_number(str(v))
