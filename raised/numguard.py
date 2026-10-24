"""Number guard: every number in generated prose must trace back to the table, the axis ticks or the facts.

    check(text, table, facts, sem, tolerance=0.01) -> [{"text", "start", "end", "value", "unit", "sentence"}, ...]
    strip_unsupported(text, offenders) -> text without the sentences that hold an offender

A mention is accepted when an allowed number of its own kind lies within the rounding its own precision implies
("4.5 million" accepts 4.45 to 4.55 million, "4,500" accepts 4,450 to 4,550, "12" accepts 11.5 to 12.5). A hedged
mention ("about 4,470") also gets the relative `tolerance`; an unhedged one does not, because a number printed to four
digits claims four digits (measured on the 300 bench charts: a blanket 1% tolerance let 34% of values planted 10% off
a table value through, precision-only lets 9% through, the rest being coincidences with other table values).
Kinds keep a "45%" from being satisfied by a plain table value of 45:
  pct     "45%", "45 percent"   fact percentages and derived percentage changes (plus table values on a % axis)
  pp      "3 percentage points" amounts: table values, ticks, fact values and differences between them
  amount  "$3.4k", "1.2 million" same pool as pp
  plain   "1,234", "1995"       the amounts, plus x values and structural counts (series, points)
  ratio   "3x", "2.5 times"     ratios between facts
  span    "21 years"            gaps between x values
Derived values come only from facts of the same series (first, last, max, min and the start and end of every run or
step), from ranked values and from the cross-series comparison. Nothing is derived from arbitrary pairs of table
values, so an invented percentage cannot hide among thousands of combinations.
Years: a four-digit mention must equal an x value or tick (to the year); tolerance never applies to it.
Ignored: figure and table references ("Figure 4.2"), powers of ten (10^3), ordinals ("3rd"), list markers ("1."),
labels that mix letters and digits (series names, "Q3", "COVID-19"), and bare number words ("three lines"). Spelled-out
numbers up to 99 are checked only when followed by a unit ("forty percent", "two million").
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from .facts import axis_scale, axis_style, value_ticks
from .model import Semantics, Table, parse_number

_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9,
         "t": 1e12, "trillion": 1e12}
_ONES = "one two three four five six seven eight nine".split()
_WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                  "sixteen seventeen eighteen nineteen".split(), range(20)))
_WORDS.update({w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)})
_WORD = "|".join(sorted(_WORDS, key=len, reverse=True))

_UNIT = re.compile(r"""(?:
    (?P<pct>\s?%|\s?(?:percent|per\s?cent)\b)
  | (?P<pp>\s?(?:percentage\s+points?|pp)\b)
  | (?P<mult>\s?(?:thousand|million|billion|trillion|bn|mn)\b|[kKmMbBtT]\b)
  | (?P<ratio>x\b|\s?times\b|-?fold\b)
  | (?P<span>[\s-]?(?:years?|decades?)\b)
)""", re.X | re.I)
# The lookbehinds keep a digit run from starting inside a word, a decimal or an acronym-with-hyphen ("Q3", "COVID-19").
_MENTION = re.compile(
    r"(?P<cur>[£$€])?(?<![\w.])(?-i:(?<![A-Z]-))(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    rf"|(?<![\w-])(?P<word>(?:{_WORD})(?:[- ](?:{'|'.join(_ONES)}))?)(?=\s)", re.I)
_REFERENCE = re.compile(r"10(?:\^\(?-?\d+\)?|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)|\b(?:figures?|fig\.?|tables?|charts?|graphs?|panels?|pages?|sections?|chapters?|exhibits?|box(?:es)?)"
                        r"\s+\d+(?:\.\d+)*[a-z]?\b|^\s*\d+[.)]\s", re.I | re.M)
_HEDGE = re.compile(r"(?:about|around|roughly|approximately|approx\.?|nearly|almost|close to|circa|~)\s*$", re.I)
_ORDINAL = re.compile(r"(?:st|nd|rd|th)\b", re.I)
_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bapprox)(?<!\bU\.S)(?<!\bU\.K)(?<!\bNo)[.!?]+[\"”')\]]*(?=\s|$)|\n{2,}")


def sentences(text: str) -> list[tuple[int, int, str]]:
    """(start, end, separator) per sentence; the separator is the whitespace that follows it."""
    out, pos = [], 0
    for m in [*_END.finditer(text), re.search(r"$", text)]:
        end = m.start() if m.group().startswith("\n") else m.end()
        if end > pos and text[pos:end].strip():
            sep = re.match(r"\s*", text[end:]).group()
            out.append((pos, end, sep))
            pos = end + len(sep)
    return out


def _label_masks(sem: Semantics | None) -> list[re.Pattern]:
    """Labels that mix letters and digits (and are not themselves numbers) hide their digits from the scan."""
    if sem is None:
        return []
    names = [sem.title, sem.x_title, sem.y_title, *sem.categories, *sem.x_ticks, *sem.legend, *(s.name for s in sem.series)]
    names = {n for n in names if any(c.isdigit() for c in n) and parse_number(n) is None}
    return [re.compile(re.escape(n), re.I) for n in sorted(names, key=len, reverse=True)]


def _half(num: str) -> float:
    """Half the unit of the last printed significant digit: '4.5' -> 0.05, '12' -> 0.5, '4,500' -> 50.
    Trailing zeros of a number of 1,000 or more are placeholders, except in year-like numbers ('2000')."""
    if "." in num:
        return 0.5 * 10 ** -len(num.partition(".")[2])
    digits = num.replace(",", "")
    year = "," not in num and len(digits) == 4 and 1000 <= int(digits) <= 2200
    zeros = len(digits) - len(digits.rstrip("0")) if len(digits) >= 4 and not year else 0
    return 0.5 * 10 ** (zeros if zeros >= 2 else 0)


def mentions(text: str, sem: Semantics | None = None) -> list[dict]:
    """Every checkable number in `text`: {text, start, end, value, half, unit, hedged}. `value` includes the multiplier
    (1.2 million -> 1.2e6) and ignores the sign; `half` is the rounding the mention implies, in the same units."""
    masked = text
    for pat in [*_label_masks(sem), _REFERENCE]:
        masked = pat.sub(lambda m: " " * len(m.group()), masked)
    out = []
    for m in _MENTION.finditer(masked):
        end = m.end()
        if m["word"]:
            tens, *ones = re.split(r"[- ]", m["word"].lower())
            value, num = _WORDS[tens] + sum(_WORDS[o] for o in ones), "0"
        else:
            if _ORDINAL.match(masked, end):
                continue
            value, num = float(m["num"].replace(",", "")), m["num"]
        u = _UNIT.match(masked, end)
        kind = u.lastgroup if u else None
        if m["word"] and kind not in ("pct", "pp", "mult"):
            continue
        mult = _MULT[u["mult"].strip().lower()] if kind == "mult" else 1.0
        unit = kind if kind in ("pct", "pp", "ratio", "span") else "amount" if m["cur"] or kind == "mult" else "plain"
        end = u.end() if u else end
        out.append({"text": text[m.start():end], "start": m.start(), "end": end, "value": value * mult,
                    "half": _half(num) * mult, "mult": mult, "unit": unit,
                    "hedged": bool(_HEDGE.search(text[max(0, m.start() - 14):m.start()]))})
    return out


def _derive(facts: list[dict]) -> tuple[list, list, list]:
    """(differences, percentage changes, ratios) between facts of one series and between ranked values."""
    diffs, pcts, ratios = [], [], []

    def pair(a, b):
        diffs.append(abs(a - b))
        for p, q in ((a, b), (b, a)):
            if p:
                pcts.append(abs(a - b) / abs(p) * 100)
                ratios.append(abs(q / p))

    anchors = defaultdict(list)
    for f in facts:
        v = f["values"]
        if f["kind"] in ("max", "min", "first", "last"):
            anchors[f["id"].split(".")[0]].append(v["value"])
        elif "start" in v and "end" in v:  # change, run, step
            pair(v["start"], v["end"])
        elif "order" in v:  # rank, compare_last
            vals = [o["value"] for o in v["order"]]
            for a, b in [*zip(vals, vals[1:]), (vals[0], vals[-1])]:
                pair(a, b)
    for vals in anchors.values():
        for a, b in combinations(vals, 2):
            pair(a, b)
    return diffs, pcts, ratios


def _flatten(v, key=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _flatten(x, k)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x, key)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        yield key, abs(float(v))


def _pools(table: Table, facts: list[dict], sem: Semantics) -> dict:
    nums = lambda ts: [abs(n) for t in ts if (n := parse_number(str(t))) is not None]
    other_ticks = sem.y_ticks if sem.value_axis == "x" else sem.x_ticks
    amounts = [abs(float(v)) for vs in table.series.values() for v in vs if v is not None and v == v] + nums(value_ticks(sem))
    xs, pcts = nums(other_ticks) + nums(table.x), []
    counts = [len(table.series), len(table.x), len(sem.categories)] + [sum(v is not None for v in vs) for vs in table.series.values()]
    for f in facts:
        values = dict(f["values"])
        counts += values.pop("counts", {}).values()
        for k, v in _flatten(values):
            (xs if k.startswith("x") else pcts if k == "pct" else counts if k in ("points", "total") else amounts).append(v)
    diffs, dpcts, ratios = _derive(facts)
    xsorted = sorted(set(xs))
    spans = [b - a for a, b in combinations(xsorted, 2)] if len(xsorted) <= 12 else \
        [xsorted[-1] - xsorted[0], *(b - a for a, b in zip(xsorted, xsorted[1:]))]
    amounts += diffs
    return {"amount": amounts, "pp": amounts, "plain": amounts + counts, "ratio": ratios, "span": spans + counts, "x": xs,
            "pct": pcts + dpcts + (amounts if axis_style(sem)["suf"] == "%" else [])}


def _supported(m: dict, pools: dict, scale: float, tol: float) -> bool:
    unit, v, half = m["unit"], m["value"], m["half"]
    year = unit == "plain" and 1000 <= v <= 2200 and half == 0.5 and m["text"].isdigit()
    if year and any(abs(v - p) < 1.0 for p in pools["x"]):
        return True
    if unit == "span" and "decade" in m["text"].lower():
        v, half = v * 10, half * 10
    cands = [(v, half)]
    if unit in ("plain", "amount", "pp"):
        if m["mult"] != 1.0:  # "30 million" when the table keeps its values in millions
            cands.append((v / m["mult"], half / m["mult"]))
        elif scale != 1.0:    # "19.4" quoted in the axis unit (millions) while the table holds 19,400,000
            cands.append((v * scale, half * scale))
    t = tol if m["hedged"] and not year else 0.0
    if any(abs(c - p) <= max(t * p, h) + 1e-9 * p for c, h in cands for p in pools[unit]):
        return True
    return unit == "plain" and not year and any(abs(v - p) <= half + 1e-9 for p in pools["x"])


def check(text: str, table: Table, facts: list[dict], sem: Semantics, tolerance: float = 0.01) -> list[dict]:
    """Mentions in `text` that no allowed number supports, with character spans and the sentence that holds them."""
    found = mentions(text, sem)
    if not found:
        return []
    pools, scale, spans = _pools(table, facts, sem), axis_scale(sem), sentences(text)
    return [{"text": m["text"], "start": m["start"], "end": m["end"], "value": m["value"], "unit": m["unit"],
             "sentence": next((text[a:b] for a, b, _ in spans if a <= m["start"] < b), text)}
            for m in found if not _supported(m, pools, scale, tolerance)]


def strip_unsupported(text: str, offenders: list[dict]) -> str:
    """Drop every sentence that contains an offender; paragraph breaks between kept sentences survive."""
    return "".join(text[a:b] + sep for a, b, sep in sentences(text)
                   if not any(a <= o["start"] < b for o in offenders)).strip()
