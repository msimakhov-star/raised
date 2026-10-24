"""Shared data contract. Every module reads and writes these shapes; the synthetic generator emits them as truth.

Coordinates: pixel (px, py) with origin top-left of the chart CROP image, y growing downwards.
Tick lists are in reading order: x left to right, y bottom to top.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

ChartType = Literal["line", "bar", "grouped_bar", "hbar", "scatter", "unsupported"]
Scale = Literal["linear", "log", "category"]


@dataclass
class Series:
    name: str
    color: str = ""          # "#1f77b4" or a colour name as the model reads it ("dark blue")
    style: str = "solid"     # solid | dashed | dotted
    marker: str = ""         # "", "o", "s", "^", ...


@dataclass
class Semantics:
    """What the chart means. From the vision model (three votes) or, in tests, from the generator (oracle)."""
    chart_type: ChartType
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    x_scale: Scale = "linear"
    y_scale: Scale = "linear"
    x_ticks: list[str] = field(default_factory=list)   # label strings exactly as printed
    y_ticks: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)  # bar charts: category labels in order (hbar: top to bottom)
    series: list[Series] = field(default_factory=list)
    legend: list[str] = field(default_factory=list)
    source: str = ""
    caption: str = ""

    @property
    def value_axis(self) -> str:
        """Which pixel axis carries the measured values."""
        return "x" if self.chart_type == "hbar" else "y"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Semantics":
        d = dict(d)
        d["series"] = [s if isinstance(s, Series) else Series(**{k: v for k, v in s.items() if k in Series.__dataclass_fields__})
                       for s in d.get("series", [])]
        return Semantics(**{k: v for k, v in d.items() if k in Semantics.__dataclass_fields__})


@dataclass
class Table:
    """A data reading. x holds numbers (line/scatter) or category strings (bar). One list per series, aligned with x;
    None marks a value that could not be read."""
    x: list
    series: dict[str, list[float | None]]

    def to_dict(self) -> dict:
        return {"x": self.x, "series": self.series}

    @staticmethod
    def from_dict(d: dict) -> "Table":
        return Table(list(d["x"]), {k: list(v) for k, v in d["series"].items()})


@dataclass
class Calibration:
    """value = a * px + b (linear) or log10(value) = a * px + b (log). residual is in value units (log10 units for log)."""
    kind: Literal["linear", "log"]
    a: float
    b: float
    residual: float
    n: int  # ticks used

    def value(self, px: float) -> float:
        v = self.a * px + self.b
        return 10 ** v if self.kind == "log" else v

    def px(self, value: float) -> float:
        v = math.log10(value) if self.kind == "log" else value
        return (v - self.b) / self.a

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Point:
    """One adjudicated value with provenance, for the review UI."""
    series: str
    x: object
    value: float | None
    reading_a: float | None = None   # vision model's direct reading
    reading_b: float | None = None   # pixel measurement
    status: Literal["agreed", "pixel", "model", "disputed", "missing"] = "agreed"
    note: str = ""


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_SUFFIX = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "mn": 1e6, "bn": 1e9, "B": 1e9, "b": 1e9, "T": 1e12, "t": 1e12}


def parse_number(s: str) -> float | None:
    """'1,200' -> 1200; '−5' (unicode minus) -> -5; '40%' -> 40; '$3.5k' -> 3500; '10^3' / '10³' -> 1000; '1e-2' -> 0.01.
    Returns None for non-numeric labels (years are numbers: '1990' -> 1990)."""
    if s is None:
        return None
    t = str(s).strip().replace("−", "-").replace("–", "-").replace("—", "-").replace(",", "").replace(" ", "")
    t = t.replace("£", "").replace("$", "").replace("€", "").replace("%", "")
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
    m = re.fullmatch(r"10(?:\^|\*\*)?([-\d⁰¹²³⁴-⁹⁻]+)", t)
    if m and not t.isdecimal():  # "100" is a number, "10³" and "10^3" are powers
        return 10.0 ** int(m.group(1).translate(sup))
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([kKmMbBtT]|mn|bn)?", t)
    if not m:
        return None
    v = float(m.group(1))
    return v * _SUFFIX[m.group(2)] if m.group(2) else v


def _selftest():
    assert parse_number("1,200") == 1200
    assert parse_number("−5") == -5
    assert parse_number("40%") == 40
    assert parse_number("$3.5k") == 3500
    assert parse_number("10³") == 1000 and parse_number("10^-2") == 0.01
    assert parse_number("1990") == 1990 and parse_number("Q1") is None
    c = Calibration("log", 0.01, -1.0, 0.0, 3)
    assert abs(c.px(c.value(250)) - 250) < 1e-9
    s = Semantics.from_dict({"chart_type": "line", "series": [{"name": "a", "color": "#000"}]})
    assert s.series[0].name == "a" and s.value_axis == "y"


if __name__ == "__main__":
    _selftest()
    print("model selftest ok")
