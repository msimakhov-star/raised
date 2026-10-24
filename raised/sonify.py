"""Sound for the keyboard explorer and the downloadable WAV: one pitch mapping, two renderers.

    mapping(table, sem, seconds_per_point=0.25, low_hz=220, high_hz=880, log=True) -> dict (JSON-ready)
    to_wav(table, sem, series, seconds_per_point=0.25, low_hz=220, high_hz=880, log=True) -> bytes

The mapping is what web/app.js reads (`series[name][i] = {x, value, hz}`), so the browser plays exactly the pitches
that the WAV contains.
  * The range is taken over ALL series, so a series keeps its pitch whether it is played alone or mixed.
  * The lowest value maps to low_hz and the highest to high_hz. With log=True the frequency in between is geometric
    (equal value steps give equal musical intervals; 220 to 880 Hz is two octaves), with log=False it is linear in Hz.
  * On a logarithmic value axis the position between the extremes follows log10(value), as the picture does.
  * A missing value is a rest; a constant series sits at the middle of the range.
Rendering: one note per point (a triangle wave for the first series, a different harmonic mix for each further series),
a 10 ms attack and a 30 ms release so notes do not click, and a short soft tick at the start of every point so the
listener hears the x position even through a rest. Several series are mixed at once. 16-bit mono PCM through the
stdlib wave module.
"""
from __future__ import annotations

import io
import math
import wave

import numpy as np

from .facts import value_scale
from .model import Semantics, Table

RATE = 22050  # Hz; the fifth harmonic of the highest default note is 4.4 kHz, well under Nyquist
TICK_HZ, TICK_MS, TICK_GAIN = 1800, 6, 0.12
ATTACK_S, RELEASE_S = 0.010, 0.030
# Harmonic amplitudes per timbre: triangle (odd harmonics, 1/n squared; what the browser oscillator plays), sawtooth-like,
# square-like, hollow. A series uses TIMBRES[index % 4].
TIMBRES = ((1, 0, 1 / 9, 0, 1 / 25), (1, 1 / 2, 1 / 3, 1 / 4), (1, 0, 1 / 3, 0, 1 / 5), (1, 0.6, 0, 0.4))


def _read(v) -> bool:
    return v is not None and v == v  # None and NaN both mean "not read"


def _position(table: Table, sem: Semantics):
    """value -> 0..1 between the lowest and highest value of the whole table (unread values pass through as None)."""
    vals = [v for vs in table.series.values() for v in vs if _read(v)]
    if not vals:
        return lambda v: None
    f = math.log10 if value_scale(sem) == "log" and min(vals) > 0 else float
    lo, hi = f(min(vals)), f(max(vals))
    return lambda v: None if not _read(v) else 0.5 if hi == lo else (f(v) - lo) / (hi - lo)


def mapping(table: Table, sem: Semantics, seconds_per_point: float = 0.25, low_hz: float = 220, high_hz: float = 880,
            log: bool = True) -> dict:
    pos = _position(table, sem)

    def hz(v):
        r = pos(v)
        return None if r is None else round(low_hz * (high_hz / low_hz) ** r if log else low_hz + r * (high_hz - low_hz), 2)

    return {"low_hz": low_hz, "high_hz": high_hz, "seconds_per_point": seconds_per_point, "pitch": "log" if log else "linear",
            "value_scale": value_scale(sem),
            "series": {n: [{"x": x, "value": v if _read(v) else None, "hz": hz(v)} for x, v in zip(table.x, vs)]
                       for n, vs in table.series.items()}}


def _note(hz: float, n: int, amps: tuple) -> np.ndarray:
    t = np.arange(n) / RATE
    wave_ = sum(a * np.sin(2 * np.pi * hz * (k + 1) * t) for k, a in enumerate(amps) if a) / sum(amps)
    env = np.ones(n)
    for ramp, side in ((int(ATTACK_S * RATE), slice(None)), (int(RELEASE_S * RATE), slice(None, None, -1))):
        r = min(ramp, n // 3)
        env[side][:r] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / max(r, 1))
    return 0.5 * wave_ * env


def _tick(n: int) -> np.ndarray:
    m = min(n, int(TICK_MS / 1000 * RATE))
    return np.pad(TICK_GAIN * np.hanning(m) * np.sin(2 * np.pi * TICK_HZ * np.arange(m) / RATE), (0, n - m))


def to_wav(table: Table, sem: Semantics, series: str | list[str] | None = None, seconds_per_point: float = 0.25,
           low_hz: float = 220, high_hz: float = 880, log: bool = True) -> bytes:
    """WAV of one series (a name), several mixed (a list) or all (None). Length is exactly len(x) * seconds_per_point."""
    names = list(table.series) if series is None else [series] if isinstance(series, str) else list(series)
    unknown = [n for n in names if n not in table.series]
    if unknown:
        raise KeyError(f"no such series: {unknown}")
    mp = mapping(table, sem, seconds_per_point, low_hz, high_hz, log)["series"]
    edges = [round(i * seconds_per_point * RATE) for i in range(len(table.x) + 1)]
    out = np.zeros(edges[-1])
    for j, name in enumerate(names):
        amps = TIMBRES[j % len(TIMBRES)]
        for i, pt in enumerate(mp[name]):
            a, b = edges[i], edges[i + 1]
            if pt["hz"] is not None:
                out[a:b] += _note(pt["hz"], b - a, amps) / len(names)
    for a, b in zip(edges, edges[1:]):
        out[a:b] += _tick(b - a)
    pcm = (np.clip(out, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
