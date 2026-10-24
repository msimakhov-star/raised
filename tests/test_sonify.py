"""sonify: WAV header and length, pitch that follows the data, no clicks, rests, mixing, and a mapping the browser can use."""
import io
import json
import wave

import numpy as np
import pytest

from raised.model import Semantics, Table
from raised.sonify import RATE, TIMBRES, mapping, to_wav

SEM = Semantics("line", y_ticks=["0", "10"])
UP = Table([1, 2, 3, 4, 5, 6], {"up": [1, 2, 3, 4, 5, 6], "down": [6, 5, 4, 3, 2, 1]})


def read(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        params = w.getparams()
        return params, np.frombuffer(w.readframes(params.nframes), dtype="<i2").astype(float) / 32768


def note_pitches(samples: np.ndarray, n_points: int, spp: float) -> list[float]:
    """Dominant frequency of each note, read from the steady middle of the note (after attack, before release)."""
    out = []
    for i in range(n_points):
        a = int((i * spp + 0.05) * RATE)
        seg = samples[a:int((i * spp + spp - 0.06) * RATE)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16))
        out.append(np.argmax(spec[100:]) + 100)
        out[-1] *= RATE / (1 << 16)
    return out


def test_header_and_length():
    params, x = read(to_wav(UP, SEM, "up"))
    assert (params.nchannels, params.sampwidth, params.framerate, params.comptype) == (1, 2, RATE, "NONE")
    assert params.nframes == pytest.approx(6 * 0.25 * RATE, abs=1)
    _, slow = read(to_wav(UP, SEM, "up", seconds_per_point=0.5))
    assert len(slow) == pytest.approx(3 * RATE, abs=1)
    assert np.abs(x).max() < 0.9 and np.abs(x).max() > 0.2  # audible, not clipping


def test_pitch_rises_with_rising_data_and_matches_the_mapping():
    _, x = read(to_wav(UP, SEM, "up"))
    got = note_pitches(x, 6, 0.25)
    want = [p["hz"] for p in mapping(UP, SEM)["series"]["up"]]
    assert all(b > a for a, b in zip(got, got[1:]))
    assert got == pytest.approx(want, rel=0.03)


def test_pitch_falls_with_falling_data():
    _, x = read(to_wav(UP, SEM, "down"))
    got = note_pitches(x, 6, 0.25)
    assert all(b < a for a, b in zip(got, got[1:]))


def test_mapping_endpoints_and_geometric_midpoint():
    t = Table([1, 2, 3], {"a": [0, 5, 10]})
    hz = [p["hz"] for p in mapping(t, SEM)["series"]["a"]]
    assert hz == [220.0, 440.0, 880.0]  # two octaves, the middle value one octave up
    lin = [p["hz"] for p in mapping(t, SEM, log=False)["series"]["a"]]
    assert lin == [220.0, 550.0, 880.0]
    custom = [p["hz"] for p in mapping(t, SEM, low_hz=100, high_hz=400)["series"]["a"]]
    assert custom == [100.0, 200.0, 400.0]


def test_pitch_range_is_shared_by_all_series():
    t = Table([1, 2], {"small": [1, 2], "big": [100, 200]})
    m = mapping(t, SEM)["series"]
    assert m["small"][0]["hz"] == 220.0 and m["big"][1]["hz"] == 880.0 and m["small"][1]["hz"] < 230


def test_log_value_axis_spaces_pitch_by_decade():
    t = Table([1, 2, 3], {"a": [1, 10, 100]})
    sem = Semantics("line", y_scale="log", y_ticks=["1", "10", "100"])
    assert [p["hz"] for p in mapping(t, sem)["series"]["a"]] == [220.0, 440.0, 880.0]
    assert mapping(t, sem)["value_scale"] == "log"
    assert [p["hz"] for p in mapping(t, SEM)["series"]["a"]][1] < 260  # the same data on a linear axis crowds the low end


def test_horizontal_bars_use_the_x_axis_scale():
    t = Table(["a", "b", "c"], {"v": [1, 10, 100]})
    sem = Semantics("hbar", x_scale="log", y_scale="category", x_ticks=["1", "10", "100"])
    assert [p["hz"] for p in mapping(t, sem)["series"]["v"]] == [220.0, 440.0, 880.0]


def test_constant_series_sits_mid_range_and_missing_values_are_rests():
    assert [p["hz"] for p in mapping(Table([1, 2, 3], {"a": [5, 5, 5]}), SEM)["series"]["a"]] == [440.0] * 3
    t = Table([1, 2, 3], {"a": [5, 5, 5], "b": [1, None, 9]})
    m = mapping(t, SEM)["series"]["b"]
    assert m[1] == {"x": 2, "value": None, "hz": None}
    _, x = read(to_wav(t, SEM, "b"))
    rest = x[int(0.25 * RATE + 0.02 * RATE):int(0.5 * RATE)]  # after the 6 ms tick
    assert np.abs(rest).max() < 1e-3


def test_nan_is_a_rest_and_the_mapping_stays_valid_json():
    t = Table([1, 2, 3], {"a": [1, float("nan"), 9]})
    m = json.loads(json.dumps(mapping(t, SEM), allow_nan=False))["series"]["a"]
    assert m[1] == {"x": 2, "value": None, "hz": None} and m[2]["hz"] == 880.0
    assert len(read(to_wav(t, SEM, "a"))[1]) == pytest.approx(0.75 * RATE, abs=1)


def test_notes_do_not_click():
    _, x = read(to_wav(UP, SEM, "up"))
    for i in range(1, 6):
        edge = int(i * 0.25 * RATE)
        assert abs(x[edge] - x[edge - 1]) < 0.05  # smooth across the note boundary
    assert abs(x[0]) < 0.05 and abs(x[-1]) < 0.05


def test_a_soft_tick_marks_every_point_even_a_rest():
    t = Table([1, 2], {"a": [None, None]})
    _, x = read(to_wav(t, SEM, "a"))
    for i in range(2):
        start = int(i * 0.25 * RATE)
        assert np.abs(x[start:start + int(0.006 * RATE)]).max() > 0.05
        assert np.abs(x[start:start + int(0.006 * RATE)]).max() < 0.2  # soft


def test_series_get_different_timbres_when_mixed():
    t = Table([1, 2, 3], {"a": [5, 5, 5], "b": [5, 5, 5]})
    assert len(set(TIMBRES)) == len(TIMBRES)

    def ratio(samples):  # 2nd harmonic over fundamental in the first note
        seg = samples[int(0.05 * RATE):int(0.19 * RATE)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16))
        f = lambda hz: spec[int(hz * (1 << 16) / RATE) - 20:int(hz * (1 << 16) / RATE) + 20].max()
        return f(880) / f(440)

    single = ratio(read(to_wav(t, SEM, "a"))[1])
    mixed = ratio(read(to_wav(t, SEM, ["a", "b"]))[1])
    assert single < 0.01 and mixed > 0.05  # a triangle has no 2nd harmonic; the second series' timbre adds one
    both, none = read(to_wav(t, SEM, None))[1], read(to_wav(t, SEM, ["a", "b"]))[1]
    assert np.allclose(both, none)  # None means every series, in table order


def test_unknown_series_is_an_error():
    with pytest.raises(KeyError):
        to_wav(UP, SEM, "nope")


def test_mapping_is_json_and_has_the_shape_web_app_js_reads():
    m = json.loads(json.dumps(mapping(UP, SEM)))
    assert (m["low_hz"], m["high_hz"], m["seconds_per_point"], m["pitch"]) == (220, 880, 0.25, "log")
    assert list(m["series"]) == ["up", "down"]
    assert m["series"]["up"][0] == {"x": 1, "value": 1, "hz": 220.0}
    assert len(m["series"]["up"]) == len(UP.x)


def test_empty_table_gives_an_empty_but_valid_wav():
    params, x = read(to_wav(Table([], {"a": []}), SEM, "a"))
    assert params.nframes == 0 and len(x) == 0
