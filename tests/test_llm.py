import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_tf import FakeTF
from PIL import Image

from raised.llm import MODELS, Meter, TFError, conforms, image_part, parse_json, shrink

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
MSG = [{"role": "user", "content": "TASK: ADD two and two"}]


def test_bare_schema_fallback_is_remembered():
    with FakeTF({"TASK: ADD": {"answer": 4}}) as tf:
        c = tf.client()
        assert c.chat_json(MODELS["fast"], MSG, SCHEMA) == {"answer": 4}
        assert c.chat_json(MODELS["fast"], MSG, SCHEMA) == {"answer": 4}
    sent = [r["response_format"]["json_schema"] for r in tf.requests]
    assert sent[0]["schema"] == SCHEMA and sent[1:] == [SCHEMA, SCHEMA]  # one 400, then straight to the bare form
    assert len(c.meter.calls) == 2


def test_rejected_reasoning_effort_is_dropped():
    with FakeTF({"TASK: ADD": {"answer": 4}}, reject_wrapper=False, reject_params=("reasoning_effort",)) as tf:
        c = tf.client()
        assert c.chat_json(MODELS["reason"], MSG, SCHEMA, reasoning_effort="medium") == {"answer": 4}
        c.chat_json(MODELS["reason"], MSG, SCHEMA, reasoning_effort="medium")
    # the first 400 is blamed on the wrapper (bare form next), the second on reasoning_effort; both stick
    assert ["reasoning_effort" in r for r in tf.requests] == [True, True, False, False]


def test_ignored_n_falls_back_to_separate_calls_at_temperature_07():
    with FakeTF({"TASK: ADD": [{"answer": 1}, {"answer": 2}, {"answer": 3}]}, reject_wrapper=False) as tf:
        c = tf.client()
        out = c.chat_json_n(MODELS["vl"], MSG, SCHEMA, 3)
    assert sorted(o["answer"] for o in out) == [1, 2, 3]
    assert [r.get("n") for r in tf.requests] == [3, None, None]
    assert {r["temperature"] for r in tf.requests} == {0.7}
    assert [m["cached"] for m in c.meter.calls] == [0, 800, 800]  # the first call warms the cache for the others


def test_rejected_n_falls_back_to_separate_calls():
    with FakeTF({"TASK: ADD": {"answer": 2}}, reject_wrapper=False, reject_params=("n",)) as tf:
        assert len(tf.client().chat_json_n(MODELS["vl"], MSG, SCHEMA, 3)) == 3
    assert [r.get("n") for r in tf.requests] == [3, 3, None, None, None]  # wrapper and bare both 400 on n


def test_honoured_n_is_one_request():
    with FakeTF({"TASK: ADD": [{"answer": 1}, {"answer": 2}, {"answer": 3}]}, reject_wrapper=False, honour_n=True) as tf:
        out = tf.client().chat_json_n(MODELS["vl"], MSG, SCHEMA, 3)
    assert len(tf.requests) == 1 and [o["answer"] for o in out] == [1, 2, 3]


def test_unparseable_vote_is_dropped_and_all_bad_raises():
    with FakeTF({"TASK: ADD": ["no json here", {"answer": 2}, {"answer": 2}]}, reject_wrapper=False) as tf:
        assert len(tf.client().chat_json_n(MODELS["vl"], MSG, SCHEMA, 3)) == 2
    with FakeTF({"TASK: ADD": "still no json"}, reject_wrapper=False) as tf, pytest.raises(TFError):
        tf.client().chat_json_n(MODELS["vl"], MSG, SCHEMA, 3)


def test_cached_tokens_billed_at_cached_price():
    m = Meter()
    usage = {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 800}}
    m.add(MODELS["vl"], usage, 5, "s")
    assert m.calls[0]["usd"] == round((200 * 0.30 + 800 * 0.06 + 200 * 0.90) / 1e6, 6)
    m.add(MODELS["fast"], usage, 5, "s")  # Lightning lists no cached price: all input at the full price
    assert m.calls[1]["usd"] == round((1000 * 0.06 + 200 * 0.24) / 1e6, 6)
    m.add(MODELS["reason"], {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": None}, 5, "s")
    assert m.calls[2]["cached"] == 0 and m.cached_share == round(1600 / 2010, 3)


def test_cached_tokens_reach_the_meter_through_the_client():
    with FakeTF({"TASK: ADD": {"answer": 4}}, reject_wrapper=False) as tf:
        c = tf.client()
        c.chat_json(MODELS["vl"], MSG, SCHEMA)
        c.chat_json(MODELS["vl"], MSG, SCHEMA)
    assert [m["cached"] for m in c.meter.calls] == [0, 800]
    assert c.meter.calls[1]["usd"] < c.meter.calls[0]["usd"]


def test_retry_after_429_and_rate_headers():
    with FakeTF({"TASK: ADD": {"answer": 4}}, reject_wrapper=False, fail=[429]) as tf:
        c = tf.client()
        assert c.chat_json(MODELS["vl"], MSG, SCHEMA) == {"answer": 4}
    assert len(tf.requests) == 2 and c.last_headers["x-ratelimit-remaining-requests"] == "99"
    with FakeTF(fail=[401]) as tf, pytest.raises(TFError) as e:
        tf.client().chat(MODELS["vl"], MSG)
    assert e.value.status == 401


def test_parse_json_skips_reasoning_and_prose():
    assert parse_json({"content": '<think>maybe {"answer": 3}</think>Sure: {"answer": 4} done'}) == {"answer": 4}
    with pytest.raises(ValueError):
        parse_json({"content": None})


def test_conforms():
    s = {"type": "object", "properties": {"v": {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
                                          "k": {"type": "string", "enum": ["a", "b"]}},
         "required": ["v", "k"], "additionalProperties": False}
    assert conforms({"v": [1, 2.5, None], "k": "a"}, s)
    assert not conforms({"v": [1, "x"], "k": "a"}, s)
    assert not conforms({"v": [], "k": "c"}, s)
    assert not conforms({"v": [], "k": "a", "extra": 1}, s)
    assert not conforms({"v": [True], "k": "a"}, s)


def _image(size, fmt) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, fmt)
    return buf.getvalue()


def test_shrink_caps_charts_at_2000_and_keeps_png():
    out, mime = shrink(_image((3000, 1000), "PNG"), "image/png")
    assert mime == "image/png" and max(Image.open(io.BytesIO(out)).size) == 2000
    small = _image((800, 600), "JPEG")
    assert shrink(small, "image/jpeg") == (small, "image/jpeg")
    assert image_part(_image((10, 10), "PNG"))["image_url"]["url"].startswith("data:image/png;base64,")


def test_probe_exits_clearly_without_key():
    env = {k: v for k, v in os.environ.items() if k != "NEBIUS_API_KEY"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "probe.py")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "NEBIUS_API_KEY is not set" in r.stderr


def test_probe_runs_every_check_against_the_fake(tmp_path):
    routes = {"TASK: PROBE_OK": "OK", "TASK: PROBE_COUNT": "2", "TASK: PROBE_MATH": {"answer": 391},
              "TASK: PROBE_EXTRACT": {"city": "Leeds", "year": 2021}}
    out = tmp_path / "probe_report.json"
    with FakeTF(routes, honour_n=True) as tf:
        env = {**os.environ, "NEBIUS_API_KEY": "test-key", "NEBIUS_BASE_URL": tf.url}
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "probe.py"), "--charts", "2", "--out", str(out)],
                           env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())
    checks = ("schema_forms", "n", "cached", "image_sizes", "multi_image", "vl_reasoning", "ultra", "lightning", "minicpm",
              "find_figures", "semantics")
    assert all("error" not in rep[k] for k in checks), {k: rep[k].get("error") for k in checks}
    assert rep["schema_forms"]["openai"]["status"] == 400 and rep["schema_forms"]["bare"]["valid"] is True
    assert rep["schema_form_used"] == "bare" and rep["n"]["honoured"] is True
    assert rep["cached"] == {"first": 0, "second": 800, "prompt_tokens": 1000, "statuses": [200, 200]}
    assert rep["ultra"]["correct"] is True and rep["semantics"]["charts"] == 2
    assert rep["rate_limits"]["last_response"]["x-ratelimit-remaining-requests"] == "99"


def test_probe_scores_reading_a_against_truth():
    from scripts.probe import reading_errors

    from raised.model import Semantics, Table
    line = Semantics("line", y_ticks=["0", "5", "10"])
    got = reading_errors(line, Table([1, 2, 3], {"A": [1, 2, 3]}), Table([1.0, 2.1, 3.0, 9.0], {"a": [1, 2, 4, 5]}))
    assert got == [0, 0, 0.1]  # x pairs to the nearest reading within half the spacing; 9.0 is unmatched
    log = Semantics("line", y_scale="log", y_ticks=["1", "10", "100"])
    assert reading_errors(log, Table([1], {"A": [10]}), Table([1], {"B": [100]})) == [0.5]  # one series each: paired
    bars = Semantics("hbar", x_ticks=["0", "50"], y_ticks=["UK", "France"], categories=["France", "UK"])
    assert reading_errors(bars, Table(["France", "UK"], {"S": [10, 20]}), Table(["uk", "france"], {"S": [25, 10]})) == [0, 0.1]
