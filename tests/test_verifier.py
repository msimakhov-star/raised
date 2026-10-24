import base64
import io

from fake_tf import VERIFY_ANSWER, FakeTF, images
from PIL import Image

from raised import prompts
from raised.llm import conforms
from raised.verifier import verify


def _png(mode: str, color) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (120, 80), color).save(buf, "PNG")
    return buf.getvalue()


def _decode(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_two_images_and_issues_through_the_bare_schema_fallback():
    with FakeTF() as tf:
        issues = verify(tf.client(), _png("RGB", "white"), _png("RGBA", (214, 0, 122, 0)))
    assert issues == VERIFY_ANSWER["issues"]
    assert tf.requests[0]["response_format"]["json_schema"]["schema"] == prompts.VERIFY_SCHEMA  # 400, then bare
    body = tf.requests[-1]
    assert body["response_format"]["json_schema"] == prompts.VERIFY_SCHEMA
    original, redraw = (_decode(u) for u in images(body))
    assert redraw.mode == "RGB" and redraw.getpixel((0, 0)) == (255, 255, 255)  # transparent redraw flattened on white
    assert "TASK: VERIFY_REDRAW" in body["messages"][0]["content"][2]["text"]


def test_loose_answers_are_normalised():
    loose = '{"issues": [{"x": 2010, "series": "A", "issue": " too low "}, "junk", {"x": "1", "series": "", "issue": ""}]}'
    assert not conforms({"issues": [{"x": 2010}]}, prompts.VERIFY_SCHEMA)
    with FakeTF({"TASK: VERIFY_REDRAW": loose}) as tf:
        assert verify(tf.client(), _png("RGB", "white"), _png("RGB", "white")) == [{"x": "2010", "series": "A", "issue": "too low"}]
    with FakeTF({"TASK: VERIFY_REDRAW": {"issues": []}}) as tf:
        assert verify(tf.client(), _png("RGB", "white"), _png("RGB", "white")) == []
