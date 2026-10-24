"""Step 7: Super VL compares the original chart with the redraw of the extracted table.

The two go as two images in one message, not as one side-by-side composite: each keeps its full resolution under
the 2000 px shrink (a composite of two 1100 px crops would be downscaled), and the prompt can name them as image 1 and
image 2 instead of describing halves. Multi-image support on Token Factory is checked by scripts/probe.py.
A transparent redraw (the pipeline's magenta overlay) is flattened onto white first, so it reads as a chart.
"""
from __future__ import annotations

import io

from PIL import Image

from . import prompts
from .llm import MODELS, image_part


def verify(client, original_png: bytes, rerender_png: bytes, model: str = MODELS["vl"]) -> list[dict]:
    """-> [{x, series, issue}] where the redraw visibly departs from the original; [] when they match."""
    msgs = [{"role": "user", "content": [image_part(original_png), image_part(_on_white(rerender_png)),
                                         {"type": "text", "text": prompts.VERIFY}]}]
    out = client.chat_json(model, msgs, prompts.VERIFY_SCHEMA, step="verify", max_tokens=2048)
    return [{"x": str(i.get("x", "")).strip(), "series": str(i.get("series", "")).strip(), "issue": str(i.get("issue", "")).strip()}
            for i in out.get("issues") or [] if isinstance(i, dict) and str(i.get("issue", "")).strip()][:20]


def _on_white(png: bytes) -> bytes:
    im = Image.open(io.BytesIO(png))
    if im.mode not in ("RGBA", "LA", "P"):
        return png
    rgba = im.convert("RGBA")
    base = Image.new("RGB", rgba.size, "white")
    base.paste(rgba, (0, 0), rgba)
    buf = io.BytesIO()
    base.save(buf, "PNG")
    return buf.getvalue()
