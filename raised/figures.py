"""Steps 0 and 1: PDF pages to images plus text layer (pypdfium2), then figure boxes and captions from Super VL.

pypdfium2 API as documented in the installed package (5.14.0): PdfDocument(input) takes bytes; PdfPage.render(scale)
where scale is pixels per PDF canvas unit, "usually 1/72in", so scale = dpi / 72; PdfBitmap.to_pil();
PdfPage.get_textpage().get_text_range() returns the page text with CRLF line breaks.
"""
from __future__ import annotations

import io
import re

import pypdfium2 as pdfium
from PIL import Image

from . import prompts
from .llm import MODELS, image_part


def render_pdf(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 40) -> list[tuple[int, Image.Image, str]]:
    """-> [(page_index, RGB page image, text layer)] for the first max_pages pages."""
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        pages = []
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            img = page.render(scale=dpi / 72).to_pil().convert("RGB")
            pages.append((i, img, page.get_textpage().get_text_range().replace("\r\n", "\n")))
        return pages
    finally:
        doc.close()


def find_figures(client, page_img: Image.Image, text: str = "", model: str = MODELS["vl"]) -> list[dict]:
    """-> [{bbox: [x0, y0, x1, y1] on a 0..1000 page grid, caption, figure_number, is_data_chart}]."""
    buf = io.BytesIO()
    page_img.save(buf, "PNG")
    msgs = [{"role": "user", "content": [image_part(buf.getvalue()), {"type": "text", "text": prompts.figures_prompt(text)}]}]
    out = client.chat_json(model, msgs, prompts.FIGURES_SCHEMA, step="find", max_tokens=4096)
    figures = []
    for f in out.get("figures") or []:
        b = f.get("box") if isinstance(f, dict) else None
        try:
            x0, x1 = sorted(min(1000, max(0, int(b[k]))) for k in ("x0", "x1"))
            y0, y1 = sorted(min(1000, max(0, int(b[k]))) for k in ("y0", "y1"))
        except (TypeError, KeyError, ValueError):
            continue
        if x1 - x0 < 20 or y1 - y0 < 20:  # under 2% of the page: not a figure
            continue
        figures.append({"bbox": [x0, y0, x1, y1], "caption": str(f.get("caption") or "").strip(),
                        "figure_number": _number(f.get("figure_number")), "is_data_chart": f.get("is_data_chart") is True})
    return figures


def crop(page_img: Image.Image, bbox: list[int], pad: float = 0.04) -> Image.Image:
    """Cut a 0..1000-grid box out of the page, grown by pad times the page size on every side (model boxes tend to
    clip tick labels, which tick localisation needs), clamped to the page."""
    w, h = page_img.size
    x0, y0, x1, y1 = (v / 1000 for v in bbox)
    return page_img.crop((max(0, round((x0 - pad) * w)), max(0, round((y0 - pad) * h)),
                          min(w, round((x1 + pad) * w)), min(h, round((y1 + pad) * h))))


def paragraph_citing(text: str, figure_number: str) -> str:
    """The body text that cites "Figure <n>": each citing sentence with one sentence either side, without the
    caption (a line starting "Figure <n>" followed by a capitalised title). Lines are joined first, since PDF text
    layers break lines but rarely mark paragraphs."""
    num = _number(figure_number)
    if not num:
        return ""
    label = rf"Fig(?:ure|\.)?\s*{re.escape(num)}\b"
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    body = " ".join(ln for ln in lines if ln and not re.match(rf"{label}\s*[.:]?\s+[A-Z]", ln))
    body = re.sub(r"(\w)- (?=[a-z])", r"\1", body)  # re-join words hyphenated across lines
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"“])", body)
    keep = sorted({j for i, s in enumerate(sents) if re.search(label, s)
                   for j in range(max(0, i - 1), min(len(sents), i + 2))})
    return " ".join(sents[j] for j in keep)


def _number(v) -> str:
    """'Figure 19.10' / 'Fig. 3' / '19.10' -> '19.10' / '3'."""
    return re.sub(r"^(?:fig(?:ure)?\.?)\s*", "", str(v or "").strip(), flags=re.I).rstrip(".:")
