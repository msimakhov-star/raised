import io

import matplotlib
import pytest

matplotlib.use("Agg")
from fake_tf import FakeTF, images  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from raised.figures import crop, find_figures, paragraph_citing, render_pdf  # noqa: E402

BODY = "Exports grew quickly after 2000, as Figure 1.2 shows. Imports rose more slowly. Both fell in 2009."
CAPTION = "Figure 1.2 Exports and imports, 2000 to 2010"


@pytest.fixture(scope="module")
def pdf() -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pages:
        fig = Figure(figsize=(8.5, 11))
        fig.text(0.1, 0.92, BODY, fontsize=9)
        ax = fig.add_axes((0.15, 0.5, 0.7, 0.3))
        ax.plot([2000, 2005, 2010], [2, 5.5, 7.9], label="Exports")
        ax.legend()
        fig.text(0.1, 0.42, CAPTION, fontsize=9)
        pages.savefig(fig)
        pages.savefig(Figure(figsize=(8.5, 11)))
    return buf.getvalue()


def test_render_pdf_pages_and_text_layer(pdf):
    pages = render_pdf(pdf, dpi=100)
    assert [p[0] for p in pages] == [0, 1]
    assert pages[0][1].size == (850, 1100) and pages[0][1].mode == "RGB"
    assert "Figure 1.2 shows" in pages[0][2] and CAPTION in pages[0][2]
    assert len(render_pdf(pdf, dpi=100, max_pages=1)) == 1


def test_find_figures_then_crop(pdf):
    _, page, text = render_pdf(pdf, dpi=100)[0]
    with FakeTF() as tf:
        figs = find_figures(tf.client(), page, text)
    assert figs == [{"bbox": [100, 200, 900, 600], "caption": CAPTION, "figure_number": "1.2", "is_data_chart": True}]
    body = tf.requests[-1]
    assert len(images(body)) == 1 and "Figure 1.2 shows" in body["messages"][0]["content"][1]["text"]
    # pad is 4% of the page on each side: x 100..900 of 850 px -> 51..799, y 200..600 of 1100 px -> 176..704
    assert crop(page, figs[0]["bbox"]).size == (748, 528)
    assert crop(page, [0, 0, 1000, 1000]).size == page.size


def test_find_figures_cleans_model_boxes():
    page = Image.new("RGB", (100, 100), "white")
    answer = {"figures": [
        {"figure_number": "Figure 3.", "caption": " c ", "is_data_chart": False, "box": {"x0": 900, "y0": 1200, "x1": 100, "y1": -5}},
        {"figure_number": "", "caption": "", "is_data_chart": True, "box": {"x0": 10, "y0": 10, "x1": 15, "y1": 500}},
    ]}
    with FakeTF({"TASK: FIND_FIGURES": answer}) as tf:
        figs = find_figures(tf.client(), page)
    assert figs == [{"bbox": [100, 0, 900, 1000], "caption": "c", "figure_number": "3", "is_data_chart": False}]


def test_paragraph_citing_skips_the_caption():
    text = ("Some earlier text about trade. Exports grew quickly after 2000, as Figure 1.2 shows. Imports rose\n"
            "more slowly. Figure 1.20 is something else. Unrelated end.\n" + CAPTION + " The blue line is exports.\n")
    got = paragraph_citing(text, "Figure 1.2")
    assert got == "Some earlier text about trade. Exports grew quickly after 2000, as Figure 1.2 shows. Imports rose more slowly."
    assert paragraph_citing(text, "") == "" and paragraph_citing(text, "9.9") == ""
