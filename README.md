# Raised

Charts in textbooks, checked and turned into data tables, descriptions, sound and tactile graphics for blind and low-vision students.

Live demo: https://m-simakhov--raised-web.us-east.modal.run · Benchmark: https://m-simakhov--raised-web.us-east.modal.run/bench

Built with NVIDIA Nemotron 3.5 Super VL, Nemotron 3 Ultra and Nemotron 3.5 Lightning on Nebius Token Factory.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install fastapi[standard] pillow numpy scipy matplotlib pypdfium2 pytesseract shapely cairosvg pytest
brew install tesseract liblouis cairo          # Debian: apt install tesseract-ocr liblouis-bin libcairo2
export NEBIUS_API_KEY=...                       # omit to serve recorded samples
.venv/bin/python local.py                       # http://localhost:8790
.venv/bin/python -m pytest tests
```

Benchmark: `python -m bench.make_synth --split bench --n 300 --seed 1`, then `python -m bench.run --config pixel_only --split synth-bench`.

Deploy: `modal secret create raised NEBIUS_API_KEY=... && modal deploy app.py`.

Sample pages: OpenStax, *Principles of Economics 2e* (Greenlaw, Shapiro et al.), CC BY 4.0. Licence: MIT.
