"""Raised on Modal: one scale-to-zero container serving the web app and the API.

    modal deploy app.py
    modal serve app.py
    modal secret create raised NEBIUS_API_KEY=... TAVILY_API_KEY=...   # once; without it the app runs in recorded mode
"""
from pathlib import Path

import modal

ROOT = Path(__file__).parent
app = modal.App("raised")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("tesseract-ocr", "liblouis-bin", "liblouis-data", "libcairo2", "fonts-dejavu-core")
         .pip_install("fastapi[standard]>=0.115", "pillow>=10", "numpy>=2", "scipy>=1.13", "matplotlib>=3.9",
                      "pypdfium2>=4.30", "pytesseract>=0.3.13", "shapely>=2", "cairosvg>=2.7")
         .add_local_dir(ROOT / "raised", remote_path="/root/raised")
         .add_local_dir(ROOT / "web", remote_path="/root/web")
         .add_local_dir(ROOT / "data", remote_path="/root/data"))
state = modal.Dict.from_name("raised-state", create_if_missing=True)

try:
    secrets = [modal.Secret.from_name("raised")]
except Exception:
    secrets = []


@app.function(image=image, secrets=secrets, scaledown_window=180, timeout=600, cpu=2.0, memory=2048)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from raised.server import build
    return build(state, Path("/root/web"), Path("/root/data"))


@app.function(image=image, schedule=modal.Cron("0 6 * * *"))
def daily_health():
    """Logs whether the public demo still answers; visible in the Modal dashboard."""
    import urllib.request
    url = web.get_web_url()
    with urllib.request.urlopen(url + "/api/health", timeout=60) as r:
        print("health", r.status, r.read()[:200])
