"""Run Raised locally:  python local.py   then open http://localhost:8790
Set NEBIUS_API_KEY (and optionally TAVILY_API_KEY) to call the models; without them it serves recorded samples."""
import os
from pathlib import Path

import uvicorn

from raised.server import build

ROOT = Path(__file__).parent
app = build({}, ROOT / "web", ROOT / "data")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8790")))
