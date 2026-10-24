"""FastAPI app for Raised. build() takes any dict-like store, so it runs the same on Modal (modal.Dict) and locally (dict).

Limits live in the app because Token Factory has no account spend cap: per-IP jobs per hour, a daily USD budget,
upload size, page and figure caps. When a limit is hit, or no key is configured, the app serves a recorded
sample and says so instead of going dark.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

MAX_UPLOAD = 15 * 1024 * 1024
PER_IP_PER_HOUR = int(os.environ.get("PER_IP_PER_HOUR", "4"))
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "0.40"))
JOB_TTL = 24 * 3600
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".pdf": "application/pdf", ".wav": "audio/wav",
        ".csv": "text/csv", ".brf": "text/plain; charset=ascii", ".zip": "application/zip", ".json": "application/json"}


def build(state, web_dir: Path, data_dir: Path) -> FastAPI:
    from . import pipeline  # imported lazily so the module can be imported without heavy deps in tests
    api = FastAPI(title="Raised")
    samples_dir = data_dir / "samples"
    ver = hashlib.sha1(b"".join((web_dir / f).read_bytes() for f in ("app.js", "style.css"))).hexdigest()[:10]

    def page(name: str = "index.html") -> str:
        return (web_dir / name).read_text().replace("/static/app.js", f"/static/app.js?v={ver}").replace("/static/style.css", f"/static/style.css?v={ver}")

    @api.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        p = request.url.path
        if p.startswith("/static/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable" if request.url.query else "no-cache"
        elif p in ("/", "/bench"):
            resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    def client_ip(req: Request) -> str:
        return (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "?")).split(",")[-1].strip()

    def over_limit(req: Request) -> str | None:
        key = f"ip:{client_ip(req)}:{int(time.time() // 3600)}"
        n = (state.get(key) or 0) + 1
        state[key] = n
        if n > PER_IP_PER_HOUR:
            return "the live demo's hourly limit for your connection is reached"
        if (state.get(f"spend:{time.strftime('%Y-%m-%d')}") or 0.0) >= DAILY_BUDGET_USD:
            return "today's model budget for the public demo is spent"
        return None

    def spend(usd: float):
        k = f"spend:{time.strftime('%Y-%m-%d')}"
        state[k] = (state.get(k) or 0.0) + float(usd or 0)

    def save_job(job: str, files: dict[str, bytes], figures_state: dict):
        state[f"job:{job}"] = {"created": time.time(), "files": files, "figs": figures_state}

    def load_job(job: str) -> dict:
        rec = state.get(f"job:{job}")
        if not rec or time.time() - rec["created"] > JOB_TTL:
            raise HTTPException(404, "this result has expired; results are kept for 24 hours")
        return rec

    # ---------- pages and static ----------
    @api.get("/", response_class=HTMLResponse)
    def index():
        return page()

    @api.get("/bench", response_class=HTMLResponse)
    def bench_page():
        return page("bench.html") if (web_dir / "bench.html").exists() else HTMLResponse("<p>Benchmark page coming soon.</p>")

    @api.get("/api/bench")
    def bench_data():
        p = data_dir / "bench" / "results.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @api.get("/api/health")
    def health():
        return {"ok": True, "live": bool(os.environ.get("NEBIUS_API_KEY")), "models": pipeline.MODELS}

    @api.get("/api/samples")
    def samples():
        p = samples_dir / "index.json"
        return json.loads(p.read_text()) if p.exists() else []

    @api.get("/samples/{sid}/{name}")
    def sample_file(sid: str, name: str):
        base = samples_dir.resolve()
        p = (samples_dir / sid / name).resolve()
        if base not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, media_type=MIME.get(p.suffix, "application/octet-stream"))

    @api.get("/api/job/{job}/{name}")
    def job_file(job: str, name: str):
        rec = load_job(job)
        data = rec["files"].get(name)
        if data is None:
            raise HTTPException(404)
        suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        headers = {"Content-Disposition": f'attachment; filename="{name}"'} if suffix in (".pdf", ".wav", ".csv", ".brf", ".zip", ".svg") else {}
        return Response(data, media_type=MIME.get(suffix, "application/octet-stream"), headers=headers)

    # ---------- run ----------
    def recorded(sid: str, reason: str):
        res = json.loads((samples_dir / sid / "result.json").read_text())
        res["recorded"], res["reason"] = True, reason
        return res

    def execute(sample: str | None, upload: tuple[bytes, str] | None, request: Request, emit) -> dict:
        index = samples()
        default = index[0]["id"] if index else None
        if sample is not None:
            if sample not in {s["id"] for s in index}:
                raise ValueError("unknown sample")
            for ev in pipeline.recorded_events(sample, samples_dir):
                emit(ev)
            return recorded(sample, "this sample was read earlier with the full pipeline")
        reason = over_limit(request) or (None if pipeline.live() else "no model key is configured on this server")
        if reason is None:
            job = secrets.token_urlsafe(9)
            try:
                res, files, figs = pipeline.run_document(upload[0], upload[1], job=job, progress=emit,
                                                         budget_left=DAILY_BUDGET_USD - (state.get(f"spend:{time.strftime('%Y-%m-%d')}") or 0.0))
                spend(res["cost"]["total_usd"])
                save_job(job, files, figs)
                return res
            except pipeline.Unreadable as e:
                raise ValueError(str(e))
            except Exception as e:  # a model or platform failure must degrade, not go dark
                reason = f"the live run failed ({type(e).__name__})"
        if default is None:
            raise ValueError(reason)
        for ev in pipeline.recorded_events(default, samples_dir):
            emit(ev)
        return recorded(default, reason + ", so a recorded sample is shown instead of your document")

    @api.post("/api/run-stream")
    async def run_stream(request: Request, sample: str | None = Form(None), file: UploadFile | None = File(None)):
        upload = None
        if sample is None:
            if file is None:
                raise HTTPException(400, "send a file or choose a sample")
            data = await file.read()
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "files must be 15 MB or smaller")
            upload = (data, file.filename or "upload")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        emit = lambda ev: loop.call_soon_threadsafe(q.put_nowait, ev)

        async def worker():
            try:
                await q.put({"result": await asyncio.to_thread(execute, sample, upload, request, emit)})
            except Exception as e:
                await q.put({"error": str(e)[:300] or type(e).__name__})

        async def events():
            task = asyncio.create_task(worker())
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if "result" in ev or "error" in ev:
                    break
            await task

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.post("/api/figure/{job}/{fig}/accept")
    async def accept(job: str, fig: str, body: dict):
        values = [v for v in body.get("values", []) if isinstance(v, dict)][:200]
        if job.startswith("sample-"):
            sid = job[len("sample-"):]
            res = recorded(sid, "sample")
            f = next((x for x in res["figures"] if x["id"] == fig), None)
            if f is None:
                raise HTTPException(404)
            figs = {fig: pipeline.figure_state_from_result(f)}
            files = {}
        else:
            rec = load_job(job)
            figs, files = rec["figs"], rec["files"]
            if fig not in figs:
                raise HTTPException(404)
            sid = None
        new_fig, new_files = await asyncio.to_thread(pipeline.accept, figs[fig], values, job)
        if sid is None:
            figs[fig] = new_fig["_state"]
            files.update(new_files)
            save_job(job, files, figs)
        else:  # samples are read-only: the accepted version lives in a short-lived job of its own
            new_job = f"edit-{secrets.token_urlsafe(6)}"
            save_job(new_job, new_files, {fig: new_fig["_state"]})
            new_fig = pipeline.retarget_urls(new_fig, job, new_job)
        new_fig.pop("_state", None)
        return new_fig

    api.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    return api
