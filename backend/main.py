"""FastAPI application entrypoint."""
from __future__ import annotations

import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from api.routes import files, preview, process_v2 as process, project, settings as settings_api
from api.websocket import router as ws_router
from config import settings

settings.ensure_dirs()

LOG_DIR = settings.temp_dir / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.debug else "INFO",
)
logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="00:00",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
)
logger.add(
    LOG_DIR / "match_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG",
    filter=lambda record: "frame_matcher" in record["name"] or "process" in record["name"],
    rotation="00:00",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Temp directory: {settings.temp_dir}")
    logger.info(f"Projects directory: {settings.projects_dir}")
    logger.info(f"Videos directory: {settings.videos_dir}")
    logger.info(f"Static directory: {STATIC_DIR}")
    recovered = 0
    for file_path in settings.projects_dir.glob("*.json"):
        if file_path.name == "settings.json":
            continue
        if "." in file_path.stem:
            continue
        try:
            stale_project = project.load_project(file_path.stem)
            if stale_project and project.recover_stale_project(
                stale_project,
                reason="上次任务因服务重启中断，当前状态已重置，请重新点击开始处理或重匹配。",
            ):
                recovered += 1
        except Exception as exc:
            logger.warning(f"Failed to recover stale project state for {file_path}: {exc}")
    if recovered:
        logger.info(f"Recovered {recovered} stale in-progress project(s) on startup")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    description="Movie narration workflow tool with segment alignment, polishing, and TTS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project.router, prefix="/api/project", tags=["project"])
app.include_router(process.router, prefix="/api/process", tags=["process"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["settings"])
app.include_router(preview.router, prefix="/api/preview", tags=["preview"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(ws_router, prefix="/ws", tags=["websocket"])

if (settings.temp_dir).exists():
    app.mount("/temp", StaticFiles(directory=str(settings.temp_dir)), name="temp")

if STATIC_DIR.exists() and (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api")
async def api_info():
    return {"name": settings.app_name, "version": "1.0.0", "status": "running"}


@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    if full_path.startswith(("api/", "ws/", "temp/", "assets/")):
        return {"error": "Not found"}

    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return HTMLResponse(
        content="""
        <html>
        <head><title>Movie Narrator Tool</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>Movie Narrator Tool</h1>
            <p>Frontend assets are not built yet. Run:</p>
            <pre style="background: #f5f5f5; padding: 20px; display: inline-block; text-align: left;">cd frontend\nnpm install\nnpm run build</pre>
            <p>API docs: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """,
        status_code=200,
    )


if __name__ == "__main__":
    import threading
    import time
    import webbrowser

    import uvicorn

    def open_browser() -> None:
        time.sleep(2)
        webbrowser.open(f"http://{settings.host}:{settings.port}")

    if settings.debug:
        threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"Serving on http://{settings.host}:{settings.port}")
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
