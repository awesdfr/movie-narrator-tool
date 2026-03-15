"""Prepare video-only analysis proxies for containers with noisy side streams."""
from __future__ import annotations

import asyncio
import hashlib
import subprocess
from pathlib import Path

from loguru import logger

from config import settings

_PROXY_EXTENSIONS = {".mkv", ".webm"}
_CACHE: dict[str, str] = {}
_PROBE_CACHE: dict[str, bool] = {}


def _resolve_ffmpeg_path() -> str:
    tools_dir = Path(__file__).resolve().parents[3] / ".tools"
    candidates = sorted(tools_dir.glob("ffmpeg-*essentials_build/bin/ffmpeg.exe"))
    if candidates:
        return str(candidates[-1])
    return "ffmpeg"


def _resolve_ffprobe_path() -> str:
    tools_dir = Path(__file__).resolve().parents[3] / ".tools"
    candidates = sorted(tools_dir.glob("ffmpeg-*essentials_build/bin/ffprobe.exe"))
    if candidates:
        return str(candidates[-1])
    return "ffprobe"


def should_prepare_analysis_video(video_path: str | Path) -> bool:
    path = Path(video_path).resolve()
    if path.suffix.lower() in _PROXY_EXTENSIONS:
        return True
    cache_key = str(path)
    cached = _PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    probe_result = _probe_needs_proxy(path)
    _PROBE_CACHE[cache_key] = probe_result
    return probe_result


async def ensure_analysis_video(video_path: str | Path) -> str:
    return await asyncio.to_thread(ensure_analysis_video_sync, video_path)


def ensure_analysis_video_sync(video_path: str | Path) -> str:
    source_path = Path(video_path).resolve()
    if not should_prepare_analysis_video(source_path):
        return str(source_path)

    try:
        stat = source_path.stat()
    except OSError:
        return str(source_path)

    cache_key = f"{source_path}|{stat.st_size}|{int(stat.st_mtime)}"
    cached = _CACHE.get(cache_key)
    if cached and Path(cached).exists():
        return cached

    proxy_dir = settings.temp_dir / "analysis_video"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:16]
    proxy_path = proxy_dir / f"{proxy_hash}.mkv"
    if proxy_path.exists() and proxy_path.stat().st_size > 0:
        _CACHE[cache_key] = str(proxy_path)
        return str(proxy_path)

    ffmpeg_path = _resolve_ffmpeg_path()
    copy_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "copy",
        str(proxy_path),
    ]
    transcode_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(proxy_path),
    ]

    for cmd in (copy_cmd, transcode_cmd):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        except FileNotFoundError:
            logger.warning("FFmpeg unavailable while preparing analysis proxy; using original video")
            return str(source_path)
        if result.returncode == 0 and proxy_path.exists() and proxy_path.stat().st_size > 0:
            logger.info(f"Prepared analysis video proxy for {source_path.name}: {proxy_path.name}")
            _CACHE[cache_key] = str(proxy_path)
            return str(proxy_path)

    logger.warning(f"Analysis video proxy failed for {source_path.name}; falling back to original stream")
    return str(source_path)


def _probe_needs_proxy(source_path: Path) -> bool:
    ffprobe_path = _resolve_ffprobe_path()
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=format_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    format_name = (result.stdout or "").strip().lower()
    return "matroska" in format_name or "webm" in format_name
