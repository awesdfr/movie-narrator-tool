"""FFmpeg-based video clipping."""
from __future__ import annotations

import asyncio
import hashlib
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


def _resolve_ffmpeg_path() -> str:
    tools_dir = Path(__file__).resolve().parents[3] / '.tools'
    candidates = sorted(tools_dir.glob('ffmpeg-*essentials_build/bin/ffmpeg.exe'))
    if candidates:
        return str(candidates[-1])
    return 'ffmpeg'


class VideoClipper:
    def __init__(self, ffmpeg_path: Optional[str] = None):
        self._ffmpeg_path = ffmpeg_path or _resolve_ffmpeg_path()

    async def clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_dir: Optional[Path] = None,
        output_name: Optional[str] = None,
    ) -> Path:
        video_path = str(video_path)
        duration = max(0.0, end_time - start_time)
        if duration <= 0:
            raise ValueError('Clip duration must be positive')

        if not output_name:
            clip_hash = hashlib.md5(f'{video_path}_{start_time:.3f}_{end_time:.3f}'.encode()).hexdigest()[:12]
            output_name = f'clip_{clip_hash}'

        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{output_name}.mp4'
        if output_path.exists():
            return output_path

        cmd = [
            self._ffmpeg_path,
            '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',
            str(output_path),
        ]

        def _run() -> Path:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                logger.error(result.stderr)
                raise RuntimeError(f'Video clip failed: {result.stderr}')
            return output_path

        return await asyncio.to_thread(_run)

    async def clip_with_audio_replace(
        self,
        video_path: str,
        audio_path: str,
        start_time: float,
        end_time: float,
        output_dir: Optional[Path] = None,
        output_name: Optional[str] = None,
    ) -> Path:
        video_path = str(video_path)
        audio_path = str(audio_path)
        duration = max(0.0, end_time - start_time)
        if duration <= 0:
            raise ValueError('Clip duration must be positive')

        if not output_name:
            clip_hash = hashlib.md5(f'{video_path}_{audio_path}_{start_time:.3f}_{end_time:.3f}'.encode()).hexdigest()[:12]
            output_name = f'clip_audio_{clip_hash}'

        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{output_name}.mp4'
        if output_path.exists():
            return output_path

        cmd = [
            self._ffmpeg_path,
            '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-i', audio_path,
            '-t', str(duration),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            str(output_path),
        ]

        def _run() -> Path:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                logger.error(result.stderr)
                raise RuntimeError(f'Video/audio mux failed: {result.stderr}')
            return output_path

        return await asyncio.to_thread(_run)
