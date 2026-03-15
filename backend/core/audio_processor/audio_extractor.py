"""FFmpeg-based audio extraction."""
from __future__ import annotations

import asyncio
import hashlib
import subprocess
import wave
from pathlib import Path
from typing import Optional


def _resolve_ffmpeg_path() -> str:
    tools_dir = Path(__file__).resolve().parents[3] / '.tools'
    candidates = sorted(tools_dir.glob('ffmpeg-*essentials_build/bin/ffmpeg.exe'))
    if candidates:
        return str(candidates[-1])
    return 'ffmpeg'


class AudioExtractor:
    def __init__(self, ffmpeg_path: Optional[str] = None):
        self._ffmpeg_path = ffmpeg_path or _resolve_ffmpeg_path()

    async def extract_full(
        self,
        video_path: str,
        output_dir: Optional[Path] = None,
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> Path:
        video_path = str(video_path)
        audio_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'audio_full_{audio_hash}.wav'
        if output_path.exists():
            return output_path

        channels = '1' if mono else '2'
        cmd = [
            self._ffmpeg_path,
            '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', channels,
            str(output_path),
        ]
        return await asyncio.to_thread(self._run_ffmpeg, cmd, output_path, 'Audio extraction failed')

    async def extract_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_dir: Optional[Path] = None,
        sample_rate: int = 16000,
    ) -> Path:
        video_path = str(video_path)
        duration = max(0.0, end_time - start_time)
        if duration <= 0:
            raise ValueError('Audio segment duration must be positive')

        seg_hash = hashlib.md5(f'{video_path}_{start_time:.3f}_{end_time:.3f}'.encode()).hexdigest()[:12]
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'audio_seg_{seg_hash}.wav'
        if output_path.exists():
            return output_path

        cmd = [
            self._ffmpeg_path,
            '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            str(output_path),
        ]
        return await asyncio.to_thread(self._run_ffmpeg, cmd, output_path, 'Audio segment extraction failed')

    def _run_ffmpeg(self, cmd: list[str], output_path: Path, error_prefix: str) -> Path:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        except FileNotFoundError as exc:
            raise RuntimeError('FFmpeg is not installed or not available in PATH') from exc
        if result.returncode != 0:
            raise RuntimeError(f'{error_prefix}: {result.stderr}')
        return output_path

    async def get_duration(self, audio_path: str) -> float:
        path = Path(audio_path)
        if not path.exists():
            return 0.0

        def _read_duration() -> float:
            with wave.open(str(path), 'rb') as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                return frames / rate if rate else 0.0

        return await asyncio.to_thread(_read_duration)
