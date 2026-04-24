"""Video frame extraction helpers."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from .analysis_video import ensure_analysis_video


def read_image_unicode(path: str | Path):
    image_path = Path(path)
    if not image_path.exists():
        return None
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path: str | Path, image, params: Optional[list[int]] = None) -> bool:
    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()
    ext = '.jpg' if suffix in {'', '.jpg', '.jpeg'} else suffix
    success, encoded = cv2.imencode(ext, image, params or [])
    if not success:
        return False
    image_path.write_bytes(encoded.tobytes())
    return True


class FrameExtractor:
    def __init__(self):
        self._cache: dict[str, dict] = {}

    async def get_video_info(self, video_path: str) -> dict:
        video_path = str(video_path)
        if video_path in self._cache:
            return self._cache[video_path]
        capture_path = await ensure_analysis_video(video_path)

        def _get_info() -> dict:
            capture = cv2.VideoCapture(capture_path)
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {capture_path}')
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frame_count / fps if fps > 0 else 0.0
            capture.release()
            return {
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count,
            }

        info = await asyncio.to_thread(_get_info)
        self._cache[video_path] = info
        return info

    async def extract_frame(self, video_path: str, time_sec: float, output_dir: Optional[Path] = None) -> Path:
        video_path = str(video_path)
        frame_hash = hashlib.md5(f'{video_path}_{time_sec:.3f}'.encode()).hexdigest()[:12]
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'frame_{frame_hash}.jpg'
        if output_path.exists():
            return output_path
        capture_path = await ensure_analysis_video(video_path)

        def _extract() -> Path:
            capture = cv2.VideoCapture(capture_path)
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {capture_path}')
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            target_frame = max(0, round(time_sec * fps)) if fps > 0 else 0
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise ValueError(f'Cannot extract frame at {time_sec:.3f}s from {capture_path}')
            if not write_image_unicode(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise ValueError(f'Cannot write frame image: {output_path}')
            return output_path

        return await asyncio.to_thread(_extract)

    async def extract_thumbnail(
        self,
        video_path: str,
        time_sec: float,
        output_dir: Optional[Path] = None,
        size: tuple[int, int] = (320, 180),
    ) -> Path:
        base_frame = await self.extract_frame(video_path, time_sec, output_dir=output_dir)
        thumb_path = base_frame.with_name(base_frame.stem.replace('frame_', 'thumb_') + '.jpg')
        if thumb_path.exists():
            return thumb_path

        def _resize() -> Path:
            image = read_image_unicode(base_frame)
            if image is None:
                raise ValueError(f'Cannot read frame image: {base_frame}')
            resized = cv2.resize(image, size)
            if not write_image_unicode(thumb_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise ValueError(f'Cannot write thumbnail image: {thumb_path}')
            return thumb_path

        thumb = await asyncio.to_thread(_resize)
        logger.debug(f'Thumbnail generated: {thumb}')
        return thumb
