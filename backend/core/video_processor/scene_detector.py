"""Scene detection helpers."""
from __future__ import annotations

import asyncio
from math import ceil
from typing import Callable, Optional

import cv2
import numpy as np

from config import settings
from .analysis_video import ensure_analysis_video


class SceneDetector:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold or settings.scene_threshold

    async def detect_scenes(
        self,
        video_path: str,
        min_scene_duration: float = 1.0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[dict]:
        capture_path = await ensure_analysis_video(video_path)
        loop = asyncio.get_running_loop()

        def _detect() -> list[dict]:
            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {capture_path}')

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 else 0.0
            step = max(1, int(fps / 2)) if fps > 0 else 1
            sampled_frames = max(1, ceil(frame_count / step)) if frame_count > 0 else 1
            report_interval = max(1, sampled_frames // 20)

            prev_hist = None
            scenes: list[dict] = []
            scene_start = 0.0
            frame_idx = 0
            sampled_idx = 0
            last_reported = -1.0

            while True:
                ok = capture.grab()
                if not ok:
                    break
                if frame_idx % step != 0:
                    frame_idx += 1
                    continue

                ok, frame = capture.retrieve()
                if not ok:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(cv2.resize(frame, (256, 144)), cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist.astype('float32'), hist.astype('float32'), cv2.HISTCMP_CHISQR)
                    current_time = frame_idx / fps if fps > 0 else 0.0
                    if diff > self.threshold and current_time - scene_start >= min_scene_duration:
                        scenes.append({'start': scene_start, 'end': current_time})
                        scene_start = current_time
                prev_hist = hist
                sampled_idx += 1
                if progress_callback and (
                    sampled_idx == 1 or sampled_idx == sampled_frames or sampled_idx % report_interval == 0
                ):
                    progress = min(39.0, 25.0 + (sampled_idx / sampled_frames) * 14.0)
                    if progress - last_reported >= 0.5 or sampled_idx == sampled_frames:
                        loop.call_soon_threadsafe(
                            progress_callback,
                            progress,
                            f"Detecting scene changes... {sampled_idx}/{sampled_frames}",
                        )
                        last_reported = progress
                frame_idx += 1

            capture.release()
            if duration <= 0:
                return []
            if not scenes or scenes[-1]['end'] < duration:
                scenes.append({'start': scene_start, 'end': duration})
            return scenes

        return await asyncio.to_thread(_detect)

    async def detect_scenes_advanced(
        self,
        video_path: str,
        min_scene_duration: float = 1.0,
        use_content_aware: bool = True,
    ) -> list[dict]:
        return await self.detect_scenes(video_path, min_scene_duration=min_scene_duration)
