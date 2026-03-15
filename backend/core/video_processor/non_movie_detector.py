"""Heuristic non-movie detector."""
from __future__ import annotations

import asyncio

import cv2
import numpy as np

from .analysis_video import ensure_analysis_video


class NonMovieDetector:
    def __init__(self):
        self.solid_color_threshold = 0.9
        self.edge_density_threshold = 0.05
        self.text_like_threshold = 0.3

    async def detect(self, video_path: str, start_time: float, end_time: float, sample_count: int = 3) -> bool:
        capture_path = await ensure_analysis_video(video_path)

        def _detect() -> bool:
            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                return False
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            duration = max(0.0, end_time - start_time)
            if duration <= 0:
                capture.release()
                return False

            scores: list[float] = []
            for idx in range(sample_count):
                sample_time = start_time + duration * (idx + 0.5) / sample_count
                capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                ok, frame = capture.read()
                if ok:
                    scores.append(self._analyze_frame(frame))
            capture.release()
            return bool(scores) and float(np.mean(scores)) >= 0.5

        return await asyncio.to_thread(_detect)

    async def detect_batch(self, video_path: str, segments: list[tuple[float, float]]) -> list[bool]:
        tasks = [self.detect(video_path, start, end) for start, end in segments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [False if isinstance(item, Exception) else bool(item) for item in results]

    def _analyze_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(cv2.resize(frame, (256, 144)), cv2.COLOR_BGR2GRAY)
        std_score = max(0.0, 1.0 - float(np.std(gray)) / 64.0)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        edge_score = 1.0 if edge_density < 0.02 else 0.5 if edge_density < self.edge_density_threshold else 0.0
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_like = 0
        frame_area = gray.shape[0] * gray.shape[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h else 0.0
            if 1.5 < aspect < 20 and 60 < area < frame_area * 0.1:
                text_like += 1
        text_score = 0.8 if text_like > 18 else 0.4 if text_like > 8 else 0.0
        return max(0.0, min(1.0, std_score * 0.4 + edge_score * 0.3 + text_score * 0.3))
