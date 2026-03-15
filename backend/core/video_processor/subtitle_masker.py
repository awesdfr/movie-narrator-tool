"""Language-agnostic subtitle region masking helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from models.project import SubtitleMaskMode
from .analysis_video import ensure_analysis_video_sync

try:
    from paddleocr import PaddleOCR

    PADDLE_OCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    PaddleOCR = None
    PADDLE_OCR_AVAILABLE = False

_OCR_INSTANCE: Optional["PaddleOCR"] = None


def _get_ocr() -> "PaddleOCR":
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        if not PADDLE_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR is not available")
        logger.info("Loading PaddleOCR for subtitle masking")
        _OCR_INSTANCE = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False, use_gpu=False)
    return _OCR_INSTANCE


@dataclass
class SubtitleMaskerConfig:
    sample_count: int = 8
    bottom_ratio: float = 0.36
    fallback_bottom_ratio: float = 0.14
    vote_ratio_threshold: float = 0.34
    mask_dilate_pixels: int = 6
    min_box_width_ratio: float = 0.18
    min_box_height: int = 10
    max_box_height_ratio: float = 0.18
    median_blur_ksize: int = 21
    ocr_confidence_threshold: float = 0.55
    use_ocr: bool = True
    use_heuristic: bool = True


class SubtitleMasker:
    """Build a stable subtitle mask for a video and apply it to frames."""

    def __init__(
        self,
        config: Optional[SubtitleMaskerConfig] = None,
        manual_regions: Optional[list[dict]] = None,
        mask_mode: SubtitleMaskMode | str = SubtitleMaskMode.HYBRID,
    ):
        self.config = config or SubtitleMaskerConfig()
        self.mask_mode = mask_mode if isinstance(mask_mode, SubtitleMaskMode) else SubtitleMaskMode(str(mask_mode))
        self._manual_regions = self._normalize_regions(manual_regions or [])
        self._fixed_mask: Optional[np.ndarray] = None
        self._frame_shape: Optional[tuple[int, int]] = None

    @property
    def has_mask(self) -> bool:
        return self.has_fixed_mask or bool(self._manual_regions)

    @property
    def has_fixed_mask(self) -> bool:
        return self._fixed_mask is not None and bool(self._fixed_mask.any())

    @property
    def has_manual_regions(self) -> bool:
        return bool(self._manual_regions)

    @property
    def uses_auto_detection(self) -> bool:
        return self.mask_mode in {SubtitleMaskMode.HYBRID, SubtitleMaskMode.AUTO_ONLY}

    def detect_fixed_regions(
        self,
        video_path: str,
        sample_count: Optional[int] = None,
        allow_fallback: bool = False,
    ) -> Optional[np.ndarray]:
        if not self.uses_auto_detection:
            self._fixed_mask = None
            return None
        sample_count = sample_count or self.config.sample_count
        capture_path = ensure_analysis_video_sync(video_path)
        capture = cv2.VideoCapture(str(capture_path))
        if not capture.isOpened():
            logger.warning(f"Unable to open video for subtitle masking: {capture_path}")
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            capture.release()
            return None

        start_frame = min(int(fps * 3), max(0, frame_count // 12))
        end_frame = min(frame_count - 1, max(start_frame + 1, int(fps * 120)))
        sample_indices = np.linspace(start_frame, end_frame, sample_count, dtype=int)

        vote_map: Optional[np.ndarray] = None
        positive_votes = 0
        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            self._frame_shape = frame.shape[:2]
            frame_mask = self.detect_subtitle_regions(frame)
            if frame_mask is None or not frame_mask.any():
                continue
            if vote_map is None:
                vote_map = np.zeros(frame.shape[:2], dtype=np.float32)
            vote_map += (frame_mask > 0).astype(np.float32)
            positive_votes += 1

        capture.release()

        if vote_map is None or positive_votes == 0:
            if allow_fallback and self._frame_shape is not None:
                self._fixed_mask = self._build_fallback_mask(self._frame_shape)
                logger.info(f"Subtitle mask fallback enabled for {Path(video_path).name}")
                return self._fixed_mask
            self._fixed_mask = None
            return None

        threshold = max(1.0, positive_votes * self.config.vote_ratio_threshold)
        fixed_mask = (vote_map >= threshold).astype(np.uint8) * 255
        if self.config.mask_dilate_pixels > 0 and fixed_mask.any():
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.config.mask_dilate_pixels * 2 + 1, self.config.mask_dilate_pixels * 2 + 1),
            )
            fixed_mask = cv2.dilate(fixed_mask, kernel, iterations=1)

        if not fixed_mask.any() and allow_fallback and self._frame_shape is not None:
            fixed_mask = self._build_fallback_mask(self._frame_shape)

        self._fixed_mask = fixed_mask if fixed_mask.any() else None
        if self._fixed_mask is not None:
            coverage = float(np.count_nonzero(self._fixed_mask) / self._fixed_mask.size)
            logger.info(f"Subtitle mask ready for {Path(video_path).name}: coverage={coverage:.2%}")
        return self._fixed_mask

    def detect_subtitle_regions(self, frame: np.ndarray) -> Optional[np.ndarray]:
        masks: list[np.ndarray] = []
        if self.config.use_heuristic:
            heuristic_mask = self._detect_heuristic(frame)
            if heuristic_mask is not None and heuristic_mask.any():
                masks.append(heuristic_mask)
        if self.config.use_ocr and PADDLE_OCR_AVAILABLE:
            ocr_mask = self._detect_with_ocr(frame)
            if ocr_mask is not None and ocr_mask.any():
                masks.append(ocr_mask)
        if not masks:
            return None
        merged = np.maximum.reduce(masks)
        return merged if merged.any() else None

    def process_frame(self, frame: np.ndarray, frame_time: Optional[float] = None) -> np.ndarray:
        mask = self.mask_for_frame(frame, frame_time=frame_time)
        if mask is None or not mask.any():
            return frame
        blur_size = self.config.median_blur_ksize if self.config.median_blur_ksize % 2 == 1 else self.config.median_blur_ksize + 1
        blurred = cv2.medianBlur(frame, blur_size)
        result = frame.copy()
        result[mask > 0] = blurred[mask > 0]
        return result

    def mask_for_frame(self, frame: np.ndarray, frame_time: Optional[float] = None) -> Optional[np.ndarray]:
        fixed_mask = self._resize_fixed_mask(frame)
        manual_mask = self._build_manual_mask(frame.shape[:2], frame_time=frame_time) if self.mask_mode != SubtitleMaskMode.AUTO_ONLY else None

        if self.mask_mode == SubtitleMaskMode.MANUAL_ONLY:
            return manual_mask
        if self.mask_mode == SubtitleMaskMode.AUTO_ONLY:
            return fixed_mask
        if fixed_mask is None:
            return manual_mask
        if manual_mask is None:
            return fixed_mask
        return np.maximum(fixed_mask, manual_mask)

    def _resize_fixed_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self._fixed_mask is None:
            return None
        if self._fixed_mask.shape[:2] == frame.shape[:2]:
            return self._fixed_mask
        return cv2.resize(self._fixed_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    def _build_manual_mask(self, frame_shape: tuple[int, int], frame_time: Optional[float] = None) -> Optional[np.ndarray]:
        if not self._manual_regions:
            return None
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in self._manual_regions:
            if not region.get("enabled", True):
                continue
            if not self._region_active_for_time(region, frame_time):
                continue
            x0 = int(round(float(region["x"]) * width))
            y0 = int(round(float(region["y"]) * height))
            x1 = int(round(float(region["x"] + region["width"]) * width))
            y1 = int(round(float(region["y"] + region["height"]) * height))
            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(x0 + 1, min(width, x1))
            y1 = max(y0 + 1, min(height, y1))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        return mask if mask.any() else None

    def _normalize_regions(self, regions: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for index, region in enumerate(regions):
            try:
                x = max(0.0, min(1.0, float(region.get("x", 0.0))))
                y = max(0.0, min(1.0, float(region.get("y", 0.0))))
                width = max(0.0, min(1.0 - x, float(region.get("width", 0.0))))
                height = max(0.0, min(1.0 - y, float(region.get("height", 0.0))))
            except (TypeError, ValueError):
                continue
            if width <= 0.0 or height <= 0.0:
                continue
            start_time = self._normalize_time(region.get("start_time"))
            end_time = self._normalize_time(region.get("end_time"))
            if start_time is not None and end_time is not None and end_time < start_time:
                end_time = start_time
            normalized.append(
                {
                    "id": region.get("id") or f"manual_region_{index}",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "enabled": bool(region.get("enabled", True)),
                    "label": region.get("label"),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
        return normalized

    def _normalize_time(self, value) -> Optional[float]:
        if value in (None, "", False):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, numeric)

    def _region_active_for_time(self, region: dict, frame_time: Optional[float]) -> bool:
        if frame_time is None:
            return True
        start_time = region.get("start_time")
        end_time = region.get("end_time")
        if start_time is not None and frame_time < float(start_time):
            return False
        if end_time is not None and frame_time > float(end_time):
            return False
        return True

    def _detect_heuristic(self, frame: np.ndarray) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        bottom_start = int(height * (1.0 - self.config.bottom_ratio))
        bottom_region = frame[bottom_start:, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        )
        gradient = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        merged = cv2.addWeighted(blackhat, 0.7, gradient, 0.3, 0.0)
        _, thresh = cv2.threshold(merged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)),
            iterations=1,
        )
        thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        bottom_height = bottom_region.shape[0]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * self.config.min_box_width_ratio:
                continue
            if h < self.config.min_box_height or h > bottom_height * self.config.max_box_height_ratio:
                continue
            area_ratio = (w * h) / max(1.0, width * height)
            if area_ratio > 0.18:
                continue
            y0 = max(0, y + bottom_start - 3)
            y1 = min(height, y + bottom_start + h + 3)
            x0 = max(0, x - 4)
            x1 = min(width, x + w + 4)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

        return mask if mask.any() else None

    def _detect_with_ocr(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            ocr = _get_ocr()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"OCR initialization failed for subtitle masking: {exc}")
            return None

        height, width = frame.shape[:2]
        bottom_start = int(height * (1.0 - self.config.bottom_ratio))
        bottom_region = frame[bottom_start:, :]
        try:
            results = ocr.ocr(bottom_region, cls=False)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"OCR detection failed for subtitle masking: {exc}")
            return None

        if not results or not results[0]:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        for line in results[0]:
            box = line[0]
            confidence = float(line[1][1]) if len(line) > 1 else 0.0
            if confidence < self.config.ocr_confidence_threshold:
                continue
            points = np.array(box, dtype=np.int32)
            points[:, 1] += bottom_start
            cv2.fillPoly(mask, [points], 255)
        return mask if mask.any() else None

    def _build_fallback_mask(self, frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cutoff = int(height * (1.0 - self.config.fallback_bottom_ratio))
        mask[cutoff:, :] = 255
        return mask
