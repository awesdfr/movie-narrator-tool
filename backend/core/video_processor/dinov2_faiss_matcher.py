"""DINOv2 + FAISS retrieval matcher used as a stronger coarse search layer."""
from __future__ import annotations

import asyncio
import pickle
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import faiss
import numpy as np
import torch
from loguru import logger
from torchvision import transforms

from .analysis_video import _resolve_ffmpeg_path, ensure_analysis_video
from .frame_matcher import FrameMatcher


class DinoFaissMatcher(FrameMatcher):
    CACHE_VERSION = 4
    _MODEL_CACHE: dict[str, torch.nn.Module] = {}

    def __init__(
        self,
        dino_model_name: str = "dinov2_vits14",
        dino_batch_size: int = 24,
        dino_top_k: int = 64,
        dino_index_interval: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dino_model_name = dino_model_name
        requested_batch_size = max(1, int(dino_batch_size))
        self.dino_top_k = max(4, int(dino_top_k))
        self.dino_index_interval = max(0.5, float(dino_index_interval))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda and requested_batch_size <= 24:
            # vits14 is small enough to keep the GPU fed with larger batches on
            # consumer NVIDIA cards. If memory is insufficient, users can lower
            # dino_batch_size in settings to override this auto-tune path.
            self.dino_batch_size = 96
        else:
            self.dino_batch_size = requested_batch_size
        self._decode_width = 224
        self._flush_batch_size = max(self.dino_batch_size * (8 if self._use_cuda else 4), 64 if self._use_cuda else 32)
        self._movie_vectors: Optional[np.ndarray] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._times_array: Optional[np.ndarray] = None
        self._embedding_dim: int = 384
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _get_model(self) -> torch.nn.Module:
        cached = self._MODEL_CACHE.get(self.dino_model_name)
        if cached is not None:
            return cached
        logger.info("Loading DINOv2 model {} on {}", self.dino_model_name, self._device)
        if self._use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        hub_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if hub_repo.exists():
            logger.info("Loading DINOv2 model {} from local torch hub cache: {}", self.dino_model_name, hub_repo)
            model = torch.hub.load(str(hub_repo), self.dino_model_name, source="local", trust_repo=True)
        else:
            logger.info("Local DINOv2 torch hub cache missing; downloading {} once", self.dino_model_name)
            model = torch.hub.load("facebookresearch/dinov2", self.dino_model_name, trust_repo=True)
        model.eval()
        model.to(self._device)
        self._MODEL_CACHE[self.dino_model_name] = model
        return model

    def _encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        model = self._get_model()
        outputs: list[np.ndarray] = []
        for start in range(0, len(frames), self.dino_batch_size):
            batch_frames = frames[start : start + self.dino_batch_size]
            batch_tensor = torch.stack([self._transform(frame) for frame in batch_frames], dim=0).to(
                self._device,
                non_blocking=self._use_cuda,
            )
            with torch.inference_mode():
                if self._use_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        embeddings = model(batch_tensor).detach().float().cpu().numpy().astype(np.float32)
                else:
                    embeddings = model(batch_tensor).detach().cpu().numpy().astype(np.float32)
            outputs.append(embeddings)
        vectors = np.concatenate(outputs, axis=0)
        faiss.normalize_L2(vectors)
        return vectors

    def _sample_times_for_segment(self, start_time: float, end_time: float) -> list[float]:
        duration = max(0.2, end_time - start_time)
        if duration <= 0.9:
            sample_count = 1
        elif duration <= 2.2:
            sample_count = 2
        elif duration <= 4.2:
            sample_count = 3
        elif duration <= 8.0:
            sample_count = 4
        elif duration <= 14.0:
            sample_count = 5
        else:
            sample_count = 6
        return [start_time + duration * (i + 0.5) / sample_count for i in range(sample_count)]

    def _score_from_similarity(self, similarity: float, coverage: float) -> float:
        scaled = float(np.clip((similarity - 0.18) / 0.62, 0.0, 1.0))
        return float(np.clip(scaled * 0.90 + coverage * 0.10, 0.0, 1.0))

    def _time_window(self, duration: float, time_hint: Optional[float], relaxed: bool, strict_window: bool) -> tuple[float, float]:
        if time_hint is None:
            return 0.0, float("inf")
        if strict_window:
            base_window = max(duration * 3.0, 45.0 if duration <= 3.5 else 60.0)
            if relaxed:
                base_window *= 1.35
        elif relaxed:
            base_window = max(duration * 6.0, 90.0)
        else:
            base_window = max(duration * 8.0, 180.0)
        return max(0.0, time_hint - base_window), time_hint + base_window

    def _build_faiss(self) -> None:
        if self._movie_vectors is None or len(self._movie_vectors) == 0:
            self._faiss_index = None
            self._times_array = None
            return
        self._times_array = np.asarray(self._times, dtype=np.float32)
        index = faiss.IndexFlatIP(self._movie_vectors.shape[1])
        index.add(self._movie_vectors.astype(np.float32))
        self._faiss_index = index

    def score_precomputed_segment_identity_at(
        self,
        query_features: list[dict],
        candidate_start: float,
        search_radius: float | None = None,
    ) -> dict:
        """Score aligned DINO identity similarity at a fixed movie start time.

        FAISS retrieval can find visually similar frames from the same scene but
        still drift on motion timing. This fixed-start scorer verifies that the
        narration sample at offset t matches the movie sample at start+t.
        """
        if self._movie_vectors is None or len(self._movie_vectors) == 0 or not query_features:
            return {"identity_score": 0.0, "identity_similarity": 0.0, "coverage": 0.0, "match_count": 0}

        times = self._times_array
        if times is None or len(times) == 0:
            times = np.asarray(self._times, dtype=np.float32)
            self._times_array = times
        radius = float(search_radius) if search_radius is not None else max(0.55, float(self._sample_step_seconds) * 0.70)
        similarities: list[float] = []
        for feature in query_features:
            vector = feature.get("embedding")
            if vector is None:
                continue
            target_time = float(candidate_start) + float(feature.get("offset", 0.0))
            center = int(np.searchsorted(times, target_time))
            left = max(0, center - 2)
            right = min(len(times), center + 3)
            best_similarity: Optional[float] = None
            query_vector = np.asarray(vector, dtype=np.float32)
            for movie_index in range(left, right):
                if abs(float(times[movie_index]) - target_time) > radius:
                    continue
                similarity = float(np.dot(query_vector, self._movie_vectors[movie_index]))
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
            if best_similarity is not None:
                similarities.append(best_similarity)

        if not similarities:
            return {"identity_score": 0.0, "identity_similarity": 0.0, "coverage": 0.0, "match_count": 0}
        coverage = len(similarities) / max(1, len(query_features))
        similarity_mean = float(np.mean(similarities))
        similarity_min = float(np.min(similarities))
        score = self._score_from_similarity(similarity_mean, coverage)
        if coverage < 0.75:
            score *= max(0.55, coverage)
        if len(similarities) >= 2 and similarity_min < similarity_mean - 0.18:
            score *= 0.92
        return {
            "identity_score": float(np.clip(score, 0.0, 1.0)),
            "identity_similarity": similarity_mean,
            "identity_similarity_min": similarity_min,
            "coverage": float(coverage),
            "match_count": len(similarities),
        }

    async def build_index(
        self,
        video_path: str,
        sample_interval: int | None = None,
        cache_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        video_path = str(video_path)
        normalized_video_path = self._normalize_video_path(video_path)
        capture_path = await ensure_analysis_video(video_path)
        cache_path = Path(cache_path) if cache_path else None
        loop = asyncio.get_running_loop()
        mask_signature = self._mask_signature_for_role("movie")
        interval = float(sample_interval or self.dino_index_interval)

        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as handle:
                    payload = pickle.load(handle)
                if (
                    payload.get("video_path") == normalized_video_path
                    and payload.get("cache_version") == self.CACHE_VERSION
                    and payload.get("model_name") == self.dino_model_name
                    and self._canonicalize_mask_signature(payload.get("mask_signature")) == self._canonicalize_mask_signature(mask_signature)
                    and float(payload.get("sample_step_seconds", 0.0)) == interval
                ):
                    self._movie_vectors = payload["vectors"]
                    self._times = payload["times"]
                    self._movie_duration = float(payload["movie_duration"])
                    self._sample_step_seconds = float(payload["sample_step_seconds"])
                    self._indexed_video_path = normalized_video_path
                    self._indexed_capture_path = str(capture_path)
                    self._embedding_dim = int(self._movie_vectors.shape[1]) if len(self._movie_vectors) else self._embedding_dim
                    await asyncio.to_thread(self._build_faiss)
                    if progress_callback:
                        progress_callback(100.0, f"Loaded DINO movie index cache ({len(self._times)} samples)")
                    return
            except Exception as exc:
                logger.warning("Failed to load DINO cache {}: {}", cache_path, exc)

        self._sample_step_seconds = interval
        subtitle_masker = await asyncio.to_thread(self._get_subtitle_masker, str(capture_path), "movie", False)

        def _build() -> tuple[np.ndarray, list[float], float]:
            probe = cv2.VideoCapture(str(capture_path))
            if not probe.isOpened():
                raise ValueError(f"Cannot open movie video: {capture_path}")
            fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            orig_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            orig_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            probe.release()
            duration = frame_count / fps if fps > 0 else 0.0
            estimated_samples = max(1, int(duration / interval) + 1) if duration > 0 else 1
            report_interval = max(1, estimated_samples // 40)

            pipe_w = self._decode_width
            if orig_w > 0 and orig_h > 0:
                pipe_h = max(2, int(orig_h * pipe_w / orig_w / 2) * 2)
            else:
                pipe_h = 180
            frame_bytes = pipe_w * pipe_h * 3
            ffmpeg_bin = _resolve_ffmpeg_path()

            def _report(sampled_idx: int, started_at: float, label: str = "Building DINO") -> None:
                if not progress_callback:
                    return
                progress = min(99.0, 100.0 * sampled_idx / max(1, estimated_samples))
                elapsed = max(0.0, time.perf_counter() - started_at)
                eta = elapsed * (estimated_samples - sampled_idx) / sampled_idx if sampled_idx > 0 and estimated_samples > sampled_idx else 0.0
                loop.call_soon_threadsafe(
                    progress_callback,
                    progress,
                    f"{label} movie index... {sampled_idx}/{estimated_samples} | ETA {self._format_duration(eta)}",
                )

            def _flush_batch(
                frame_batch: list[np.ndarray],
                time_batch: list[float],
                vector_chunks: list[np.ndarray],
                valid_times: list[float],
            ) -> None:
                if not frame_batch:
                    return
                vectors = self._encode_frames(frame_batch)
                if len(vectors):
                    vector_chunks.append(vectors)
                    valid_times.extend(time_batch)
                frame_batch.clear()
                time_batch.clear()

            def _build_with_ffmpeg(hwaccel: Optional[str]) -> tuple[np.ndarray, list[float], float]:
                cmd = [ffmpeg_bin]
                if hwaccel:
                    cmd += ["-hwaccel", hwaccel]
                cmd += [
                    "-i",
                    str(capture_path),
                    "-an",
                    "-vf",
                    f"fps=1/{interval:.4f},scale={pipe_w}:{pipe_h}",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "pipe:1",
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=frame_bytes * max(4, self.dino_batch_size * 2),
                )
                started_at = time.perf_counter()
                sampled_idx = 0
                frame_batch: list[np.ndarray] = []
                time_batch: list[float] = []
                valid_times: list[float] = []
                vector_chunks: list[np.ndarray] = []
                try:
                    while True:
                        raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                        if not raw or len(raw) < frame_bytes:
                            break
                        timestamp = sampled_idx * interval
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(pipe_h, pipe_w, 3).copy()
                        processed = self._preprocess_frame(frame, subtitle_masker, frame_time=timestamp)
                        frame_batch.append(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                        time_batch.append(float(timestamp))
                        sampled_idx += 1

                        if len(frame_batch) >= self._flush_batch_size:
                            _flush_batch(frame_batch, time_batch, vector_chunks, valid_times)
                        if sampled_idx % report_interval == 0:
                            _report(sampled_idx, started_at)
                    _flush_batch(frame_batch, time_batch, vector_chunks, valid_times)
                finally:
                    if proc.stdout:
                        proc.stdout.close()
                    proc.wait(timeout=30)

                if not vector_chunks:
                    raise ValueError("No frames decoded from ffmpeg pipeline")
                if progress_callback:
                    _report(len(valid_times), started_at)
                return np.concatenate(vector_chunks, axis=0), valid_times, duration

            hwaccel_candidates = ("cuda", "d3d11va", "auto", None) if self._use_cuda else ("d3d11va", "auto", None)
            for hwaccel in hwaccel_candidates:
                try:
                    return _build_with_ffmpeg(hwaccel)
                except Exception as exc:
                    logger.warning("DINO ffmpeg index build failed (hwaccel={}): {}", hwaccel, exc)

            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open movie video: {capture_path}")
            timestamps = list(np.arange(0.0, max(duration, 0.0), interval))
            if duration > 0 and (not timestamps or abs(timestamps[-1] - duration) > interval * 0.5):
                timestamps.append(max(0.0, duration - min(interval, 1.0)))

            valid_times: list[float] = []
            vector_chunks: list[np.ndarray] = []
            frame_batch: list[np.ndarray] = []
            time_batch: list[float] = []
            started_at = time.perf_counter()
            for idx, timestamp in enumerate(timestamps, start=1):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                processed = self._preprocess_frame(frame, subtitle_masker, frame_time=timestamp)
                frame_batch.append(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                time_batch.append(float(timestamp))
                if len(frame_batch) >= self._flush_batch_size:
                    _flush_batch(frame_batch, time_batch, vector_chunks, valid_times)
                if idx % report_interval == 0:
                    _report(idx, started_at, "Fallback DINO")
            capture.release()
            _flush_batch(frame_batch, time_batch, vector_chunks, valid_times)
            if not vector_chunks:
                return np.zeros((0, self._embedding_dim), dtype=np.float32), [], duration
            return np.concatenate(vector_chunks, axis=0), valid_times, duration

        vectors, times, duration = await asyncio.to_thread(_build)
        self._movie_vectors = vectors
        self._times = times
        self._movie_duration = duration
        self._indexed_video_path = normalized_video_path
        self._indexed_capture_path = str(capture_path)
        if len(vectors):
            self._embedding_dim = int(vectors.shape[1])
        await asyncio.to_thread(self._build_faiss)

        if cache_path:
            try:
                with open(cache_path, "wb") as handle:
                    pickle.dump(
                        {
                            "cache_version": self.CACHE_VERSION,
                            "video_path": normalized_video_path,
                            "model_name": self.dino_model_name,
                            "vectors": vectors,
                            "times": times,
                            "movie_duration": duration,
                            "sample_step_seconds": self._sample_step_seconds,
                            "mask_signature": mask_signature,
                        },
                        handle,
                    )
            except Exception as exc:
                logger.warning("Failed to write DINO cache {}: {}", cache_path, exc)
        if progress_callback:
            progress_callback(100.0, f"Built DINO movie index ({len(times)} samples)")

    async def precompute_segment_features_batch(
        self,
        video_path: str,
        segments: list[dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cache_path: Optional[Path] = None,
    ) -> dict[str, list[dict]]:
        normalized_video_path = self._normalize_video_path(video_path)
        capture_path = await ensure_analysis_video(video_path)
        subtitle_masker = await asyncio.to_thread(self._get_subtitle_masker, str(capture_path), "narration", True)
        cache_path = Path(cache_path) if cache_path else None
        mask_signature = self._mask_signature_for_role("narration")
        segments_signature = self._segments_signature(segments)

        feature_map: dict[str, list[dict]] = {}
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as handle:
                    payload = pickle.load(handle)
                if (
                    payload.get("cache_version") == self.CACHE_VERSION
                    and payload.get("video_path") == normalized_video_path
                    and payload.get("model_name") == self.dino_model_name
                    and self._canonicalize_mask_signature(payload.get("mask_signature")) == self._canonicalize_mask_signature(mask_signature)
                    and payload.get("segments_signature") == segments_signature
                ):
                    cached_map = payload.get("feature_map", {})
                    for segment in segments:
                        seg_id = str(segment.get("id"))
                        features = cached_map.get(seg_id, [])
                        feature_map[seg_id] = features
                        cache_key = self._feature_cache_key(str(capture_path), float(segment["start"]), float(segment["end"]), "narration")
                        self._query_feature_cache[cache_key] = features
                    return feature_map
            except Exception as exc:
                logger.warning("Failed to load DINO narration feature cache {}: {}", cache_path, exc)

        pending: list[dict] = []
        for segment in segments:
            seg_id = str(segment.get("id"))
            start_time = float(segment["start"])
            end_time = float(segment["end"])
            cache_key = self._feature_cache_key(str(capture_path), start_time, end_time, "narration")
            cached = self._query_feature_cache.get(cache_key)
            if cached is not None:
                feature_map[seg_id] = cached
                continue
            pending.append(
                {
                    "id": seg_id,
                    "start": start_time,
                    "end": end_time,
                    "sample_times": self._sample_times_for_segment(start_time, end_time),
                    "cache_key": cache_key,
                }
            )

        if not pending:
            return feature_map

        def _build() -> dict[str, list[dict]]:
            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open narration video: {capture_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 24.0
            seek_gap_frames = max(1, int(fps * 2.0))
            built: dict[str, list[dict]] = {item["id"]: [] for item in pending}
            sample_requests: list[tuple[str, float, float]] = []
            segment_request_counts: dict[str, int] = {}
            for item in pending:
                seg_id = item["id"]
                sample_times = sorted(float(sample_time) for sample_time in item["sample_times"])
                segment_request_counts[seg_id] = len(sample_times)
                for sample_time in sample_times:
                    sample_requests.append((seg_id, sample_time, sample_time - float(item["start"])))
            sample_requests.sort(key=lambda item: item[1])
            frame_batch: list[np.ndarray] = []
            meta_batch: list[tuple[str, float]] = []
            completed = 0
            next_frame_idx = 0

            def flush_batch() -> None:
                if not frame_batch:
                    return
                vectors = self._encode_frames(frame_batch)
                for (seg_id, offset), vector in zip(meta_batch, vectors):
                    built[seg_id].append(
                        {
                            "offset": offset,
                            "embedding": vector,
                            "quality_score": 1.0,
                            "is_low_info": False,
                        }
                    )
                frame_batch.clear()
                meta_batch.clear()

            processed_per_segment: dict[str, int] = {item["id"]: 0 for item in pending}
            for seg_id, sample_time, offset in sample_requests:
                target_frame = max(0, int(round(sample_time * fps)))
                if target_frame < next_frame_idx or target_frame - next_frame_idx > seek_gap_frames:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    next_frame_idx = target_frame

                frame = None
                ok = False
                while next_frame_idx <= target_frame:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    next_frame_idx += 1
                if not ok or frame is None:
                    processed_per_segment[seg_id] += 1
                    if progress_callback and processed_per_segment[seg_id] == segment_request_counts[seg_id]:
                        completed += 1
                        progress_callback(completed, len(pending))
                    continue

                processed = self._preprocess_frame(frame, subtitle_masker, frame_time=sample_time)
                frame_batch.append(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                meta_batch.append((seg_id, offset))
                if len(frame_batch) >= self.dino_batch_size:
                    flush_batch()
                processed_per_segment[seg_id] += 1
                if progress_callback and processed_per_segment[seg_id] == segment_request_counts[seg_id]:
                    completed += 1
                    progress_callback(completed, len(pending))
            flush_batch()
            capture.release()
            return built

        built_map = await asyncio.to_thread(_build)
        for item in pending:
            seg_id = item["id"]
            features = built_map.get(seg_id, [])
            feature_map[seg_id] = features
            self._query_feature_cache[item["cache_key"]] = features

        if cache_path:
            try:
                with open(cache_path, "wb") as handle:
                    pickle.dump(
                        {
                            "cache_version": self.CACHE_VERSION,
                            "video_path": normalized_video_path,
                            "model_name": self.dino_model_name,
                            "mask_signature": mask_signature,
                            "segments_signature": segments_signature,
                            "feature_map": feature_map,
                        },
                        handle,
                    )
            except Exception as exc:
                logger.warning("Failed to write DINO narration feature cache {}: {}", cache_path, exc)
        return feature_map

    async def _extract_segment_features(self, video_path: str, start_time: float, end_time: float) -> list[dict]:
        capture_path = await ensure_analysis_video(video_path)
        cache_key = self._feature_cache_key(str(capture_path), start_time, end_time, "narration")
        cached = self._query_feature_cache.get(cache_key)
        if cached is not None:
            return cached

        subtitle_masker = await asyncio.to_thread(self._get_subtitle_masker, str(capture_path), "narration", True)

        def _extract() -> list[dict]:
            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                return []
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 24.0
            seek_gap_frames = max(1, int(fps * 2.0))
            frames: list[np.ndarray] = []
            offsets: list[float] = []
            next_frame_idx = 0
            for sample_time in sorted(self._sample_times_for_segment(start_time, end_time)):
                target_frame = max(0, int(round(float(sample_time) * fps)))
                if target_frame < next_frame_idx or target_frame - next_frame_idx > seek_gap_frames:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    next_frame_idx = target_frame

                frame = None
                ok = False
                while next_frame_idx <= target_frame:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    next_frame_idx += 1
                if not ok or frame is None:
                    continue
                processed = self._preprocess_frame(frame, subtitle_masker, frame_time=float(sample_time))
                frames.append(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                offsets.append(float(sample_time) - float(start_time))
            capture.release()
            vectors = self._encode_frames(frames)
            return [
                {
                    "offset": offset,
                    "embedding": vector,
                    "quality_score": 1.0,
                    "is_low_info": False,
                }
                for offset, vector in zip(offsets, vectors)
            ]

        features = await asyncio.to_thread(_extract)
        self._query_feature_cache[cache_key] = features
        return features

    async def match_segment_candidates(
        self,
        narration_path: str,
        start_time: float,
        end_time: float,
        time_hint: Optional[float] = None,
        relaxed: bool = False,
        strict_window: bool = False,
        precomputed_features: Optional[list[dict]] = None,
        limit: int = 6,
    ) -> list[dict]:
        if self._faiss_index is None or self._movie_vectors is None or not self._times:
            raise RuntimeError("DINO movie index has not been built")

        duration = max(0.2, end_time - start_time)
        query_features = precomputed_features if precomputed_features is not None else await self._extract_segment_features(narration_path, start_time, end_time)
        if not query_features:
            return []

        min_time, max_time = self._time_window(duration, time_hint, relaxed, strict_window)
        candidate_buckets: dict[float, dict] = {}
        query_vectors = np.stack([item["embedding"] for item in query_features]).astype(np.float32)
        query_count = max(1, len(query_features))
        search_top_k = self.dino_top_k
        if self._use_cuda:
            if strict_window and not relaxed:
                search_top_k = max(search_top_k, 48)
            elif time_hint is not None:
                search_top_k = max(search_top_k, 64)
            else:
                search_top_k = max(search_top_k, 96)
            per_query_hit_cap = 4 if strict_window and not relaxed else 6 if time_hint is not None else 8
        else:
            if strict_window and not relaxed:
                search_top_k = min(search_top_k, 18)
            elif time_hint is not None:
                search_top_k = min(search_top_k, 20)
            per_query_hit_cap = 2 if strict_window and not relaxed else 3 if time_hint is not None else 4
        search_top_k = min(max(1, int(search_top_k)), len(self._times))
        scores, indices = self._faiss_index.search(query_vectors, search_top_k)

        for query_idx, feature in enumerate(query_features):
            offset = float(feature["offset"])
            hits_for_query = 0
            for similarity, index in zip(scores[query_idx], indices[query_idx]):
                if index < 0:
                    continue
                movie_time = float(self._times[index])
                candidate_start = max(0.0, movie_time - offset)
                if candidate_start < min_time or candidate_start > max_time:
                    continue
                bucket = round(candidate_start / self._sample_step_seconds) * self._sample_step_seconds
                entry = candidate_buckets.setdefault(
                    bucket,
                    {
                        "scores": [],
                        "matched_queries": set(),
                        "offsets": [],
                    },
                )
                entry["scores"].append(float(similarity))
                entry["matched_queries"].add(query_idx)
                entry["offsets"].append(offset)
                hits_for_query += 1
                if hits_for_query >= per_query_hit_cap:
                    break

        if not candidate_buckets:
            return []

        ranked: list[tuple[float, float, int]] = []
        identity_payloads: dict[float, dict] = {}
        for candidate_start, payload in candidate_buckets.items():
            retrieval_similarity = float(np.mean(payload["scores"]))
            retrieval_coverage = len(payload["matched_queries"]) / query_count
            identity_payload = self.score_precomputed_segment_identity_at(
                query_features,
                candidate_start,
                search_radius=max(0.55, float(self._sample_step_seconds) * 0.70),
            )
            identity_payloads[candidate_start] = identity_payload
            identity_similarity = float(identity_payload.get("identity_similarity", 0.0))
            identity_coverage = float(identity_payload.get("coverage", 0.0))
            if identity_coverage > 0.0:
                similarity_mean = retrieval_similarity * 0.35 + identity_similarity * 0.65
                coverage = min(1.0, retrieval_coverage * 0.45 + identity_coverage * 0.55)
            else:
                similarity_mean = retrieval_similarity
                coverage = retrieval_coverage
            support_ratio = min(1.0, len(payload["scores"]) / max(1.0, query_count * per_query_hit_cap))
            score = self._score_from_similarity(similarity_mean, coverage)
            score = float(np.clip(score * 0.94 + support_ratio * 0.06, 0.0, 1.0))
            if identity_coverage > 0.0 and float(identity_payload.get("identity_score", 0.0)) < 0.58:
                score *= 0.86
            ranked.append((score, candidate_start, len(payload["matched_queries"])))

        ranked.sort(reverse=True)
        results: list[dict] = []
        min_score = max(0.30, self.match_threshold * (0.60 if relaxed else 0.55))
        capped_limit = max(1, int(limit))
        for rank, (score, candidate_start, match_count) in enumerate(ranked[:capped_limit], start=1):
            adjusted_score = float(score)
            second_score = ranked[rank][0] if rank < len(ranked) else 0.0
            if rank == 1 and len(ranked) > 1:
                second_start = ranked[1][1]
                if abs(candidate_start - second_start) > max(24.0, duration * 10.0) and adjusted_score - second_score < 0.035:
                    adjusted_score = max(0.0, adjusted_score - 0.08)
            if adjusted_score < min_score:
                continue

            identity = identity_payloads.get(candidate_start, {})
            stability = min(1.0, match_count / query_count)
            results.append(
                {
                    "success": True,
                    "start": float(candidate_start),
                    "end": float(min(self._movie_duration or candidate_start + duration, candidate_start + duration)),
                    "confidence": adjusted_score,
                    "rank_gap": float(max(0.0, adjusted_score - second_score)),
                    "match_count": int(match_count),
                    "stability_score": float(stability),
                    "candidate_quality": adjusted_score,
                    "query_quality": 1.0,
                    "low_info_ratio": 0.0,
                    "identity_score": float(identity.get("identity_score", 0.0)),
                    "identity_similarity": float(identity.get("identity_similarity", 0.0)),
                    "identity_coverage": float(identity.get("coverage", 0.0)),
                    "retrieval_rank": rank,
                    "confidence_level": "high" if adjusted_score >= 0.82 else "medium" if adjusted_score >= 0.65 else "low",
                }
            )
        return results

    async def match_segment(
        self,
        narration_path: str,
        start_time: float,
        end_time: float,
        time_hint: Optional[float] = None,
        relaxed: bool = False,
        strict_window: bool = False,
        precomputed_features: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        candidates = await self.match_segment_candidates(
            narration_path,
            start_time,
            end_time,
            time_hint=time_hint,
            relaxed=relaxed,
            strict_window=strict_window,
            precomputed_features=precomputed_features,
            limit=1,
        )
        return candidates[0] if candidates else None
