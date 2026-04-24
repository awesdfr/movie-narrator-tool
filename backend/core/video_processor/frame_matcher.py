"""Lightweight frame matcher used by the v2 alignment pipeline."""
from __future__ import annotations

import asyncio
import bisect
import hashlib
import os
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger

from config import settings
from .analysis_video import ensure_analysis_video, _resolve_ffmpeg_path
from .subtitle_masker import SubtitleMasker


class FrameMatcher:
    CACHE_VERSION = 17  # bumped: added gradient orientation histogram (128 dims, 4x4 grid) for encoding-robust structural matching
    # Lookup table for fast popcount (bit count) — used by vectorized prefilter & scoring
    _POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def __init__(
        self,
        phash_threshold: Optional[int] = None,
        match_threshold: Optional[float] = None,
        use_deep_learning: Optional[bool] = None,
        index_sample_fps: Optional[float] = None,
        fast_mode: Optional[bool] = None,
        enable_subtitle_masking: Optional[bool] = None,
        subtitle_mask_mode: Optional[str] = None,
        movie_subtitle_regions: Optional[list[dict]] = None,
        narration_subtitle_regions: Optional[list[dict]] = None,
        **_: dict,
    ):
        self.phash_threshold = phash_threshold if phash_threshold is not None else settings.phash_threshold
        self.match_threshold = match_threshold if match_threshold is not None else settings.frame_match_threshold
        self.use_deep_learning = bool(use_deep_learning) if use_deep_learning is not None else False
        self.index_sample_fps = index_sample_fps if index_sample_fps is not None else settings.index_sample_fps
        self.fast_mode = bool(fast_mode) if fast_mode is not None else False
        self.enable_subtitle_masking = True if enable_subtitle_masking is None else bool(enable_subtitle_masking)
        self.subtitle_mask_mode = getattr(subtitle_mask_mode, "value", subtitle_mask_mode) or "hybrid"
        self.high_confidence_threshold = 0.82
        self.medium_confidence_threshold = 0.65
        self._index: list[dict] = []
        self._times: list[float] = []
        self._movie_duration: float = 0.0
        self._sample_step_seconds: float = 1.0
        self._indexed_video_path: Optional[str] = None
        self._indexed_capture_path: Optional[str] = None
        # Precomputed packed bit arrays for vectorized prefilter & scoring
        self._idx_ahash_packed: Optional[np.ndarray] = None        # (N, 32) uint8
        self._idx_phash_packed: Optional[np.ndarray] = None        # (N, 8)  uint8
        self._idx_center_ahash_packed: Optional[np.ndarray] = None # (N, 32) uint8
        self._idx_center_phash_packed: Optional[np.ndarray] = None # (N, 8)  uint8
        self._idx_edge_packed: Optional[np.ndarray] = None         # (N, 18) uint8
        self._idx_hist: Optional[np.ndarray] = None                # (N, 48) float32 L2-normalised
        self._idx_color_hist: Optional[np.ndarray] = None         # (N, 48) float32 L2-normalised BGR color hist
        self._idx_spatial_color_hist: Optional[np.ndarray] = None # (N, 432) float32 L2-normalised 3x3 spatial color hist
        self._idx_grad_hist: Optional[np.ndarray] = None          # (N, 128) float32 L2-normalised gradient orientation hist
        self._subtitle_maskers: dict[str, SubtitleMasker] = {}
        self._query_feature_cache: dict[tuple[str, float, float, str], list[dict]] = {}
        self._subtitle_regions_by_role = {
            "movie": self._normalize_regions(movie_subtitle_regions),
            "narration": self._normalize_regions(narration_subtitle_regions),
        }
        self._crop_ratio_center = 0.82
        self._crop_ratio_side = 0.72
        # time_scales kept for fallback _score_candidate path only; vectorised path always uses 1.0
        self._time_scales = (1.0,) if self.fast_mode else (0.95, 1.0, 1.05)

    def _normalize_video_path(self, video_path: str) -> str:
        try:
            return str(Path(video_path).resolve()).lower()
        except Exception:
            return str(Path(video_path)).lower()

    def _segments_signature(self, segments: list[dict]) -> str:
        hasher = hashlib.sha1()
        for segment in segments:
            hasher.update(f"{round(float(segment['start']), 3)}:{round(float(segment['end']), 3)}|".encode("utf-8"))
        hasher.update(str(self.fast_mode).encode("utf-8"))
        return hasher.hexdigest()

    def _canonicalize_mask_signature(self, signature) -> tuple:
        if not signature:
            return tuple()
        try:
            enabled, mode, role, region_signature = signature
        except (TypeError, ValueError):
            return tuple(signature) if isinstance(signature, (list, tuple)) else (signature,)
        canonical_regions = []
        for region in region_signature or ():
            if not isinstance(region, (list, tuple)):
                continue
            if len(region) >= 8:
                _, x, y, width, height, region_enabled, start_time, end_time = region[:8]
            elif len(region) == 7:
                x, y, width, height, region_enabled, start_time, end_time = region
            else:
                continue
            canonical_regions.append(
                self._region_signature_entry(
                    str(role),
                    {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'enabled': region_enabled,
                        'start_time': start_time,
                        'end_time': end_time,
                    },
                )
            )
        return (bool(enabled), str(mode), str(role), tuple(canonical_regions))

    def _region_signature_entry(self, source_role: str, region: dict) -> tuple:
        x = float(region.get('x', 0.0))
        y = float(region.get('y', 0.0))
        width = float(region.get('width', 0.0))
        height = float(region.get('height', 0.0))

        if source_role in {"movie", "narration"}:
            x = 0.0 if width >= 0.92 or x <= 0.05 else round(round(x / 0.05) * 0.05, 2)
            width = 1.0 if width >= 0.92 else round(round(width / 0.05) * 0.05, 2)
            y = round(round(y / 0.10) * 0.10, 2)
            height = round(round(height / 0.10) * 0.10, 2)
        else:
            x = round(x, 4)
            y = round(y, 4)
            width = round(width, 4)
            height = round(height, 4)

        return (
            x,
            y,
            width,
            height,
            bool(region.get('enabled', True)),
            None if region.get('start_time') is None else round(float(region.get('start_time')), 3),
            None if region.get('end_time') is None else round(float(region.get('end_time')), 3),
        )

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
        interval = float(sample_interval or max(1.0, 1.0 / max(self.index_sample_fps, 0.1)))
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as handle:
                    payload = pickle.load(handle)
                if (
                    payload.get('video_path') == normalized_video_path
                    and payload.get('cache_version') == self.CACHE_VERSION
                    and self._canonicalize_mask_signature(payload.get('mask_signature')) == self._canonicalize_mask_signature(mask_signature)
                    and payload.get('sample_step_seconds') == interval
                ):
                    self._index = payload['index']
                    self._times = payload['times']
                    self._movie_duration = payload['movie_duration']
                    self._sample_step_seconds = payload['sample_step_seconds']
                    self._indexed_video_path = normalized_video_path
                    self._indexed_capture_path = str(capture_path)
                    await asyncio.to_thread(self._precompute_packed_hashes)
                    if progress_callback:
                        progress_callback(100.0, f"Loaded movie frame index cache ({len(self._index)} samples)")
                    return
            except Exception as exc:
                logger.warning(f'Failed to load match cache {cache_path}: {exc}')
        self._sample_step_seconds = interval
        subtitle_masker = await asyncio.to_thread(self._get_subtitle_masker, capture_path, "movie", False)

        def _build() -> tuple[list[dict], float]:
            # ── Probe video metadata ──
            cap_probe = cv2.VideoCapture(capture_path)
            if not cap_probe.isOpened():
                raise ValueError(f'Cannot open movie video: {capture_path}')
            fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            orig_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_probe.release()
            duration = frame_count / fps if fps > 0 else 0.0
            estimated_samples = max(1, int(duration / interval) + 1) if duration > 0 else 1
            report_interval = max(1, estimated_samples // 25)

            # ── Compute pipe frame dimensions (preserve aspect ratio, even pixels) ──
            pipe_w = 256
            if orig_w > 0 and orig_h > 0:
                pipe_h = max(2, int(orig_h * pipe_w / orig_w / 2) * 2)
            else:
                pipe_h = 144
            frame_bytes = pipe_w * pipe_h * 3

            def _report(sampled_idx: int, started_at: float, label: str = "Building") -> None:
                if not progress_callback:
                    return
                progress = min(99.0, 100.0 * sampled_idx / max(1, estimated_samples))
                elapsed = max(0.0, time.perf_counter() - started_at)
                eta = elapsed * (estimated_samples - sampled_idx) / sampled_idx if sampled_idx > 0 and estimated_samples > sampled_idx else 0.0
                loop.call_soon_threadsafe(
                    progress_callback,
                    progress,
                    f"{label} movie frame index... {sampled_idx}/{estimated_samples} | ETA {self._format_duration(eta)}",
                )

            # ── Fast path: ffmpeg pipe + GPU decode + parallel frame processing ──
            entries: list[dict] = []
            n_workers = min(8, max(2, (os.cpu_count() or 4)))
            batch_size = n_workers * 4  # read this many frames before dispatching

            def _process_frame_batch(batch: list[tuple[int, bytes]]) -> list[dict]:
                """Process a batch of (frame_idx, raw_bytes) tuples in parallel."""
                def _one(args: tuple[int, bytes]) -> dict:
                    idx, raw = args
                    f = np.frombuffer(raw, dtype=np.uint8).reshape(pipe_h, pipe_w, 3).copy()
                    ct = idx * interval
                    return {'time': ct, **self._frame_features_lite(f, subtitle_masker=subtitle_masker, frame_time=ct)}
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    return list(pool.map(_one, batch))

            try:
                ffmpeg_bin = _resolve_ffmpeg_path()
                vf = f'fps=1/{interval:.4f},scale={pipe_w}:{pipe_h}'
                # Try GPU decode first (d3d11va = best on Windows with NVIDIA/AMD)
                for hwaccel in ('d3d11va', 'cuda', 'auto', None):
                    cmd = [ffmpeg_bin]
                    if hwaccel:
                        cmd += ['-hwaccel', hwaccel]
                    cmd += ['-i', capture_path, '-an', '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        bufsize=frame_bytes * batch_size * 2,
                    )
                    started_at = time.perf_counter()
                    sampled_idx = 0
                    entries_tmp: list[dict] = []
                    batch_buf: list[tuple[int, bytes]] = []
                    success = False
                    while True:
                        raw = proc.stdout.read(frame_bytes)
                        if len(raw) < frame_bytes:
                            break
                        batch_buf.append((sampled_idx, raw))
                        sampled_idx += 1
                        if len(batch_buf) >= batch_size:
                            entries_tmp.extend(_process_frame_batch(batch_buf))
                            batch_buf = []
                            if sampled_idx % report_interval < batch_size:
                                _report(sampled_idx, started_at, f"Building (GPU+∥ hw={hwaccel or 'sw'})")
                    if batch_buf:
                        entries_tmp.extend(_process_frame_batch(batch_buf))
                    proc.stdout.close()
                    proc.wait()
                    if entries_tmp:
                        entries = entries_tmp
                        success = True
                        label = hwaccel or 'sw'
                        logger.info(f'Index built via ffmpeg pipe hwaccel={label}: {len(entries)} frames in {time.perf_counter()-started_at:.1f}s')
                        break
                    logger.debug(f'ffmpeg hwaccel={hwaccel} produced no frames, trying next')

                if not entries:
                    raise RuntimeError('all ffmpeg hwaccel modes produced no frames')
            except Exception as exc:
                logger.warning(f'ffmpeg pipe failed ({exc}), falling back to OpenCV')
                entries = []

            # ── Fallback: OpenCV frame-by-frame ──
            if not entries:
                capture = cv2.VideoCapture(capture_path)
                sample_step_frames = max(1, int(round(interval * fps))) if fps > 0 else 1
                started_at = time.perf_counter()
                frame_idx = 0
                sampled_idx = 0
                while True:
                    ok = capture.grab()
                    if not ok:
                        break
                    if frame_idx % sample_step_frames != 0:
                        frame_idx += 1
                        continue
                    ok, frame = capture.retrieve()
                    if not ok:
                        frame_idx += 1
                        continue
                    current_time = frame_idx / fps if fps > 0 else sampled_idx * interval
                    entries.append({
                        'time': current_time,
                        **self._frame_features_lite(frame, subtitle_masker=subtitle_masker, frame_time=current_time),
                    })
                    sampled_idx += 1
                    if sampled_idx == 1 or sampled_idx == estimated_samples or sampled_idx % report_interval == 0:
                        _report(sampled_idx, started_at)
                    frame_idx += 1
                capture.release()

            if progress_callback:
                loop.call_soon_threadsafe(
                    progress_callback,
                    100.0,
                    f"Built movie frame index ({len(entries)} samples)",
                )
            return entries, duration

        index, duration = await asyncio.to_thread(_build)
        self._index = index
        self._times = [entry['time'] for entry in index]
        self._movie_duration = duration
        self._indexed_video_path = normalized_video_path
        self._indexed_capture_path = str(capture_path)
        await asyncio.to_thread(self._precompute_packed_hashes)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as handle:
                pickle.dump(
                    {
                        'cache_version': self.CACHE_VERSION,
                        'video_path': normalized_video_path,
                        'index': self._index,
                        'times': self._times,
                        'movie_duration': self._movie_duration,
                        'sample_step_seconds': self._sample_step_seconds,
                        'mask_signature': mask_signature,
                    },
                    handle,
                )

    def _precompute_packed_hashes(self) -> None:
        """Pack all index frame hashes/histograms into matrices for vectorized scoring."""
        if not self._index:
            return
        try:
            zeros_a = np.zeros(256, dtype=np.uint8)
            zeros_p = np.zeros(64, dtype=np.uint8)
            zeros_e = np.zeros(144, dtype=np.uint8)
            zeros_h = np.zeros(48, dtype=np.float32)
            ahash_rows, phash_rows, cahash_rows, cphash_rows = [], [], [], []
            edge_rows, hist_rows = [], []
            for entry in self._index:
                full = entry['variants'].get('full')
                center = entry['variants'].get('center')
                ahash_rows.append(full['hash'] if full is not None else zeros_a)
                phash_rows.append(full['phash'] if full is not None else zeros_p)
                cahash_rows.append(center['hash'] if center is not None else zeros_a)
                cphash_rows.append(center['phash'] if center is not None else zeros_p)
                edge_rows.append(entry.get('edge_hash', zeros_e))
                h = full['hist'].reshape(-1) if full is not None else zeros_h
                hist_rows.append(h)
            self._idx_ahash_packed = np.packbits(np.stack(ahash_rows), axis=1)         # (N, 32)
            self._idx_phash_packed = np.packbits(np.stack(phash_rows), axis=1)         # (N, 8)
            self._idx_center_ahash_packed = np.packbits(np.stack(cahash_rows), axis=1) # (N, 32)
            self._idx_center_phash_packed = np.packbits(np.stack(cphash_rows), axis=1) # (N, 8)
            self._idx_edge_packed = np.packbits(np.stack(edge_rows), axis=1)           # (N, 18)
            self._idx_hist = np.stack(hist_rows).astype(np.float32)                    # (N, 48)

            # Color histogram matrices (encoding-robust, high weight in scoring)
            color_hists = np.stack([e.get('color_hist', np.zeros(48, dtype='float32')) for e in self._index])
            norms = np.linalg.norm(color_hists, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._idx_color_hist = (color_hists / norms).astype('float32')

            spatial_hists = np.stack([e.get('spatial_color_hist', np.zeros(432, dtype='float32')) for e in self._index])
            norms2 = np.linalg.norm(spatial_hists, axis=1, keepdims=True)
            norms2[norms2 == 0] = 1.0
            self._idx_spatial_color_hist = (spatial_hists / norms2).astype('float32')

            grad_hists = np.stack([e.get('grad_hist', np.zeros(128, dtype='float32')) for e in self._index])
            norms3 = np.linalg.norm(grad_hists, axis=1, keepdims=True)
            norms3[norms3 == 0] = 1.0
            self._idx_grad_hist = (grad_hists / norms3).astype('float32')

            color_info = f', color_hist {self._idx_color_hist.shape}, spatial {self._idx_spatial_color_hist.shape}, grad {self._idx_grad_hist.shape}' if self._idx_color_hist is not None else ''
            logger.debug(f'Packed hash matrices built: {len(self._index)} frames, '
                         f'ahash {self._idx_ahash_packed.shape}, hist {self._idx_hist.shape}{color_info}')
        except Exception as exc:
            logger.warning(f'Failed to precompute packed hashes: {exc}')
            self._idx_ahash_packed = None

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
        if not self._index:
            raise RuntimeError('Frame index has not been built')

        duration = max(0.2, end_time - start_time)
        if precomputed_features is not None:
            query_features = precomputed_features
        else:
            query_features = await self._extract_segment_features(narration_path, start_time, end_time)
        if not query_features:
            return None

        candidate_starts = self._candidate_starts(duration, time_hint=time_hint, relaxed=relaxed, strict_window=strict_window)
        if not candidate_starts:
            return None

        if duration <= 1.15 and not relaxed:
            search_radius = max(0.55, min(0.85, self._sample_step_seconds * 0.70))
        elif duration <= 2.20 and not relaxed:
            search_radius = max(0.80, min(1.20, self._sample_step_seconds * 0.95))
        else:
            search_radius = max(1.5, self._sample_step_seconds * (3.0 if relaxed else 2.0))
        movie_duration = self._movie_duration

        _index_ref = self._index
        _times_ref = self._times

        # Capture packed index arrays for use inside thread
        _idx_ahash_p = self._idx_ahash_packed
        _idx_phash_p = self._idx_phash_packed
        _idx_cahash_p = self._idx_center_ahash_packed
        _idx_cphash_p = self._idx_center_phash_packed
        _idx_edge_p = self._idx_edge_packed
        _idx_hist = self._idx_hist
        _idx_color_hist = self._idx_color_hist
        _idx_spatial_color_hist = self._idx_spatial_color_hist
        _idx_grad_hist = self._idx_grad_hist
        _tbl = self.__class__._POPCOUNT_TABLE

        def _run_search() -> Optional[dict]:
            cands = candidate_starts

            # ── Stage 1: vectorised prefilter (narrows large candidate set → top-200) ──
            full_score_mat: Optional[np.ndarray] = None   # (Q, N) reused in stage 2
            center_score_mat: Optional[np.ndarray] = None
            q_hashes_valid: list[dict] = []  # valid query-frame hash/hist info

            if _idx_ahash_p is not None and query_features:
                for qf in query_features:
                    qfv = qf['variants'].get('full')
                    qfc = qf['variants'].get('center') or qfv
                    if qfv is None:
                        continue
                    q_hashes_valid.append({
                        'ap': qfv['hash'], 'pp': qfv['phash'],
                        'cap': qfc['hash'], 'cpp': qfc['phash'],
                        'hist': qfv['hist'].reshape(-1),
                        'color_hist': qf.get('color_hist', np.zeros(48, dtype='float32')),
                        'spatial_color_hist': qf.get('spatial_color_hist', np.zeros(432, dtype='float32')),
                        'grad_hist': qf.get('grad_hist', np.zeros(128, dtype='float32')),
                        'edge': qf.get('edge_hash'),
                        'offset': float(qf['offset']),
                        'quality': float(qf['quality_score']),
                        'low_info': bool(qf['is_low_info']),
                    })

                if q_hashes_valid:
                    q_ap = np.packbits(np.stack([x['ap'] for x in q_hashes_valid]), axis=1)
                    q_pp = np.packbits(np.stack([x['pp'] for x in q_hashes_valid]), axis=1)
                    q_cap = np.packbits(np.stack([x['cap'] for x in q_hashes_valid]), axis=1)
                    q_cpp = np.packbits(np.stack([x['cpp'] for x in q_hashes_valid]), axis=1)

                    # (Q, N) aHash + pHash scores for full and center variants
                    ah = 1.0 - _tbl[q_ap[:, None, :] ^ _idx_ahash_p[None, :, :]].sum(2, dtype=np.int32) / 256.0
                    ph = 1.0 - _tbl[q_pp[:, None, :] ^ _idx_phash_p[None, :, :]].sum(2, dtype=np.int32) / 64.0
                    cah = 1.0 - _tbl[q_cap[:, None, :] ^ _idx_cahash_p[None, :, :]].sum(2, dtype=np.int32) / 256.0
                    cph = 1.0 - _tbl[q_cpp[:, None, :] ^ _idx_cphash_p[None, :, :]].sum(2, dtype=np.int32) / 64.0

                    full_score_mat = ah * 0.55 + ph * 0.45      # (Q, N)
                    center_score_mat = (cah * 0.55 + cph * 0.45) * 0.92  # (Q, N)

                    # Add histogram similarity for higher-quality prefilter signal
                    if _idx_hist is not None:
                        q_hist_mat = np.stack([x['hist'] for x in q_hashes_valid])  # (Q, 48)
                        hist_sim = np.clip((q_hist_mat @ _idx_hist.T + 1.0) / 2.0, 0.0, 1.0)  # (Q, N)
                        if _idx_color_hist is not None and _idx_spatial_color_hist is not None:
                            q_color_hist_mat = np.stack([x['color_hist'] for x in q_hashes_valid])  # (Q, 48)
                            q_norms = np.linalg.norm(q_color_hist_mat, axis=1, keepdims=True)
                            q_norms[q_norms == 0] = 1.0
                            q_color_hist_mat = q_color_hist_mat / q_norms
                            color_hist_sim = np.clip((q_color_hist_mat @ _idx_color_hist.T + 1.0) / 2.0, 0.0, 1.0)  # (Q, N)

                            q_spatial_color_mat = np.stack([x['spatial_color_hist'] for x in q_hashes_valid])  # (Q, 432)
                            q_norms2 = np.linalg.norm(q_spatial_color_mat, axis=1, keepdims=True)
                            q_norms2[q_norms2 == 0] = 1.0
                            q_spatial_color_mat = q_spatial_color_mat / q_norms2
                            spatial_color_sim = np.clip((q_spatial_color_mat @ _idx_spatial_color_hist.T + 1.0) / 2.0, 0.0, 1.0)  # (Q, N)

                            hash_combined = ah * 0.55 + ph * 0.45
                            if _idx_grad_hist is not None:
                                q_grad_mat = np.stack([x['grad_hist'] for x in q_hashes_valid])  # (Q, 128)
                                q_norms3 = np.linalg.norm(q_grad_mat, axis=1, keepdims=True)
                                q_norms3[q_norms3 == 0] = 1.0
                                q_grad_mat = q_grad_mat / q_norms3
                                grad_sim = np.clip((q_grad_mat @ _idx_grad_hist.T + 1.0) / 2.0, 0.0, 1.0)  # (Q, N)
                                full_score_mat = (
                                    hash_combined * 0.34
                                    + hist_sim * 0.08
                                    + color_hist_sim * 0.16
                                    + spatial_color_sim * 0.22
                                    + grad_sim * 0.20
                                )
                            else:
                                full_score_mat = hash_combined * 0.36 + hist_sim * 0.08 + color_hist_sim * 0.22 + spatial_color_sim * 0.34
                        else:
                            full_score_mat = full_score_mat * 0.80 + hist_sim * 0.20

                    if len(cands) > 250:
                        combined_prefilter = np.maximum(full_score_mat, center_score_mat)
                        quick: list[tuple[float, float]] = []
                        for cs in cands:
                            frame_scores: list[float] = []
                            for q_i, query in enumerate(q_hashes_valid):
                                target_time = cs + float(query["offset"])
                                l = bisect.bisect_left(_times_ref, target_time - search_radius)
                                r = bisect.bisect_right(_times_ref, target_time + search_radius)
                                if l < r:
                                    frame_scores.append(float(combined_prefilter[q_i, l:r].max()))
                            if frame_scores:
                                frame_scores.sort(reverse=True)
                                keep = max(1, int(len(frame_scores) * 0.70))
                                quick.append((float(np.mean(frame_scores[:keep])), cs))
                        quick.sort(reverse=True)
                        cands = [cs for _, cs in quick[:200]]

            # ── Stage 2: vectorised final scoring across all remaining candidates ──
            if full_score_mat is not None and q_hashes_valid and cands:
                Q = len(q_hashes_valid)
                n_cands = len(cands)

                # Build combined score matrix (Q, N): full + center + edge + (hist already merged)
                combined = np.maximum(full_score_mat, center_score_mat)  # (Q, N)
                if _idx_edge_p is not None:
                    edge_rows_valid = [x['edge'] for x in q_hashes_valid if x['edge'] is not None]
                    if len(edge_rows_valid) == Q:
                        q_ep = np.packbits(np.stack(edge_rows_valid), axis=1)  # (Q, 18)
                        edge_sim = 1.0 - _tbl[
                            q_ep[:, None, :] ^ _idx_edge_p[None, :, :]
                        ].sum(2, dtype=np.int32) / 144.0  # (Q, N)
                    combined = combined * 0.80 + edge_sim * 0.20

                # For each (query_frame, candidate), find max combined score in time window.
                # Use exact offset timing first; optional scale candidates are only allowed
                # when they clearly improve the phase-locked score.
                offsets = np.array([x['offset'] for x in q_hashes_valid])   # (Q,)
                q_low_info = np.array([float(x['low_info']) for x in q_hashes_valid])  # (Q,)
                q_quality = np.array([x['quality'] for x in q_hashes_valid])            # (Q,)
                scale_options = (1.0,) if self.fast_mode else (1.0, 0.88, 0.94, 1.07, 1.16)
                phase_radius = max(0.38, min(search_radius, float(self._sample_step_seconds) * 0.62))
                best_scores_by_candidate = np.zeros(n_cands, dtype=np.float32)
                best_scales_by_candidate = np.ones(n_cands, dtype=np.float32)
                best_match_count_by_candidate = np.zeros(n_cands, dtype=np.int32)
                best_stability_by_candidate = np.zeros(n_cands, dtype=np.float32)
                best_quality_by_candidate = np.zeros(n_cands, dtype=np.float32)
                best_low_info_by_candidate = np.ones(n_cands, dtype=np.float32)

                for scale in scale_options:
                    per_qc = np.zeros((Q, n_cands), dtype=np.float32)
                    for q_i in range(Q):
                        offset = float(offsets[q_i]) * float(scale)
                        row = combined[q_i]  # (N,) view
                        for c_i, cs in enumerate(cands):
                            l = bisect.bisect_left(_times_ref, cs + offset - phase_radius)
                            r = bisect.bisect_right(_times_ref, cs + offset + phase_radius)
                            if l < r:
                                per_qc[q_i, c_i] = float(row[l:r].max())

                    sorted_scores = np.sort(per_qc, axis=0)[::-1]          # (Q, n_cands)
                    keep = Q if Q <= 3 else max(2, int(Q * 0.75))
                    trimmed_mean = sorted_scores[:keep].mean(axis=0)        # (n_cands,)
                    floor_score = sorted_scores[keep - 1] if keep > 0 else trimmed_mean
                    coverage = (per_qc > 0).sum(axis=0) / Q                # (n_cands,)
                    matched_mask = per_qc > 0                               # (Q, n_cands) bool
                    matched_count_per_cand = matched_mask.sum(axis=0).clip(1)
                    low_info_ratio = (q_low_info[:, None] * matched_mask).sum(axis=0) / matched_count_per_cand
                    quality_penalty = np.maximum(0.0, low_info_ratio - 0.25) * 0.25
                    mean_c = per_qc.mean(axis=0)
                    std_c = per_qc.std(axis=0)
                    cv = std_c / (mean_c + 1e-6)
                    consistency_bonus = np.maximum(0.0, 0.06 * (1.0 - np.minimum(1.0, cv * 2.0)))
                    q_quality_weighted = (q_quality[:, None] * matched_mask).sum(axis=0) / matched_count_per_cand
                    stability = np.clip(coverage * 0.5 + q_quality_weighted * 0.5, 0.0, 1.0)

                    candidate_scores_for_scale = np.clip(
                        trimmed_mean * 0.74 + floor_score * 0.13 + coverage * 0.09 + consistency_bonus
                        + stability * 0.02 - quality_penalty,
                        0.0, 1.0,
                    )
                    if scale != 1.0:
                        candidate_scores_for_scale = np.where(
                            candidate_scores_for_scale >= best_scores_by_candidate + 0.028,
                            candidate_scores_for_scale,
                            0.0,
                        )

                    improved = candidate_scores_for_scale > best_scores_by_candidate
                    best_scores_by_candidate = np.where(improved, candidate_scores_for_scale, best_scores_by_candidate)
                    best_scales_by_candidate = np.where(improved, float(scale), best_scales_by_candidate)
                    best_match_count_by_candidate = np.where(
                        improved,
                        matched_mask.sum(axis=0).astype(np.int32),
                        best_match_count_by_candidate,
                    )
                    best_stability_by_candidate = np.where(improved, stability, best_stability_by_candidate)
                    best_quality_by_candidate = np.where(improved, q_quality_weighted, best_quality_by_candidate)
                    best_low_info_by_candidate = np.where(improved, low_info_ratio, best_low_info_by_candidate)

                candidate_scores = best_scores_by_candidate
                best_c = int(candidate_scores.argmax())
                best_cs = cands[best_c]
                best_scale = float(best_scales_by_candidate[best_c])
                second_score = float(np.partition(candidate_scores, -2)[-2]) if len(candidate_scores) > 1 else 0.0
                best_confidence = float(candidate_scores[best_c])
                rank_gap = max(0.0, best_confidence - second_score)
                if rank_gap < 0.005 and best_confidence < 0.93:
                    best_confidence *= 0.82
                elif rank_gap < 0.015 and best_confidence < 0.90:
                    best_confidence *= 0.90
                elif rank_gap < 0.030 and best_confidence < 0.86:
                    best_confidence *= 0.95
                return {
                    'start': max(0.0, best_cs),
                    'end': min(movie_duration or best_cs + duration * best_scale, best_cs + duration * best_scale),
                    'confidence': best_confidence,
                    'rank_gap': rank_gap,
                    'match_count': int(best_match_count_by_candidate[best_c]),
                    'stability_score': float(best_stability_by_candidate[best_c]),
                    'candidate_quality': float(best_quality_by_candidate[best_c]),
                    'query_quality': float(q_quality.mean()),
                    'low_info_ratio': float(best_low_info_by_candidate[best_c]),
                    'time_scale': best_scale,
                }

            # ── Fallback: original per-candidate Python loop (when precomputed data unavailable) ──
            result: Optional[dict] = None
            second_best_confidence = 0.0
            for candidate_start in cands:
                score_info = self._score_candidate(candidate_start, query_features, search_radius)
                if result is None or score_info['score'] > result['confidence']:
                    if result is not None:
                        second_best_confidence = max(second_best_confidence, float(result['confidence']))
                    result = {
                        'start': max(0.0, candidate_start),
                        'end': min(movie_duration or candidate_start + duration, candidate_start + duration),
                        'confidence': score_info['score'],
                        'rank_gap': 0.0,
                        'match_count': score_info['match_count'],
                        'stability_score': score_info['stability_score'],
                        'candidate_quality': score_info['candidate_quality'],
                        'query_quality': score_info['query_quality'],
                        'low_info_ratio': score_info['low_info_ratio'],
                    }
                else:
                    second_best_confidence = max(second_best_confidence, float(score_info['score']))
            if result is not None:
                result['rank_gap'] = max(0.0, float(result['confidence']) - second_best_confidence)
            return result

        best = await asyncio.to_thread(_run_search)

        if not best:
            return None

        min_success = max(0.32, self.match_threshold * (0.65 if relaxed else 0.55))
        if best['match_count'] <= 0 or best['confidence'] < min_success:
            return None

        confidence = float(best['confidence'])
        if confidence >= self.high_confidence_threshold:
            level = 'high'
        elif confidence >= self.medium_confidence_threshold:
            level = 'medium'
        else:
            level = 'low'

        return {
            'success': True,
            'start': float(best['start']),
            'end': float(best['end']),
            'confidence': confidence,
            'rank_gap': float(best.get('rank_gap', 0.0)),
            'match_count': int(best['match_count']),
            'stability_score': float(best.get('stability_score', 0.0)),
            'candidate_quality': float(best.get('candidate_quality', 0.0)),
            'query_quality': float(best.get('query_quality', 0.0)),
            'low_info_ratio': float(best.get('low_info_ratio', 0.0)),
            'confidence_level': level,
        }

    async def match_all_segments_fast(
        self,
        narration_path: str,
        segments: list[dict],
        sample_fps: float = 3.0,
        progress_callback=None,
        movie_duration: float | None = None,
        narration_duration: float | None = None,
        allow_non_sequential: bool = False,
        max_concurrent: int = 8,
        precomputed_features_map: Optional[dict[str, list[dict]]] = None,
        expected_time_map: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        total = max(1, len(segments))

        if not allow_non_sequential:
            # 严格顺序模式：保留 last_movie_time 约束
            results: list[dict] = []
            last_movie_time: Optional[float] = None
            for idx, segment in enumerate(segments, start=1):
                expected_time = (expected_time_map or {}).get(segment.get('id'))
                if expected_time is None and movie_duration and narration_duration and narration_duration > 0:
                    expected_time = float(segment['start']) / narration_duration * movie_duration
                if last_movie_time is not None:
                    expected_time = max(expected_time or 0.0, last_movie_time)
                result = await self.match_segment(
                    narration_path, float(segment['start']), float(segment['end']),
                    time_hint=expected_time, relaxed=False, strict_window=expected_time is not None,
                    precomputed_features=(precomputed_features_map or {}).get(segment.get('id')),
                )
                payload = {'id': segment.get('id'), 'success': False, 'start': None, 'end': None, 'confidence': 0.0, 'match_count': 0}
                if result:
                    payload.update(result)
                    last_movie_time = float(result['start'])
                results.append(payload)
                if progress_callback:
                    progress_callback('batch_match', int(idx * 100 / total), f'Matched {idx}/{total} segments')
            return results

        # 并行模式：allow_non_sequential=True，无顺序依赖，可全并发
        ordered: list[Optional[dict]] = [None] * len(segments)
        completed = 0
        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        async def _match_one(i: int, segment: dict) -> None:
            nonlocal completed
            async with semaphore:
                expected_time = (expected_time_map or {}).get(segment.get('id'))
                if expected_time is None and movie_duration and narration_duration and narration_duration > 0:
                    expected_time = float(segment['start']) / narration_duration * movie_duration
                result = await self.match_segment(
                    narration_path, float(segment['start']), float(segment['end']),
                    time_hint=expected_time,
                    relaxed=expected_time is not None,
                    strict_window=expected_time is not None,
                    precomputed_features=(precomputed_features_map or {}).get(segment.get('id')),
                )
                payload = {'id': segment.get('id'), 'success': False, 'start': None, 'end': None, 'confidence': 0.0, 'match_count': 0}
                if result:
                    payload.update(result)
                ordered[i] = payload
                completed += 1
                if progress_callback:
                    progress_callback('batch_match', int(completed * 100 / total), f'Matched {completed}/{total} segments')

        await asyncio.gather(*(_match_one(i, seg) for i, seg in enumerate(segments)))
        return [
            r if r is not None else {'id': seg.get('id'), 'success': False, 'start': None, 'end': None, 'confidence': 0.0, 'match_count': 0}
            for r, seg in zip(ordered, segments)
        ]

    def _feature_cache_key(self, capture_path: str, start_time: float, end_time: float, role: str) -> tuple[str, float, float, str]:
        return (str(capture_path), round(float(start_time), 3), round(float(end_time), 3), role)

    def _sample_times_for_segment(self, start_time: float, end_time: float) -> list[float]:
        duration = max(0.2, end_time - start_time)
        sample_count = max(2, min(6, int(duration * (2.0 if not self.fast_mode else 1.0))))
        return [start_time + duration * (i + 0.5) / sample_count for i in range(sample_count)]

    def _finalize_segment_features(self, features: list[dict]) -> list[dict]:
        if not features:
            return []
        strong_features = [feature for feature in features if feature['quality_score'] >= 0.20 and not feature['is_low_info']]
        if len(strong_features) >= max(2, len(features) // 3):
            return strong_features
        features = sorted(features, key=lambda item: item['quality_score'], reverse=True)
        keep_count = max(2, min(len(features), max(3, int(len(features) * 0.75))))
        return features[:keep_count]

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
        pending: list[dict] = []
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as handle:
                    payload = pickle.load(handle)
                if (
                    payload.get("cache_version") == self.CACHE_VERSION
                    and payload.get("video_path") == normalized_video_path
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
                    if feature_map:
                        return feature_map
            except Exception as exc:
                logger.warning(f"Failed to load narration feature cache {cache_path}: {exc}")

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

        def _extract_batch() -> dict[str, list[dict]]:
            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                raise ValueError(f'Cannot open narration video: {capture_path}')
            built: dict[str, list[dict]] = {}
            try:
                total = len(pending)
                for idx, item in enumerate(pending, start=1):
                    features: list[dict] = []
                    for sample_time in item["sample_times"]:
                        capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                        ok, frame = capture.read()
                        if not ok:
                            continue
                        features.append(
                            {
                                "offset": sample_time - item["start"],
                                **self._frame_features_lite(frame, subtitle_masker=subtitle_masker, frame_time=sample_time),
                            }
                        )
                    built[item["id"]] = self._finalize_segment_features(features)
                    if progress_callback and (idx == 1 or idx == total or idx % 25 == 0):
                        progress_callback(idx, total)
            finally:
                capture.release()
            return built

        built = await asyncio.to_thread(_extract_batch)
        for item in pending:
            segment_features = built.get(item["id"], [])
            self._query_feature_cache[item["cache_key"]] = segment_features
            feature_map[item["id"]] = segment_features
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as handle:
                pickle.dump(
                    {
                        "cache_version": self.CACHE_VERSION,
                        "video_path": normalized_video_path,
                        "mask_signature": mask_signature,
                        "segments_signature": segments_signature,
                        "feature_map": feature_map,
                    },
                    handle,
                )
        return feature_map

    async def _extract_segment_features(self, video_path: str, start_time: float, end_time: float) -> list[dict]:
        capture_path = await ensure_analysis_video(video_path)
        cache_key = self._feature_cache_key(str(capture_path), start_time, end_time, "narration")
        cached = self._query_feature_cache.get(cache_key)
        if cached is not None:
            return cached

        subtitle_masker = await asyncio.to_thread(self._get_subtitle_masker, str(capture_path), "narration", True)
        sample_times = self._sample_times_for_segment(start_time, end_time)

        def _extract() -> list[dict]:
            capture = cv2.VideoCapture(str(capture_path))
            if not capture.isOpened():
                raise ValueError(f'Cannot open narration video: {capture_path}')
            features: list[dict] = []
            for sample_time in sample_times:
                capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                features.append(
                    {
                        'offset': sample_time - start_time,
                        **self._frame_features_lite(frame, subtitle_masker=subtitle_masker, frame_time=sample_time),
                    }
                )
            capture.release()
            return self._finalize_segment_features(features)

        features = await asyncio.to_thread(_extract)
        self._query_feature_cache[cache_key] = features
        return features

    async def verify_candidates(
        self,
        narration_path: str,
        segment_start: float,
        segment_end: float,
        candidates: list[dict],
    ) -> dict[tuple[float, float], dict]:
        if not candidates or not self._indexed_capture_path:
            return {}

        narration_capture_path = await ensure_analysis_video(narration_path)
        movie_capture_path = self._indexed_capture_path
        narration_masker = await asyncio.to_thread(self._get_subtitle_masker, str(narration_capture_path), "narration", True)
        movie_masker = await asyncio.to_thread(self._get_subtitle_masker, str(movie_capture_path), "movie", False)

        duration = max(0.2, segment_end - segment_start)
        sample_positions = [0.35, 0.65] if duration < 2.4 else [0.2, 0.5, 0.8]

        def _verify() -> dict[tuple[float, float], dict]:
            narr_cap = cv2.VideoCapture(str(narration_capture_path))
            movie_cap = cv2.VideoCapture(str(movie_capture_path))
            if not narr_cap.isOpened() or not movie_cap.isOpened():
                raise ValueError("Cannot open videos for candidate verification")
            orb = cv2.ORB_create(nfeatures=480, fastThreshold=16)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            results: dict[tuple[float, float], dict] = {}

            def _read_processed_frame(cap, t: float, masker, role: str):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
                ok, frame = cap.read()
                if not ok or frame is None:
                    return None
                h, w = frame.shape[:2]
                if w > 512:
                    nh = max(32, int(h * 512.0 / w))
                    frame = cv2.resize(frame, (512, nh), interpolation=cv2.INTER_AREA)
                processed = self._preprocess_frame(frame, masker, frame_time=t)
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                return gray

            try:
                for candidate in candidates:
                    candidate_start = float(candidate["start"])
                    candidate_end = float(candidate["end"])
                    candidate_duration = max(0.2, candidate_end - candidate_start)
                    frame_scores: list[float] = []
                    inlier_counts: list[int] = []
                    inlier_ratios: list[float] = []
                    for pos in sample_positions:
                        q_time = segment_start + duration * pos
                        c_time = candidate_start + candidate_duration * pos
                        q_gray = _read_processed_frame(narr_cap, q_time, narration_masker, "narration")
                        c_gray = _read_processed_frame(movie_cap, c_time, movie_masker, "movie")
                        if q_gray is None or c_gray is None:
                            continue

                        q_hist = cv2.calcHist([q_gray], [0], None, [32], [0, 256]).astype("float32")
                        c_hist = cv2.calcHist([c_gray], [0], None, [32], [0, 256]).astype("float32")
                        q_hist /= max(float(q_hist.sum()), 1.0)
                        c_hist /= max(float(c_hist.sum()), 1.0)
                        hist_score = float((cv2.compareHist(q_hist, c_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)

                        kp1, des1 = orb.detectAndCompute(q_gray, None)
                        kp2, des2 = orb.detectAndCompute(c_gray, None)
                        inliers = 0
                        inlier_ratio = 0.0
                        warp_score = 0.0
                        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                            matches = matcher.knnMatch(des1, des2, k=2)
                            good = []
                            for pair in matches:
                                if len(pair) < 2:
                                    continue
                                m, n = pair
                                if m.distance < 0.78 * n.distance:
                                    good.append(m)
                            if len(good) >= 8:
                                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                if mask is not None:
                                    inliers = int(mask.sum())
                                    inlier_ratio = inliers / max(1, len(good))
                                    if H is not None and inliers >= 8:
                                        try:
                                            warped = cv2.warpPerspective(c_gray, H, (q_gray.shape[1], q_gray.shape[0]))
                                            diff = float(np.mean(np.abs(q_gray.astype(np.float32) - warped.astype(np.float32)))) / 255.0
                                            warp_score = max(0.0, 1.0 - diff)
                                        except cv2.error:
                                            warp_score = 0.0
                            elif good:
                                inliers = len(good)
                                inlier_ratio = min(1.0, len(good) / max(12.0, min(len(kp1), len(kp2))))
                        geom_score = min(1.0, inliers / 40.0) * 0.30 + inlier_ratio * 0.30 + warp_score * 0.40
                        frame_scores.append(max(hist_score * 0.20 + geom_score * 0.80, hist_score * 0.45))
                        inlier_counts.append(inliers)
                        inlier_ratios.append(inlier_ratio)

                    if frame_scores:
                        results[(candidate_start, candidate_end)] = {
                            "verification_score": float(np.mean(frame_scores)),
                            "geometric_inliers": int(round(float(np.mean(inlier_counts)))) if inlier_counts else 0,
                            "geometric_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
                        }
                    else:
                        results[(candidate_start, candidate_end)] = {
                            "verification_score": 0.0,
                            "geometric_inliers": 0,
                            "geometric_inlier_ratio": 0.0,
                        }
            finally:
                narr_cap.release()
                movie_cap.release()
            return results

        return await asyncio.to_thread(_verify)

    async def verify_segment_matches(
        self,
        narration_path: str,
        segments: list[dict],
    ) -> dict[str, dict]:
        """Batch verify already-selected matches while reusing video handles.

        This is intentionally stricter than the retrieval score. Retrieval can find
        visually similar frames, but auto-accept must prove local geometry or strong
        structural similarity on the exact selected timeline.
        """
        if not segments or not self._indexed_capture_path:
            return {}

        narration_capture_path = await ensure_analysis_video(narration_path)
        movie_capture_path = self._indexed_capture_path
        narration_masker = await asyncio.to_thread(self._get_subtitle_masker, str(narration_capture_path), "narration", True)
        movie_masker = await asyncio.to_thread(self._get_subtitle_masker, str(movie_capture_path), "movie", False)

        payload = [
            {
                "id": str(item["id"]),
                "narration_start": float(item["narration_start"]),
                "narration_end": float(item["narration_end"]),
                "movie_start": float(item["movie_start"]),
                "movie_end": float(item["movie_end"]),
            }
            for item in segments
            if item.get("id") and item.get("movie_start") is not None and item.get("movie_end") is not None
        ]
        if not payload:
            return {}

        def _verify() -> dict[str, dict]:
            narr_cap = cv2.VideoCapture(str(narration_capture_path))
            movie_cap = cv2.VideoCapture(str(movie_capture_path))
            if not narr_cap.isOpened() or not movie_cap.isOpened():
                raise ValueError("Cannot open videos for batch match verification")

            orb = cv2.ORB_create(nfeatures=640, fastThreshold=14)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            results: dict[str, dict] = {}

            def _read_processed_frame(cap, t: float, masker):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
                ok, frame = cap.read()
                if not ok or frame is None:
                    return None
                h, w = frame.shape[:2]
                if w > 512:
                    nh = max(32, int(h * 512.0 / w))
                    frame = cv2.resize(frame, (512, nh), interpolation=cv2.INTER_AREA)
                processed = self._preprocess_frame(frame, masker, frame_time=t)
                return cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            def _score_pair(q_gray, c_gray) -> tuple[float, int, float, float, float]:
                if q_gray.shape != c_gray.shape:
                    c_gray = cv2.resize(c_gray, (q_gray.shape[1], q_gray.shape[0]), interpolation=cv2.INTER_AREA)

                q_hist = cv2.calcHist([q_gray], [0], None, [32], [0, 256]).astype("float32")
                c_hist = cv2.calcHist([c_gray], [0], None, [32], [0, 256]).astype("float32")
                q_hist /= max(float(q_hist.sum()), 1.0)
                c_hist /= max(float(c_hist.sum()), 1.0)
                hist_score = float((cv2.compareHist(q_hist, c_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)

                q_edges = cv2.Canny(q_gray, 60, 140)
                c_edges = cv2.Canny(c_gray, 60, 140)
                edge_diff = float(np.mean(np.abs(q_edges.astype(np.float32) - c_edges.astype(np.float32)))) / 255.0
                edge_score = max(0.0, 1.0 - edge_diff)

                kp1, des1 = orb.detectAndCompute(q_gray, None)
                kp2, des2 = orb.detectAndCompute(c_gray, None)
                inliers = 0
                inlier_ratio = 0.0
                warp_score = 0.0
                if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                    matches = matcher.knnMatch(des1, des2, k=2)
                    good = []
                    for pair in matches:
                        if len(pair) < 2:
                            continue
                        m, n = pair
                        if m.distance < 0.78 * n.distance:
                            good.append(m)
                    if len(good) >= 8:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if mask is not None:
                            inliers = int(mask.sum())
                            inlier_ratio = inliers / max(1, len(good))
                            if H is not None and inliers >= 8:
                                try:
                                    warped = cv2.warpPerspective(c_gray, H, (q_gray.shape[1], q_gray.shape[0]))
                                    diff = float(np.mean(np.abs(q_gray.astype(np.float32) - warped.astype(np.float32)))) / 255.0
                                    warp_score = max(0.0, 1.0 - diff)
                                except cv2.error:
                                    warp_score = 0.0
                    elif good:
                        inliers = len(good)
                        inlier_ratio = min(1.0, len(good) / max(12.0, min(len(kp1), len(kp2))))

                geom_score = min(1.0, inliers / 42.0) * 0.30 + inlier_ratio * 0.34 + warp_score * 0.36
                structural_score = hist_score * 0.55 + edge_score * 0.45
                score = max(
                    structural_score * 0.38 + geom_score * 0.62,
                    structural_score * 0.55 if inliers < 5 else 0.0,
                )
                return float(score), int(inliers), float(inlier_ratio), float(hist_score), float(edge_score)

            try:
                for item in payload:
                    duration = max(0.2, item["narration_end"] - item["narration_start"])
                    movie_duration = max(0.2, item["movie_end"] - item["movie_start"])
                    if duration < 1.4:
                        sample_positions = [0.50]
                    elif duration < 2.8:
                        sample_positions = [0.35, 0.65]
                    else:
                        sample_positions = [0.25, 0.50, 0.75]

                    frame_scores: list[float] = []
                    inlier_counts: list[int] = []
                    inlier_ratios: list[float] = []
                    hist_scores: list[float] = []
                    edge_scores: list[float] = []

                    for pos in sample_positions:
                        q_time = item["narration_start"] + duration * pos
                        c_time = item["movie_start"] + movie_duration * pos
                        q_gray = _read_processed_frame(narr_cap, q_time, narration_masker)
                        c_gray = _read_processed_frame(movie_cap, c_time, movie_masker)
                        if q_gray is None or c_gray is None:
                            continue
                        score, inliers, inlier_ratio, hist_score, edge_score = _score_pair(q_gray, c_gray)
                        frame_scores.append(score)
                        inlier_counts.append(inliers)
                        inlier_ratios.append(inlier_ratio)
                        hist_scores.append(hist_score)
                        edge_scores.append(edge_score)

                    if not frame_scores:
                        results[item["id"]] = {
                            "verification_score": 0.0,
                            "geometric_inliers": 0,
                            "geometric_inlier_ratio": 0.0,
                            "hist_score": 0.0,
                            "edge_score": 0.0,
                            "sample_count": 0,
                        }
                        continue

                    results[item["id"]] = {
                        "verification_score": float(np.mean(frame_scores)),
                        "geometric_inliers": int(round(float(np.mean(inlier_counts)))) if inlier_counts else 0,
                        "geometric_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
                        "hist_score": float(np.mean(hist_scores)) if hist_scores else 0.0,
                        "edge_score": float(np.mean(edge_scores)) if edge_scores else 0.0,
                        "sample_count": len(frame_scores),
                    }
            finally:
                narr_cap.release()
                movie_cap.release()

            return results

        return await asyncio.to_thread(_verify)

    def _candidate_starts(self, duration: float, time_hint: Optional[float], relaxed: bool, strict_window: bool) -> list[float]:
        if not self._times:
            return []
        if time_hint is None:
            return self._times

        base_window = max(duration * (4 if strict_window else 8), 120.0 if strict_window else 99999.0)
        if relaxed:
            base_window *= 1.6
        left = bisect.bisect_left(self._times, max(0.0, time_hint - base_window))
        right = bisect.bisect_right(self._times, time_hint + base_window)
        subset = self._times[left:right]
        return subset if subset else self._times

    def _score_candidate(
        self,
        candidate_start: float,
        query_features: list[dict],
        search_radius: float,
        time_scales: Optional[tuple[float, ...]] = None,
    ) -> dict:
        best_score = 0.0
        best_matched = 0
        best_query_quality = 0.0
        best_candidate_quality = 0.0
        best_low_info_ratio = 1.0
        best_stability = 0.0
        best_time_scale = 1.0
        scales = time_scales if time_scales is not None else self._time_scales
        for time_scale in scales:
            scores: list[float] = []
            weighted_query_quality = 0.0
            weighted_candidate_quality = 0.0
            low_info_matches = 0
            matched = 0
            for query in query_features:
                target_time = candidate_start + float(query['offset']) * time_scale
                left = bisect.bisect_left(self._times, target_time - search_radius)
                right = bisect.bisect_right(self._times, target_time + search_radius)
                if right <= left:
                    continue
                best_local = 0.0
                best_candidate_quality_local = 0.0
                best_local_low_info = True
                for idx in range(left, right):
                    candidate = self._index[idx]
                    feature_score = self._feature_score(query, candidate)
                    feature_score *= 0.82 + 0.18 * min(float(query['quality_score']), float(candidate['quality_score']))
                    if feature_score > best_local:
                        best_local = feature_score
                        best_candidate_quality_local = float(candidate['quality_score'])
                        best_local_low_info = bool(candidate['is_low_info'])
                if best_local > 0:
                    scores.append(best_local)
                    matched += 1
                    weighted_query_quality += float(query['quality_score'])
                    weighted_candidate_quality += best_candidate_quality_local
                    if best_local_low_info or query['is_low_info']:
                        low_info_matches += 1
            if not scores:
                continue
            coverage = matched / max(1, len(query_features))
            avg_query_quality = weighted_query_quality / max(1, matched)
            avg_candidate_quality = weighted_candidate_quality / max(1, matched)
            low_info_ratio = low_info_matches / max(1, matched)
            stability = max(0.0, min(1.0, coverage * 0.5 + avg_query_quality * 0.25 + avg_candidate_quality * 0.25))
            quality_penalty = max(0.0, low_info_ratio - 0.25) * 0.25
            # 截尾均值（取分数最高的70%）减少噪声帧影响
            sorted_scores = sorted(scores, reverse=True)
            keep = len(sorted_scores) if len(sorted_scores) <= 3 else max(2, int(len(sorted_scores) * 0.75))
            top_scores = sorted_scores[:keep]
            trimmed_mean = float(np.mean(top_scores))
            floor_score = float(top_scores[-1]) if top_scores else trimmed_mean
            # 时序一致性：正确匹配时各帧得分应相近（变异系数小）
            # 错误匹配往往只有少数帧碰巧相似，方差大
            if len(top_scores) >= 2:
                score_cv = float(np.std(top_scores)) / (trimmed_mean + 1e-6)
                consistency_bonus = max(0.0, 0.06 * (1.0 - min(1.0, score_cv * 2.0)))
            else:
                consistency_bonus = 0.0
            score = float(trimmed_mean * 0.74 + floor_score * 0.13 + coverage * 0.09 + stability * 0.02 + consistency_bonus - quality_penalty)
            if score > best_score:
                best_score = score
                best_matched = matched
                best_query_quality = avg_query_quality
                best_candidate_quality = avg_candidate_quality
                best_low_info_ratio = low_info_ratio
                best_stability = stability
                best_time_scale = float(time_scale)
        return {
            'score': max(0.0, min(1.0, best_score)),
            'match_count': best_matched,
            'query_quality': max(0.0, min(1.0, best_query_quality)),
            'candidate_quality': max(0.0, min(1.0, best_candidate_quality)),
            'low_info_ratio': max(0.0, min(1.0, best_low_info_ratio if best_matched else 1.0)),
            'stability_score': max(0.0, min(1.0, best_stability)),
            'time_scale': best_time_scale,
        }

    def score_precomputed_segment_at(
        self,
        precomputed_features: list[dict],
        candidate_start: float,
        *,
        search_radius: Optional[float] = None,
        time_scales: Optional[tuple[float, ...]] = None,
    ) -> dict:
        """Score a fixed movie timestamp using the already built frame index."""
        if not self._index or not precomputed_features:
            return {
                "confidence": 0.0,
                "match_count": 0,
                "stability_score": 0.0,
                "candidate_quality": 0.0,
                "query_quality": 0.0,
                "low_info_ratio": 1.0,
                "time_scale": 1.0,
            }
        radius = (
            float(search_radius)
            if search_radius is not None
            else max(0.75, min(1.25, float(self._sample_step_seconds) * 0.9))
        )
        scored = self._score_candidate(max(0.0, float(candidate_start)), precomputed_features, radius, time_scales)
        return {
            "confidence": float(scored.get("score", 0.0)),
            "match_count": int(scored.get("match_count", 0)),
            "stability_score": float(scored.get("stability_score", 0.0)),
            "candidate_quality": float(scored.get("candidate_quality", 0.0)),
            "query_quality": float(scored.get("query_quality", 0.0)),
            "low_info_ratio": float(scored.get("low_info_ratio", 1.0)),
            "time_scale": float(scored.get("time_scale", 1.0)),
        }

    def _feature_score(self, query: dict, candidate: dict) -> float:
        query_variants = query["variants"]
        candidate_variants = candidate["variants"]

        full_score = self._variant_score(query_variants, "full", candidate_variants, "full")
        center_score = max(
            self._variant_score(query_variants, "full", candidate_variants, "center"),
            self._variant_score(query_variants, "center", candidate_variants, "full"),
            self._variant_score(query_variants, "center", candidate_variants, "center"),
        )
        side_score = max(
            self._variant_score(query_variants, "left", candidate_variants, "full"),
            self._variant_score(query_variants, "right", candidate_variants, "full"),
            self._variant_score(query_variants, "top", candidate_variants, "full"),
            self._variant_score(query_variants, "bottom", candidate_variants, "full"),
            self._variant_score(query_variants, "full", candidate_variants, "left"),
            self._variant_score(query_variants, "full", candidate_variants, "right"),
            self._variant_score(query_variants, "full", candidate_variants, "top"),
            self._variant_score(query_variants, "full", candidate_variants, "bottom"),
            self._variant_score(query_variants, "left", candidate_variants, "left"),
            self._variant_score(query_variants, "right", candidate_variants, "right"),
            self._variant_score(query_variants, "top", candidate_variants, "top"),
            self._variant_score(query_variants, "bottom", candidate_variants, "bottom"),
        )
        mirror_score = max(
            self._variant_score(query_variants, "flip_full", candidate_variants, "full"),
            self._variant_score(query_variants, "flip_center", candidate_variants, "center"),
            self._variant_score(query_variants, "flip_full", candidate_variants, "center"),
            self._variant_score(query_variants, "flip_center", candidate_variants, "full"),
            self._variant_score(query_variants, "full", candidate_variants, "flip_full"),
            self._variant_score(query_variants, "center", candidate_variants, "flip_center"),
            self._variant_score(query_variants, "full", candidate_variants, "flip_center"),
            self._variant_score(query_variants, "center", candidate_variants, "flip_full"),
        )
        layout_score = self._layout_similarity(query["layout_vector"], candidate["layout_vector"])
        mirror_layout_score = max(
            self._layout_similarity(query["flip_layout_vector"], candidate["layout_vector"]),
            self._layout_similarity(query["layout_vector"], candidate["flip_layout_vector"]),
        )
        edge_score = max(
            self._edge_similarity(query["edge_hash"], candidate["edge_hash"]),
            self._edge_similarity(query["flip_edge_hash"], candidate["edge_hash"]),
            self._edge_similarity(query["edge_hash"], candidate["flip_edge_hash"]),
        )

        direct_combined = full_score * 0.52 + center_score * 0.18 + side_score * 0.08 + edge_score * 0.14 + layout_score * 0.08
        crop_combined = center_score * 0.34 + side_score * 0.32 + full_score * 0.14 + edge_score * 0.14 + layout_score * 0.06
        mirror_combined = mirror_score * 0.36 + full_score * 0.22 + center_score * 0.14 + edge_score * 0.16 + mirror_layout_score * 0.12
        return max(0.0, min(1.0, max(direct_combined, crop_combined, mirror_combined)))

    def _frame_features(
        self,
        frame: np.ndarray,
        subtitle_masker: Optional[SubtitleMasker] = None,
        frame_time: Optional[float] = None,
    ) -> dict:
        processed = self._preprocess_frame(frame, subtitle_masker, frame_time=frame_time)
        flipped = cv2.flip(processed, 1)
        variant_frames = {
            "full": processed,
            "center": self._crop_frame(processed, ratio=self._crop_ratio_center, anchor="center"),
            "left": self._crop_frame(processed, ratio=self._crop_ratio_side, anchor="left"),
            "right": self._crop_frame(processed, ratio=self._crop_ratio_side, anchor="right"),
            "top": self._crop_frame(processed, ratio=self._crop_ratio_side, anchor="top"),
            "bottom": self._crop_frame(processed, ratio=self._crop_ratio_side, anchor="bottom"),
            "flip_full": flipped,
            "flip_center": self._crop_frame(flipped, ratio=self._crop_ratio_center, anchor="center"),
        }
        variants: dict[str, dict] = {}
        edge_hash = None
        flip_edge_hash = None
        layout_vector = None
        flip_layout_vector = None
        for name, variant_frame in variant_frames.items():
            resized = cv2.resize(variant_frame, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            ahash, phash, hist = self._hash_and_hist(gray)
            variants[name] = {"hash": ahash, "phash": phash, "hist": hist}
            if name == "full":
                edge_hash = self._edge_hash(gray)
                layout_vector = self._layout_descriptor(gray)
            elif name == "flip_full":
                flip_edge_hash = self._edge_hash(gray)
                flip_layout_vector = self._layout_descriptor(gray)
        return {
            "variants": variants,
            "edge_hash": edge_hash if edge_hash is not None else np.zeros(64, dtype=np.uint8),
            "flip_edge_hash": flip_edge_hash if flip_edge_hash is not None else np.zeros(64, dtype=np.uint8),
            "layout_vector": layout_vector if layout_vector is not None else np.zeros(16, dtype=np.float32),
            "flip_layout_vector": flip_layout_vector if flip_layout_vector is not None else np.zeros(16, dtype=np.float32),
            **self._quality_features(variant_frames["full"]),
        }

    def _frame_features_lite(
        self,
        frame: np.ndarray,
        subtitle_masker: Optional[SubtitleMasker] = None,
        frame_time: Optional[float] = None,
    ) -> dict:
        """Lightweight variant for index building and query extraction.

        Computes only 'full' and 'center' variants (no flip/side crops).
        Pre-downscales the input frame to ≤256px wide so that all subsequent
        ops work on a small image regardless of the source resolution.
        ~3-4× faster than _frame_features.
        """
        # Downscale to a fixed small size before any processing
        h, w = frame.shape[:2]
        if w > 256:
            nh = max(16, int(h * 256.0 / w))
            frame = cv2.resize(frame, (256, nh), interpolation=cv2.INTER_AREA)

        processed = self._preprocess_frame(frame, subtitle_masker, frame_time=frame_time)
        center_frame = self._crop_frame(processed, ratio=self._crop_ratio_center, anchor="center")

        variants: dict[str, dict] = {}
        full_gray: Optional[np.ndarray] = None
        edge_hash = None
        layout_vector = None

        color_hist = np.zeros(48, dtype='float32')
        spatial_color_hist = np.zeros(432, dtype='float32')
        grad_hist = np.zeros(128, dtype='float32')
        for name, vframe in (("full", processed), ("center", center_frame)):
            resized = cv2.resize(vframe, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            ahash, phash, hist = self._hash_and_hist(gray)
            variants[name] = {"hash": ahash, "phash": phash, "hist": hist}
            if name == "full":
                full_gray = gray
                edge_hash = self._edge_hash(gray)
                layout_vector = self._layout_descriptor(gray)
                color_hist, spatial_color_hist = self._color_and_spatial_hist(resized)
                grad_hist = self._gradient_orientation_hist(gray)

        # Quality features — reuse full_gray to avoid a second Canny call
        if full_gray is not None:
            edge_map = cv2.Canny(full_gray, 40, 140)
            edge_density = float(np.count_nonzero(edge_map)) / float(edge_map.size)
            std_value = float(np.std(full_gray))
            black_ratio = float(np.mean(full_gray <= 18))
            quality_score = (
                min(1.0, std_value / 42.0) * 0.42
                + min(1.0, edge_density / 0.08) * 0.38
                + max(0.0, 1.0 - black_ratio) * 0.20
            )
            is_low_info = bool(black_ratio >= 0.80 or (std_value < 12.0 and edge_density < 0.018))
        else:
            quality_score = 0.0
            edge_density = 0.0
            black_ratio = 0.0
            is_low_info = True

        zeros144 = np.zeros(144, dtype=np.uint8)
        zeros16 = np.zeros(16, dtype=np.float32)
        return {
            "variants": variants,
            "edge_hash": edge_hash if edge_hash is not None else zeros144,
            "flip_edge_hash": zeros144,
            "layout_vector": layout_vector if layout_vector is not None else zeros16,
            "flip_layout_vector": zeros16,
            "quality_score": max(0.0, min(1.0, quality_score)),
            "edge_density": edge_density,
            "black_ratio": black_ratio,
            "is_low_info": is_low_info,
            "color_hist": color_hist,
            "spatial_color_hist": spatial_color_hist,
            "grad_hist": grad_hist,
        }

    def _get_subtitle_masker(self, video_path: str, source_role: str, allow_fallback: bool) -> Optional[SubtitleMasker]:
        if not self.enable_subtitle_masking:
            return None
        cache_key = f'{source_role}:{video_path}'
        masker = self._subtitle_maskers.get(cache_key)
        if masker is None:
            masker = SubtitleMasker(
                manual_regions=self._subtitle_regions_by_role.get(source_role, []),
                mask_mode=self.subtitle_mask_mode,
            )
            self._subtitle_maskers[cache_key] = masker
        if masker.uses_auto_detection and not masker.has_fixed_mask:
            try:
                masker.detect_fixed_regions(video_path, allow_fallback=allow_fallback)
            except Exception as exc:
                logger.debug(f'Subtitle mask detection failed for {video_path}: {exc}')
        return masker

    def _normalize_regions(self, regions: Optional[list[dict]]) -> list[dict]:
        normalized: list[dict] = []
        for index, region in enumerate(regions or []):
            try:
                x = max(0.0, min(1.0, float(region.get('x', 0.0))))
                y = max(0.0, min(1.0, float(region.get('y', 0.0))))
                width = max(0.0, min(1.0 - x, float(region.get('width', 0.0))))
                height = max(0.0, min(1.0 - y, float(region.get('height', 0.0))))
            except (TypeError, ValueError, AttributeError):
                continue
            if width <= 0.0 or height <= 0.0:
                continue
            start_time = self._normalize_time(region.get('start_time'))
            end_time = self._normalize_time(region.get('end_time'))
            if start_time is not None and end_time is not None and end_time < start_time:
                end_time = start_time
            normalized.append(
                {
                    'id': region.get('id') or f'{index}',
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'enabled': bool(region.get('enabled', True)),
                    'start_time': start_time,
                    'end_time': end_time,
                }
            )
        return normalized

    def _normalize_time(self, value) -> Optional[float]:
        if value in (None, '', False):
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    def _mask_signature_for_role(self, source_role: str) -> tuple:
        regions = self._subtitle_regions_by_role.get(source_role, [])
        region_signature = tuple(
            self._region_signature_entry(source_role, region)
            for region in regions
        )
        return (
            bool(self.enable_subtitle_masking),
            str(self.subtitle_mask_mode),
            source_role,
            region_signature,
        )

    def _preprocess_frame(
        self,
        frame: np.ndarray,
        subtitle_masker: Optional[SubtitleMasker],
        frame_time: Optional[float] = None,
    ) -> np.ndarray:
        processed = frame
        if subtitle_masker is None:
            return self._crop_content_region(processed)
        try:
            # Get mask first, then zero-fill masked region with neutral gray (128)
            # This ensures movie and narration hashes are computed from identical
            # constant values in the subtitle area — blurring alone leaves different
            # pixel distributions that degrade hash similarity scores.
            mask = subtitle_masker.mask_for_frame(frame, frame_time=frame_time)
            if mask is not None and mask.any():
                processed = frame.copy()
                processed[mask > 0] = 128
            else:
                processed = frame
        except Exception as exc:
            logger.debug(f'Subtitle mask preprocessing failed: {exc}')
            processed = frame
        return self._crop_content_region(processed)

    def _center_crop(self, frame: np.ndarray, ratio: float = 0.82) -> np.ndarray:
        return self._crop_frame(frame, ratio=ratio, anchor="center")

    def _crop_frame(self, frame: np.ndarray, ratio: float, anchor: str) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_height = max(8, int(height * ratio))
        crop_width = max(8, int(width * ratio))
        if anchor == "left":
            top = max(0, (height - crop_height) // 2)
            left = 0
        elif anchor == "right":
            top = max(0, (height - crop_height) // 2)
            left = max(0, width - crop_width)
        elif anchor == "top":
            top = 0
            left = max(0, (width - crop_width) // 2)
        elif anchor == "bottom":
            top = max(0, height - crop_height)
            left = max(0, (width - crop_width) // 2)
        else:
            top = max(0, (height - crop_height) // 2)
            left = max(0, (width - crop_width) // 2)
        return frame[top : top + crop_height, left : left + crop_width]

    def _dct_hash(self, gray_frame: np.ndarray) -> np.ndarray:
        """64-bit DCT perceptual hash — frequency-based, far more robust than aHash
        for frames with compression artifacts or minor encoding differences."""
        img32 = cv2.resize(gray_frame, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct_mat = cv2.dct(img32)
        # Take the 8×8 low-frequency block (top-left), skip DC component
        dct_low = dct_mat[:8, :8].reshape(-1)
        median_val = float(np.median(dct_low))
        return (dct_low >= median_val).astype(np.uint8)

    def _edge_hash(self, gray_frame: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray_frame, 40, 140)
        # 12×12 = 144 bit 边缘哈希，细节更丰富
        small = cv2.resize(edges, (12, 12), interpolation=cv2.INTER_AREA)
        avg = float(np.mean(small))
        return (small >= avg).astype(np.uint8).reshape(-1)

    def _hash_and_hist(self, gray_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # aHash: 16×16 = 256 bits — 全局亮度分布
        small = cv2.resize(gray_frame, (16, 16), interpolation=cv2.INTER_AREA)
        avg = float(np.mean(small))
        ahash = (small >= avg).astype(np.uint8).reshape(-1)
        # DCT pHash: 64 bits — 频域特征，对压缩/编码差异鲁棒性更强
        phash = self._dct_hash(gray_frame)
        # 灰度直方图: 48 bins (kept for backward compat in scoring)
        hist = cv2.calcHist([gray_frame], [0], None, [48], [0, 256])
        hist = cv2.normalize(hist, hist).astype('float32')
        return ahash, phash, hist

    def _color_and_spatial_hist(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR color histogram (48 dims) + 3×3 spatial color histogram (432 dims).
        Color histograms are encoding-robust: same scene at different bitrates → ~0.92-0.95 similarity.
        3×3 grid (9 regions) provides finer spatial discrimination than 2×2 (4 regions).
        """
        h, w = bgr_frame.shape[:2]
        # Global BGR color histogram: 16 bins × 3 channels = 48 dims
        parts = []
        for ch in range(3):
            hc = cv2.calcHist([bgr_frame], [ch], None, [16], [0, 256]).flatten().astype('float32')
            parts.append(hc)
        color_hist = np.concatenate(parts)
        norm = float(np.linalg.norm(color_hist))
        if norm > 0:
            color_hist /= norm

        # Spatial color histogram: 3×3 grid × 16 bins × 3 channels = 432 dims
        rows = [0, h // 3, 2 * h // 3, h]
        cols = [0, w // 3, 2 * w // 3, w]
        spatial_parts = []
        for r in range(3):
            for c in range(3):
                region = bgr_frame[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
                if region.size == 0:
                    spatial_parts.append(np.zeros(48, dtype='float32'))
                    continue
                for ch in range(3):
                    hr = cv2.calcHist([region], [ch], None, [16], [0, 256]).flatten().astype('float32')
                    spatial_parts.append(hr)
        spatial_hist = np.concatenate(spatial_parts)  # (432,)
        norm2 = float(np.linalg.norm(spatial_hist))
        if norm2 > 0:
            spatial_hist /= norm2
        return color_hist, spatial_hist

    def _gradient_orientation_hist(self, gray_frame: np.ndarray) -> np.ndarray:
        """4×4 spatial grid × 8 orientation bins = 128-dim gradient orientation histogram.
        Gradient orientations are stable across re-encodings (depend on contrast structure, not absolute pixel values).
        Complements color histograms: distinguishes scenes with similar colors but different structure.
        """
        gx = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx * gx + gy * gy)
        # Unsigned orientation: 0-180 degrees
        angle = (np.arctan2(np.abs(gy), np.abs(gx)) * 180.0 / np.pi)  # 0-90, mirror to 0-180
        h, w = gray_frame.shape[:2]
        rows = [0, h // 4, h // 2, 3 * h // 4, h]
        cols = [0, w // 4, w // 2, 3 * w // 4, w]
        parts = []
        for r in range(4):
            for c in range(4):
                mag_r = magnitude[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
                ang_r = angle[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
                if mag_r.size == 0:
                    parts.append(np.zeros(8, dtype='float32'))
                    continue
                hist, _ = np.histogram(ang_r.ravel(), bins=8, range=(0.0, 90.0), weights=mag_r.ravel())
                parts.append(hist.astype('float32'))
        result = np.concatenate(parts)  # (128,)
        norm = float(np.linalg.norm(result))
        if norm > 0:
            result /= norm
        return result

    def _hash_hist_score(
        self,
        query_hash: np.ndarray,
        query_phash: np.ndarray,
        query_hist: np.ndarray,
        candidate_hash: np.ndarray,
        candidate_phash: np.ndarray,
        candidate_hist: np.ndarray,
    ) -> float:
        # aHash: 256 bits — 全局亮度感知（XOR+sum 等价于 count_nonzero on 0/1 array，但更快）
        ahash_dist = int(np.sum(query_hash ^ candidate_hash))
        ahash_score = max(0.0, 1.0 - ahash_dist / 256.0)
        # DCT pHash: 64 bits — 频域结构感知，对压缩更鲁棒，权重更高
        phash_dist = int(np.sum(query_phash ^ candidate_phash))
        phash_score = max(0.0, 1.0 - phash_dist / 64.0)
        # 灰度直方图
        hist_score = cv2.compareHist(query_hist, candidate_hist, cv2.HISTCMP_CORREL)
        hist_score = float(max(0.0, min(1.0, (hist_score + 1.0) / 2.0)))
        return max(0.0, min(1.0, ahash_score * 0.35 + phash_score * 0.45 + hist_score * 0.20))

    def _variant_score(self, query_variants: dict, query_name: str, candidate_variants: dict, candidate_name: str) -> float:
        query_feature = query_variants.get(query_name)
        candidate_feature = candidate_variants.get(candidate_name)
        if not query_feature or not candidate_feature:
            return 0.0
        return self._hash_hist_score(
            query_feature["hash"],
            query_feature["phash"],
            query_feature["hist"],
            candidate_feature["hash"],
            candidate_feature["phash"],
            candidate_feature["hist"],
        )

    def _edge_similarity(self, query_edge_hash: np.ndarray, candidate_edge_hash: np.ndarray) -> float:
        edge_distance = int(np.sum(query_edge_hash ^ candidate_edge_hash))
        # 除数与 edge hash 位数一致：12×12 = 144 bits
        return max(0.0, 1.0 - edge_distance / 144.0)

    def _layout_descriptor(self, gray_frame: np.ndarray, grid_size: int = 4) -> np.ndarray:
        small = cv2.resize(gray_frame, (grid_size, grid_size), interpolation=cv2.INTER_AREA).astype("float32")
        vector = small.reshape(-1)
        vector -= float(vector.mean())
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            vector /= norm
        return vector

    def _layout_similarity(self, query_layout: np.ndarray, candidate_layout: np.ndarray) -> float:
        denom = float(np.linalg.norm(query_layout) * np.linalg.norm(candidate_layout))
        if denom <= 1e-6:
            return 0.5
        cosine = float(np.dot(query_layout, candidate_layout) / denom)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _quality_features(self, frame: np.ndarray) -> dict:
        resized = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 40, 140)
        edge_density = float(np.count_nonzero(edge_map)) / float(edge_map.size)
        std_value = float(np.std(gray))
        black_ratio = float(np.mean(gray <= 18))
        quality_score = (
            min(1.0, std_value / 42.0) * 0.42
            + min(1.0, edge_density / 0.08) * 0.38
            + max(0.0, 1.0 - black_ratio) * 0.20
        )
        is_low_info = bool(black_ratio >= 0.80 or (std_value < 12.0 and edge_density < 0.018))
        return {
            "quality_score": max(0.0, min(1.0, quality_score)),
            "edge_density": edge_density,
            "black_ratio": black_ratio,
            "is_low_info": is_low_info,
        }

    def _crop_content_region(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height < 40 or width < 40:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_dark_ratio = np.mean(gray <= 26, axis=1)
        col_dark_ratio = np.mean(gray <= 26, axis=0)
        row_std = np.std(gray, axis=1)
        col_std = np.std(gray, axis=0)
        row_is_bar = (row_dark_ratio >= 0.97) & (row_std <= 10.0)
        col_is_bar = (col_dark_ratio >= 0.97) & (col_std <= 10.0)

        top = self._count_bar_prefix(row_is_bar)
        bottom = self._count_bar_suffix(row_is_bar)
        left = self._count_bar_prefix(col_is_bar)
        right = self._count_bar_suffix(col_is_bar)

        max_vertical_crop = int(height * 0.28)
        max_horizontal_crop = int(width * 0.18)
        top = min(top, max_vertical_crop)
        bottom = min(bottom, max_vertical_crop)
        left = min(left, max_horizontal_crop)
        right = min(right, max_horizontal_crop)

        if top + bottom >= int(height * 0.35):
            top = bottom = 0
        if left + right >= int(width * 0.28):
            left = right = 0
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return frame

        cropped = frame[top : height - bottom if bottom > 0 else height, left : width - right if right > 0 else width]
        if cropped.shape[0] < 24 or cropped.shape[1] < 24:
            return frame
        return cropped

    def _count_bar_prefix(self, mask: np.ndarray) -> int:
        count = 0
        for value in mask:
            if not value:
                break
            count += 1
        return count

    def _count_bar_suffix(self, mask: np.ndarray) -> int:
        count = 0
        for value in mask[::-1]:
            if not value:
                break
            count += 1
        return count

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"
