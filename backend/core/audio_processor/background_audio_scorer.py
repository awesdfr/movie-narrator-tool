"""Lightweight background audio similarity scoring without extra heavy deps."""
from __future__ import annotations

import asyncio
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_extractor import AudioExtractor


@dataclass
class _PreparedAudio:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int


@dataclass
class AudioSimilarityResult:
    confidence: float
    envelope_confidence: float
    band_confidence: float
    onset_confidence: float
    narration_activity: str
    movie_activity: str
    note: str = ""


class AudioSimilarityScorer:
    """Compare two timeline ranges using coarse audio dynamics and band energy."""

    def __init__(
        self,
        sample_rate: int = 16000,
        target_steps: int = 48,
    ):
        self.sample_rate = sample_rate
        self.target_steps = max(24, target_steps)
        self._bands = ((40, 180), (180, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400))
        self._extractor = AudioExtractor()
        self._movie_audio: Optional[_PreparedAudio] = None
        self._narration_audio: Optional[_PreparedAudio] = None
        self._feature_cache: dict[tuple[str, float, float], dict] = {}

    async def prepare(self, movie_path: str, narration_path: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        movie_audio_path, narration_audio_path = await asyncio.gather(
            self._extractor.extract_full(movie_path, output_dir=output_dir, sample_rate=self.sample_rate, mono=True),
            self._extractor.extract_full(narration_path, output_dir=output_dir, sample_rate=self.sample_rate, mono=True),
        )
        self._movie_audio = await asyncio.to_thread(self._inspect_audio, Path(movie_audio_path))
        self._narration_audio = await asyncio.to_thread(self._inspect_audio, Path(narration_audio_path))

    async def score_pair(
        self,
        narration_start: float,
        narration_end: float,
        movie_start: float,
        movie_end: float,
    ) -> AudioSimilarityResult:
        if self._movie_audio is None or self._narration_audio is None:
            raise RuntimeError("AudioSimilarityScorer is not prepared")
        return await asyncio.to_thread(
            self._score_pair_sync,
            narration_start,
            narration_end,
            movie_start,
            movie_end,
        )

    def _score_pair_sync(
        self,
        narration_start: float,
        narration_end: float,
        movie_start: float,
        movie_end: float,
    ) -> AudioSimilarityResult:
        narration_features = self._segment_features(self._narration_audio, narration_start, narration_end, "narration")
        movie_features = self._segment_features(self._movie_audio, movie_start, movie_end, "movie")

        if narration_features["signal_length"] < self.sample_rate * 0.35 or movie_features["signal_length"] < self.sample_rate * 0.35:
            return AudioSimilarityResult(
                confidence=0.0,
                envelope_confidence=0.0,
                band_confidence=0.0,
                onset_confidence=0.0,
                narration_activity=narration_features["activity_label"],
                movie_activity=movie_features["activity_label"],
                note="audio_too_short",
            )

        if narration_features["activity_label"] == "silent" and movie_features["activity_label"] == "silent":
            return AudioSimilarityResult(
                confidence=0.5,
                envelope_confidence=0.5,
                band_confidence=0.5,
                onset_confidence=0.5,
                narration_activity="silent",
                movie_activity="silent",
                note="both_quiet",
            )

        if narration_features["activity_label"] == "silent" or movie_features["activity_label"] == "silent":
            return AudioSimilarityResult(
                confidence=0.18,
                envelope_confidence=0.12,
                band_confidence=0.18,
                onset_confidence=0.12,
                narration_activity=narration_features["activity_label"],
                movie_activity=movie_features["activity_label"],
                note="one_side_quiet",
            )

        envelope_confidence = self._cosine_similarity(
            narration_features["envelope_vector"],
            movie_features["envelope_vector"],
        )
        band_confidence = self._cosine_similarity(
            narration_features["band_vector"],
            movie_features["band_vector"],
        )
        band_profile_confidence = self._cosine_similarity(
            narration_features["band_profile"],
            movie_features["band_profile"],
        )
        band_confidence = band_confidence * 0.6 + band_profile_confidence * 0.4
        onset_confidence = self._cosine_similarity(
            narration_features["onset_vector"],
            movie_features["onset_vector"],
        )
        confidence = max(
            0.0,
            min(
                1.0,
                envelope_confidence * 0.35 + band_confidence * 0.45 + onset_confidence * 0.20,
            ),
        )
        note = ""
        if narration_features["activity_label"] == "weak" or movie_features["activity_label"] == "weak":
            confidence *= 0.92
            note = "weak_audio"
        return AudioSimilarityResult(
            confidence=confidence,
            envelope_confidence=envelope_confidence,
            band_confidence=band_confidence,
            onset_confidence=onset_confidence,
            narration_activity=narration_features["activity_label"],
            movie_activity=movie_features["activity_label"],
            note=note,
        )

    def _inspect_audio(self, audio_path: Path) -> _PreparedAudio:
        with wave.open(str(audio_path), "rb") as handle:
            return _PreparedAudio(
                path=audio_path,
                sample_rate=handle.getframerate(),
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                frame_count=handle.getnframes(),
            )

    def _segment_features(self, prepared: _PreparedAudio, start_time: float, end_time: float, source_key: str) -> dict:
        cache_key = (source_key, round(max(0.0, start_time), 3), round(max(start_time, end_time), 3))
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            return cached

        samples = self._read_segment(prepared, start_time, end_time)
        features = self._extract_features(samples, prepared.sample_rate)
        self._feature_cache[cache_key] = features
        return features

    def _read_segment(self, prepared: _PreparedAudio, start_time: float, end_time: float) -> np.ndarray:
        start_frame = max(0, min(prepared.frame_count, int(start_time * prepared.sample_rate)))
        end_frame = max(start_frame, min(prepared.frame_count, int(end_time * prepared.sample_rate)))
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            return np.zeros(0, dtype=np.float32)

        with wave.open(str(prepared.path), "rb") as handle:
            handle.setpos(start_frame)
            raw = handle.readframes(frame_count)

        if prepared.sample_width == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif prepared.sample_width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

        if prepared.channels > 1:
            samples = samples.reshape(-1, prepared.channels).mean(axis=1)
        return samples.astype(np.float32, copy=False)

    def _extract_features(self, samples: np.ndarray, sample_rate: int) -> dict:
        if samples.size == 0:
            empty = np.zeros(self.target_steps, dtype=np.float32)
            return {
                "signal_length": 0,
                "activity_label": "silent",
                "envelope_vector": empty,
                "band_vector": np.zeros(self.target_steps * len(self._bands), dtype=np.float32),
                "band_profile": np.zeros(len(self._bands), dtype=np.float32),
                "onset_vector": empty,
            }

        slices = np.array_split(samples, self.target_steps)
        envelope: list[float] = []
        band_rows: list[np.ndarray] = []
        for chunk in slices:
            if chunk.size == 0:
                envelope.append(0.0)
                band_rows.append(np.zeros(4, dtype=np.float32))
                continue
            envelope.append(float(np.sqrt(np.mean(np.square(chunk))) + 1e-8))
            band_rows.append(self._band_energies(chunk, sample_rate))

        envelope_vec = np.asarray(envelope, dtype=np.float32)
        band_matrix = np.vstack(band_rows).astype(np.float32)
        onset_vec = np.abs(np.diff(envelope_vec, prepend=envelope_vec[0])).astype(np.float32)
        active_ratio = float(np.mean(envelope_vec > max(np.percentile(envelope_vec, 25) * 1.4, 10 ** (-48 / 20))))
        rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(np.square(samples))) + 1e-8))
        if rms_db < -48.0 and active_ratio < 0.04:
            activity_label = "silent"
        elif rms_db < -36.0 and active_ratio < 0.12:
            activity_label = "weak"
        else:
            activity_label = "active"

        return {
            "signal_length": int(samples.size),
            "activity_label": activity_label,
            "envelope_vector": self._normalize_series(envelope_vec),
            "band_vector": self._normalize_positive_series(np.log1p(band_matrix).reshape(-1)),
            "band_profile": self._normalize_positive_series(np.log1p(band_matrix.mean(axis=0))),
            "onset_vector": self._normalize_series(onset_vec),
        }

    def _band_energies(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        if chunk.size < 16:
            return np.zeros(4, dtype=np.float32)
        window = np.hanning(chunk.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(chunk * window)).astype(np.float32)
        freqs = np.fft.rfftfreq(chunk.size, d=1.0 / sample_rate)
        energies = []
        for low, high in self._bands:
            mask = (freqs >= low) & (freqs < high)
            energies.append(float(spectrum[mask].mean()) if np.any(mask) else 0.0)
        return np.asarray(energies, dtype=np.float32)

    def _normalize_series(self, values: np.ndarray) -> np.ndarray:
        arr = values.astype(np.float32, copy=True)
        if arr.size == 0:
            return arr
        arr -= float(arr.mean())
        scale = float(arr.std())
        if scale > 1e-6:
            arr /= scale
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            arr /= norm
        return arr

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-6:
            return 0.5
        cosine = float(np.dot(a, b) / denom)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _normalize_positive_series(self, values: np.ndarray) -> np.ndarray:
        arr = values.astype(np.float32, copy=True)
        if arr.size == 0:
            return arr
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            arr /= norm
        return arr
