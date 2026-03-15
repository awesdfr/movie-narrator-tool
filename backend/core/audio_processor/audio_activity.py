"""Lightweight audio activity analysis for segment classification."""
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
class AudioActivityStats:
    rms_db: float
    peak_db: float
    active_ratio: float
    speech_likelihood: float
    label: str


class AudioActivityAnalyzer:
    """Estimate whether a segment contains likely speech or only weak/background audio."""

    def __init__(self, frame_ms: int = 30):
        self.frame_ms = max(10, frame_ms)
        self._audio_extractor = AudioExtractor()

    async def analyze_segment(self, video_path: str, start_time: float, end_time: float, output_dir: Optional[Path] = None) -> AudioActivityStats:
        audio_path = await self._audio_extractor.extract_segment(video_path, start_time, end_time, output_dir=output_dir)
        return await self.analyze_audio_file(str(audio_path))

    async def analyze_audio_file(self, audio_path: str) -> AudioActivityStats:
        return await asyncio.to_thread(self._analyze_audio_file_sync, audio_path)

    def _analyze_audio_file_sync(self, audio_path: str) -> AudioActivityStats:
        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw_bytes = handle.readframes(frame_count)

        if sample_width == 1:
            samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 3:
            # 24-bit PCM：每3字节扩展为4字节int32（带符号位扩展）
            raw_np = np.frombuffer(raw_bytes, dtype=np.uint8)
            n_samples = len(raw_np) // 3
            raw_np = raw_np[: n_samples * 3].reshape(n_samples, 3)
            padded = np.zeros((n_samples, 4), dtype=np.uint8)
            padded[:, :3] = raw_np  # 小端序：[LSB, MID, MSB, sign_ext]
            padded[:, 3] = np.where(raw_np[:, 2] & 0x80, np.uint8(0xFF), np.uint8(0x00))
            samples = np.frombuffer(padded.tobytes(), dtype=np.int32).astype(np.float32) / 8388608.0
        elif sample_width == 4:
            samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw_bytes, dtype=np.float32)

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if samples.size == 0:
            return AudioActivityStats(rms_db=-80.0, peak_db=-80.0, active_ratio=0.0, speech_likelihood=0.0, label="silent")

        rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-8)
        peak = float(np.max(np.abs(samples)) + 1e-8)
        rms_db = 20.0 * math.log10(rms)
        peak_db = 20.0 * math.log10(peak)

        window_size = max(1, int(sample_rate * self.frame_ms / 1000))
        window_count = max(1, math.ceil(samples.size / window_size))
        padded = np.pad(samples, (0, window_count * window_size - samples.size))
        windows = padded.reshape(window_count, window_size)
        frame_rms = np.sqrt(np.mean(np.square(windows), axis=1) + 1e-8)

        noise_floor = max(np.percentile(frame_rms, 20) * 1.6, 10 ** (-48 / 20))
        active_ratio = float(np.mean(frame_rms >= noise_floor))

        loudness_score = min(1.0, max(0.0, (rms_db + 48.0) / 24.0))
        activity_score = min(1.0, active_ratio / 0.35)
        speech_likelihood = max(0.0, min(1.0, loudness_score * 0.4 + activity_score * 0.6))

        if rms_db < -48.0 or active_ratio < 0.04:
            label = "silent"
        elif rms_db < -36.0 or active_ratio < 0.12:
            label = "weak"
        else:
            label = "active"

        return AudioActivityStats(
            rms_db=rms_db,
            peak_db=peak_db,
            active_ratio=active_ratio,
            speech_likelihood=speech_likelihood,
            label=label,
        )
