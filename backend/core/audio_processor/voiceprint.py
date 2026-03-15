"""Voiceprint-based narrator identification."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config import settings
from models.segment import SegmentType
from .audio_extractor import AudioExtractor
from .audio_activity import AudioActivityAnalyzer

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate as RESEMBLYZER_SR

    RESEMBLYZER_AVAILABLE = True
    RESEMBLYZER_IMPORT_ERROR = None
except Exception as exc:
    RESEMBLYZER_AVAILABLE = False
    RESEMBLYZER_SR = 16000
    RESEMBLYZER_IMPORT_ERROR = exc
    logger.warning(f'Resemblyzer is unavailable; narrator identification will fall back to narration-only mode: {exc}')


class VoiceprintRecognizer:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold or settings.voiceprint_threshold
        self._encoder = None
        self._reference_embedding = None
        self._audio_extractor = AudioExtractor()
        self._activity_analyzer = AudioActivityAnalyzer()

    def _load_encoder(self):
        if not RESEMBLYZER_AVAILABLE:
            raise RuntimeError(f'Resemblyzer is unavailable: {RESEMBLYZER_IMPORT_ERROR}')
        if self._encoder is None:
            self._encoder = VoiceEncoder()
        return self._encoder

    async def load_reference(self, audio_path: str):
        if not RESEMBLYZER_AVAILABLE:
            logger.warning('Skipping voiceprint reference load because Resemblyzer is unavailable')
            return

        def _load():
            encoder = self._load_encoder()
            wav = preprocess_wav(Path(audio_path))
            self._reference_embedding = encoder.embed_utterance(wav)

        await asyncio.to_thread(_load)

    async def identify_narrator(self, video_path: str, transcription: list[dict]) -> list[dict]:
        if not transcription:
            return []

        results: list[dict] = []
        for segment in transcription:
            text = str(segment.get('text', '') or '').strip()
            try:
                audio_path = await self._audio_extractor.extract_segment(
                    video_path,
                    float(segment['start']),
                    float(segment['end']),
                    output_dir=settings.temp_dir / 'voiceprint',
                )
                activity = await self._activity_analyzer.analyze_audio_file(str(audio_path))
                if RESEMBLYZER_AVAILABLE and self._reference_embedding is not None:
                    similarity = await self._compute_similarity(str(audio_path))
                else:
                    similarity = 1.0 if text else 0.0
            except Exception as exc:
                logger.warning(f'Voiceprint match failed for segment {segment.get("start")}-{segment.get("end")}: {exc}')
                activity = None
                similarity = 1.0 if text else 0.0

            seg_type = self._infer_segment_type(text=text, similarity=similarity, activity=activity)
            results.append(
                {
                    **segment,
                    'type': seg_type,
                    'voiceprint_similarity': similarity,
                    'audio_activity_label': activity.label if activity else 'unknown',
                    'speech_likelihood': activity.speech_likelihood if activity else 0.0,
                    'rms_db': activity.rms_db if activity else -80.0,
                }
            )
        return results

    def _infer_segment_type(self, text: str, similarity: float, activity) -> SegmentType:
        has_text = bool(text)
        speech_likelihood = activity.speech_likelihood if activity else (0.5 if has_text else 0.0)
        activity_label = activity.label if activity else 'unknown'

        if activity_label == 'silent' and not has_text:
            return SegmentType.NO_NARRATION

        if self._reference_embedding is None or not RESEMBLYZER_AVAILABLE:
            if not has_text and speech_likelihood < 0.18:
                return SegmentType.NO_NARRATION
            if len(text) < 2 and speech_likelihood < 0.24:
                return SegmentType.NO_NARRATION
            return SegmentType.HAS_NARRATION

        if similarity >= self.threshold and speech_likelihood >= 0.12:
            return SegmentType.HAS_NARRATION
        if similarity >= max(0.18, self.threshold - 0.08) and has_text and speech_likelihood >= 0.28:
            return SegmentType.HAS_NARRATION
        if not has_text and speech_likelihood < 0.2:
            return SegmentType.NO_NARRATION
        if similarity < max(0.18, self.threshold * 0.72):
            return SegmentType.NO_NARRATION
        return SegmentType.HAS_NARRATION

    async def _compute_similarity(self, audio_path: str) -> float:
        if self._reference_embedding is None:
            return 0.0

        def _compute() -> float:
            wav = preprocess_wav(Path(audio_path))
            if len(wav) < RESEMBLYZER_SR * 0.5:
                return 0.0
            encoder = self._load_encoder()
            embedding = encoder.embed_utterance(wav)
            similarity = float(np.dot(self._reference_embedding, embedding))
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        return await asyncio.to_thread(_compute)

    async def verify_speaker(self, audio_path: str, reference_path: str) -> tuple[bool, float]:
        if not RESEMBLYZER_AVAILABLE:
            raise RuntimeError('Resemblyzer is not installed')

        def _verify() -> tuple[bool, float]:
            encoder = self._load_encoder()
            emb_a = encoder.embed_utterance(preprocess_wav(Path(audio_path)))
            emb_b = encoder.embed_utterance(preprocess_wav(Path(reference_path)))
            similarity = float(np.dot(emb_a, emb_b))
            confidence = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            return confidence >= self.threshold, confidence

        return await asyncio.to_thread(_verify)
