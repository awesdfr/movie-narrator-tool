"""Whisper-based speech recognition."""
import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from .audio_extractor import AudioExtractor

try:
    import whisper

    WHISPER_AVAILABLE = True
    WHISPER_IMPORT_ERROR = None
except Exception as exc:
    WHISPER_AVAILABLE = False
    WHISPER_IMPORT_ERROR = exc
    logger.warning(f"Whisper is unavailable; speech recognition will be disabled: {exc}")


class SpeechRecognizer:
    """Speech recognizer powered by OpenAI Whisper."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        language: Optional[str] = None,
        word_timestamps: Optional[bool] = None,
    ):
        self.model_name = model_name or settings.whisper_model
        self.device = device or settings.whisper_device
        self.language = language or settings.whisper_language
        self.word_timestamps = word_timestamps if word_timestamps is not None else True

        self._model = None
        self._audio_extractor = AudioExtractor()

    def _load_model(self):
        if not WHISPER_AVAILABLE:
            raise RuntimeError(f"Whisper is unavailable: {WHISPER_IMPORT_ERROR}")

        if self._model is None:
            import torch

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA is unavailable; falling back to CPU")
                self.device = "cpu"

            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper model loaded")

        return self._model

    async def transcribe(self, audio_or_video_path: str, language: Optional[str] = None) -> list[dict]:
        """Transcribe audio or video and keep word timestamps when available."""

        if not WHISPER_AVAILABLE:
            raise RuntimeError(f"Whisper is unavailable: {WHISPER_IMPORT_ERROR}")

        path = Path(audio_or_video_path)
        if path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
            audio_path = await self._audio_extractor.extract_full(str(path), output_dir=settings.temp_dir)
        else:
            audio_path = path

        def _transcribe():
            model = self._load_model()
            result = model.transcribe(
                str(audio_path),
                language=language or self.language,
                task="transcribe",
                verbose=False,
                word_timestamps=self.word_timestamps,
            )

            segments = []
            for seg in result.get("segments", []):
                words = []
                for word in seg.get("words", []) or []:
                    if word.get("start") is None or word.get("end") is None:
                        continue
                    words.append(
                        {
                            "start": float(word["start"]),
                            "end": float(word["end"]),
                            "word": str(word.get("word", "")),
                            "probability": float(word.get("probability", 0.0)),
                        }
                    )
                segments.append(
                    {
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": seg.get("text", "").strip(),
                        "words": words,
                    }
                )
            return segments

        return await asyncio.to_thread(_transcribe)

    async def transcribe_segment(
        self,
        audio_or_video_path: str,
        start_time: float,
        end_time: float,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe a specific sub-range."""

        segment_audio = await self._audio_extractor.extract_segment(
            audio_or_video_path,
            start_time,
            end_time,
            output_dir=settings.temp_dir,
        )

        def _transcribe():
            model = self._load_model()
            result = model.transcribe(
                str(segment_audio),
                language=language or self.language,
                task="transcribe",
                verbose=False,
                word_timestamps=self.word_timestamps,
            )
            return result.get("text", "").strip()

        return await asyncio.to_thread(_transcribe)

    async def detect_language(self, audio_path: str) -> tuple[str, float]:
        """Detect audio language."""

        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper is not installed")

        def _detect():
            model = self._load_model()
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            detected = max(probs, key=probs.get)
            return detected, float(probs[detected])

        return await asyncio.to_thread(_detect)

    def get_supported_languages(self) -> list[str]:
        """Return a subset of common Whisper languages."""

        if not WHISPER_AVAILABLE:
            return []
        return ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "it", "pt"]
