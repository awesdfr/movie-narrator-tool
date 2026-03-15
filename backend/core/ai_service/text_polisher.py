"""Narration polishing helpers."""
import asyncio
import re
from typing import Optional

from loguru import logger

from .api_manager import APIManager


class TextPolisher:
    """Polish narration text with optional AI assistance and local safeguards."""

    SUPPORTED_LANGUAGES = {"zh": "Chinese", "en": "English"}

    STYLE_PRESETS = {
        "natural": "Keep the tone natural, restrained, and spoken. Avoid exaggerated transitions or summary phrases.",
        "movie_pro": "Write like a professional movie narrator: concise, cinematic, confident, and smooth between shots.",
        "short_video": "Write with stronger short-video energy, but keep it human and not templated.",
    }

    DEFAULT_TEMPLATE = (
        "You are a movie narration editor. Rewrite the narration to sound natural and spoken, "
        "without changing facts or adding plot details.\n\nOriginal:\n{text}\n\nOutput only the rewritten narration."
    )

    def __init__(
        self,
        api_manager: Optional[APIManager] = None,
        template: Optional[str] = None,
        language: str = "zh",
        temperature: float = 0.4,
        max_tokens: int = 2000,
        default_style_preset: str = "movie_pro",
        enable_de_ai_pass: bool = True,
        enable_self_review: bool = True,
    ):
        self._api_manager = api_manager
        self.language = language
        self.template = template or self.DEFAULT_TEMPLATE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_style_preset = default_style_preset
        self.enable_de_ai_pass = enable_de_ai_pass
        self.enable_self_review = enable_self_review

    def _get_api_manager(self) -> APIManager:
        if self._api_manager is None:
            self._api_manager = APIManager()
        return self._api_manager

    def _has_remote_model(self) -> bool:
        manager = self._api_manager or self._get_api_manager()
        return bool(manager.api_key)

    def _local_de_ai_cleanup(self, text: str) -> str:
        replacements = {
            "让我们一起来": "",
            "可以看到": "",
            "值得一提的是": "",
            "总的来说": "",
            "不得不说": "",
            "接下来": "",
            "首先": "",
            "其次": "",
            "最后": "",
            "In summary,": "",
            "overall,": "",
            "it is worth mentioning that": "",
        }
        cleaned = text.strip()
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        cleaned = re.sub(r"([，。！？!?])\1+", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _self_review_cleanup(self, text: str) -> str:
        cleaned = re.sub(r"(突然之间|镜头一转|不得不说|总之|总的来说)", "", text)
        cleaned = re.sub(r"([，。！？!?；;])\1+", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    async def polish(self, text: str, style: Optional[str] = None, language: Optional[str] = None) -> str:
        if not text or not text.strip():
            return text

        target_lang = language or self.language
        working = self._local_de_ai_cleanup(text) if self.enable_de_ai_pass else text.strip()
        style_text = style or self.STYLE_PRESETS.get(self.default_style_preset, self.STYLE_PRESETS["movie_pro"])
        prompt = (
            f"Language: {target_lang}\n"
            f"Style: {style_text}\n"
            "Constraints: keep facts, keep names, do not add information, avoid AI-like summary phrases.\n\n"
            f"Original:\n{working}\n\n"
            "Output only the rewritten narration."
        )

        try:
            if self._has_remote_model():
                result = await self._get_api_manager().chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                result = working
            result = result.strip().strip('"').strip("'")
            if len(result) > max(len(text) * 1.6, len(text) + 40):
                result = working
            if self.enable_self_review:
                result = self._self_review_cleanup(result)
            return result or working
        except Exception as exc:
            logger.warning(f"Polish failed, using local cleanup: {exc}")
            return self._self_review_cleanup(working)

    async def rewrite_segment(
        self,
        text: str,
        target_duration: Optional[float] = None,
        prev_text: str = "",
        next_text: str = "",
        match_reason: str = "",
        style_preset: Optional[str] = None,
    ) -> str:
        if not text or not text.strip():
            return text

        preset = style_preset or self.default_style_preset
        style_hint = self.STYLE_PRESETS.get(preset, self.STYLE_PRESETS["movie_pro"])
        working = self._local_de_ai_cleanup(text) if self.enable_de_ai_pass else text.strip()

        if not self._has_remote_model():
            result = self._self_review_cleanup(working)
            return self._fit_duration(result, target_duration)

        duration_hint = "unknown" if target_duration is None else f"about {target_duration:.1f} seconds"
        prompt = (
            "You are rewriting one segment of a movie narration script.\n"
            f"Style preset: {preset}. {style_hint}\n"
            f"Target spoken duration: {duration_hint}.\n"
            f"Previous segment: {prev_text or '[none]'}\n"
            f"Next segment: {next_text or '[none]'}\n"
            f"Match note: {match_reason or '[none]'}\n"
            "Rules:\n"
            "1. Keep names, places, and facts.\n"
            "2. Do not add plot details or moral commentary.\n"
            "3. Avoid AI-like fillers, staged transitions, and neat essay structure.\n"
            "4. Make it sound like a human narrator speaking over a matched shot.\n"
            "5. Keep the length close to the original and suitable for the target duration.\n\n"
            f"Original text:\n{working}\n\n"
            "Output only the rewritten segment."
        )
        try:
            result = await self._get_api_manager().chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning(f"Contextual rewrite failed, using fallback: {exc}")
            result = working

        result = result.strip().strip('"').strip("'")
        if self.enable_self_review:
            result = self._self_review_cleanup(result)
        return self._fit_duration(result or working, target_duration)

    def _fit_duration(self, text: str, target_duration: Optional[float]) -> str:
        if target_duration is None or target_duration <= 0 or not text:
            return text
        current_duration = max(len(text) / 4.0, 0.1)
        ratio = target_duration / current_duration
        if ratio < 0.75:
            return text[: max(1, int(len(text) * ratio * 1.05))].rstrip("，, ")
        return text

    async def translate_and_polish(
        self,
        text: str,
        source_language: str,
        target_language: str,
        style: Optional[str] = None,
    ) -> str:
        if not text or not text.strip():
            return text
        style_text = style or self.STYLE_PRESETS.get(self.default_style_preset, self.STYLE_PRESETS["movie_pro"])
        prompt = (
            f"Translate from {source_language} to {target_language} and rewrite as a natural movie narration.\n"
            f"Style: {style_text}\n"
            "Keep facts, keep names, do not add information.\n\n"
            f"Original:\n{text}\n\nOutput only the translated narration."
        )
        try:
            if self._has_remote_model():
                result = await self._get_api_manager().chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return self._self_review_cleanup(result.strip())
        except Exception as exc:
            logger.warning(f"Translate and polish failed: {exc}")
        return self._self_review_cleanup(text)

    async def polish_batch(self, texts: list[str], language: Optional[str] = None, concurrency: int = 3) -> list[str]:
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(text: str) -> str:
            async with semaphore:
                return await self.polish(text, language=language)

        return await asyncio.gather(*(worker(text) for text in texts))

    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
        concurrency: int = 3,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(text: str) -> str:
            async with semaphore:
                return await self.translate_and_polish(text, source_language, target_language)

        return await asyncio.gather(*(worker(text) for text in texts))

    async def rewrite_for_tts(self, text: str, target_duration: Optional[float] = None) -> str:
        return await self.rewrite_segment(text=text, target_duration=target_duration, style_preset=self.default_style_preset)

    def set_language(self, language: str):
        if language in self.SUPPORTED_LANGUAGES:
            self.language = language

    @classmethod
    def get_supported_languages(cls) -> dict[str, str]:
        return cls.SUPPORTED_LANGUAGES.copy()

    def set_template(self, template: str):
        if "{text}" not in template and "Original:" not in template:
            raise ValueError("Template must contain either {text} or an equivalent original-text placeholder")
        self.template = template

    async def close(self):
        if self._api_manager:
            await self._api_manager.close()
