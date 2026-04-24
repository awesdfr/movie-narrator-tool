"""Narration polishing helpers."""
import asyncio
import math
import re
from typing import Optional

from loguru import logger

from .api_manager import APIManager


class TextPolisher:
    """Polish narration text with optional AI assistance and local safeguards."""

    SUPPORTED_LANGUAGES = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "it": "Italian",
        "pt": "Portuguese",
        "ar": "Arabic",
        "th": "Thai",
        "vi": "Vietnamese",
    }

    _CJK_LANGUAGES = {"zh", "ja", "ko"}
    _SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[。！？!?；;])")
    _CLAUSE_SPLIT_REGEX = re.compile(r"(?<=[，,、：:；;])")

    # Per-language AI-filler phrases to strip before polishing.
    _DE_AI_PHRASES: dict[str, list[str]] = {
        "zh": ["让我们一起来", "可以看到", "值得一提的是", "总的来说", "不得不说", "接下来", "首先", "其次", "最后"],
        "en": ["let's take a look", "it is worth noting that", "overall,", "in summary,",
               "first of all,", "next,", "finally,", "it is worth mentioning that",
               "as we can see,", "needless to say,"],
        "ja": ["まず最初に", "次に", "最後に", "言うまでもなく", "注目すべきは", "総じて言えば"],
        "ko": ["먼저", "다음으로", "마지막으로", "주목할 만한 것은", "전반적으로"],
        "fr": ["tout d'abord,", "ensuite,", "enfin,", "il convient de noter que", "en résumé,", "dans l'ensemble,"],
        "de": ["zunächst,", "dann,", "schließlich,", "es sei darauf hingewiesen,", "insgesamt,", "zusammenfassend,"],
        "es": ["primero,", "luego,", "finalmente,", "cabe destacar que", "en resumen,", "en general,"],
        "ru": ["во-первых,", "затем,", "наконец,", "стоит отметить,", "в целом,", "подводя итог,"],
        "it": ["prima di tutto,", "poi,", "infine,", "vale la pena notare che", "in generale,", "in sintesi,"],
        "pt": ["primeiramente,", "em seguida,", "por fim,", "vale notar que", "no geral,", "em resumo,"],
    }

    # Per-language self-review cleanup patterns.
    _REVIEW_PATTERNS: dict[str, str] = {
        "zh": r"(突然之间|镜头一转|不得不说|总之|总的来说|让我们看看|可以看到)",
        "en": r"(suddenly,?\s|the camera cuts to|it must be said|in conclusion,?\s|all in all,?\s|let's take a look|as we can see,?\s)",
        "ja": r"(突然|カメラが切り替わり|言うまでもなく|結論として)",
        "ko": r"(갑자기|카메라가 전환되며|말할 것도 없이|결론적으로)",
        "fr": r"(soudainement,?\s|la caméra coupe|il faut dire|en conclusion,?\s)",
        "de": r"(plötzlich,?\s|die Kamera schwenkt|muss man sagen|abschließend,?\s)",
        "es": r"(de repente,?\s|la cámara corta|hay que decir|en conclusión,?\s)",
        "ru": r"(внезапно,?\s|камера переключается|надо сказать|в заключение,?\s)",
    }

    _META_PATTERNS: dict[str, str] = {
        "zh": r"(作为旁白|这段文案|这段解说|下面这段|润色后|改写后)",
        "en": r"(as a narrator|this narration|rewritten version|polished version)",
    }

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

    def _get_language(self, language: Optional[str] = None) -> str:
        return language or self.language

    def _is_cjk(self, language: Optional[str] = None) -> bool:
        return self._get_language(language) in self._CJK_LANGUAGES

    def _estimate_duration(self, text: str, language: Optional[str] = None) -> float:
        if not text:
            return 0.0
        if self._is_cjk(language):
            return max(len(re.sub(r"\s+", "", text)) / 4.1, 0.1)
        return max(len(text.split()) / 2.6, 0.1)

    def _target_unit_count(self, target_duration: float, language: Optional[str] = None) -> int:
        if self._is_cjk(language):
            return max(1, int(round(target_duration * 4.1)))
        return max(1, int(round(target_duration * 2.6)))

    def _duration_ratio(self, text: str, target_duration: Optional[float], language: Optional[str] = None) -> float:
        if not target_duration or target_duration <= 0:
            return 1.0
        return self._estimate_duration(text, language) / max(target_duration, 0.1)

    def _local_de_ai_cleanup(self, text: str, language: Optional[str] = None) -> str:
        lang = self._get_language(language)
        phrases = self._DE_AI_PHRASES.get(lang, self._DE_AI_PHRASES.get("en", []))
        cleaned = text.strip()
        for phrase in phrases:
            cleaned = cleaned.replace(phrase, "")
        cleaned = re.sub(r"([，。！？!?；;,.])\1+", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _self_review_cleanup(self, text: str, language: Optional[str] = None) -> str:
        lang = self._get_language(language)
        pattern = self._REVIEW_PATTERNS.get(lang, self._REVIEW_PATTERNS.get("en", ""))
        meta_pattern = self._META_PATTERNS.get(lang, self._META_PATTERNS.get("en", ""))
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE) if pattern else text
        cleaned = re.sub(meta_pattern, "", cleaned, flags=re.IGNORECASE) if meta_pattern else cleaned
        cleaned = re.sub(r"([，。！？!?；;,.])\1+", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_result(self, text: str) -> str:
        cleaned = (text or "").strip().strip('"').strip("'").strip()
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        cleaned = re.sub(r"^\s*(输出|结果|rewrite|rewritten)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _split_units(self, text: str, language: Optional[str] = None) -> list[str]:
        if not text:
            return []
        if self._is_cjk(language):
            return list(re.sub(r"\s+", "", text))
        return [token for token in text.split() if token]

    def _join_units(self, units: list[str], language: Optional[str] = None) -> str:
        return "".join(units) if self._is_cjk(language) else " ".join(units)

    def _trim_fragment(self, text: str, max_units: int, language: Optional[str] = None) -> str:
        units = self._split_units(text, language)
        if len(units) <= max_units:
            return text.strip()
        trimmed = self._join_units(units[:max_units], language).strip()
        return trimmed.rstrip("，,、：:；; ")

    def _compress_to_duration(self, text: str, target_duration: float, language: Optional[str] = None) -> str:
        if not text:
            return text
        target_units = self._target_unit_count(target_duration, language)
        sentences = [part.strip() for part in self._SENTENCE_SPLIT_REGEX.split(text) if part.strip()]
        if not sentences:
            return self._trim_fragment(text, target_units, language)

        kept: list[str] = []
        current_units = 0
        for sentence in sentences:
            sentence_units = len(self._split_units(sentence, language))
            if current_units + sentence_units <= target_units:
                kept.append(sentence)
                current_units += sentence_units
                continue
            remaining = target_units - current_units
            if remaining <= 0:
                break
            clauses = [part.strip() for part in self._CLAUSE_SPLIT_REGEX.split(sentence) if part.strip()]
            if clauses:
                fragment_parts: list[str] = []
                fragment_units = 0
                for clause in clauses:
                    clause_units = len(self._split_units(clause, language))
                    if fragment_units + clause_units <= remaining:
                        fragment_parts.append(clause)
                        fragment_units += clause_units
                        continue
                    tail_budget = remaining - fragment_units
                    if tail_budget > 0:
                        fragment_parts.append(self._trim_fragment(clause, tail_budget, language))
                    break
                fragment = "".join(fragment_parts) if self._is_cjk(language) else " ".join(fragment_parts)
                if fragment.strip():
                    kept.append(fragment.strip())
            else:
                kept.append(self._trim_fragment(sentence, remaining, language))
            break

        compressed = ("".join(kept) if self._is_cjk(language) else " ".join(kept)).strip()
        return compressed or self._trim_fragment(text, target_units, language)

    def _choose_best_candidate(
        self,
        candidates: list[str],
        original: str,
        target_duration: Optional[float],
        language: Optional[str] = None,
    ) -> str:
        if not candidates:
            return original

        original_len = max(len(original.strip()), 1)
        best_text = original
        best_score = float("-inf")

        for candidate in candidates:
            text = self._normalize_result(candidate)
            if self.enable_self_review:
                text = self._self_review_cleanup(text, language)
            if not text:
                continue

            ratio = self._duration_ratio(text, target_duration, language)
            length_ratio = len(text) / original_len
            score = 0.0

            if target_duration and target_duration > 0:
                score -= abs(math.log(max(ratio, 0.01)))
                if 0.78 <= ratio <= 1.18:
                    score += 0.7
                elif 0.65 <= ratio <= 1.35:
                    score += 0.2
                else:
                    score -= 0.8

            if 0.55 <= length_ratio <= 1.45:
                score += 0.3
            else:
                score -= abs(length_ratio - 1.0)

            meta_pattern = self._META_PATTERNS.get(self._get_language(language), "")
            if meta_pattern and re.search(meta_pattern, text, flags=re.IGNORECASE):
                score -= 1.0

            if score > best_score:
                best_score = score
                best_text = text

        return best_text or original

    def _fit_duration(self, text: str, target_duration: Optional[float], language: Optional[str] = None) -> str:
        if target_duration is None or target_duration <= 0 or not text:
            return text

        ratio = self._duration_ratio(text, target_duration, language)
        if ratio <= 1.18:
            return text
        return self._compress_to_duration(text, target_duration, language)

    def _render_template(self, text: str) -> str:
        if "{text}" in self.template:
            return self.template.format(text=text)
        rendered = self.template.replace("Original:\n", f"Original:\n{text}\n")
        if rendered == self.template:
            rendered = f"{self.template.rstrip()}\n\nOriginal:\n{text}\n"
        return rendered

    def _build_general_prompt(self, text: str, style: str, language: str) -> str:
        base = self._render_template(text)
        return (
            f"Language: {language}\n"
            f"Style: {style}\n"
            "Constraints: keep facts, keep names, do not add information, avoid AI-like summary phrases, "
            "avoid meta commentary, output only the final narration.\n\n"
            f"{base}"
        )

    def _build_segment_prompt(
        self,
        text: str,
        target_duration: Optional[float],
        prev_text: str,
        next_text: str,
        match_reason: str,
        style_preset: str,
        style_hint: str,
        language: str,
    ) -> str:
        lang_name = self.SUPPORTED_LANGUAGES.get(language, language.upper())
        if target_duration and target_duration > 0:
            if self._is_cjk(language):
                unit_label = "characters"
            else:
                unit_label = "words"
            target_units = self._target_unit_count(target_duration, language)
            duration_line = f"Target spoken duration: about {target_duration:.1f} seconds ({target_units} {unit_label} max, close to original length)."
        else:
            duration_line = "Target spoken duration: unknown, keep close to the original length."

        return (
            f"You are rewriting one segment of a movie narration script in {lang_name}.\n"
            f"Style preset: {style_preset}. {style_hint}\n"
            f"{duration_line}\n"
            f"Previous segment: {prev_text or '[none]'}\n"
            f"Next segment: {next_text or '[none]'}\n"
            f"Match note: {match_reason or '[none]'}\n"
            "Rules:\n"
            "1. Keep names, places, events, and facts exactly consistent with the original.\n"
            "2. Do not add plot details, interpretation, or moral commentary.\n"
            "3. Avoid AI-like fillers, staged transitions, topic-summary structure, and phrases like 'we can see'.\n"
            "4. Make it sound like one natural spoken line for voice-over, not an essay.\n"
            f"5. Output ONLY in {lang_name}. Keep the pacing tight and suitable for dubbing.\n"
            "6. Prefer concrete, direct wording over decorative adjectives.\n\n"
            f"Original text:\n{text}\n\n"
            "Output only the rewritten segment."
        )

    def _build_retry_prompt(self, text: str, previous_output: str, target_duration: Optional[float], language: str) -> str:
        ratio = self._duration_ratio(previous_output, target_duration, language)
        if ratio > 1.18:
            duration_rule = "The previous output was too long. Rewrite it shorter and tighter."
        elif ratio < 0.72:
            duration_rule = "The previous output was too short. Restore the missing core information while staying concise."
        else:
            duration_rule = "The previous output was acceptable in length but still needs cleaner spoken delivery."

        return (
            f"Rewrite the narration again in {self.SUPPORTED_LANGUAGES.get(language, language.upper())}.\n"
            f"{duration_rule}\n"
            "Keep facts and names unchanged. Do not add new details. Remove any meta commentary. "
            "Output only the final narration.\n\n"
            f"Original:\n{text}\n\n"
            f"Previous output:\n{previous_output}"
        )

    async def polish(self, text: str, style: Optional[str] = None, language: Optional[str] = None) -> str:
        if not text or not text.strip():
            return text

        target_lang = self._get_language(language)
        working = self._local_de_ai_cleanup(text, target_lang) if self.enable_de_ai_pass else text.strip()
        style_text = style or self.STYLE_PRESETS.get(self.default_style_preset, self.STYLE_PRESETS["movie_pro"])
        prompt = self._build_general_prompt(working, style_text, target_lang)

        candidates = [working, text.strip()]
        try:
            if self._has_remote_model():
                result = await self._get_api_manager().chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                candidates.append(result)
            return self._choose_best_candidate(candidates, working, target_duration=None, language=target_lang)
        except Exception as exc:
            logger.warning(f"Polish failed, using local cleanup: {exc}")
            return self._self_review_cleanup(working, target_lang)

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
        language = self.language
        working = self._local_de_ai_cleanup(text, language) if self.enable_de_ai_pass else text.strip()

        if not self._has_remote_model():
            result = self._choose_best_candidate([working, text.strip()], working, target_duration, language)
            return self._fit_duration(result, target_duration, language)

        prompt = self._build_segment_prompt(
            text=working,
            target_duration=target_duration,
            prev_text=prev_text,
            next_text=next_text,
            match_reason=match_reason,
            style_preset=preset,
            style_hint=style_hint,
            language=language,
        )

        candidates = [working, text.strip()]
        try:
            result = await self._get_api_manager().chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            candidates.append(result)

            normalized = self._normalize_result(result)
            duration_ratio = self._duration_ratio(normalized, target_duration, language)
            if normalized and (duration_ratio > 1.18 or duration_ratio < 0.72):
                retry_prompt = self._build_retry_prompt(working, normalized, target_duration, language)
                retry_result = await self._get_api_manager().chat(
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=min(self.temperature, 0.25),
                    max_tokens=self.max_tokens,
                )
                candidates.append(retry_result)
        except Exception as exc:
            logger.warning(f"Contextual rewrite failed, using fallback: {exc}")

        best = self._choose_best_candidate(candidates, working, target_duration, language)
        return self._fit_duration(best or working, target_duration, language)

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
            "Keep facts, keep names, do not add information, avoid meta commentary.\n\n"
            f"Original:\n{text}\n\nOutput only the translated narration."
        )
        candidates = [text.strip()]
        try:
            if self._has_remote_model():
                result = await self._get_api_manager().chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                candidates.append(result)
                return self._choose_best_candidate(candidates, text.strip(), target_duration=None, language=target_language)
        except Exception as exc:
            logger.warning(f"Translate and polish failed: {exc}")
        return self._self_review_cleanup(text, target_language)

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
