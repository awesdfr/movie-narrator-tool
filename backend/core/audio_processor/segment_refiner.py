"""Narration segmentation refinement helpers."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from models.segment import SegmentType


_SENTENCE_ENDINGS = set("。！？!?；;…")
_WORD_SPLIT_RE = re.compile(r"([。！？!?；;…])")


@dataclass
class SegmenterConfig:
    min_segment_duration: float = 1.2
    max_segment_duration: float = 8.0
    split_pause_seconds: float = 0.55
    merge_gap_seconds: float = 0.35
    sentence_snap_tolerance: float = 0.4
    enable_scene_snap: bool = True
    prefer_word_timestamps: bool = True


class SegmentRefiner:
    """Split raw ASR chunks into edit-ready narration segments."""

    def __init__(self, config: Optional[SegmenterConfig] = None):
        self.config = config or SegmenterConfig()

    def refine(self, raw_segments: list[dict], scenes: Optional[list[dict]] = None) -> list[dict]:
        """Refine ASR or subtitle chunks into stable segments."""

        if not raw_segments:
            return []

        scene_boundaries = self._scene_boundaries(scenes)
        refined: list[dict] = []

        for raw in raw_segments:
            refined.extend(self._split_raw_segment(raw))

        refined = self._merge_short_segments(refined)
        refined = self._snap_to_scenes(refined, scene_boundaries)
        refined = self._force_max_duration(refined)

        cleaned: list[dict] = []
        for item in refined:
            text = (item.get("text") or "").strip()
            if not text and item.get("type") not in {SegmentType.NON_MOVIE, SegmentType.NO_NARRATION}:
                continue
            if item["end"] <= item["start"]:
                continue
            cleaned.append(item)

        logger.info("Segment refinement complete: {} -> {}".format(len(raw_segments), len(cleaned)))
        return cleaned

    def _split_raw_segment(self, raw: dict) -> list[dict]:
        words = raw.get("words") or []
        if words and self.config.prefer_word_timestamps:
            segments = self._split_with_words(raw, words)
            if segments:
                return segments
        return self._split_without_words(raw)

    def _split_with_words(self, raw: dict, words: list[dict]) -> list[dict]:
        pieces: list[dict] = []
        current_words: list[dict] = []

        for idx, word in enumerate(words):
            if word.get("start") is None or word.get("end") is None:
                continue
            current_words.append(word)
            next_word = words[idx + 1] if idx + 1 < len(words) else None
            should_flush = False

            text = str(word.get("word", "")).strip()
            piece_start = current_words[0]["start"]
            piece_end = current_words[-1]["end"]
            piece_duration = piece_end - piece_start

            if text and text[-1] in _SENTENCE_ENDINGS and piece_duration >= self.config.min_segment_duration:
                should_flush = True
            elif next_word is not None:
                gap = max(0.0, next_word.get("start", piece_end) - piece_end)
                if gap >= self.config.split_pause_seconds and piece_duration >= self.config.min_segment_duration:
                    should_flush = True
                elif piece_duration >= self.config.max_segment_duration:
                    should_flush = True
            else:
                should_flush = True

            if should_flush:
                pieces.append(self._build_piece(raw, current_words))
                current_words = []

        if current_words:
            pieces.append(self._build_piece(raw, current_words))

        return pieces

    def _build_piece(self, raw: dict, words: list[dict]) -> dict:
        text = "".join(str(word.get("word", "")) for word in words).strip()
        return {
            "start": float(words[0]["start"]),
            "end": float(words[-1]["end"]),
            "text": text or raw.get("text", "").strip(),
            "type": raw.get("type", SegmentType.HAS_NARRATION),
            "words": words,
            "voiceprint_similarity": raw.get("voiceprint_similarity"),
            "audio_activity_label": raw.get("audio_activity_label"),
            "speech_likelihood": raw.get("speech_likelihood"),
            "rms_db": raw.get("rms_db"),
        }

    def _split_without_words(self, raw: dict) -> list[dict]:
        text = (raw.get("text") or "").strip()
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", start))
        duration = max(0.0, end - start)
        if duration <= 0:
            return []

        pieces_text = self._split_text(text) if text else [""]
        pieces_text = [piece for piece in pieces_text if piece is not None]
        if not pieces_text:
            pieces_text = [text]

        if len(pieces_text) == 1 and duration <= self.config.max_segment_duration:
            return [
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "type": raw.get("type", SegmentType.HAS_NARRATION),
                    "words": raw.get("words") or [],
                    "voiceprint_similarity": raw.get("voiceprint_similarity"),
                    "audio_activity_label": raw.get("audio_activity_label"),
                    "speech_likelihood": raw.get("speech_likelihood"),
                    "rms_db": raw.get("rms_db"),
                }
            ]

        weights = [max(1, len(piece.strip())) for piece in pieces_text]
        total_weight = sum(weights) or len(pieces_text)
        cursor = start
        pieces: list[dict] = []

        for idx, piece in enumerate(pieces_text):
            if idx == len(pieces_text) - 1:
                piece_end = end
            else:
                piece_duration = duration * (weights[idx] / total_weight)
                piece_end = min(end, cursor + piece_duration)
            pieces.append(
                {
                    "start": cursor,
                    "end": piece_end,
                    "text": piece.strip(),
                    "type": raw.get("type", SegmentType.HAS_NARRATION),
                    "words": [],
                    "voiceprint_similarity": raw.get("voiceprint_similarity"),
                    "audio_activity_label": raw.get("audio_activity_label"),
                    "speech_likelihood": raw.get("speech_likelihood"),
                    "rms_db": raw.get("rms_db"),
                }
            )
            cursor = piece_end

        return pieces

    def _split_text(self, text: str) -> list[str]:
        if not text:
            return [""]
        fragments = _WORD_SPLIT_RE.split(text)
        pieces: list[str] = []
        current = ""
        for frag in fragments:
            if not frag:
                continue
            current += frag
            if frag in _SENTENCE_ENDINGS:
                pieces.append(current)
                current = ""
        if current:
            pieces.append(current)
        return pieces or [text]

    def _merge_short_segments(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return []

        merged: list[dict] = []
        for segment in segments:
            if not merged:
                merged.append(segment)
                continue

            prev = merged[-1]
            prev_duration = prev["end"] - prev["start"]
            curr_duration = segment["end"] - segment["start"]
            gap = max(0.0, segment["start"] - prev["end"])
            same_type = prev.get("type") == segment.get("type")

            if (
                same_type
                and gap <= self.config.merge_gap_seconds
                and (prev_duration < self.config.min_segment_duration or curr_duration < self.config.min_segment_duration)
                and (segment["end"] - prev["start"]) <= self.config.max_segment_duration
            ):
                prev["end"] = segment["end"]
                prev["text"] = (prev.get("text", "") + segment.get("text", "")).strip()
                prev["words"] = (prev.get("words") or []) + (segment.get("words") or [])
                prev["speech_likelihood"] = max(
                    float(prev.get("speech_likelihood") or 0.0),
                    float(segment.get("speech_likelihood") or 0.0),
                )
                prev["voiceprint_similarity"] = max(
                    float(prev.get("voiceprint_similarity") or 0.0),
                    float(segment.get("voiceprint_similarity") or 0.0),
                )
                prev_rms = float(prev.get("rms_db") or -80.0)
                segment_rms = float(segment.get("rms_db") or -80.0)
                prev["rms_db"] = max(prev_rms, segment_rms)
                prev["audio_activity_label"] = self._stronger_activity_label(
                    prev.get("audio_activity_label"),
                    segment.get("audio_activity_label"),
                )
            else:
                merged.append(segment)
        return merged

    def _stronger_activity_label(self, a: Optional[str], b: Optional[str]) -> str:
        order = {"silent": 0, "weak": 1, "active": 2}
        a_score = order.get(a or "", -1)
        b_score = order.get(b or "", -1)
        return a if a_score >= b_score else (b or a or "unknown")

    def _snap_to_scenes(self, segments: list[dict], boundaries: list[float]) -> list[dict]:
        if not segments or not boundaries or not self.config.enable_scene_snap:
            return segments

        snapped: list[dict] = []
        for segment in segments:
            start = self._nearest_boundary(segment["start"], boundaries)
            end = self._nearest_boundary(segment["end"], boundaries)
            if abs(start - segment["start"]) <= self.config.sentence_snap_tolerance:
                segment["start"] = start
            if abs(end - segment["end"]) <= self.config.sentence_snap_tolerance:
                segment["end"] = max(segment["start"], end)
            snapped.append(segment)
        return snapped

    def _force_max_duration(self, segments: list[dict]) -> list[dict]:
        capped: list[dict] = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration <= self.config.max_segment_duration:
                capped.append(segment)
                continue

            text_pieces = self._split_text(segment.get("text", ""))
            chunk_count = max(2, math.ceil(duration / self.config.max_segment_duration))
            if len(text_pieces) < chunk_count:
                text_pieces = self._split_evenly(segment.get("text", ""), chunk_count)
            weight_total = sum(max(1, len(piece.strip())) for piece in text_pieces) or len(text_pieces)
            cursor = segment["start"]

            for idx, piece in enumerate(text_pieces):
                if idx == len(text_pieces) - 1:
                    piece_end = segment["end"]
                else:
                    piece_duration = duration * (max(1, len(piece.strip())) / weight_total)
                    piece_end = min(segment["end"], cursor + piece_duration)
                capped.append(
                    {
                        **segment,
                        "start": cursor,
                        "end": piece_end,
                        "text": piece.strip(),
                        "words": [],
                    }
                )
                cursor = piece_end
        return capped

    def _split_evenly(self, text: str, chunks: int) -> list[str]:
        if chunks <= 1:
            return [text]
        if not text:
            return [""] * chunks
        step = max(1, math.ceil(len(text) / chunks))
        return [text[i : i + step] for i in range(0, len(text), step)]

    def _scene_boundaries(self, scenes: Optional[list[dict]]) -> list[float]:
        if not scenes:
            return []
        boundaries = set()
        for scene in scenes:
            start = scene.get("start") if isinstance(scene, dict) else None
            end = scene.get("end") if isinstance(scene, dict) else None
            if start is not None:
                boundaries.add(float(start))
            if end is not None:
                boundaries.add(float(end))
        return sorted(boundaries)

    def _nearest_boundary(self, value: float, boundaries: list[float]) -> float:
        return min(boundaries, key=lambda boundary: abs(boundary - value))
