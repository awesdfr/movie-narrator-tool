"""Repair source jumps that happen without a narration visual cut.

This is a conservative playback-smoothness pass. It never performs global
retrieval; it only tests whether the current segment should continue from the
previous movie source when the narration picture itself did not cut.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from models.project import Project
from models.segment import AlignmentStatus, MatchCandidate, Segment, SegmentType


@dataclass
class Repair:
    segment_id: str
    narration_start: float
    old_start: float
    old_end: float
    new_start: float
    new_end: float
    source_jump: float
    current_score: float
    candidate_score: float
    changed: bool


def _load_project(path: Path) -> Project:
    return Project.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _save_project(path: Path, project: Project) -> None:
    project.updated_at = datetime.now()
    path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _usable_segments(project: Project) -> list[Segment]:
    return [
        segment
        for segment in sorted(project.segments, key=lambda item: (float(item.narration_start), float(item.narration_end)))
        if segment.use_segment
        and not segment.skip_matching
        and segment.segment_type != SegmentType.NON_MOVIE
        and segment.movie_start is not None
        and segment.movie_end is not None
        and segment.movie_end > segment.movie_start
        and segment.narration_end > segment.narration_start
    ]


def _duration(segment: Segment) -> float:
    return max(0.05, float(segment.narration_end) - float(segment.narration_start))


def _is_visual_cut(scorer: "FrameScorer", previous: Segment, current: Segment) -> bool:
    gap = float(current.narration_start) - float(previous.narration_end)
    if gap > 0.45:
        return True
    boundary = max(float(current.narration_start), float(previous.narration_end))
    left_time = max(float(previous.narration_start) + 0.04, min(float(previous.narration_end) - 0.08, boundary - 0.08))
    right_time = min(float(current.narration_end) - 0.04, max(float(current.narration_start) + 0.08, boundary + 0.08))
    left = scorer._read_gray("n", left_time)
    right = scorer._read_gray("n", right_time)
    if left is None or right is None:
        return True
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    left_hist = cv2.calcHist([left], [0], None, [40], [0, 256]).astype("float32")
    right_hist = cv2.calcHist([right], [0], None, [40], [0, 256]).astype("float32")
    left_hist /= max(float(left_hist.sum()), 1.0)
    right_hist /= max(float(right_hist.sum()), 1.0)
    hist_corr = (float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
    mean_abs = float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))) / 255.0
    edge_diff = float(
        np.mean(np.abs(cv2.Canny(left, 70, 150).astype(np.float32) - cv2.Canny(right, 70, 150).astype(np.float32)))
    ) / 255.0
    return bool(
        (hist_corr < 0.56 and mean_abs > 0.080)
        or mean_abs > 0.230
        or (edge_diff > 0.260 and mean_abs > 0.090)
        or (edge_diff > 0.190 and mean_abs > 0.125)
    )


class FrameScorer:
    def __init__(self, narration_path: str, movie_path: str, width: int = 192, crop_ratio: float = 0.78):
        self.narration = cv2.VideoCapture(str(narration_path))
        self.movie = cv2.VideoCapture(str(movie_path))
        self.width = int(width)
        self.crop_ratio = float(crop_ratio)
        self.cache: dict[tuple[str, float], np.ndarray] = {}

    def close(self) -> None:
        self.narration.release()
        self.movie.release()

    def _read_gray(self, role: str, timestamp: float) -> np.ndarray | None:
        key = (role, round(max(0.0, float(timestamp)), 2))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        capture = self.narration if role == "n" else self.movie
        capture.set(cv2.CAP_PROP_POS_MSEC, key[1] * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        height, frame_width = frame.shape[:2]
        if height <= 0 or frame_width <= 0:
            return None
        frame = frame[: max(1, int(height * self.crop_ratio)), int(frame_width * 0.04) : int(frame_width * 0.96)]
        resized_height = max(32, int(frame.shape[0] * self.width / max(1, frame.shape[1])))
        frame = cv2.resize(frame, (self.width, resized_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        self.cache[key] = gray
        if len(self.cache) > 1400:
            self.cache.pop(next(iter(self.cache)))
        return gray

    @staticmethod
    def _center_crop(frame: np.ndarray, ratio: float) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_h = max(8, int(height * ratio))
        crop_w = max(8, int(width * ratio))
        y = max(0, (height - crop_h) // 2)
        x = max(0, (width - crop_w) // 2)
        return frame[y : y + crop_h, x : x + crop_w]

    def _score_pair(self, query: np.ndarray | None, movie: np.ndarray | None) -> float:
        if query is None or movie is None:
            return 0.0
        if query.shape != movie.shape:
            movie = cv2.resize(movie, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)

        def score_one(left: np.ndarray, right: np.ndarray) -> float:
            if left.shape != right.shape:
                right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            left_edges = cv2.Canny(left, 70, 150)
            right_edges = cv2.Canny(right, 70, 150)
            edge = 1.0 - float(np.mean(np.abs(left_edges.astype(np.float32) - right_edges.astype(np.float32)))) / 255.0
            left_hist = cv2.calcHist([left], [0], None, [48], [0, 256]).astype("float32")
            right_hist = cv2.calcHist([right], [0], None, [48], [0, 256]).astype("float32")
            left_hist /= max(float(left_hist.sum()), 1.0)
            right_hist /= max(float(right_hist.sum()), 1.0)
            hist = (float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
            return float(max(0.0, min(1.0, edge * 0.58 + hist * 0.42)))

        return max(
            score_one(query, movie),
            score_one(self._center_crop(query, 0.82), self._center_crop(movie, 0.82)),
            score_one(self._center_crop(query, 0.68), self._center_crop(movie, 0.68)),
        )

    @staticmethod
    def _sample_positions(duration: float) -> tuple[float, ...]:
        if duration < 0.65:
            return (0.50,)
        if duration < 1.35:
            return (0.35, 0.72)
        return (0.22, 0.52, 0.82)

    def score_range(self, segment: Segment, source_start: float, source_end: float) -> float:
        duration = _duration(segment)
        source_duration = max(0.05, float(source_end) - float(source_start))
        scores: list[float] = []
        for position in self._sample_positions(duration):
            query_time = float(segment.narration_start) + duration * position
            movie_time = float(source_start) + source_duration * position
            scores.append(self._score_pair(self._read_gray("n", query_time), self._read_gray("m", movie_time)))
        if not scores:
            return 0.0
        return float(np.mean(scores) * 0.72 + min(scores) * 0.28)


def _should_accept(
    *,
    source_jump: float,
    current_score: float,
    candidate_score: float,
    candidate_shift: float,
    min_score: float,
    min_gain: float,
) -> bool:
    if candidate_score < min_score:
        return False
    if candidate_score >= current_score + min_gain:
        return True
    jump_is_bad = abs(source_jump) >= 1.20 and candidate_shift >= 0.35
    if jump_is_bad and candidate_score >= max(min_score + 0.03, current_score - 0.025):
        return True
    return False


def repair_project(
    project_path: Path,
    *,
    dry_run: bool,
    max_segments: int,
    max_time: float | None,
    jump_threshold: float,
    min_score: float,
    min_gain: float,
) -> dict[str, Any]:
    project = _load_project(project_path)
    usable = _usable_segments(project)
    scorer = FrameScorer(str(project.narration_path), str(project.movie_path))
    repairs: list[Repair] = []

    try:
        previous: Segment | None = None
        accepted_count = 0
        for segment in usable:
            if max_time is not None and float(segment.narration_start) >= max_time:
                break
            if previous is None:
                previous = segment
                continue
            if _is_visual_cut(scorer, previous, segment):
                previous = segment
                continue
            if previous.movie_end is None or segment.movie_start is None or segment.movie_end is None:
                previous = segment
                continue

            target_gap = max(0.0, float(segment.narration_start) - float(previous.narration_end))
            source_jump = float(segment.movie_start) - float(previous.movie_end)
            if abs(source_jump - target_gap) < max(jump_threshold, _duration(segment) * 0.45):
                previous = segment
                continue

            source_duration = max(0.05, float(segment.movie_end) - float(segment.movie_start))
            candidate_start = max(0.0, float(previous.movie_end) + target_gap)
            candidate_end = candidate_start + source_duration
            movie_duration = float(project.movie_duration or 0.0)
            if movie_duration > 0.0 and candidate_end > movie_duration:
                previous = segment
                continue

            current_score = scorer.score_range(segment, float(segment.movie_start), float(segment.movie_end))
            candidate_score = scorer.score_range(segment, candidate_start, candidate_end)
            candidate_shift = abs(float(segment.movie_start) - candidate_start)
            changed = _should_accept(
                source_jump=source_jump,
                current_score=current_score,
                candidate_score=candidate_score,
                candidate_shift=candidate_shift,
                min_score=min_score,
                min_gain=min_gain,
            )
            repairs.append(
                Repair(
                    segment_id=segment.id,
                    narration_start=float(segment.narration_start),
                    old_start=float(segment.movie_start),
                    old_end=float(segment.movie_end),
                    new_start=float(candidate_start),
                    new_end=float(candidate_end),
                    source_jump=float(source_jump),
                    current_score=float(current_score),
                    candidate_score=float(candidate_score),
                    changed=bool(changed),
                )
            )
            if changed:
                accepted_count += 1
            if changed and not dry_run:
                confidence = min(0.93, max(float(segment.match_confidence or 0.0), candidate_score))
                candidate = MatchCandidate(
                    id=f"{segment.id}_noncut_continuity",
                    start=float(candidate_start),
                    end=float(candidate_end),
                    score=confidence,
                    confidence=confidence,
                    visual_confidence=confidence,
                    audio_confidence=float(segment.audio_confidence or 0.0),
                    temporal_confidence=max(float(segment.temporal_confidence or 0.0), 0.90),
                    verification_score=float(candidate_score),
                    stability_score=max(float(segment.stability_score or 0.0), 0.78),
                    duration_gap=abs(_duration(segment) - source_duration),
                    match_count=max(1, len(FrameScorer._sample_positions(_duration(segment)))),
                    reason=(
                        f"Non-cut continuity repair; score={candidate_score:.3f}, "
                        f"before={current_score:.3f}, jump={source_jump:.2f}s"
                    ),
                    source="noncut_continuity",
                    rank=1,
                )
                segment.match_candidates = [candidate] + [item for item in segment.match_candidates if item.id != candidate.id][:7]
                segment.selected_candidate_id = candidate.id
                segment.movie_start = float(candidate_start)
                segment.movie_end = float(candidate_end)
                segment.match_confidence = confidence
                segment.visual_confidence = confidence
                segment.temporal_confidence = candidate.temporal_confidence
                segment.stability_score = candidate.stability_score
                segment.duration_gap = candidate.duration_gap
                segment.match_reason = candidate.reason
                segment.evidence_summary = f"noncut_continuity={candidate_score:.2f}, before={current_score:.2f}"
                segment.match_type = "exact" if candidate_score >= 0.74 else "inferred"
                segment.review_required = candidate_score < 0.78
                segment.alignment_status = AlignmentStatus.AUTO_ACCEPTED if not segment.review_required else AlignmentStatus.NEEDS_REVIEW
            if changed and accepted_count >= max_segments:
                break
            previous = segment
    finally:
        scorer.close()

    changed_repairs = [item for item in repairs if item.changed]
    result = {
        "dry_run": dry_run,
        "checked": len(repairs),
        "changed": len(changed_repairs),
        "changes": [item.__dict__ for item in changed_repairs[:120]],
    }
    if changed_repairs and not dry_run:
        backup = project_path.with_name(
            f"{project_path.stem}.before_noncut_jump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(project_path, backup)
        _save_project(project_path, project)
        result["backup"] = str(backup)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair non-cut visual source jumps.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-segments", type=int, default=80)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--jump-threshold", type=float, default=0.85)
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--min-gain", type=float, default=0.035)
    args = parser.parse_args()
    print(
        json.dumps(
            repair_project(
                args.project,
                dry_run=args.dry_run,
                max_segments=max(1, int(args.max_segments)),
                max_time=args.max_time if args.max_time and args.max_time > 0 else None,
                jump_threshold=max(0.1, float(args.jump_threshold)),
                min_score=max(0.0, min(1.0, float(args.min_score))),
                min_gain=max(0.0, min(1.0, float(args.min_gain))),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
