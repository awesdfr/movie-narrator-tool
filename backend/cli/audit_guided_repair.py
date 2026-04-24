"""Repair only the timeline segments proven bad by a visual audit report."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from api.routes.process_v2 import (
    _apply_selected_candidate,
    _calculate_expected_movie_time,
    _candidate_from_result,
    _collect_candidates_for_segment,
    _compute_neighbor_hint_for_segment,
    _dedupe_candidates,
    _infer_sequence_mode_from_segments,
    _segment_duration,
)
from api.routes.settings import load_settings
from core.video_processor.frame_matcher import FrameMatcher
from models.segment import AlignmentStatus, MatchCandidate
from models.project import Project

TIME_SCALE = 1_000_000


@dataclass
class RepairResult:
    segment_id: str
    before: float
    after: float
    changed: bool


def _load_project(path: Path) -> Project:
    return Project.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _save_project(path: Path, project: Project) -> None:
    project.updated_at = datetime.now()
    path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _usable_movie_segments(project: Project) -> list:
    return [
        segment
        for segment in project.segments
        if segment.use_segment
        and segment.segment_type.value != "non_movie"
        and segment.movie_start is not None
        and segment.movie_end is not None
        and segment.movie_end > segment.movie_start
        and segment.narration_end > segment.narration_start
    ]


def _audit_target_indices(report: dict[str, Any], max_segments: int) -> list[int]:
    scores: dict[int, float] = {}
    for group in report.get("low_groups", []):
        index = group.get("worst_segment_index")
        if index is None:
            continue
        score = float(group.get("min_score", group.get("average_score", 0.0)) or 0.0)
        scores[int(index)] = min(score, scores.get(int(index), 1.0))
    for sample in report.get("worst_samples", []):
        index = sample.get("segment_index")
        if index is None:
            continue
        score = float(sample.get("score", 0.0) or 0.0)
        scores[int(index)] = min(score, scores.get(int(index), 1.0))
    return [index for index, _ in sorted(scores.items(), key=lambda item: item[1])[:max_segments]]


def _read_gray(cap: cv2.VideoCapture, timestamp: float, width: int = 192, crop_ratio: float = 0.78):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    height, frame_width = frame.shape[:2]
    if height <= 0 or frame_width <= 0:
        return None
    frame = frame[: max(1, int(height * crop_ratio)), :]
    if frame_width != width:
        resized_height = max(24, int(frame.shape[0] * width / max(1, frame_width)))
        frame = cv2.resize(frame, (width, resized_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _score_pair(query, movie) -> float:
    if query is None or movie is None:
        return 0.0
    if query.shape != movie.shape:
        movie = cv2.resize(movie, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)
    query_f = query.astype(np.float32)
    movie_f = movie.astype(np.float32)
    diff_score = 1.0 - float(np.mean(np.abs(query_f - movie_f))) / 255.0
    query_edges = cv2.Canny(query, 60, 140)
    movie_edges = cv2.Canny(movie, 60, 140)
    edge_score = 1.0 - float(np.mean(np.abs(query_edges.astype(np.float32) - movie_edges.astype(np.float32)))) / 255.0
    q_hist = cv2.calcHist([query], [0], None, [32], [0, 256]).astype("float32")
    m_hist = cv2.calcHist([movie], [0], None, [32], [0, 256]).astype("float32")
    q_hist /= max(float(q_hist.sum()), 1.0)
    m_hist /= max(float(m_hist.sum()), 1.0)
    hist_score = float((cv2.compareHist(q_hist, m_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
    return max(0.0, min(1.0, diff_score * 0.42 + edge_score * 0.34 + hist_score * 0.24))


def _sample_times(segment) -> list[float]:
    duration = _segment_duration(segment)
    if duration < 0.75:
        positions = (0.50,)
    elif duration < 1.75:
        positions = (0.35, 0.70)
    else:
        positions = (0.25, 0.55, 0.82)
    return [float(segment.narration_start) + duration * pos for pos in positions]


def _score_candidate_frames(project: Project, segment, candidate: MatchCandidate) -> float:
    duration = _segment_duration(segment)
    source_duration = max(0.05, float(candidate.end) - float(candidate.start))
    narr_cap = cv2.VideoCapture(str(project.narration_path))
    movie_cap = cv2.VideoCapture(str(project.movie_path))
    try:
        scores: list[float] = []
        for query_time in _sample_times(segment):
            offset = (query_time - float(segment.narration_start)) / max(duration, 0.05)
            movie_time = float(candidate.start) + source_duration * offset
            scores.append(_score_pair(_read_gray(narr_cap, query_time), _read_gray(movie_cap, movie_time)))
        if not scores:
            return 0.0
        return float(np.mean(scores) * 0.70 + min(scores) * 0.30)
    finally:
        narr_cap.release()
        movie_cap.release()


def _current_candidate(segment) -> MatchCandidate | None:
    if segment.movie_start is None or segment.movie_end is None:
        return None
    return MatchCandidate(
        id=f"{segment.id}_current",
        start=float(segment.movie_start),
        end=float(segment.movie_end),
        score=float(segment.match_confidence or 0.0),
        confidence=float(segment.match_confidence or 0.0),
        visual_confidence=float(segment.visual_confidence or segment.match_confidence or 0.0),
        audio_confidence=float(segment.audio_confidence or 0.0),
        temporal_confidence=float(segment.temporal_confidence or 0.0),
        stability_score=float(segment.stability_score or 0.0),
        duration_gap=float(segment.duration_gap or 0.0),
        reason="Current audit baseline",
        source="audit_current",
        rank=1,
    )


async def repair_project(project_path: Path, report_path: Path, max_segments: int, min_gain: float, dry_run: bool) -> dict[str, Any]:
    project = _load_project(project_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    usable = _usable_movie_segments(project)
    target_indices = _audit_target_indices(report, max_segments=max_segments)
    targets = [usable[index] for index in target_indices if 0 <= index < len(usable)]

    app_settings = load_settings()
    frame_matcher = FrameMatcher(
        phash_threshold=app_settings.match.phash_threshold,
        match_threshold=app_settings.match.frame_match_threshold,
        use_deep_learning=app_settings.match.use_deep_learning,
        index_sample_fps=app_settings.match.index_sample_fps,
        fast_mode=app_settings.match.fast_mode,
        subtitle_mask_mode=getattr(project, "subtitle_mask_mode", "hybrid"),
        movie_subtitle_regions=[region.model_dump() for region in getattr(project, "movie_subtitle_regions", [])],
        narration_subtitle_regions=[region.model_dump() for region in getattr(project, "narration_subtitle_regions", [])],
    )
    cache_dir = Path(__file__).resolve().parents[1] / "temp" / "match_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_cache_path = cache_dir / f"{Path(project.movie_path).stem}_frame.pkl"
    await frame_matcher.build_index(project.movie_path, sample_interval=app_settings.match.sample_interval, cache_path=frame_cache_path)

    prefer_sequential, _ = _infer_sequence_mode_from_segments(project.segments, app_settings.match.allow_non_sequential)
    results: list[RepairResult] = []
    for segment in targets:
        idx = project.segments.index(segment)
        neighbor_hint = _compute_neighbor_hint_for_segment(project.segments, idx, segment)
        precomputed = await frame_matcher._extract_segment_features(project.narration_path, segment.narration_start, segment.narration_end)
        candidates = await _collect_candidates_for_segment(
            frame_matcher,
            project,
            segment,
            app_settings,
            None,
            neighbor_hint=neighbor_hint,
            audio_scorer=None,
            precomputed_features=precomputed,
            prefer_sequential=prefer_sequential,
            aggressive=True,
            extra_hints=[float(segment.movie_start)] if segment.movie_start is not None else None,
        )
        current = _current_candidate(segment)
        if current is not None:
            candidates = _dedupe_candidates([current] + candidates, tolerance=0.18)
        if not candidates:
            continue

        scored = [(candidate, _score_candidate_frames(project, segment, candidate)) for candidate in candidates[:8]]
        current_score = _score_candidate_frames(project, segment, current) if current is not None else 0.0
        best_candidate, best_score = max(scored, key=lambda item: item[1])
        changed = best_candidate.source != "audit_current" and best_score >= current_score + min_gain
        if changed and not dry_run:
            best_candidate.confidence = max(float(best_candidate.confidence or 0.0), min(0.94, best_score))
            best_candidate.score = best_candidate.confidence
            best_candidate.visual_confidence = max(float(best_candidate.visual_confidence or 0.0), min(0.94, best_score))
            best_candidate.reason = f"Audit-guided repair; audit_score={best_score:.3f}, before={current_score:.3f}"
            _apply_selected_candidate(segment, best_candidate, AlignmentStatus.REMATCHED, best_score < 0.82)
            segment.is_manual_match = False
        results.append(RepairResult(segment.id, current_score, best_score, changed))

    changed_count = sum(1 for item in results if item.changed)
    if changed_count and not dry_run:
        backup = project_path.with_name(f"{project_path.stem}.before_audit_guided_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(project_path, backup)
        _save_project(project_path, project)
    return {
        "targets": len(targets),
        "changed": changed_count,
        "results": [item.__dict__ for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair project segments selected from visual audit failures.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-segments", type=int, default=24)
    parser.add_argument("--min-gain", type=float, default=0.035)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(repair_project(args.project, args.report, args.max_segments, args.min_gain, args.dry_run))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
