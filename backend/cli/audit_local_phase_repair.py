"""Repair audit low-score segments by local source-time phase search."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cli.noncut_jump_repair import FrameScorer, _duration, _is_visual_cut, _load_project, _save_project, _usable_segments
from models.segment import AlignmentStatus, MatchCandidate


@dataclass
class PhaseRepair:
    segment_id: str
    narration_start: float
    old_start: float
    new_start: float
    current_score: float
    best_score: float
    changed: bool


class CachedFrameScorer(FrameScorer):
    """FrameScorer backed by dense ffmpeg-decoded gray frame arrays."""

    def __init__(self, narration_path: str, movie_path: str):
        self.width = 180
        self.crop_ratio = 0.78
        self.cache = {}
        self.series = {
            "n": self._load_series(narration_path, cache_role="narration", step=0.50),
            "m": self._load_series(movie_path, cache_role="movie", step=1.00),
        }

    def close(self) -> None:
        return None

    def _cache_path(self, video_path: str, cache_role: str, step: float, width: int, height: int, crop_h: int) -> Path:
        path = Path(video_path)
        stat = path.stat()
        cache_key = hashlib.md5(
            "|".join(
                [
                    str(path.resolve()).lower(),
                    str(int(stat.st_mtime)),
                    str(int(stat.st_size)),
                    cache_role,
                    f"{step:.4f}",
                    str(width),
                    str(height),
                ]
            ).encode("utf-8", errors="ignore")
        ).hexdigest()
        cache_dir = Path(__file__).resolve().parents[1] / "temp" / "visual_gray_cache"
        return cache_dir / f"{cache_key}.npy"

    def _load_series(self, video_path: str, cache_role: str, step: float) -> dict[str, Any]:
        path = Path(video_path)
        probe = cv2.VideoCapture(str(path))
        try:
            source_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            source_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            probe.release()
        if source_w <= 0 or source_h <= 0:
            raise RuntimeError(f"Cannot probe video: {video_path}")
        width = self.width
        height = max(2, int(source_h * width / source_w / 2) * 2)
        crop_h = max(1, int(height * self.crop_ratio))
        cache_path = self._cache_path(video_path, cache_role, step, width, height, crop_h)
        if cache_path.exists():
            return {"step": float(step), "frames": np.load(str(cache_path), mmap_mode="r")}

        # Missing cache means the exporter has not pre-warmed this material yet.
        # Building the whole movie cache here is too slow for an audit repair
        # pass, so fail fast and let the caller fall back to FrameScorer.
        if cache_role == "movie":
            raise RuntimeError(f"Movie gray cache missing: {cache_path}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame_bytes = width * height
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"fps=1/{step:.4f},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 128)
        frames: list[np.ndarray] = []
        try:
            while True:
                raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                if not raw or len(raw) < frame_bytes:
                    break
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width)[:crop_h, :].copy())
            proc.wait(timeout=10)
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
        if not frames:
            raise RuntimeError(f"Cannot build gray cache for {video_path}")
        array = np.stack(frames, axis=0)
        np.save(str(cache_path), array)
        return {"step": float(step), "frames": np.load(str(cache_path), mmap_mode="r")}

    def _read_gray(self, role: str, timestamp: float) -> np.ndarray | None:
        series = self.series["n" if role == "n" else "m"]
        step = float(series["step"])
        frames = series["frames"]
        index = int(round(max(0.0, float(timestamp)) / step))
        if 0 <= index < len(frames):
            return np.asarray(frames[index])
        return None


def _segment_at_time(usable: list, timestamp: float):
    for segment in usable:
        if float(segment.narration_start) - 1e-6 <= timestamp <= float(segment.narration_end) + 1e-6:
            return segment
    return None


def _target_segments(report: dict[str, Any], usable: list, max_segments: int) -> list:
    scores: dict[str, tuple[float, Any]] = {}

    def add_target(timestamp: float | None, index: int | None, score: float) -> None:
        segment = None
        if timestamp is not None:
            segment = _segment_at_time(usable, float(timestamp))
        if segment is None and index is not None and 0 <= int(index) < len(usable):
            segment = usable[int(index)]
        if segment is None:
            return
        current = scores.get(segment.id)
        if current is None or score < current[0]:
            scores[segment.id] = (score, segment)

    for group in report.get("low_groups", []):
        add_target(
            group.get("worst_time"),
            group.get("worst_segment_index"),
            float(group.get("min_score", 0.0) or 0.0),
        )
    for sample in report.get("worst_samples", []):
        add_target(
            sample.get("time"),
            sample.get("segment_index"),
            float(sample.get("score", 0.0) or 0.0),
        )
    return [segment for _segment_id, (_score, segment) in sorted(scores.items(), key=lambda item: item[1][0])[:max_segments]]


def _candidate_starts(current_start: float, source_duration: float, movie_duration: float) -> list[float]:
    values: set[float] = {round(max(0.0, current_start), 3)}
    for delta in (-2.4, -1.8, -1.2, -0.8, -0.5, -0.3, -0.15, 0.15, 0.3, 0.5, 0.8, 1.2, 1.8, 2.4):
        start = max(0.0, current_start + delta)
        if movie_duration <= 0.0 or start + source_duration <= movie_duration:
            values.add(round(start, 3))
    return sorted(values)


def repair_project(
    project_path: Path,
    report_path: Path,
    *,
    dry_run: bool,
    max_segments: int,
    min_score: float,
    min_gain: float,
    max_shift: float,
) -> dict[str, Any]:
    project = _load_project(project_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    usable = _usable_segments(project)
    targets = _target_segments(report, usable, max_segments)
    movie_duration = float(project.movie_duration or 0.0)
    try:
        scorer: FrameScorer = CachedFrameScorer(str(project.narration_path), str(project.movie_path))
    except Exception:
        scorer = FrameScorer(str(project.narration_path), str(project.movie_path))
    repairs: list[PhaseRepair] = []

    try:
        usable_index = {segment.id: idx for idx, segment in enumerate(usable)}
        for segment in targets:
            if segment.movie_start is None or segment.movie_end is None:
                continue
            current_start = float(segment.movie_start)
            source_duration = max(0.05, float(segment.movie_end) - current_start)
            current_score = scorer.score_range(segment, current_start, current_start + source_duration)

            candidates = _candidate_starts(current_start, source_duration, movie_duration)
            index = usable_index.get(segment.id)
            if index is not None and index > 0:
                previous = usable[index - 1]
                if previous.movie_end is not None and not _is_visual_cut(scorer, previous, segment):
                    gap = max(0.0, float(segment.narration_start) - float(previous.narration_end))
                    continuity_start = float(previous.movie_end) + gap
                    if movie_duration <= 0.0 or continuity_start + source_duration <= movie_duration:
                        candidates.append(round(max(0.0, continuity_start), 3))
            if index is not None and index + 1 < len(usable):
                next_segment = usable[index + 1]
                if next_segment.movie_start is not None and not _is_visual_cut(scorer, segment, next_segment):
                    gap = max(0.0, float(next_segment.narration_start) - float(segment.narration_end))
                    next_continuity_start = float(next_segment.movie_start) - gap - source_duration
                    if next_continuity_start >= 0.0:
                        candidates.append(round(next_continuity_start, 3))

            best_start = current_start
            best_score = current_score
            for candidate_start in sorted(set(candidates)):
                shift = abs(candidate_start - current_start)
                if shift > max_shift:
                    continue
                score = scorer.score_range(segment, candidate_start, candidate_start + source_duration)
                if score > best_score:
                    best_score = score
                    best_start = candidate_start

            changed = (
                abs(best_start - current_start) >= 0.10
                and best_score >= min_score
                and best_score >= current_score + min_gain
            )
            repairs.append(
                PhaseRepair(
                    segment_id=segment.id,
                    narration_start=float(segment.narration_start),
                    old_start=current_start,
                    new_start=float(best_start),
                    current_score=float(current_score),
                    best_score=float(best_score),
                    changed=bool(changed),
                )
            )
            if changed and not dry_run:
                candidate = MatchCandidate(
                    id=f"{segment.id}_audit_local_phase",
                    start=float(best_start),
                    end=float(best_start + source_duration),
                    score=min(0.93, max(float(segment.match_confidence or 0.0), best_score)),
                    confidence=min(0.93, max(float(segment.match_confidence or 0.0), best_score)),
                    visual_confidence=min(0.93, max(float(segment.visual_confidence or 0.0), best_score)),
                    audio_confidence=float(segment.audio_confidence or 0.0),
                    temporal_confidence=max(float(segment.temporal_confidence or 0.0), 0.88),
                    verification_score=float(best_score),
                    stability_score=max(float(segment.stability_score or 0.0), 0.76),
                    duration_gap=float(segment.duration_gap or 0.0),
                    match_count=max(1, len(FrameScorer._sample_positions(_duration(segment)))),
                    reason=f"Audit local phase repair; score={best_score:.3f}, before={current_score:.3f}",
                    source="audit_local_phase",
                    rank=1,
                )
                segment.match_candidates = [candidate] + [item for item in segment.match_candidates if item.id != candidate.id][:7]
                segment.selected_candidate_id = candidate.id
                segment.movie_start = candidate.start
                segment.movie_end = candidate.end
                segment.match_confidence = candidate.confidence
                segment.visual_confidence = candidate.visual_confidence
                segment.temporal_confidence = candidate.temporal_confidence
                segment.stability_score = candidate.stability_score
                segment.match_reason = candidate.reason
                segment.evidence_summary = f"audit_local_phase={best_score:.2f}, before={current_score:.2f}"
                segment.match_type = "exact" if best_score >= 0.76 else "inferred"
                segment.review_required = best_score < 0.80
                segment.alignment_status = AlignmentStatus.AUTO_ACCEPTED if not segment.review_required else AlignmentStatus.NEEDS_REVIEW
    finally:
        scorer.close()

    changed_repairs = [item for item in repairs if item.changed]
    result = {
        "dry_run": dry_run,
        "targets": len(targets),
        "changed": len(changed_repairs),
        "changes": [item.__dict__ for item in changed_repairs[:120]],
    }
    if changed_repairs and not dry_run:
        backup = project_path.with_name(
            f"{project_path.stem}.before_audit_local_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(project_path, backup)
        _save_project(project_path, project)
        result["backup"] = str(backup)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair audit failures by local phase search.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-segments", type=int, default=32)
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--min-gain", type=float, default=0.035)
    parser.add_argument("--max-shift", type=float, default=1.25)
    args = parser.parse_args()
    print(
        json.dumps(
            repair_project(
                args.project,
                args.report,
                dry_run=args.dry_run,
                max_segments=max(1, int(args.max_segments)),
                min_score=max(0.0, min(1.0, float(args.min_score))),
                min_gain=max(0.0, min(1.0, float(args.min_gain))),
                max_shift=max(0.1, float(args.max_shift)),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
