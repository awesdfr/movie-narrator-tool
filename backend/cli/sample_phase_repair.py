"""Batch repair low visual-audit samples by DINO local phase search."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cli.noncut_jump_repair import _load_project, _save_project, _usable_segments
from cli.visual_match_audit import DinoIdentityScorer, _read_frame, _score_pair


@dataclass
class SampleProposal:
    segment_id: str
    sample_time: float
    old_score: float
    best_score: float
    current_source: float
    best_source: float
    proposed_start: float


def _segment_at_time(segments: list[Any], timestamp: float):
    for segment in segments:
        if float(segment.narration_start) - 1e-6 <= timestamp <= float(segment.narration_end) + 1e-6:
            return segment
    return None


def _target_samples(report: dict[str, Any], usable: list[Any], max_groups: int, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    def add_row(timestamp: Any, score: float) -> None:
        if timestamp is None or score >= threshold:
            return
        segment = _segment_at_time(usable, float(timestamp))
        if segment is None or segment.movie_start is None or segment.movie_end is None:
            return
        key = (str(segment.id), round(float(timestamp), 3))
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "time": float(timestamp),
                "score": score,
                "segment_id": segment.id,
            }
        )

    for group in report.get("low_groups", []):
        score = float(group.get("min_score", group.get("average_score", 0.0)) or 0.0)
        add_row(group.get("worst_time"), score)
    for sample in report.get("worst_samples", []):
        score = float(sample.get("score", 0.0) or 0.0)
        add_row(sample.get("time"), score)
    rows.sort(key=lambda item: float(item["score"]))
    return rows[: max(1, int(max_groups))]


def _mark_repaired(segment: dict[str, Any], reason: str) -> None:
    segment["match_reason"] = reason
    segment["evidence_summary"] = "sample_phase_verified; inferred=False"
    segment["match_type"] = "exact"
    segment["review_required"] = False
    segment["alignment_status"] = "auto_accepted"
    segment["match_confidence"] = max(float(segment.get("match_confidence") or 0.0), 0.90)
    segment["visual_confidence"] = max(float(segment.get("visual_confidence") or 0.0), 0.90)


def _apply_proposals(project_path: Path, proposals: list[SampleProposal], dry_run: bool) -> dict[str, Any]:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    by_id: dict[str, list[SampleProposal]] = {}
    for proposal in proposals:
        by_id.setdefault(proposal.segment_id, []).append(proposal)

    new_segments: list[dict[str, Any]] = []
    changed = 0
    split_count = 0
    for segment in data.get("segments", []):
        segment_proposals = sorted(by_id.get(segment.get("id"), []), key=lambda item: item.sample_time)
        if not segment_proposals:
            new_segments.append(segment)
            continue

        if len(segment_proposals) == 1:
            proposal = segment_proposals[0]
            duration = float(segment.get("movie_end") or 0.0) - float(segment.get("movie_start") or 0.0)
            if duration > 0:
                segment["movie_start"] = proposal.proposed_start
                segment["movie_end"] = proposal.proposed_start + duration
                segment["selected_candidate_id"] = f"{segment['id']}_sample_phase"
                _mark_repaired(segment, "DINO sample-phase repair")
                changed += 1
            new_segments.append(segment)
            continue

        # Multiple samples in one segment can indicate a cut or speed edit inside
        # the narration clip. Split at sample midpoints so each short piece can
        # follow its own verified source phase.
        boundaries = [float(segment["narration_start"])]
        for left, right in zip(segment_proposals, segment_proposals[1:]):
            boundaries.append((left.sample_time + right.sample_time) / 2.0)
        boundaries.append(float(segment["narration_end"]))

        for index, proposal in enumerate(segment_proposals):
            start = boundaries[index]
            end = boundaries[index + 1]
            if end - start <= 0.05:
                continue
            piece = copy.deepcopy(segment)
            piece["id"] = f"{segment['id']}_p{index + 1:02d}"
            piece["narration_start"] = start
            piece["narration_end"] = end
            piece_duration = end - start
            offset = proposal.sample_time - start
            source_start = max(0.0, proposal.best_source - offset)
            piece["movie_start"] = source_start
            piece["movie_end"] = source_start + piece_duration
            piece["selected_candidate_id"] = f"{piece['id']}_sample_phase"
            _mark_repaired(piece, "DINO sample-phase split repair")
            new_segments.append(piece)
            changed += 1
        split_count += 1

    for index, segment in enumerate(new_segments, start=1):
        segment["index"] = index
    data["segments"] = new_segments
    data["updated_at"] = datetime.now().isoformat()

    if changed and not dry_run:
        backup = project_path.with_name(f"{project_path.stem}.before_sample_phase_repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(project_path, backup)
        project_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        backup_path = str(backup)
    else:
        backup_path = None

    return {
        "changed": changed,
        "split_segments": split_count,
        "backup": backup_path,
    }


def repair_project(
    project_path: Path,
    report_path: Path,
    *,
    dry_run: bool,
    max_groups: int,
    search_window: float,
    search_step: float,
    threshold: float,
    min_score: float,
    min_gain: float,
) -> dict[str, Any]:
    project = _load_project(project_path)
    usable = _usable_segments(project)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    targets = _target_samples(report, usable, max_groups=max_groups, threshold=threshold)
    if not targets:
        return {"targets": 0, "accepted": 0, "changed": 0, "split_segments": 0, "proposals": []}

    width = 320
    crop_ratio = 0.76
    narration_cap = cv2.VideoCapture(str(project.narration_path))
    movie_cap = cv2.VideoCapture(str(project.movie_path))
    if not narration_cap.isOpened() or not movie_cap.isOpened():
        raise RuntimeError("Cannot open narration or movie video")

    query_frames: list[np.ndarray] = []
    query_meta: list[tuple[dict[str, Any], Any, float, float]] = []
    movie_frames: list[np.ndarray] = []
    movie_times: list[float] = []
    seen_movie_times: set[float] = set()

    try:
        for target in targets:
            sample_time = float(target["time"])
            segment = _segment_at_time(usable, sample_time)
            if segment is None or segment.movie_start is None:
                continue
            offset = sample_time - float(segment.narration_start)
            current_source = float(segment.movie_start) + offset
            query_frame = _read_frame(narration_cap, sample_time, width=width, crop_ratio=crop_ratio)
            if query_frame is None:
                continue
            query_meta.append((target, segment, offset, current_source))
            query_frames.append(query_frame)
            start = max(0.0, current_source - search_window)
            end = current_source + search_window
            for movie_time in np.arange(start, end + 1e-6, search_step):
                movie_time = round(float(movie_time), 3)
                if movie_time in seen_movie_times:
                    continue
                movie_frame = _read_frame(movie_cap, movie_time, width=width, crop_ratio=crop_ratio)
                if movie_frame is None:
                    continue
                seen_movie_times.add(movie_time)
                movie_times.append(movie_time)
                movie_frames.append(movie_frame)
    finally:
        narration_cap.release()
        movie_cap.release()

    if not query_frames or not movie_frames:
        return {"targets": len(targets), "accepted": 0, "changed": 0, "split_segments": 0, "proposals": []}

    scorer = DinoIdentityScorer("dinov2_vits14", batch_size=128)
    query_vectors = scorer.encode(query_frames)
    movie_vectors = scorer.encode(movie_frames)

    proposals: list[SampleProposal] = []
    for query_index, (target, segment, offset, current_source) in enumerate(query_meta):
        query_gray = cv2.cvtColor(query_frames[query_index], cv2.COLOR_BGR2GRAY)
        similarities = movie_vectors @ query_vectors[query_index]
        best: tuple[float, float, float, float] | None = None
        for similarity, movie_time, movie_frame in zip(similarities, movie_times, movie_frames):
            if abs(movie_time - current_source) > search_window + 1e-6:
                continue
            pixel = _score_pair(query_gray, cv2.cvtColor(movie_frame, cv2.COLOR_BGR2GRAY))["score"]
            score = scorer.calibrated_score(float(similarity), float(pixel))
            row = (float(score), float(similarity), float(pixel), float(movie_time))
            if best is None or row > best:
                best = row
        if best is None:
            continue
        best_score, _similarity, _pixel, best_source = best
        old_score = float(target["score"])
        if best_score < min_score or best_score < old_score + min_gain:
            continue
        proposals.append(
            SampleProposal(
                segment_id=str(segment.id),
                sample_time=float(target["time"]),
                old_score=old_score,
                best_score=best_score,
                current_source=current_source,
                best_source=best_source,
                proposed_start=max(0.0, best_source - offset),
            )
        )

    apply_result = _apply_proposals(project_path, proposals, dry_run=dry_run)
    return {
        "targets": len(targets),
        "accepted": len(proposals),
        **apply_result,
        "proposals": [proposal.__dict__ for proposal in proposals],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair visual-audit low samples with DINO local phase search.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-groups", type=int, default=48)
    parser.add_argument("--search-window", type=float, default=8.0)
    parser.add_argument("--search-step", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--min-score", type=float, default=0.98)
    parser.add_argument("--min-gain", type=float, default=0.08)
    args = parser.parse_args()
    result = repair_project(
        args.project,
        args.report,
        dry_run=args.dry_run,
        max_groups=args.max_groups,
        search_window=args.search_window,
        search_step=max(0.05, args.search_step),
        threshold=args.threshold,
        min_score=args.min_score,
        min_gain=args.min_gain,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
