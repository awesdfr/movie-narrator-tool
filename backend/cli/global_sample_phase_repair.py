"""Repair low visual-audit samples with global DINO retrieval + phase refine."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cli.noncut_jump_repair import _load_project, _usable_segments
from cli.sample_phase_repair import SampleProposal, _apply_proposals, _segment_at_time
from cli.visual_match_audit import DinoIdentityScorer, _read_frame, _score_pair


def _default_movie_index_path(project: Any, model_name: str) -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / "temp" / "match_cache"
    return cache_dir / f"{Path(project.movie_path).stem}_dino_{model_name}.pkl"


def _target_samples(report: dict[str, Any], usable: list[Any], max_samples: int, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    def add(timestamp: Any, score: Any, source_time: Any = None) -> None:
        if timestamp is None:
            return
        score_value = float(score or 0.0)
        if score_value >= threshold:
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
                "score": score_value,
                "source_time": source_time,
                "segment_id": str(segment.id),
            }
        )

    for group in report.get("low_groups", []):
        add(group.get("worst_time"), group.get("min_score"), group.get("worst_source_time"))
    for sample in report.get("worst_samples", []):
        add(sample.get("time"), sample.get("score"), sample.get("source_time"))

    rows.sort(key=lambda item: float(item["score"]))
    return rows[: max(1, int(max_samples))]


def repair_project(
    project_path: Path,
    report_path: Path,
    *,
    dry_run: bool,
    max_samples: int,
    threshold: float,
    min_score: float,
    min_gain: float,
    top_k: int,
    verify_top_k: int,
    refine_radius: float,
    refine_step: float,
    dino_model: str,
    movie_index_path: Path | None,
) -> dict[str, Any]:
    project = _load_project(project_path)
    usable = _usable_segments(project)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    targets = _target_samples(report, usable, max_samples=max_samples, threshold=threshold)
    if not targets:
        return {"targets": 0, "accepted": 0, "changed": 0, "split_segments": 0, "proposals": []}

    index_path = movie_index_path or _default_movie_index_path(project, dino_model)
    if not index_path.exists():
        raise FileNotFoundError(f"Movie DINO index not found: {index_path}")
    with index_path.open("rb") as handle:
        index_payload = pickle.load(handle)
    movie_vectors = np.asarray(index_payload["vectors"], dtype=np.float32)
    movie_times = np.asarray(index_payload["times"], dtype=np.float32)
    if movie_vectors.ndim != 2 or len(movie_times) != len(movie_vectors):
        raise RuntimeError(f"Invalid DINO movie index: {index_path}")

    width = 320
    crop_ratio = 0.76
    narration_cap = cv2.VideoCapture(str(project.narration_path))
    movie_cap = cv2.VideoCapture(str(project.movie_path))
    if not narration_cap.isOpened() or not movie_cap.isOpened():
        raise RuntimeError("Cannot open narration or movie video")

    query_frames: list[np.ndarray] = []
    query_meta: list[tuple[dict[str, Any], Any, float, float]] = []
    try:
        for target in targets:
            sample_time = float(target["time"])
            segment = _segment_at_time(usable, sample_time)
            if segment is None or segment.movie_start is None:
                continue
            offset = sample_time - float(segment.narration_start)
            current_source = float(segment.movie_start) + offset
            frame = _read_frame(narration_cap, sample_time, width=width, crop_ratio=crop_ratio)
            if frame is None:
                continue
            query_frames.append(frame)
            query_meta.append((target, segment, offset, current_source))
    finally:
        narration_cap.release()

    if not query_frames:
        movie_cap.release()
        return {"targets": len(targets), "accepted": 0, "changed": 0, "split_segments": 0, "proposals": []}

    scorer = DinoIdentityScorer(dino_model, batch_size=128)
    query_vectors = scorer.encode(query_frames)

    proposals: list[SampleProposal] = []
    try:
        for query_index, (target, segment, offset, current_source) in enumerate(query_meta):
            similarities = movie_vectors @ query_vectors[query_index]
            candidate_count = min(max(1, int(top_k)), len(similarities))
            top_indices = np.argpartition(-similarities, candidate_count - 1)[:candidate_count]
            top_indices = sorted(top_indices, key=lambda item: float(similarities[int(item)]), reverse=True)[
                : max(1, int(verify_top_k))
            ]
            query_gray = cv2.cvtColor(query_frames[query_index], cv2.COLOR_BGR2GRAY)

            best: tuple[float, float, float, float] | None = None
            checked_times: set[float] = set()
            for movie_index in top_indices:
                base_time = float(movie_times[int(movie_index)])
                dino_similarity = float(similarities[int(movie_index)])
                for delta in np.arange(-refine_radius, refine_radius + 1e-6, refine_step):
                    movie_time = round(max(0.0, base_time + float(delta)), 3)
                    if movie_time in checked_times:
                        continue
                    checked_times.add(movie_time)
                    movie_frame = _read_frame(movie_cap, movie_time, width=width, crop_ratio=crop_ratio)
                    if movie_frame is None:
                        continue
                    pixel = _score_pair(query_gray, cv2.cvtColor(movie_frame, cv2.COLOR_BGR2GRAY))["score"]
                    score = scorer.calibrated_score(dino_similarity, float(pixel))
                    row = (float(score), dino_similarity, float(pixel), movie_time)
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
    finally:
        movie_cap.release()

    apply_result = _apply_proposals(project_path, proposals, dry_run=dry_run)
    return {
        "targets": len(targets),
        "accepted": len(proposals),
        **apply_result,
        "proposals": [proposal.__dict__ for proposal in proposals],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair low audit samples with global DINO retrieval.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--min-score", type=float, default=0.98)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--verify-top-k", type=int, default=6)
    parser.add_argument("--refine-radius", type=float, default=0.5)
    parser.add_argument("--refine-step", type=float, default=0.5)
    parser.add_argument("--dino-model", default="dinov2_vits14")
    parser.add_argument("--movie-index", type=Path, default=None)
    args = parser.parse_args()
    result = repair_project(
        args.project,
        args.report,
        dry_run=args.dry_run,
        max_samples=max(1, int(args.max_samples)),
        threshold=max(0.0, min(1.0, float(args.threshold))),
        min_score=max(0.0, min(1.0, float(args.min_score))),
        min_gain=max(0.0, min(1.0, float(args.min_gain))),
        top_k=max(1, int(args.top_k)),
        verify_top_k=max(1, int(args.verify_top_k)),
        refine_radius=max(0.0, float(args.refine_radius)),
        refine_step=max(0.05, float(args.refine_step)),
        dino_model=args.dino_model,
        movie_index_path=args.movie_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
