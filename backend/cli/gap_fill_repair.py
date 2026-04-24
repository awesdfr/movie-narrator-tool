"""Fill visual-audit timeline gaps with globally verified movie snippets."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cli.visual_match_audit import DinoIdentityScorer, _read_frame, _score_pair


def _default_movie_index_path(project: dict[str, Any], model_name: str) -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / "temp" / "match_cache"
    return cache_dir / f"{Path(project['movie_path']).stem}_dino_{model_name}.pkl"


def _load_gap_targets(report: dict[str, Any], max_gaps: int) -> list[float]:
    targets: list[float] = []
    seen: set[float] = set()
    for group in report.get("low_groups", []):
        if group.get("worst_segment_index") is not None:
            continue
        timestamp = group.get("worst_time")
        if timestamp is None:
            continue
        key = round(float(timestamp), 3)
        if key in seen:
            continue
        seen.add(key)
        targets.append(float(timestamp))
    return targets[: max(1, int(max_gaps))]


def _find_container(segments: list[dict[str, Any]], timestamp: float) -> tuple[int | None, int | None, int | None]:
    previous_index: int | None = None
    next_index: int | None = None
    containing_index: int | None = None
    for index, segment in enumerate(segments):
        start = float(segment.get("narration_start") or 0.0)
        end = float(segment.get("narration_end") or 0.0)
        if start <= timestamp <= end:
            containing_index = index
            break
        if end <= timestamp:
            previous_index = index
        if start > timestamp and next_index is None:
            next_index = index
            break
    if containing_index is not None:
        return previous_index, containing_index, next_index
    return previous_index, None, next_index


def _verified_source_for_time(
    *,
    narration_cap: cv2.VideoCapture,
    movie_cap: cv2.VideoCapture,
    scorer: DinoIdentityScorer,
    movie_vectors: np.ndarray,
    movie_times: np.ndarray,
    timestamp: float,
    width: int,
    crop_ratio: float,
    top_k: int,
    verify_top_k: int,
    refine_radius: float,
    refine_step: float,
) -> tuple[float, float] | None:
    query_frame = _read_frame(narration_cap, timestamp, width=width, crop_ratio=crop_ratio)
    if query_frame is None:
        return None
    query_vector = scorer.encode([query_frame])[0]
    similarities = movie_vectors @ query_vector
    candidate_count = min(max(1, int(top_k)), len(similarities))
    top_indices = np.argpartition(-similarities, candidate_count - 1)[:candidate_count]
    top_indices = sorted(top_indices, key=lambda item: float(similarities[int(item)]), reverse=True)[
        : max(1, int(verify_top_k))
    ]

    query_gray = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
    best: tuple[float, float] | None = None
    checked: set[float] = set()
    for movie_index in top_indices:
        base_time = float(movie_times[int(movie_index)])
        dino_similarity = float(similarities[int(movie_index)])
        for delta in np.arange(-refine_radius, refine_radius + 1e-6, refine_step):
            movie_time = round(max(0.0, base_time + float(delta)), 3)
            if movie_time in checked:
                continue
            checked.add(movie_time)
            movie_frame = _read_frame(movie_cap, movie_time, width=width, crop_ratio=crop_ratio)
            if movie_frame is None:
                continue
            pixel = _score_pair(query_gray, cv2.cvtColor(movie_frame, cv2.COLOR_BGR2GRAY))["score"]
            score = scorer.calibrated_score(dino_similarity, float(pixel))
            row = (float(score), movie_time)
            if best is None or row > best:
                best = row
    return best


def _mark_segment(segment: dict[str, Any], source_start: float, source_end: float, score: float) -> None:
    segment["movie_start"] = float(source_start)
    segment["movie_end"] = float(source_end)
    segment["segment_type"] = "has_narration"
    segment["status"] = "completed"
    segment["match_confidence"] = max(float(segment.get("match_confidence") or 0.0), float(score))
    segment["visual_confidence"] = max(float(segment.get("visual_confidence") or 0.0), float(score))
    segment["temporal_confidence"] = max(float(segment.get("temporal_confidence") or 0.0), 0.86)
    segment["stability_score"] = max(float(segment.get("stability_score") or 0.0), 0.72)
    segment["duration_gap"] = abs((source_end - source_start) - (float(segment["narration_end"]) - float(segment["narration_start"])))
    segment["match_reason"] = f"Timeline gap fill by global DINO; score={score:.3f}"
    segment["match_type"] = "exact"
    segment["evidence_summary"] = f"gap_fill_global_dino={score:.2f}"
    segment["alignment_status"] = "auto_accepted"
    segment["review_required"] = score < 0.98
    segment["selected_candidate_id"] = f"{segment['id']}_gap_fill"
    segment["match_candidates"] = [
        {
            "id": segment["selected_candidate_id"],
            "start": float(source_start),
            "end": float(source_end),
            "score": float(score),
            "confidence": float(score),
            "visual_confidence": float(score),
            "audio_confidence": float(segment.get("audio_confidence") or 0.0),
            "temporal_confidence": float(segment.get("temporal_confidence") or 0.86),
            "verification_score": float(score),
            "stability_score": float(segment.get("stability_score") or 0.72),
            "duration_gap": float(segment["duration_gap"]),
            "match_count": 1,
            "reason": segment["match_reason"],
            "source": "gap_fill_global_dino",
            "rank": 1,
        }
    ]


def repair_project(
    project_path: Path,
    report_path: Path,
    *,
    dry_run: bool,
    max_gaps: int,
    min_score: float,
    top_k: int,
    verify_top_k: int,
    refine_radius: float,
    refine_step: float,
    dino_model: str,
    movie_index_path: Path | None,
) -> dict[str, Any]:
    project = json.loads(project_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    targets = _load_gap_targets(report, max_gaps=max_gaps)
    if not targets:
        return {"targets": 0, "changed": 0, "changes": []}

    index_path = movie_index_path or _default_movie_index_path(project, dino_model)
    with index_path.open("rb") as handle:
        index_payload = pickle.load(handle)
    movie_vectors = np.asarray(index_payload["vectors"], dtype=np.float32)
    movie_times = np.asarray(index_payload["times"], dtype=np.float32)

    width = 320
    crop_ratio = 0.76
    scorer = DinoIdentityScorer(dino_model, batch_size=128)
    narration_cap = cv2.VideoCapture(str(project["narration_path"]))
    movie_cap = cv2.VideoCapture(str(project["movie_path"]))
    if not narration_cap.isOpened() or not movie_cap.isOpened():
        raise RuntimeError("Cannot open narration or movie video")

    segments = sorted(project.get("segments") or [], key=lambda item: (float(item.get("narration_start") or 0.0), float(item.get("narration_end") or 0.0)))
    changes: list[dict[str, Any]] = []
    try:
        for timestamp in targets:
            previous_index, containing_index, next_index = _find_container(segments, timestamp)
            if containing_index is not None:
                segment = segments[containing_index]
                gap_start = float(segment["narration_start"])
                gap_end = float(segment["narration_end"])
                insert_index = containing_index
                created = False
            else:
                if previous_index is None or next_index is None:
                    continue
                previous = segments[previous_index]
                next_segment = segments[next_index]
                gap_start = float(previous["narration_end"])
                gap_end = float(next_segment["narration_start"])
                if gap_end - gap_start <= 0.04:
                    continue
                template = previous if previous.get("movie_end") is not None else next_segment
                segment = copy.deepcopy(template)
                segment["id"] = f"gapfill_{len(changes) + 1:03d}_{int(round(gap_start * 1000))}"
                segment["narration_start"] = gap_start
                segment["narration_end"] = gap_end
                segment["original_text"] = ""
                segment["polished_text"] = ""
                insert_index = next_index
                created = True

            midpoint = (gap_start + gap_end) / 2.0
            verified = _verified_source_for_time(
                narration_cap=narration_cap,
                movie_cap=movie_cap,
                scorer=scorer,
                movie_vectors=movie_vectors,
                movie_times=movie_times,
                timestamp=midpoint,
                width=width,
                crop_ratio=crop_ratio,
                top_k=top_k,
                verify_top_k=verify_top_k,
                refine_radius=refine_radius,
                refine_step=refine_step,
            )
            if verified is None:
                continue
            score, source_midpoint = verified
            if score < min_score:
                continue
            duration = gap_end - gap_start
            source_start = max(0.0, source_midpoint - duration / 2.0)
            source_end = source_start + duration
            _mark_segment(segment, source_start, source_end, score)
            if created:
                segments.insert(insert_index, segment)
            changes.append(
                {
                    "time": timestamp,
                    "segment_id": segment["id"],
                    "created": created,
                    "narration_start": gap_start,
                    "narration_end": gap_end,
                    "movie_start": source_start,
                    "movie_end": source_end,
                    "score": score,
                }
            )
    finally:
        narration_cap.release()
        movie_cap.release()

    if changes and not dry_run:
        for index, segment in enumerate(sorted(segments, key=lambda item: (float(item["narration_start"]), float(item["narration_end"]))), start=1):
            segment["index"] = index
        project["segments"] = segments
        project["updated_at"] = datetime.now().isoformat()
        backup = project_path.with_name(f"{project_path.stem}.before_gap_fill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(project_path, backup)
        project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        backup_path = str(backup)
    else:
        backup_path = None

    return {"targets": len(targets), "changed": len(changes), "backup": backup_path, "changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill visual timeline gaps from audit failures.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-gaps", type=int, default=24)
    parser.add_argument("--min-score", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--verify-top-k", type=int, default=4)
    parser.add_argument("--refine-radius", type=float, default=0.5)
    parser.add_argument("--refine-step", type=float, default=0.5)
    parser.add_argument("--dino-model", default="dinov2_vits14")
    parser.add_argument("--movie-index", type=Path, default=None)
    args = parser.parse_args()
    result = repair_project(
        args.project,
        args.report,
        dry_run=args.dry_run,
        max_gaps=max(1, int(args.max_gaps)),
        min_score=max(0.0, min(1.0, float(args.min_score))),
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
