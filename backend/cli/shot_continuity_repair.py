"""Apply narration-shot continuity repair with a dry-run first.

This pass does not search unrelated movie scenes. It only retimes adjacent
segments that the narration video itself treats as one continuous visual shot.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from core.matcher.shot_continuity import apply_shot_continuity, plan_shot_continuity
from models.project import Project


def _load_project(path: Path) -> Project:
    return Project.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _save_project(path: Path, project: Project) -> None:
    project.updated_at = datetime.now()
    path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _change_report(project: Project) -> dict[str, Any]:
    plan = plan_shot_continuity(project, project.segments)
    changes: list[dict[str, Any]] = []
    for segment in project.segments:
        planned = plan.segment_ranges.get(segment.id)
        if planned is None or segment.is_manual_match:
            continue
        source_start, source_end, is_inferred, score = planned
        old_start = float(segment.movie_start) if segment.movie_start is not None else None
        old_end = float(segment.movie_end) if segment.movie_end is not None else None
        if old_start is None or old_end is None:
            shift = None
        else:
            shift = abs(old_start - source_start)
        changed = (
            old_start is None
            or old_end is None
            or abs(old_start - source_start) > 0.18
            or abs(old_end - source_end) > 0.18
        )
        if not changed:
            continue
        changes.append(
            {
                "segment_id": segment.id,
                "narration_start": float(segment.narration_start),
                "narration_end": float(segment.narration_end),
                "old_start": old_start,
                "old_end": old_end,
                "new_start": float(source_start),
                "new_end": float(source_end),
                "shift": shift,
                "inferred": bool(is_inferred),
                "score": float(score),
            }
        )
    changes.sort(key=lambda item: (float(item["narration_start"]), item["segment_id"]))
    large_changes = [item for item in changes if item["shift"] is not None and float(item["shift"]) > 8.0]
    return {
        "groups": plan.groups,
        "planned_changes": len(changes),
        "planned_inferred": plan.inferred_segments,
        "max_change_seconds": plan.max_change_seconds,
        "large_changes": len(large_changes),
        "changes_preview": changes[:80],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair non-cut source jumps by narration-shot continuity.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = _load_project(args.project)
    before = _change_report(project)
    result: dict[str, Any] = {"dry_run": bool(args.dry_run), "before": before}
    if not args.dry_run and before["planned_changes"] > 0:
        backup = args.project.with_name(
            f"{args.project.stem}.before_shot_continuity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(args.project, backup)
        applied = apply_shot_continuity(project, project.segments)
        _save_project(args.project, project)
        result["backup"] = str(backup)
        result["applied"] = applied
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
