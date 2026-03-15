"""Benchmark CLI for segment alignment evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes.project import load_project, save_project
from models.segment import AlignmentStatus, SegmentType

DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_BOUNDARY_TOLERANCE = 1.5


def _resolve_segment(project, item: dict[str, Any]):
    segment_id = item.get('segment_id')
    if segment_id:
        return next((segment for segment in project.segments if segment.id == segment_id), None)

    index = item.get('index')
    if index is None:
        return None
    return next((segment for segment in project.segments if segment.index == index), None)


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def _segment_is_non_movie(segment) -> bool:
    return (
        segment.segment_type == SegmentType.NON_MOVIE
        or segment.alignment_status == AlignmentStatus.NON_MOVIE
        or (segment.movie_start is None and segment.movie_end is None and not segment.original_text)
    )


def evaluate_manifest(project, manifest: dict[str, Any]) -> dict[str, Any]:
    defaults = manifest.get('defaults', {})
    default_iou = float(defaults.get('iou_threshold', DEFAULT_IOU_THRESHOLD))
    default_boundary = float(defaults.get('boundary_tolerance', DEFAULT_BOUNDARY_TOLERANCE))

    rows: list[dict[str, Any]] = []
    by_scenario: dict[str, dict[str, int]] = defaultdict(lambda: {'total': 0, 'correct': 0})

    correct_count = 0
    false_match_count = 0
    missing_match_count = 0
    review_hit_count = 0
    incorrect_count = 0

    for item in manifest.get('segments', []):
        segment = _resolve_segment(project, item)
        scenario = item.get('scenario', 'default')
        expected_label = item.get('label', 'movie')
        iou_threshold = float(item.get('iou_threshold', default_iou))
        boundary_tolerance = float(item.get('boundary_tolerance', default_boundary))

        if segment is None:
            rows.append({
                'segment_id': item.get('segment_id'),
                'index': item.get('index'),
                'scenario': scenario,
                'label': expected_label,
                'correct': False,
                'error': 'segment_not_found',
            })
            by_scenario[scenario]['total'] += 1
            incorrect_count += 1
            continue

        row: dict[str, Any] = {
            'segment_id': segment.id,
            'index': segment.index,
            'scenario': scenario,
            'label': expected_label,
            'alignment_status': segment.alignment_status,
            'review_required': bool(segment.review_required),
            'match_confidence': float(segment.match_confidence or 0.0),
            'movie_start': segment.movie_start,
            'movie_end': segment.movie_end,
        }

        correct = False
        false_match = False
        missing_match = False

        if expected_label == 'non_movie':
            predicted_non_movie = _segment_is_non_movie(segment)
            has_wrong_movie_match = segment.movie_start is not None or segment.movie_end is not None
            correct = predicted_non_movie and not has_wrong_movie_match
            false_match = has_wrong_movie_match
            row['predicted_non_movie'] = predicted_non_movie
        else:
            gt_start = item.get('movie_start')
            gt_end = item.get('movie_end')
            if gt_start is None or gt_end is None:
                row['error'] = 'missing_ground_truth_time'
            elif segment.movie_start is None or segment.movie_end is None:
                missing_match = True
                row['error'] = 'missing_prediction'
            else:
                iou = _interval_iou(float(segment.movie_start), float(segment.movie_end), float(gt_start), float(gt_end))
                start_error = abs(float(segment.movie_start) - float(gt_start))
                end_error = abs(float(segment.movie_end) - float(gt_end))
                correct = iou >= iou_threshold or (start_error <= boundary_tolerance and end_error <= boundary_tolerance)
                false_match = not correct
                row.update(
                    {
                        'gt_movie_start': float(gt_start),
                        'gt_movie_end': float(gt_end),
                        'iou': iou,
                        'start_error': start_error,
                        'end_error': end_error,
                    }
                )

        row['correct'] = correct
        row['false_match'] = false_match
        row['missing_match'] = missing_match
        rows.append(row)

        by_scenario[scenario]['total'] += 1
        if correct:
            correct_count += 1
            by_scenario[scenario]['correct'] += 1
        else:
            incorrect_count += 1
            if row['review_required'] or row['alignment_status'] in {
                AlignmentStatus.NEEDS_REVIEW,
                AlignmentStatus.UNMATCHED,
            }:
                review_hit_count += 1
        if false_match:
            false_match_count += 1
        if missing_match:
            missing_match_count += 1

    total = len(rows)
    accuracy = correct_count / total if total else 0.0
    false_match_rate = false_match_count / total if total else 0.0
    low_confidence_recall = review_hit_count / incorrect_count if incorrect_count else 1.0

    scenario_breakdown = {
        scenario: {
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': stats['correct'] / stats['total'] if stats['total'] else 0.0,
        }
        for scenario, stats in sorted(by_scenario.items())
    }

    return {
        'project_id': project.id,
        'project_name': project.name,
        'manifest_name': manifest.get('name', 'unnamed_benchmark'),
        'pipeline_version': project.match_version,
        'metrics': {
            'total_segments': total,
            'correct_segments': correct_count,
            'incorrect_segments': incorrect_count,
            'accuracy': accuracy,
            'false_match_count': false_match_count,
            'false_match_rate': false_match_rate,
            'missing_match_count': missing_match_count,
            'review_required_on_errors': review_hit_count,
            'low_confidence_recall_on_errors': low_confidence_recall,
            'targets': {
                'accuracy': 0.99,
                'false_match_rate': 0.01,
                'low_confidence_recall_on_errors': 0.95,
            },
        },
        'scenario_breakdown': scenario_breakdown,
        'rows': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate project segment alignment against a benchmark manifest.')
    parser.add_argument('--project', required=True, help='Project id to evaluate')
    parser.add_argument('--manifest', required=True, help='Path to benchmark manifest JSON')
    parser.add_argument('--report', help='Optional output report path')
    args = parser.parse_args()

    project = load_project(args.project)
    if not project:
        print(f'Project not found: {args.project}', file=sys.stderr)
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f'Manifest not found: {manifest_path}', file=sys.stderr)
        return 1

    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)

    report = evaluate_manifest(project, manifest)

    project.benchmark_accuracy = report['metrics']['accuracy']
    project.benchmark_manifest = str(manifest_path.resolve())
    save_project(project)

    report_path = Path(args.report) if args.report else manifest_path.with_name(f"{manifest_path.stem}_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    metrics = report['metrics']
    print(f"Benchmark: {report['manifest_name']}")
    print(f"Project: {project.id} ({project.name})")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"False match rate: {metrics['false_match_rate']:.2%}")
    print(f"Low-confidence recall on errors: {metrics['low_confidence_recall_on_errors']:.2%}")
    print(f"Report written to: {report_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
