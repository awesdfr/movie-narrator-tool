"""Preview and export routes."""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from api.routes.project import load_project, save_project
from config import settings
from models.project import ExportMode, VideoSourceInfo
from models.segment import SegmentType

router = APIRouter()


class BenchmarkEvaluateRequest(BaseModel):
    manifest_path: str | None = Field(default=None, description="Benchmark manifest JSON path")


class VisualAuditRequest(BaseModel):
    draft_path: str | None = Field(default=None, description="draft_content.json path or Jianying draft directory")
    step: float = Field(default=1.0, ge=0.1, le=10.0)
    threshold: float = Field(default=0.66, ge=0.0, le=1.0)
    crop_ratio: float = Field(default=0.76, ge=0.2, le=1.0)
    width: int = Field(default=320, ge=80, le=1920)
    max_time: float | None = Field(default=None, ge=1.0)
    metric: str = Field(default="identity")
    dino_model: str = Field(default="dinov2_vits14")


def _get_project_and_segment(project_id: str, segment_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
    segment = next((item for item in project.segments if item.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail='片段不存在')
    return project, segment


def _get_video_path(project, source: str) -> str:
    if source not in {'movie', 'narration'}:
        raise HTTPException(status_code=400, detail='source 必须是 movie 或 narration')
    video_path = project.narration_path if source == 'narration' else project.movie_path
    if not video_path:
        raise HTTPException(status_code=400, detail='视频路径不存在')
    return video_path


def _resolve_project_json_path(project_id: str) -> Path:
    return settings.projects_dir / f'{project_id}.json'


def _resolve_draft_content_path(draft_path: str | None, project) -> Path:
    if draft_path:
        candidate = Path(draft_path)
    elif getattr(project, 'last_jianying_draft_path', None):
        candidate = Path(project.last_jianying_draft_path)
    else:
        raise HTTPException(status_code=400, detail='没有可用的剪映草稿路径，请先导出草稿或手动传入 draft_path。')

    if candidate.is_dir():
        candidate = candidate / 'draft_content.json'
    if not candidate.exists():
        raise HTTPException(status_code=400, detail=f'草稿文件不存在: {candidate}')
    return candidate


def _commercial_readiness_payload(project) -> dict:
    segments = [
        segment
        for segment in project.segments
        if segment.use_segment and getattr(segment.segment_type, 'value', segment.segment_type) != SegmentType.NON_MOVIE.value
    ]
    matched = [segment for segment in segments if segment.movie_start is not None and segment.movie_end is not None]
    review_required = [segment for segment in matched if segment.review_required]
    low_confidence = [segment for segment in matched if float(segment.match_confidence or 0.0) < 0.78]
    very_low_confidence = [segment for segment in matched if float(segment.match_confidence or 0.0) < 0.68]

    blockers: list[dict[str, str]] = []
    if review_required:
        blockers.append({
            'code': 'review_required_segments',
            'message': f'仍有 {len(review_required)} 个片段待复核，自动商用会直接破坏观感。',
        })
    if very_low_confidence:
        blockers.append({
            'code': 'very_low_confidence_segments',
            'message': f'仍有 {len(very_low_confidence)} 个片段置信度低于 0.68，存在明显错画面风险。',
        })
    if project.benchmark_accuracy is None:
        blockers.append({
            'code': 'benchmark_missing',
            'message': '尚未绑定 benchmark，当前高匹配率数字不能证明真实准确率。',
        })
    elif float(project.benchmark_accuracy or 0.0) < 0.98:
        blockers.append({
            'code': 'benchmark_below_target',
            'message': f'Benchmark accuracy 仅 {float(project.benchmark_accuracy):.2%}，未达到 98% 商业化目标。',
        })
    if project.visual_audit_score is None:
        blockers.append({
            'code': 'visual_audit_missing',
            'message': '尚未运行导出画面审计，无法确认草稿导出后的真实视觉一致性。',
        })
    elif float(project.visual_audit_score or 0.0) < 0.98 or int(project.visual_audit_below_threshold or 0) > 0:
        blockers.append({
            'code': 'visual_audit_below_target',
            'message': (
                f'导出画面审计均分仅 {float(project.visual_audit_score):.2%}，'
                f'且仍有 {int(project.visual_audit_below_threshold or 0)} 个低于目标阈值的采样点，'
                '未达到 98% 商业化目标。'
            ),
        })
    if not bool(getattr(project, 'rights_confirmed', False)):
        blockers.append({
            'code': 'rights_unconfirmed',
            'message': '商业版权状态未确认，无法判定是否可合法商用。',
        })
    if not bool(getattr(project, 'platform_risk_acknowledged', False)):
        blockers.append({
            'code': 'platform_risk_unacknowledged',
            'message': '平台查重/风控风险尚未确认并接受，不能视为可直接商业发布。',
        })

    status = 'ready' if not blockers else ('conditional' if len(blockers) <= 2 else 'blocked')
    return {
        'status': status,
        'summary': {
            'usable_segments': len(segments),
            'matched_segments': len(matched),
            'review_required_segments': len(review_required),
            'low_confidence_segments': len(low_confidence),
            'very_low_confidence_segments': len(very_low_confidence),
            'benchmark_accuracy': project.benchmark_accuracy,
            'benchmark_false_match_rate': getattr(project, 'benchmark_false_match_rate', None),
            'visual_audit_score': getattr(project, 'visual_audit_score', None),
            'rights_confirmed': bool(getattr(project, 'rights_confirmed', False)),
            'platform_risk_acknowledged': bool(getattr(project, 'platform_risk_acknowledged', False)),
        },
        'blockers': blockers,
    }


@router.post('/{project_id}/benchmark/evaluate')
async def evaluate_benchmark(project_id: str, payload: BenchmarkEvaluateRequest):
    from cli.benchmark_cli import evaluate_manifest

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    manifest_path_str = payload.manifest_path or getattr(project, 'benchmark_manifest', None)
    if not manifest_path_str:
        raise HTTPException(status_code=400, detail='没有可用的 benchmark manifest，请先传入 manifest_path。')

    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail=f'Benchmark manifest 不存在: {manifest_path}')

    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)

    report = await asyncio.to_thread(evaluate_manifest, project, manifest)
    metrics = report.get('metrics', {})

    report_dir = settings.temp_dir / project_id / 'exports' / 'benchmark'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{manifest_path.stem}_report.json"
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    project.benchmark_accuracy = float(metrics.get('accuracy') or 0.0)
    project.benchmark_manifest = str(manifest_path.resolve())
    project.benchmark_false_match_rate = float(metrics.get('false_match_rate') or 0.0)
    project.benchmark_low_confidence_recall = float(metrics.get('low_confidence_recall_on_errors') or 0.0)
    project.benchmark_report_path = str(report_path)
    project.benchmark_updated_at = datetime.now()
    save_project(project)

    return {
        'message': 'benchmark 评测完成',
        'report_path': str(report_path),
        'metrics': metrics,
        'manifest_name': report.get('manifest_name'),
        'scenario_breakdown': report.get('scenario_breakdown', {}),
    }


@router.get('/{project_id}/benchmark/report')
async def get_benchmark_report(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    report_path_str = getattr(project, 'benchmark_report_path', None)
    if not report_path_str:
        raise HTTPException(status_code=404, detail='当前项目还没有 benchmark 报告。')

    report_path = Path(report_path_str)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f'Benchmark 报告不存在: {report_path}')
    return FileResponse(report_path, media_type='application/json', filename=report_path.name)


@router.post('/{project_id}/audit/visual')
async def run_visual_audit(project_id: str, payload: VisualAuditRequest):
    from cli.visual_match_audit import audit_visual_match

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    draft_content_path = _resolve_draft_content_path(payload.draft_path, project)
    project_json_path = _resolve_project_json_path(project_id)
    if not project_json_path.exists():
        raise HTTPException(status_code=400, detail=f'项目文件不存在: {project_json_path}')

    report = await asyncio.to_thread(
        audit_visual_match,
        project_path=project_json_path,
        draft_path=draft_content_path,
        step=float(payload.step),
        threshold=float(payload.threshold),
        crop_ratio=float(payload.crop_ratio),
        width=int(payload.width),
        max_time=payload.max_time,
        metric=payload.metric,
        dino_model=payload.dino_model,
    )
    summary = report.get('summary', {})

    report_dir = settings.temp_dir / project_id / 'exports' / 'visual_audit'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'{draft_content_path.parent.name}_visual_audit.json'
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    project.visual_audit_score = float(summary.get('score_average') or 0.0)
    project.visual_audit_metric = str(summary.get('metric') or payload.metric)
    project.visual_audit_threshold = float(summary.get('threshold') or payload.threshold)
    project.visual_audit_below_threshold = int(summary.get('below_threshold') or 0)
    project.visual_audit_report_path = str(report_path)
    project.visual_audit_updated_at = datetime.now()
    save_project(project)

    return {
        'message': '导出画面审计完成',
        'report_path': str(report_path),
        'summary': summary,
        'low_groups': report.get('low_groups', []),
        'source_jumps': report.get('source_jumps', []),
    }


@router.get('/{project_id}/audit/report')
async def get_visual_audit_report(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    report_path_str = getattr(project, 'visual_audit_report_path', None)
    if not report_path_str:
        raise HTTPException(status_code=404, detail='当前项目还没有导出画面审计报告。')

    report_path = Path(report_path_str)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f'导出画面审计报告不存在: {report_path}')
    return FileResponse(report_path, media_type='application/json', filename=report_path.name)


@router.get('/{project_id}/commercial-readiness')
async def get_commercial_readiness(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
    return _commercial_readiness_payload(project)


@router.get('/{project_id}/thumbnail/{segment_id}')
async def get_segment_thumbnail(project_id: str, segment_id: str):
    from core.video_processor.frame_extractor import FrameExtractor

    project, segment = _get_project_and_segment(project_id, segment_id)
    if segment.thumbnail_path and Path(segment.thumbnail_path).exists():
        return FileResponse(segment.thumbnail_path, media_type='image/jpeg')

    extractor = FrameExtractor()
    thumbnail_path = await extractor.extract_thumbnail(
        project.narration_path,
        segment.narration_start,
        output_dir=settings.temp_dir / project_id,
    )
    segment.thumbnail_path = str(thumbnail_path)
    save_project(project)
    return FileResponse(thumbnail_path, media_type='image/jpeg')


@router.get('/{project_id}/frame')
async def get_frame(
    project_id: str,
    source: str,
    time: float,
    masked: bool = False,
    mask_mode: str | None = None,
    manual_regions: str | None = None,
):
    from core.video_processor.frame_extractor import FrameExtractor, read_image_unicode, write_image_unicode
    from core.video_processor.subtitle_masker import SubtitleMasker

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    video_path = _get_video_path(project, source)

    extractor = FrameExtractor()
    frame_path = await extractor.extract_frame(video_path, time, output_dir=settings.temp_dir / project_id / 'frames')
    if not masked:
        return FileResponse(frame_path, media_type='image/jpeg')

    manual_payload = None
    if manual_regions:
        try:
            manual_payload = json.loads(manual_regions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail='manual_regions 必须是合法 JSON')
    if manual_payload is None:
        saved_regions = getattr(project, f'{source}_subtitle_regions', None) or []
        manual_payload = [region.model_dump() if hasattr(region, 'model_dump') else region for region in saved_regions]
    effective_mask_mode = mask_mode or getattr(project, 'subtitle_mask_mode', 'hybrid')
    cache_signature = hashlib.md5(
        json.dumps(
            {
                'video_path': video_path,
                'source': source,
                'time': round(float(time), 2),
                'mask_mode': effective_mask_mode,
                'manual_regions': manual_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()[:16]
    masked_dir = settings.temp_dir / project_id / 'masked_frames'
    masked_dir.mkdir(parents=True, exist_ok=True)
    masked_path = masked_dir / f'{source}_{cache_signature}.jpg'
    if masked_path.exists():
        return FileResponse(masked_path, media_type='image/jpeg')

    def _render_masked_frame() -> Path:
        import cv2

        image = read_image_unicode(frame_path)
        if image is None:
            raise ValueError(f'Cannot read frame image: {frame_path}')
        masker = SubtitleMasker(
            manual_regions=manual_payload,
            mask_mode=effective_mask_mode,
        )
        if masker.uses_auto_detection:
            masker.detect_fixed_regions(video_path, allow_fallback=True)
        processed = masker.process_frame(image, frame_time=float(time))
        if not write_image_unicode(masked_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f'Cannot write masked frame image: {masked_path}')
        return masked_path

    rendered_path = await asyncio.to_thread(_render_masked_frame)
    return FileResponse(rendered_path, media_type='image/jpeg')


@router.get('/{project_id}/source-info', response_model=VideoSourceInfo)
async def get_source_info(project_id: str, source: str):
    from core.video_processor.frame_extractor import FrameExtractor

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    video_path = _get_video_path(project, source)

    extractor = FrameExtractor()
    info = await extractor.get_video_info(video_path)

    changed = False
    if source == 'movie':
        resolution = (info['width'], info['height'])
        changed = (
            project.movie_duration != info['duration']
            or project.movie_fps != info['fps']
            or project.movie_resolution != resolution
        )
        project.movie_duration = info['duration']
        project.movie_fps = info['fps']
        project.movie_resolution = resolution
    else:
        resolution = (info['width'], info['height'])
        changed = (
            project.narration_duration != info['duration']
            or project.narration_fps != info['fps']
            or project.narration_resolution != resolution
        )
        project.narration_duration = info['duration']
        project.narration_fps = info['fps']
        project.narration_resolution = resolution
    if changed:
        save_project(project)

    return VideoSourceInfo(
        source=source,
        path=video_path,
        duration=float(info['duration'] or 0.0),
        fps=float(info['fps'] or 0.0),
        width=int(info['width'] or 0),
        height=int(info['height'] or 0),
    )


@router.get('/{project_id}/creative-plan')
async def get_creative_plan(project_id: str):
    from core.composition.creative_planner import CreativePlanner

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    planner = CreativePlanner(template=getattr(project, 'creative_template', 'story_mix'))
    plan = planner.build(project)
    segment_map = {segment.id: segment for segment in project.segments}

    counts = {'exact': 0, 'inferred': 0, 'fallback': 0}
    unit_count = 0
    segments_payload = []

    for item in plan.segments:
        segment = segment_map.get(item.segment_id)
        counts[item.match_type] = counts.get(item.match_type, 0) + 1
        unit_count += len(item.units)
        segments_payload.append(
            {
                'segment_id': item.segment_id,
                'index': segment.index if segment else None,
                'match_type': item.match_type,
                'summary_text': item.summary_text,
                'notes': list(item.notes),
                'duration_us': item.duration_us,
                'match_confidence': segment.match_confidence if segment else 0.0,
                'evidence_summary': segment.evidence_summary if segment else '',
                'units': [
                    {
                        'unit_type': unit.unit_type,
                        'timeline_start_us': unit.timeline_start_us,
                        'duration_us': unit.duration_us,
                        'source_start': unit.source_start,
                        'source_end': unit.source_end,
                        'text': unit.text,
                        'label': unit.label,
                    }
                    for unit in item.units
                ],
            }
        )

    return {
        'project_id': project.id,
        'template': getattr(project, 'creative_template', 'story_mix'),
        'default_export_mode': getattr(project, 'default_export_mode', ExportMode.CREATIVE_DRAFT),
        'total_duration_us': plan.total_duration_us,
        'segment_count': len(plan.segments),
        'unit_count': unit_count,
        'counts': counts,
        'segments': segments_payload,
    }


@router.get('/{project_id}/segment/{segment_id}/audio')
async def get_segment_audio(project_id: str, segment_id: str, source: str = 'tts'):
    from core.audio_processor.audio_extractor import AudioExtractor

    project, segment = _get_project_and_segment(project_id, segment_id)

    if source == 'tts':
        if not segment.tts_audio_path or not Path(segment.tts_audio_path).exists():
            raise HTTPException(status_code=404, detail='TTS 音频不存在')
        return FileResponse(segment.tts_audio_path, media_type='audio/wav')

    extractor = AudioExtractor()
    audio_path = await extractor.extract_segment(
        project.narration_path,
        segment.narration_start,
        segment.narration_end,
        output_dir=settings.temp_dir / project_id / 'audio',
    )
    return FileResponse(audio_path, media_type='audio/wav')


@router.get('/{project_id}/segment/{segment_id}/video')
async def get_segment_video(project_id: str, segment_id: str, source: str = 'narration'):
    from core.video_processor.video_clipper import VideoClipper

    project, segment = _get_project_and_segment(project_id, segment_id)
    clipper = VideoClipper()

    if source == 'narration':
        video_path = await clipper.clip(
            project.narration_path,
            segment.narration_start,
            segment.narration_end,
            output_dir=settings.temp_dir / project_id / 'clips',
        )
    else:
        if segment.movie_start is None or segment.movie_end is None:
            raise HTTPException(status_code=400, detail='该片段尚未匹配到电影片段')
        video_path = await clipper.clip(
            project.movie_path,
            segment.movie_start,
            segment.movie_end,
            output_dir=settings.temp_dir / project_id / 'clips',
        )

    return FileResponse(video_path, media_type='video/mp4')


@router.post('/{project_id}/export/jianying')
async def export_to_jianying(project_id: str, mode: str | None = None):
    from api.routes.settings import load_settings
    from core.exporter.jianying_exporter import JianyingExporter

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    app_settings = load_settings()
    audio_source = getattr(app_settings.export, 'audio_source', 'original')
    export_mode = mode or getattr(project, 'default_export_mode', ExportMode.RESTORE_DRAFT)
    export_mode_value = export_mode.value if isinstance(export_mode, ExportMode) else str(export_mode)
    matched_count = sum(
        1
        for segment in project.segments
        if segment.use_segment and segment.movie_start is not None and segment.movie_end is not None
    )
    if export_mode_value == ExportMode.RESTORE_DRAFT.value and project.segments and matched_count == 0:
        raise HTTPException(
            status_code=400,
            detail='还没有完成画面匹配，不能导出剪映草稿。请先点击“开始处理/重新匹配”，等状态进入待润色或完成后再导出。',
        )

    if audio_source == 'tts':
        missing = [segment.id for segment in project.segments if segment.use_segment and (not segment.tts_audio_path or not Path(segment.tts_audio_path).exists())]
        if missing:
            logger.warning(f'导出剪映：{len(missing)} 个片段缺少 TTS，将自动使用原始解说音频替代')

    exporter = JianyingExporter(
        drafts_dir=app_settings.export.jianying_drafts_dir,
        output_fps=app_settings.export.output_fps,
        output_resolution=app_settings.export.output_resolution,
        audio_source=audio_source,
        min_playback_speed=app_settings.export.min_playback_speed,
        max_playback_speed=app_settings.export.max_playback_speed,
    )
    draft_path = await exporter.export(project, export_mode=export_mode)
    project.last_jianying_draft_path = str(draft_path)
    save_project(project)
    return {'message': '导出成功', 'draft_path': str(draft_path), 'mode': export_mode_value}


@router.post('/{project_id}/export/material-basket')
async def export_material_basket(project_id: str):
    from core.composition.creative_planner import CreativePlanner

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    planner = CreativePlanner(template=getattr(project, 'creative_template', 'story_mix'))
    plan = planner.build(project)
    export_dir = settings.temp_dir / project_id / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    basket_path = export_dir / 'material_basket.json'

    payload = {
        'project_id': project.id,
        'project_name': project.name,
        'template': getattr(project, 'creative_template', 'story_mix'),
        'movie_path': project.movie_path,
        'narration_path': project.narration_path,
        'total_duration_us': plan.total_duration_us,
        'segments': [
            {
                'segment_id': item.segment_id,
                'match_type': item.match_type,
                'summary_text': item.summary_text,
                'notes': item.notes,
                'units': [
                    {
                        'unit_type': unit.unit_type,
                        'timeline_start_us': unit.timeline_start_us,
                        'duration_us': unit.duration_us,
                        'source_start': unit.source_start,
                        'source_end': unit.source_end,
                        'text': unit.text,
                        'label': unit.label,
                    }
                    for unit in item.units
                ],
            }
            for item in plan.segments
        ],
    }
    with open(basket_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return {'message': '导出成功', 'basket_path': str(basket_path)}


@router.post('/{project_id}/export/subtitle')
async def export_subtitles(project_id: str, format: str = 'srt'):
    from core.exporter.subtitle_exporter import SubtitleExporter

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    exporter = SubtitleExporter()
    subtitle_path = await exporter.export(project, format=format, output_dir=settings.temp_dir / project_id)
    media_type = 'application/x-subrip' if format == 'srt' else 'text/x-ass'
    return FileResponse(subtitle_path, media_type=media_type)


@router.get('/{project_id}/export/report')
async def export_match_report(project_id: str):
    from core.exporter.match_report_exporter import MatchReportExporter

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    exporter = MatchReportExporter()
    try:
        report_path = await exporter.export(
            project,
            output_dir=settings.temp_dir / project_id / 'exports' / 'report',
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(report_path, media_type='text/html')


@router.get('/{project_id}/export/davinci')
async def export_davinci_xml(project_id: str):
    from core.exporter.davinci_xml_exporter import DaVinciXMLExporter

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    exporter = DaVinciXMLExporter()
    try:
        xml_path = await exporter.export(
            project,
            output_dir=settings.temp_dir / project_id / 'exports' / 'davinci',
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(xml_path, media_type='application/xml', filename=xml_path.name)
