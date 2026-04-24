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

router = APIRouter()


class VisualAuditRequest(BaseModel):
    draft_path: str | None = Field(default=None, description="draft_content.json path or Jianying draft directory")
    step: float = Field(default=1.0, ge=0.1, le=10.0)
    threshold: float = Field(default=0.66, ge=0.0, le=1.0)
    crop_ratio: float = Field(default=0.76, ge=0.2, le=1.0)
    width: int = Field(default=320, ge=80, le=1920)
    max_time: float | None = Field(default=None, ge=1.0)
    metric: str = Field(default="identity")
    dino_model: str = Field(default="dinov2_vits14")


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
