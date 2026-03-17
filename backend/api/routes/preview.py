"""Preview and export routes."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from api.routes.project import load_project, save_project
from config import settings
from models.project import VideoSourceInfo

router = APIRouter()


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
    from core.video_processor.frame_extractor import FrameExtractor
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

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f'Cannot read frame image: {frame_path}')
        masker = SubtitleMasker(
            manual_regions=manual_payload,
            mask_mode=effective_mask_mode,
        )
        if masker.uses_auto_detection:
            masker.detect_fixed_regions(video_path, allow_fallback=True)
        processed = masker.process_frame(image, frame_time=float(time))
        cv2.imwrite(str(masked_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
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
async def export_to_jianying(project_id: str):
    from api.routes.settings import load_settings
    from core.exporter.jianying_exporter import JianyingExporter

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    app_settings = load_settings()
    audio_source = getattr(app_settings.export, 'audio_source', 'original')

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
    draft_path = await exporter.export(project)
    return {'message': '导出成功', 'draft_path': str(draft_path)}


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
