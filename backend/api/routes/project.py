"""Project management routes."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from config import settings
from models.project import (
    Project,
    ProjectCreate,
    ProcessingProgress,
    ProjectStatus,
    ProjectSummary,
    SubtitleRegionsUpdate,
)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}

router = APIRouter()
_projects: dict[str, Project] = {}
ACTIVE_PROJECT_STATUSES = {
    ProjectStatus.ANALYZING,
    ProjectStatus.RECOGNIZING,
    ProjectStatus.MATCHING,
}


def get_project_file_path(project_id: str) -> Path:
    return settings.projects_dir / f"{project_id}.json"


def save_project(project: Project) -> None:
    project.updated_at = datetime.now()
    file_path = get_project_file_path(project.id)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(project.to_dict(), ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=file_path.parent,
        prefix=f"{file_path.stem}.",
        suffix='.tmp',
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    for attempt in range(8):
        try:
            os.replace(temp_path, file_path)
            break
        except PermissionError:
            if attempt == 7:
                try:
                    temp_path.unlink(missing_ok=True)
                finally:
                    raise
            time.sleep(0.05 * (attempt + 1))
    _projects[project.id] = project


def load_project(project_id: str) -> Optional[Project]:
    if project_id in _projects:
        return _projects[project_id]

    file_path = get_project_file_path(project_id)
    if not file_path.exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    project = Project.from_dict(data)
    _projects[project.id] = project
    return project


def recover_stale_project(project: Project, reason: str = "任务因服务重启或异常中断，请重新点击开始处理或重匹配。") -> bool:
    if project.status not in ACTIVE_PROJECT_STATUSES:
        return False

    matched_count = len([segment for segment in project.segments if segment.movie_start is not None and segment.movie_end is not None])
    if project.status == ProjectStatus.MATCHING and matched_count > 0:
        project.status = ProjectStatus.COMPLETED
        project.progress = ProcessingProgress(
            stage="completed",
            progress=100.0,
            message=f"上次匹配在服务重启时中断，已保留 {matched_count} 个已匹配片段；可继续重匹配或导出草稿。",
        )
    else:
        project.status = ProjectStatus.ERROR
        project.progress = ProcessingProgress(
            stage="error",
            progress=0.0,
            message=reason,
        )
    save_project(project)
    return True


def validate_video_file(path: Path, name: str) -> None:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"{name} 格式不支持: {path.suffix}")

    try:
        if path.stat().st_size <= 0:
            raise HTTPException(status_code=400, detail=f"{name} 文件为空: {path}")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"{name} 无法访问: {path}") from exc

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            logger.warning(f"Skipped strict open validation for {name}: {path}")
        capture.release()
    except ImportError:
        logger.warning(f"OpenCV not available, skipped validation for {name}: {path}")


@router.get('/list', response_model=list[ProjectSummary])
async def list_projects():
    projects: list[ProjectSummary] = []
    settings.projects_dir.mkdir(parents=True, exist_ok=True)

    for file_path in settings.projects_dir.glob('*.json'):
        if file_path.name == 'settings.json':
            continue
        if "." in file_path.stem:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            project = Project.from_dict(data)
            projects.append(
                ProjectSummary(
                    id=project.id,
                    name=project.name,
                    status=project.status,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                    movie_path=project.movie_path,
                    narration_path=project.narration_path,
                    segment_count=len(project.segments),
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to load project file {file_path}: {exc}")

    projects.sort(key=lambda item: item.updated_at, reverse=True)
    return projects


@router.post('/create', response_model=Project)
async def create_project(request: ProjectCreate):
    name = request.name.strip()
    movie_path_str = request.movie_path.strip()
    narration_path_str = request.narration_path.strip()
    reference_audio_path = request.reference_audio_path.strip() if request.reference_audio_path else None
    tts_reference_audio_path = request.tts_reference_audio_path.strip() if request.tts_reference_audio_path else None
    subtitle_path = request.subtitle_path.strip() if request.subtitle_path else None

    if not name:
        raise HTTPException(status_code=400, detail='项目名称不能为空')
    if not movie_path_str:
        raise HTTPException(status_code=400, detail='请选择原电影文件')
    if not narration_path_str:
        raise HTTPException(status_code=400, detail='请选择解说视频文件')

    movie_path = Path(movie_path_str)
    narration_path = Path(narration_path_str)

    if not movie_path.exists():
        raise HTTPException(status_code=400, detail=f"原电影文件不存在: {movie_path_str}")
    if not narration_path.exists():
        raise HTTPException(status_code=400, detail=f"解说视频文件不存在: {narration_path_str}")

    await asyncio.gather(
        asyncio.to_thread(validate_video_file, movie_path, '原电影'),
        asyncio.to_thread(validate_video_file, narration_path, '解说视频'),
    )

    for optional_path, label in [
        (reference_audio_path, '参考音频'),
        (tts_reference_audio_path, 'TTS 参考音频'),
        (subtitle_path, '字幕文件'),
    ]:
        if optional_path and not Path(optional_path).exists():
            raise HTTPException(status_code=400, detail=f"{label}不存在: {optional_path}")

    if subtitle_path and Path(subtitle_path).suffix.lower() != '.srt':
        raise HTTPException(status_code=400, detail='当前仅支持 SRT 字幕文件')

    project = Project(
        id=f"proj_{uuid.uuid4().hex[:12]}",
        name=name,
        status=ProjectStatus.CREATED,
        movie_path=movie_path_str,
        narration_path=narration_path_str,
        reference_audio_path=reference_audio_path,
        tts_reference_audio_path=tts_reference_audio_path,
        subtitle_path=subtitle_path,
    )
    save_project(project)
    logger.info(f"Created project {project.id}: {project.name}")
    return project


@router.get('/{project_id}', response_model=Project)
async def get_project(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
    return project


@router.put('/{project_id}', response_model=Project)
async def update_project(project_id: str, updates: dict):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    for key, value in updates.items():
        if hasattr(project, key):
            setattr(project, key, value)

    save_project(project)
    return project


@router.put('/{project_id}/subtitle-regions', response_model=Project)
async def update_subtitle_regions(project_id: str, payload: SubtitleRegionsUpdate):
    """Update project-level manual subtitle regions."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    project.subtitle_mask_mode = payload.subtitle_mask_mode
    project.narration_subtitle_regions = list(payload.narration_subtitle_regions)
    project.movie_subtitle_regions = list(payload.movie_subtitle_regions)
    save_project(project)
    return project


@router.delete('/{project_id}')
async def delete_project(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    file_path = get_project_file_path(project_id)
    if file_path.exists():
        file_path.unlink()
    _projects.pop(project_id, None)
    logger.info(f"Deleted project {project_id}")
    return {'message': '项目已删除'}


@router.post('/{project_id}/duplicate', response_model=Project)
async def duplicate_project(project_id: str, new_name: Optional[str] = None):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    copy = project.model_copy(deep=True)
    copy.id = f"proj_{uuid.uuid4().hex[:12]}"
    copy.name = new_name or f"{project.name} - 副本"
    copy.created_at = datetime.now()
    copy.updated_at = datetime.now()
    save_project(copy)
    logger.info(f"Duplicated project {project_id} -> {copy.id}")
    return copy
