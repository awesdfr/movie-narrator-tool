"""Project management routes."""
from __future__ import annotations

import json
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
    ProjectStatus,
    ProjectSummary,
    SubtitleRegionsUpdate,
)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}

router = APIRouter()
_projects: dict[str, Project] = {}


def get_project_file_path(project_id: str) -> Path:
    return settings.projects_dir / f"{project_id}.json"


def save_project(project: Project) -> None:
    project.updated_at = datetime.now()
    file_path = get_project_file_path(project.id)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as handle:
        json.dump(project.to_dict(), handle, ensure_ascii=False, indent=2)
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


def validate_video_file(path: Path, name: str) -> None:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"{name} 格式不支持: {path.suffix}")

    try:
        from core.video_processor.analysis_video import ensure_analysis_video_sync
        import cv2

        capture = cv2.VideoCapture(ensure_analysis_video_sync(str(path)))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail=f"{name} 无法打开，文件可能已损坏: {path}")
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
    movie_path = Path(request.movie_path)
    narration_path = Path(request.narration_path)

    if not movie_path.exists():
        raise HTTPException(status_code=400, detail=f"原电影文件不存在: {request.movie_path}")
    if not narration_path.exists():
        raise HTTPException(status_code=400, detail=f"解说视频文件不存在: {request.narration_path}")

    validate_video_file(movie_path, '原电影')
    validate_video_file(narration_path, '解说视频')

    for optional_path, label in [
        (request.reference_audio_path, '参考音频'),
        (request.tts_reference_audio_path, 'TTS 参考音频'),
        (request.subtitle_path, '字幕文件'),
    ]:
        if optional_path and not Path(optional_path).exists():
            raise HTTPException(status_code=400, detail=f"{label}不存在: {optional_path}")

    if request.subtitle_path and Path(request.subtitle_path).suffix.lower() != '.srt':
        raise HTTPException(status_code=400, detail='当前仅支持 SRT 字幕文件')

    project = Project(
        id=f"proj_{uuid.uuid4().hex[:12]}",
        name=request.name,
        status=ProjectStatus.CREATED,
        movie_path=request.movie_path,
        narration_path=request.narration_path,
        reference_audio_path=request.reference_audio_path,
        tts_reference_audio_path=request.tts_reference_audio_path,
        subtitle_path=request.subtitle_path,
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
