"""Minimal video-matching API for external integrations."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.routes.preview import export_to_jianying
from api.routes.process_v2 import _processing_tasks, process_project_task
from api.routes.project import load_project, save_project, validate_video_file
from models.project import ProcessingProgress, Project, ProjectStatus, SubtitleRegion

router = APIRouter()


class MatchJobCreate(BaseModel):
    """Payload for starting a movie/narration visual matching job."""

    movie_path: str = Field(..., min_length=1, description="Original movie file path")
    narration_path: str = Field(..., min_length=1, description="Narration/edited video file path")
    name: Optional[str] = Field(default=None, description="Optional project name")
    subtitle_path: Optional[str] = Field(default=None, description="Optional SRT path to skip ASR")
    narration_subtitle_regions: list[SubtitleRegion] = Field(default_factory=list)
    movie_subtitle_regions: list[SubtitleRegion] = Field(default_factory=list)


@router.post("/jobs")
async def create_match_job(payload: MatchJobCreate):
    """Create a project and start visual matching immediately."""

    movie_path = Path(payload.movie_path)
    narration_path = Path(payload.narration_path)
    if not movie_path.exists():
        raise HTTPException(status_code=400, detail=f"原电影文件不存在: {movie_path}")
    if not narration_path.exists():
        raise HTTPException(status_code=400, detail=f"解说视频文件不存在: {narration_path}")

    await asyncio.gather(
        asyncio.to_thread(validate_video_file, movie_path, "原电影"),
        asyncio.to_thread(validate_video_file, narration_path, "解说视频"),
    )

    if payload.subtitle_path:
        subtitle_path = Path(payload.subtitle_path)
        if not subtitle_path.exists():
            raise HTTPException(status_code=400, detail=f"字幕文件不存在: {subtitle_path}")
        if subtitle_path.suffix.lower() != ".srt":
            raise HTTPException(status_code=400, detail="当前仅支持 SRT 字幕文件")

    project = Project(
        id=f"proj_{uuid.uuid4().hex[:12]}",
        name=payload.name or movie_path.stem,
        status=ProjectStatus.ANALYZING,
        movie_path=str(movie_path),
        narration_path=str(narration_path),
        subtitle_path=payload.subtitle_path,
        narration_subtitle_regions=payload.narration_subtitle_regions,
        movie_subtitle_regions=payload.movie_subtitle_regions,
        progress=ProcessingProgress(stage="analyzing", progress=0, message="Queued video matching job"),
    )
    save_project(project)

    task = asyncio.create_task(process_project_task(project.id))
    _processing_tasks[project.id] = task
    return {"project_id": project.id, "status": project.status, "progress": project.progress}


@router.get("/jobs/{project_id}")
async def get_match_job(project_id: str):
    """Return project status and summary for a matching job."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    matched = sum(1 for item in project.segments if item.movie_start is not None and item.movie_end is not None)
    return {
        "project_id": project.id,
        "name": project.name,
        "status": project.status,
        "progress": project.progress,
        "movie_path": project.movie_path,
        "narration_path": project.narration_path,
        "segments_total": len(project.segments),
        "segments_matched": matched,
        "visual_audit_score": project.visual_audit_score,
        "visual_audit_below_threshold": project.visual_audit_below_threshold,
        "last_jianying_draft_path": project.last_jianying_draft_path,
    }


@router.get("/jobs/{project_id}/segments")
async def get_match_segments(project_id: str):
    """Return the matched timeline segments."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.segments


@router.post("/jobs/{project_id}/export/jianying")
async def export_match_to_jianying(project_id: str):
    """Export the matched movie timeline as a Jianying draft."""

    return await export_to_jianying(project_id, mode="restore_draft")
