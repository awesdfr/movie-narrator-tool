"""File browsing routes."""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from config import settings

router = APIRouter()

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
SUBTITLE_EXTENSIONS = {'.srt'}


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    size_display: str
    modified_time: str


def format_file_size(size_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} {units[-1]}"  # fallback，理论上由循环覆盖


def list_files(directory: Path, extensions: set[str]) -> list[FileInfo]:
    directory.mkdir(parents=True, exist_ok=True)
    files: list[FileInfo] = []
    for file_path in directory.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in extensions:
            continue
        try:
            stat = file_path.stat()
            files.append(
                FileInfo(
                    name=file_path.name,
                    path=str(file_path.resolve()),
                    size=stat.st_size,
                    size_display=format_file_size(stat.st_size),
                    modified_time=datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to read file info for {file_path}: {exc}")
    files.sort(key=lambda item: item.modified_time, reverse=True)
    return files


@router.get('/movies', response_model=list[FileInfo])
async def list_movies():
    return list_files(settings.videos_dir / 'movies', VIDEO_EXTENSIONS)


@router.get('/narrations', response_model=list[FileInfo])
async def list_narrations():
    return list_files(settings.videos_dir / 'narrations', VIDEO_EXTENSIONS)


@router.get('/reference_audio', response_model=list[FileInfo])
async def list_reference_audio():
    return list_files(settings.videos_dir / 'reference_audio', AUDIO_EXTENSIONS)


@router.get('/subtitles', response_model=list[FileInfo])
async def list_subtitles():
    return list_files(settings.videos_dir / 'subtitles', SUBTITLE_EXTENSIONS)


@router.post('/open_folder')
async def open_folder(folder_type: Literal['movies', 'narrations', 'reference_audio', 'subtitles']):
    folder_map = {
        'movies': settings.videos_dir / 'movies',
        'narrations': settings.videos_dir / 'narrations',
        'reference_audio': settings.videos_dir / 'reference_audio',
        'subtitles': settings.videos_dir / 'subtitles',
    }
    folder_path = folder_map[folder_type]
    folder_path.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.Popen(f'explorer "{folder_path}"', shell=True)
        return {'message': f'已打开文件夹: {folder_path}'}
    except Exception as exc:
        logger.error(f"Failed to open folder {folder_path}: {exc}")
        raise HTTPException(status_code=500, detail=f'打开文件夹失败: {exc}')


@router.post('/validate_video')
async def validate_video(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f'文件不存在: {path}')
    if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'不支持的视频格式: {file_path.suffix}')

    try:
        from core.video_processor.analysis_video import ensure_analysis_video_sync
        import cv2

        capture = cv2.VideoCapture(ensure_analysis_video_sync(str(file_path)))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail='无法打开视频文件，文件可能已损坏')

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        capture.release()

        return {
            'valid': True,
            'info': {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'duration_display': f"{int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}",
            },
        }
    except ImportError:
        return {'valid': True, 'info': None}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Video validation failed for {file_path}: {exc}")
        raise HTTPException(status_code=400, detail=f'视频验证失败: {exc}')


@router.get('/videos_dir')
async def get_videos_dir():
    return {
        'videos_dir': str(settings.videos_dir),
        'movies_dir': str(settings.videos_dir / 'movies'),
        'narrations_dir': str(settings.videos_dir / 'narrations'),
        'reference_audio_dir': str(settings.videos_dir / 'reference_audio'),
        'subtitles_dir': str(settings.videos_dir / 'subtitles'),
    }
