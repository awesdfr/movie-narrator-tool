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


@router.get('/subtitles', response_model=list[FileInfo])
async def list_subtitles():
    return list_files(settings.videos_dir / 'subtitles', SUBTITLE_EXTENSIONS)


@router.post('/open_folder')
async def open_folder(folder_type: Literal['movies', 'narrations', 'subtitles']):
    folder_map = {
        'movies': settings.videos_dir / 'movies',
        'narrations': settings.videos_dir / 'narrations',
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
