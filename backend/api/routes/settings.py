"""Application settings routes."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, get_args, get_origin

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, ValidationError

from config import settings as app_config
from models.settings import AISettings, AppSettings, ExportSettings, MatchSettings, TTSSettings, UISettings

router = APIRouter()
SETTINGS_FILE = app_config.projects_dir / 'settings.json'


def _resolve_ffmpeg() -> str | None:
    local_tools = Path(__file__).resolve().parents[3] / '.tools'
    local_candidates = sorted(local_tools.glob('ffmpeg-*essentials_build/bin/ffmpeg.exe'))
    if local_candidates:
        return str(local_candidates[-1])
    return shutil.which('ffmpeg')


def _migrate_settings(app_settings: AppSettings) -> bool:
    changed = False
    if app_settings.ai.temperature == 0.7:
        app_settings.ai.temperature = 0.4
        changed = True
    if not getattr(app_settings.ai, 'polish_style_preset', None):
        app_settings.ai.polish_style_preset = 'movie_pro'
        changed = True
    return changed


def load_settings() -> AppSettings:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            app_settings = AppSettings.model_validate(data)
            if _migrate_settings(app_settings):
                save_settings(app_settings)
            return app_settings
        except Exception as exc:
            logger.warning(f'Failed to load settings: {exc}')
    return AppSettings()


def save_settings(app_settings: AppSettings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as handle:
        json.dump(app_settings.model_dump(mode='json'), handle, ensure_ascii=False, indent=2)


def _allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return annotation is type(None)
    return any(arg is type(None) for arg in get_args(annotation))


def _normalize_for_model(payload: Any, model_type: type[BaseModel]) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized: dict[str, Any] = {}
    for field_name, field_info in model_type.model_fields.items():
        if field_name not in payload:
            continue

        value = payload[field_name]
        annotation = field_info.annotation

        if isinstance(value, dict) and 'path' in value and (annotation is str or _allows_none(annotation)):
            value = value.get('path')

        if isinstance(value, dict) and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            normalized[field_name] = _normalize_for_model(value, annotation)
            continue

        if value is None:
            if _allows_none(annotation):
                normalized[field_name] = None
            continue

        if isinstance(value, str) and value.strip() == '' and _allows_none(annotation):
            normalized[field_name] = None
            continue

        normalized[field_name] = value

    return normalized


def _merge_nested_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_settings_payload(payload: dict[str, Any], model_type: type[BaseModel], current_value: BaseModel):
    normalized = _normalize_for_model(payload or {}, model_type)
    merged_payload = _merge_nested_dicts(current_value.model_dump(mode='python'), normalized)
    try:
        return model_type.model_validate(merged_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@router.get('', response_model=AppSettings)
async def get_settings():
    return load_settings()


@router.put('', response_model=AppSettings)
async def update_settings(payload: dict[str, Any]):
    app_settings = _validate_settings_payload(payload, AppSettings, load_settings())
    save_settings(app_settings)
    return app_settings


@router.get('/ai', response_model=AISettings)
async def get_ai_settings():
    return load_settings().ai


@router.put('/ai', response_model=AISettings)
async def update_ai_settings(payload: dict[str, Any]):
    app_settings = load_settings()
    ai_settings = _validate_settings_payload(payload, AISettings, app_settings.ai)
    app_settings.ai = ai_settings
    save_settings(app_settings)
    return ai_settings


@router.get('/tts', response_model=TTSSettings)
async def get_tts_settings():
    return load_settings().tts


@router.put('/tts', response_model=TTSSettings)
async def update_tts_settings(payload: dict[str, Any]):
    app_settings = load_settings()
    tts_settings = _validate_settings_payload(payload, TTSSettings, app_settings.tts)
    app_settings.tts = tts_settings
    save_settings(app_settings)
    return tts_settings


@router.get('/match', response_model=MatchSettings)
async def get_match_settings():
    return load_settings().match


@router.put('/match', response_model=MatchSettings)
async def update_match_settings(payload: dict[str, Any]):
    app_settings = load_settings()
    match_settings = _validate_settings_payload(payload, MatchSettings, app_settings.match)
    app_settings.match = match_settings
    save_settings(app_settings)
    return match_settings


@router.get('/export', response_model=ExportSettings)
async def get_export_settings():
    return load_settings().export


@router.put('/export', response_model=ExportSettings)
async def update_export_settings(payload: dict[str, Any]):
    app_settings = load_settings()
    export_settings = _validate_settings_payload(payload, ExportSettings, app_settings.export)
    app_settings.export = export_settings
    save_settings(app_settings)
    return export_settings


@router.get('/ui', response_model=UISettings)
async def get_ui_settings():
    return load_settings().ui


@router.put('/ui', response_model=UISettings)
async def update_ui_settings(payload: dict[str, Any]):
    app_settings = load_settings()
    ui_settings = _validate_settings_payload(payload, UISettings, app_settings.ui)
    app_settings.ui = ui_settings
    save_settings(app_settings)
    return ui_settings


@router.post('/ai/test')
async def test_ai_connection():
    from core.ai_service.api_tester import APITester

    app_settings = load_settings()
    tester = APITester()
    return await tester.test_ai_api(
        api_base=app_settings.ai.api_base,
        api_key=app_settings.ai.api_key,
        model=app_settings.ai.model,
    )


@router.post('/tts/test')
async def test_tts_connection():
    from core.tts_service.tts_client import TTSClient

    app_settings = load_settings()
    client = TTSClient(api_base=app_settings.tts.api_base, api_endpoint=app_settings.tts.api_endpoint)
    return await client.test_connection()


@router.get('/check_ffmpeg')
async def check_ffmpeg():
    ffmpeg_path = _resolve_ffmpeg()
    if not ffmpeg_path:
        return {'installed': False, 'version': None, 'path': None}

    try:
        result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0] if result.stdout else ''
            return {'installed': True, 'version': version_line, 'path': ffmpeg_path}
    except Exception as exc:
        logger.warning(f'FFmpeg check failed: {exc}')
    return {'installed': False, 'version': None, 'path': ffmpeg_path}
