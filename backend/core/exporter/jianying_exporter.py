"""鍓槧鑽夌瀵煎嚭妯″潡

鐢熸垚鍓槧Pro鍙瘑鍒殑鑽夌鏂囦欢
"""
import bisect
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings
from core.composition.creative_planner import CreativePlan, CreativePlanner, CreativeUnit
from core.matcher.shot_continuity import plan_shot_continuity
from models.project import ExportMode, Project
from models.segment import Segment, SegmentType, compute_segment_duration


class JianyingExporter:
    """鍓槧鑽夌瀵煎嚭鍣?
    鐢熸垚鍓槧Pro Drafts鏍煎紡鐨勮崏绋挎枃浠?
    杞ㄩ亾缁撴瀯:
    - 杞ㄩ亾1: 鎴愬搧瑙嗛 + TTS闊抽
    - 杞ㄩ亾2: 鍘熻В璇磋棰戯紙瀹屾暣锛?    - 杞ㄩ亾3: 琚壀鎺夌殑鍘熺數褰辩墖娈?    """

    # 鏃堕棿鍗曚綅杞崲锛堝壀鏄犱娇鐢ㄥ井绉掞級
    TIME_SCALE = 1_000_000

    def __init__(
        self,
        drafts_dir: Optional[Path] = None,
        output_fps: int = 0,
        output_resolution: str = "original",
        audio_source: str = "original",
        min_playback_speed: float = 0.5,
        max_playback_speed: float = 2.0
    ):
        """鍒濆鍖?
        Args:
            drafts_dir: 鍓槧鑽夌鐩綍
            output_fps: 杈撳嚭甯х巼锛?琛ㄧず淇濇寔鍘熷
            output_resolution: 杈撳嚭鍒嗚鲸鐜?            audio_source: 闊抽鏉ユ簮 "original"(鍘熷瑙ｈ) 鎴?"tts"(TTS鐢熸垚)
            min_playback_speed: 鏈€灏忔挱鏀鹃€熷害锛堟參鏀句笅闄愶級
            max_playback_speed: 鏈€澶ф挱鏀鹃€熷害锛堝揩鏀句笂闄愶級
        """
        self.drafts_dir = Path(drafts_dir or settings.jianying_drafts_dir)
        self.output_fps = output_fps
        self.output_resolution = output_resolution
        self.audio_source = audio_source
        self.min_playback_speed = min_playback_speed
        self.max_playback_speed = max_playback_speed

        # Parse output resolution.
        if output_resolution and output_resolution != "original":
            parts = output_resolution.split("x")
            self.width = int(parts[0])
            self.height = int(parts[1])
        else:
            self.width = 1920
            self.height = 1080

    async def export(self, project: Project, export_mode: str | ExportMode | None = None) -> Path:
        """瀵煎嚭椤圭洰鍒板壀鏄犺崏绋?
        Args:
            project: 椤圭洰鏁版嵁

        Returns:
            鑽夌鐩綍璺緞
        """
        # 鍒涘缓鑽夌鐩綍
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        draft_name = f"{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        draft_name, draft_dir, staged_dir = self._create_staged_draft_dir(draft_name)
        export_project = self._prepare_project_for_export(project, staged_dir)

        logger.info(f"鍒涘缓鍓槧鑽夌: {draft_dir}")

        # 鐢熸垚鑽夌鍐呭
        resolved_mode = self._resolve_export_mode(project, export_mode)
        draft_content = self._generate_draft_content(export_project, draft_id, resolved_mode)

        # 鍐欏叆鑽夌鏂囦欢
        draft_file = staged_dir / "draft_content.json"
        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

        # Copy material files into the draft directory.
        await self._copy_materials(export_project, staged_dir)

        # Generate draft metadata.
        meta_content = self._generate_meta_content(
            export_project,
            draft_id,
            draft_name,
            resolved_mode,
            draft_content["duration"],
        )
        self._apply_draft_meta_paths(meta_content, draft_dir)
        meta_file = staged_dir / "draft_meta_info.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, ensure_ascii=False, indent=2)

        self._publish_staged_draft(staged_dir, draft_dir)
        self._verify_draft_integrity(draft_dir)
        logger.info(f"鍓槧鑽夌瀵煎嚭瀹屾垚: {draft_dir}")
        return draft_dir

    def _create_staged_draft_dir(self, draft_name: str) -> tuple[str, Path, Path]:
        self.drafts_dir.mkdir(parents=True, exist_ok=True)

        base_name = draft_name
        final_dir = self.drafts_dir / draft_name
        suffix = 1
        while final_dir.exists():
            draft_name = f"{base_name}_{suffix}"
            final_dir = self.drafts_dir / draft_name
            suffix += 1

        staging_root = self.drafts_dir / ".draft_export_staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staged_dir = staging_root / f"{draft_name}_{uuid.uuid4().hex[:8]}"
        staged_dir.mkdir(parents=True, exist_ok=False)
        return draft_name, final_dir, staged_dir

    def _publish_staged_draft(self, staged_dir: Path, draft_dir: Path) -> None:
        if draft_dir.exists():
            raise FileExistsError(f"Jianying draft already exists: {draft_dir}")
        shutil.move(str(staged_dir), str(draft_dir))

    def _apply_draft_meta_paths(self, meta: dict, draft_dir: Path) -> None:
        meta["draft_fold_path"] = draft_dir.as_posix()
        meta["draft_root_path"] = str(self.drafts_dir)
        meta["draft_removable_storage_device"] = draft_dir.drive
        meta["tm_draft_removed"] = 0

    def _verify_draft_integrity(self, draft_dir: Path) -> None:
        content_file = draft_dir / "draft_content.json"
        meta_file = draft_dir / "draft_meta_info.json"
        if not content_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Incomplete Jianying draft: {draft_dir}")

        with open(content_file, "r", encoding="utf-8") as f:
            draft_content = json.load(f)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_content = json.load(f)

        if meta_content.get("tm_draft_removed", 0) != 0:
            raise ValueError(f"Jianying draft is marked removed: {draft_dir}")
        if ".recycle_bin" in str(meta_content.get("draft_fold_path", "")):
            raise ValueError(f"Jianying draft points to recycle bin: {draft_dir}")
        if int(draft_content.get("duration", 0) or 0) <= 0:
            raise ValueError(f"Jianying draft has invalid duration: {draft_dir}")

    def _prepare_project_for_export(self, project: Project, staged_dir: Path) -> Project:
        export_project = project.model_copy(deep=True)
        export_project.movie_path = self._prepare_video_source_for_export(
            export_project.movie_path,
            staged_dir,
            f"movie_{project.id}",
        )
        export_project.narration_path = self._prepare_video_source_for_export(
            export_project.narration_path,
            staged_dir,
            f"narration_{project.id}",
        )
        return export_project

    def _prepare_video_source_for_export(
        self,
        source_path: Optional[str],
        staged_dir: Path,
        label: str,
    ) -> Optional[str]:
        if not source_path:
            return source_path
        path = Path(source_path)
        if not path.exists() or not self._should_normalize_video_source(path):
            return source_path

        normalized_path = self._shared_normalized_video_cache_path(path, label)

        if normalized_path.exists():
            try:
                if normalized_path.stat().st_mtime >= path.stat().st_mtime:
                    logger.info("导出剪映：复用稳定化素材 {}", normalized_path)
                    return str(normalized_path)
            except OSError:
                pass

        temp_output_path = normalized_path.with_suffix(".tmp.mp4")
        try:
            self._normalize_video_source_for_export(path, temp_output_path)
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            temp_output_path.replace(normalized_path)
            logger.info("导出剪映：{} 已稳定化为 {}", path.name, normalized_path.name)
            return str(normalized_path)
        except Exception as exc:
            temp_output_path.unlink(missing_ok=True)
            logger.warning("导出剪映：稳定化 {} 失败，回退原始路径。原因: {}", path, exc)
            return source_path

    def _shared_normalized_video_cache_path(self, path: Path, label: str) -> Path:
        cache_key = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
        cache_dir = settings.temp_dir / "normalized_export_sources"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{label}_{cache_key}.mp4"

    def _should_normalize_video_source(self, path: Path) -> bool:
        if path.suffix.lower() in {".mkv", ".webm"}:
            return True
        format_name = self._probe_video_container(path)
        return bool(format_name) and any(token in format_name for token in ("matroska", "webm"))

    def _probe_video_container(self, path: Path) -> str:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=format_name",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            return ""
        return result.stdout.strip().lower()

    def _normalize_video_source_for_export(self, source_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)

        remux_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(
                remux_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=True,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            logger.warning("导出剪映：快速 remux {} 失败，降级转码。{}", source_path.name, exc)

        transcode_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(
            transcode_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )

    def _resolve_export_mode(self, project: Project, export_mode: str | ExportMode | None) -> ExportMode:
        if export_mode is None:
            export_mode = getattr(project, "default_export_mode", ExportMode.RESTORE_DRAFT)
        if isinstance(export_mode, ExportMode):
            return export_mode
        try:
            return ExportMode(export_mode)
        except ValueError:
            logger.warning(f"Unknown export mode {export_mode}, fallback to restore_draft")
            return ExportMode.RESTORE_DRAFT

    def _generate_draft_content(self, project: Project, draft_id: str, export_mode: ExportMode) -> dict:
        """鐢熸垚鑽夌鍐呭JSON"""
        # Calculate output aspect ratio.
        ratio = f"{self.width}:{self.height}"
        if self.width == 1920 and self.height == 1080:
            ratio = "16:9"
        elif self.width == 1280 and self.height == 720:
            ratio = "16:9"
        elif self.width == 1080 and self.height == 1920:
            ratio = "9:16"

        # 鍩虹缁撴瀯
        content = {
            "canvas_config": {
                "height": self.height,
                "ratio": ratio,
                "width": self.width
            },
            "color_space": 0,
            "config": {
                "adjust_max_index": 1,
                "attachment_info": [],
                "combination_max_index": 1,
                "export_range": None,
                "extract_audio_last_index": 1,
                "lyrics_recognition_id": "",
                "lyrics_sync": True,
                "lyrics_taskinfo": [],
                "maintrack_adsorb": True,
                "material_save_mode": 0,
                "original_sound_last_index": 1,
                "record_audio_last_index": 1,
                "sticker_max_index": 1,
                "subtitle_recognition_id": "",
                "subtitle_sync": True,
                "subtitle_taskinfo": [],
                "system_font_list": [],
                "video_mute": False,
                "zoom_info_params": None
            },
            "cover": None,
            "create_time": int(datetime.now().timestamp() * 1000000),
            "duration": 0,
            "extra_info": None,
            "fps": float(self.output_fps) if self.output_fps > 0 else 30.0,
            "free_render_index_mode_on": False,
            "group_container": None,
            "id": draft_id,
            "keyframe_graph_list": [],
            "keyframes": {"adjusts": [], "audios": [], "effects": [], "filters": [], "handwrites": [], "stickers": [], "texts": [], "videos": []},
            "last_modified_platform": {"app_id": 3704, "app_source": "lv", "app_version": "5.0.0", "device_id": "", "hard_disk_id": "", "mac_address": "", "os": "windows", "os_version": "10.0.19045"},
            "materials": self._generate_materials(project),
            "mutable_config": None,
            "name": project.name,
            "new_version": "92.0.0",
            "platform": {"app_id": 3704, "app_source": "lv", "app_version": "5.0.0", "device_id": "", "hard_disk_id": "", "mac_address": "", "os": "windows", "os_version": "10.0.19045"},
            "relationships": [],
            "render_index_track_mode_on": False,
            "retouch_cover": None,
            "source": "default",
            "static_cover_image_path": "",
            "tracks": [],
            "update_time": int(datetime.now().timestamp() * 1000000),
            "version": 360000
        }

        # Build tracks and duration for the selected export mode.
        if export_mode == ExportMode.CREATIVE_DRAFT:
            creative_plan = self._build_creative_plan(project)
            content["tracks"] = self._generate_creative_tracks(project, content["materials"], creative_plan)
            content["duration"] = creative_plan.total_duration_us
        else:
            content["tracks"] = self._generate_tracks(project)
            content["duration"] = self._calculate_duration(project)

        return content

    def _generate_materials(self, project: Project) -> dict:
        """鐢熸垚绱犳潗鍒楄〃"""
        materials = {
            "audios": [],
            "beats": [],
            "canvases": [],
            "chromas": [],
            "color_curves": [],
            "digital_humans": [],
            "drafts": [],
            "effects": [],
            "flowers": [],
            "green_screens": [],
            "handwrites": [],
            "hsl": [],
            "images": [],
            "log_color_wheels": [],
            "loudnesses": [],
            "manual_deformations": [],
            "masks": [],
            "material_animations": [],
            "material_colors": [],
            "multi_language_refs": [],
            "placeholders": [],
            "plugin_effects": [],
            "primary_color_wheels": [],
            "realtime_denoises": [],
            "shapes": [],
            "smart_crops": [],
            "smart_relights": [],
            "sound_channel_mappings": [],
            "speeds": [],
            "stickers": [],
            "tail_leaders": [],
            "text_templates": [],
            "texts": [],
            "time_marks": [],
            "transitions": [],
            "video_effects": [],
            "video_trackings": [],
            "videos": [],
            "vocal_beautifys": [],
            "vocal_separations": []
        }

        # Add the source movie as a video material when available.
        uses_original_narration_audio = self.audio_source == "original" or any(
            segment.use_segment and not segment.tts_audio_path
            for segment in project.segments
        )
        if project.movie_path:
            movie_material = self._create_video_material(
                project.movie_path,
                f"movie_{project.id}",
                project.movie_duration or 0
            )
            materials["videos"].append(movie_material)

        # 娣诲姞瑙ｈ瑙嗛绱犳潗
        if project.narration_path:
            narration_material = self._create_video_material(
                project.narration_path,
                f"narration_{project.id}",
                project.narration_duration or 0
            )
            materials["videos"].append(narration_material)

            # 娣诲姞鍘熷瑙ｈ闊抽绱犳潗锛堢敤浜庢彁鍙栭煶棰戯級
            if uses_original_narration_audio:
                narration_audio = {
                    "app_id": 0,
                    "category_id": "",
                    "category_name": "",
                    "check_flag": 1,
                    "duration": int((project.narration_duration or 0) * self.TIME_SCALE),
                    "effect_id": "",
                    "formula_id": "",
                    "id": f"narration_audio_{project.id}",
                    "intensifies_path": "",
                    "local_material_id": "",
                    "music_id": "",
                    "name": Path(project.narration_path).name,
                    "path": project.narration_path,
                    "request_id": "",
                    "resource_id": "",
                    "search_id": "",
                    "source_from": "",
                    "source_platform": 0,
                    "team_id": "",
                    "text_id": "",
                    "tone_category_id": "",
                    "tone_category_name": "",
                    "tone_effect_id": "",
                    "tone_effect_name": "",
                    "tone_platform": "",
                    "tone_second_category_id": "",
                    "tone_second_category_name": "",
                    "tone_speaker": "",
                    "tone_type": "",
                    "type": "extract_music",
                    "video_id": "",
                    "wave_points": []
                }
                materials["audios"].append(narration_audio)

        # 娣诲姞TTS闊抽绱犳潗锛堜粎鍦ㄤ娇鐢═TS妯″紡鏃讹級
        if self.audio_source == "tts":
            for segment in project.segments:
                if segment.tts_audio_path and segment.use_segment:
                    audio_material = self._create_audio_material(
                        segment.tts_audio_path,
                        f"tts_{segment.id}",
                        segment.tts_duration or 0
                    )
                    materials["audios"].append(audio_material)

        return materials

    def _create_video_material(self, path: str, material_id: str, duration: float) -> dict:
        """鍒涘缓瑙嗛绱犳潗"""
        return {
            "audio_fade": None,
            "cartoon_path": "",
            "category_id": "",
            "category_name": "local",
            "check_flag": 63487,
            "crop": {"lower_left_x": 0.0, "lower_left_y": 1.0, "lower_right_x": 1.0, "lower_right_y": 1.0, "upper_left_x": 0.0, "upper_left_y": 0.0, "upper_right_x": 1.0, "upper_right_y": 0.0},
            "crop_ratio": "free",
            "crop_scale": 1.0,
            "duration": int(duration * self.TIME_SCALE),
            "extra_type_option": 0,
            "formula_id": "",
            "freeze": None,
            "gameplay": None,
            "has_audio": True,
            "height": self.height,
            "id": material_id,
            "intensifies_audio_path": "",
            "intensifies_path": "",
            "is_ai_generate_content": False,
            "is_unified_beauty_mode": False,
            "local_id": "",
            "local_material_id": "",
            "material_id": material_id,
            "material_name": Path(path).name,
            "material_url": "",
            "matting": {"flag": 0, "has_use_quick_brush": False, "has_use_quick_eraser": False, "interactiveTime": [], "path": "", "strokes": []},
            "media_path": "",
            "object_locked": None,
            "origin_material_id": "",
            "path": path,
            "picture_from": "none",
            "picture_set_category_id": "",
            "picture_set_category_name": "",
            "request_id": "",
            "reverse_intensifies_path": "",
            "reverse_path": "",
            "smart_motion": None,
            "source": 0,
            "source_platform": 0,
            "stable": {"matrix_path": "", "stable_level": 0, "time_range": {"duration": 0, "start": 0}},
            "team_id": "",
            "type": "video",
            "video_algorithm": {"algorithms": [], "deflicker": None, "motion_blur_config": None, "noise_reduction": None, "path": "", "quality_enhance": None, "time_range": None},
            "width": self.width
        }

    def _create_audio_material(self, path: str, material_id: str, duration: float) -> dict:
        """鍒涘缓闊抽绱犳潗"""
        return {
            "app_id": 0,
            "category_id": "",
            "category_name": "local",
            "check_flag": 1,
            "duration": int(duration * self.TIME_SCALE),
            "effect_id": "",
            "formula_id": "",
            "id": material_id,
            "intensifies_path": "",
            "local_material_id": "",
            "music_id": "",
            "name": Path(path).name,
            "path": path,
            "request_id": "",
            "resource_id": "",
            "search_id": "",
            "source_from": "",
            "source_platform": 0,
            "team_id": "",
            "text_id": "",
            "tone_category_id": "",
            "tone_category_name": "",
            "tone_effect_id": "",
            "tone_effect_name": "",
            "tone_platform": "",
            "tone_second_category_id": "",
            "tone_second_category_name": "",
            "tone_speaker": "",
            "tone_type": "",
            "type": "extract_music",
            "video_id": "",
            "wave_points": []
        }

    # 浣庣疆淇″害闃堝€硷紝浣庝簬姝ゅ€肩殑鐗囨鏀惧埌"寰呮鏌?杞ㄩ亾
    def _build_creative_plan(self, project: Project) -> CreativePlan:
        planner = CreativePlanner(
            audio_source=self.audio_source,
            template=getattr(project, "creative_template", "story_mix"),
        )
        return planner.build(project)

    def _generate_creative_tracks(self, project: Project, materials: dict, creative_plan: CreativePlan) -> list:
        segment_map = {segment.id: segment for segment in project.segments}
        tracks = []

        main_video_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": False,
            "name": "鍒涗綔鐢婚潰",
            "segments": [],
            "type": "video",
        }
        main_audio_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": False,
            "name": "瑙ｈ闊抽",
            "segments": [],
            "type": "audio",
        }
        text_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": False,
            "name": "信息卡",
            "segments": [],
            "type": "text",
        }
        reference_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": False,
            "name": "解说参考",
            "segments": [],
            "type": "video",
        }

        for segment_plan in creative_plan.segments:
            segment = segment_map.get(segment_plan.segment_id)
            if not segment:
                continue

            should_use_original = (
                self.audio_source == "original"
                or segment.segment_type == SegmentType.NO_NARRATION
                or not segment.tts_audio_path
            )

            timeline_start_us = segment_plan.timeline_start_us
            duration_s = segment_plan.duration_us / self.TIME_SCALE

            if should_use_original:
                audio_seg = self._create_original_audio_segment(
                    f"narration_audio_{project.id}",
                    project.narration_path,
                    timeline_start_us,
                    segment.narration_start,
                    segment.narration_end,
                )
            elif segment.tts_audio_path and segment.tts_duration:
                audio_seg = self._create_audio_segment(
                    f"tts_{segment.id}",
                    segment.tts_audio_path,
                    timeline_start_us,
                    segment.tts_duration,
                )
            else:
                audio_seg = self._create_original_audio_segment(
                    f"narration_audio_{project.id}",
                    project.narration_path,
                    timeline_start_us,
                    segment.narration_start,
                    segment.narration_end,
                )
            main_audio_track["segments"].append(audio_seg)

            for unit in segment_plan.units:
                if unit.unit_type == "movie_clip":
                    main_video_track["segments"].append(
                        self._create_video_segment(
                            f"movie_{project.id}",
                            project.movie_path,
                            unit.timeline_start_us,
                            float(unit.source_start or 0.0),
                            float(unit.source_end or 0.0),
                            segment.mute_movie_audio,
                            target_duration=unit.duration_us / self.TIME_SCALE,
                            allow_speed_change=self._segment_allows_movie_speed_change(segment),
                        )
                    )
                elif unit.unit_type == "narration_clip":
                    main_video_track["segments"].append(
                        self._create_video_segment(
                            f"narration_{project.id}",
                            project.narration_path,
                            unit.timeline_start_us,
                            float(unit.source_start or segment.narration_start),
                            float(unit.source_end or segment.narration_end),
                            True,
                            target_duration=unit.duration_us / self.TIME_SCALE,
                        )
                    )
                elif unit.unit_type == "text_card" and unit.text:
                    text_id, animation_id = self._create_text_material(
                        materials,
                        unit.text,
                        font_size=10.5 if "待替换画面" in unit.text else 9.0,
                    )
                    text_track["segments"].append(
                        self._create_text_segment(
                            text_id,
                            animation_id,
                            unit.timeline_start_us,
                            unit.duration_us,
                            position_y=-0.12 if "待替换画面" in unit.text else -0.72,
                        )
                    )

            if project.narration_path:
                reference_track["segments"].append(
                    self._create_video_segment(
                        f"narration_{project.id}",
                        project.narration_path,
                        timeline_start_us,
                        segment.narration_start,
                        segment.narration_end,
                        True,
                        visible=False,
                        target_duration=duration_s,
                    )
                )

        tracks.extend([main_video_track, main_audio_track])
        if text_track["segments"]:
            tracks.append(text_track)
        if reference_track["segments"]:
            tracks.append(reference_track)
        return tracks

    def _create_text_material(
        self,
        materials: dict,
        text: str,
        font_size: float = 9.0,
        color: Optional[list[float]] = None,
    ) -> tuple[str, str]:
        if color is None:
            color = [1.0, 1.0, 1.0]

        text_id = str(uuid.uuid4()).upper()
        animation_id = str(uuid.uuid4()).upper()
        content_str = json.dumps(
            {
                "styles": [
                    {
                        "fill": {
                            "alpha": 1.0,
                            "content": {
                                "render_type": "solid",
                                "solid": {"alpha": 1.0, "color": color},
                            },
                        },
                        "range": [0, len(text)],
                        "size": font_size,
                        "bold": False,
                        "italic": False,
                        "underline": False,
                        "strokes": [],
                    }
                ],
                "text": text,
            },
            ensure_ascii=False,
        )

        materials["texts"].append(
            {
                "id": text_id,
                "content": content_str,
                "typesetting": 0,
                "alignment": 1,
                "letter_spacing": 0.0,
                "line_spacing": 0.02,
                "line_feed": 1,
                "line_max_width": 0.82,
                "force_apply_line_max_width": False,
                "check_flag": 7,
                "type": "text",
                "global_alpha": 1.0,
            }
        )
        materials["material_animations"].append(
            {
                "animations": [],
                "id": animation_id,
                "type": "sticker_animation",
            }
        )
        return text_id, animation_id

    def _create_text_segment(
        self,
        material_id: str,
        animation_id: str,
        timeline_start_us: int,
        duration_us: int,
        position_y: float = -0.72,
    ) -> dict:
        return {
            "enable_adjust": True,
            "enable_color_correct_adjust": False,
            "enable_color_curves": True,
            "enable_color_match_adjust": False,
            "enable_color_wheels": True,
            "enable_lut": True,
            "enable_smart_color_adjust": False,
            "last_nonzero_volume": 1.0,
            "reverse": False,
            "track_attribute": 0,
            "track_render_index": 0,
            "visible": True,
            "id": str(uuid.uuid4()),
            "material_id": material_id,
            "target_timerange": {"start": timeline_start_us, "duration": duration_us},
            "common_keyframes": [],
            "keyframe_refs": [],
            "source_timerange": None,
            "speed": 1.0,
            "volume": 1.0,
            "extra_material_refs": [animation_id],
            "is_tone_modify": False,
            "clip": {
                "alpha": 1.0,
                "flip": {"horizontal": False, "vertical": False},
                "rotation": 0.0,
                "scale": {"x": 1.0, "y": 1.0},
                "transform": {"x": 0.0, "y": position_y},
            },
            "uniform_scale": {"on": True, "value": 1.0},
            "render_index": 15000,
        }

    LOW_CONFIDENCE_THRESHOLD = 0.78
    VISUAL_RESTORE_MIN_CUT_GAP = 0.80
    VISUAL_RESTORE_CUT_SAMPLE_STEP = 0.50
    VISUAL_RESTORE_MAX_SEEDS = 4
    VISUAL_RESTORE_CONTINUITY_TOLERANCE = 0.80
    VISUAL_RESTORE_MIN_SMOOTH_SEGMENT = 0.55
    VISUAL_RESTORE_SHORT_MERGE_TOLERANCE = 4.0
    VISUAL_RESTORE_TINY_SEGMENT = 0.35
    VISUAL_RESTORE_FLICKER_MAX_DURATION = 0.90
    VISUAL_RESTORE_FLICKER_DIRECT_TOLERANCE = 1.25
    VISUAL_RESTORE_FLICKER_JUMP_MIN = 1.50
    VISUAL_RESTORE_REPEAT_BACKTRACK_TOLERANCE = 0.35
    VISUAL_RESTORE_REPEAT_MIN_OVERLAP_RATIO = 0.45
    VISUAL_RESTORE_REPEAT_MIN_OVERLAP_SECONDS = 1.00
    VISUAL_RESTORE_PROTECTED_SOURCE_JUMP = 8.0
    VISUAL_RESTORE_ANCHOR_LOCK_SCORE = 0.76
    VISUAL_RESTORE_ANCHOR_LOCK_RADIUS = 3.0
    VISUAL_RESTORE_ANCHOR_ESCAPE_MARGIN = 0.12
    VISUAL_RESTORE_PRECUT_RETIMER_MAX_DURATION = 2.60
    VISUAL_RESTORE_VISUAL_CUT_TOLERANCE = 0.38
    VISUAL_RESTORE_NONCUT_CONTINUITY_MARGIN = 0.18
    VISUAL_RESTORE_NONCUT_MAX_JUMP = 18.0
    VISUAL_RESTORE_CANDIDATE_PROTECTED_SOURCE_JUMP = 25.0
    VISUAL_RESTORE_LOCAL_BACKTRACK_MAX = 120.0
    VISUAL_RESTORE_LOCAL_BACKTRACK_MARGIN = 0.13
    VISUAL_RESTORE_GLOBAL_FALLBACK_SCORE = 0.78
    VISUAL_RESTORE_GLOBAL_FALLBACK_GAIN = 0.025
    VISUAL_RESTORE_GLOBAL_CANDIDATES = 96
    VISUAL_RESTORE_GLOBAL_COARSE_FLOOR = 0.58
    VISUAL_RESTORE_RETIME_MIN_DURATION = 1.40
    VISUAL_RESTORE_RETIME_SEARCH_RADIUS = 1.35
    VISUAL_RESTORE_RETIME_MIN_SCORE = 0.64
    VISUAL_RESTORE_RETIME_MIN_SPEED = 0.55
    VISUAL_RESTORE_RETIME_MAX_SPEED = 1.85
    VISUAL_RESTORE_PROBE_SEGMENT = 1.00
    VISUAL_RESTORE_EXCURSION_MIN_JUMP = 20.0
    VISUAL_RESTORE_EXCURSION_MAX_DURATION = 2.25
    VISUAL_RESTORE_EXCURSION_RETURN_TOLERANCE = 8.0

    def _find_neighbor_matched_segment(
        self,
        segments: list[Segment],
        index: int,
        direction: int,
    ) -> Optional[Segment]:
        cursor = index + direction
        while 0 <= cursor < len(segments):
            segment = segments[cursor]
            if (
                segment.use_segment
                and segment.segment_type != SegmentType.NON_MOVIE
                and segment.movie_start is not None
                and segment.movie_end is not None
            ):
                return segment
            cursor += direction
        return None

    def _resolve_restore_source_range(
        self,
        project: Project,
        segments: list[Segment],
        index: int,
        segment: Segment,
        timeline_duration: float,
    ) -> Optional[tuple[float, float, bool]]:
        if segment.movie_start is not None and segment.movie_end is not None:
            return float(segment.movie_start), float(segment.movie_end), False

        prev_segment = self._find_neighbor_matched_segment(segments, index, -1)
        next_segment = self._find_neighbor_matched_segment(segments, index, 1)
        inferred_start: Optional[float] = None

        if prev_segment and next_segment:
            prev_narr = float(prev_segment.narration_end)
            next_narr = float(next_segment.narration_start)
            prev_movie = float(prev_segment.movie_end)
            next_movie = max(prev_movie, float(next_segment.movie_start) - timeline_duration)
            narr_span = max(0.2, next_narr - prev_narr)
            ratio = (float(segment.narration_start) - prev_narr) / narr_span
            ratio = max(0.0, min(1.0, ratio))
            inferred_start = prev_movie + ratio * max(0.0, next_movie - prev_movie)
        elif prev_segment:
            inferred_start = float(prev_segment.movie_end)
        elif next_segment:
            inferred_start = max(0.0, float(next_segment.movie_start) - timeline_duration)

        if inferred_start is None:
            return None

        inferred_end = inferred_start + timeline_duration
        movie_duration = float(project.movie_duration or 0.0)
        if movie_duration > 0.0 and inferred_end > movie_duration:
            inferred_end = movie_duration
            inferred_start = max(0.0, inferred_end - timeline_duration)

        return inferred_start, inferred_end, True

    def _is_restore_anchor_segment(self, segment: Segment) -> bool:
        if not segment.use_segment or segment.segment_type == SegmentType.NON_MOVIE:
            return False
        if segment.movie_start is None or segment.movie_end is None:
            return False
        status = getattr(segment.alignment_status, "value", segment.alignment_status)
        if status == "manual":
            return True
        return (
            status in {"auto_accepted", "rematched"}
            and not segment.review_required
            and segment.match_type == "exact"
            and float(segment.match_confidence or 0.0) >= 0.85
        )

    def _build_continuous_restore_ranges(
        self,
        project: Project,
        segments: list[Segment],
    ) -> dict[str, tuple[float, float, bool]]:
        """Build movie-only source ranges with low-confidence matches smoothed.

        High-confidence matches act as anchors. Other segments are filled from
        the previous movie source end so they do not jump to unreliable matches.
        """
        usable = [
            segment for segment in segments
            if segment.use_segment and segment.segment_type != SegmentType.NON_MOVIE
        ]
        if not usable:
            return {}

        boundary_cuts = self._detect_narration_boundary_cuts(project, usable)
        anchor_indices = {
            idx for idx, segment in enumerate(usable)
            if self._is_restore_anchor_segment(segment)
        }

        next_anchor: list[Optional[int]] = [None] * len(usable)
        cursor_anchor: Optional[int] = None
        for idx in range(len(usable) - 1, -1, -1):
            if idx in anchor_indices:
                cursor_anchor = idx
            next_anchor[idx] = cursor_anchor

        ranges: dict[str, tuple[float, float, bool]] = {}
        source_cursor: Optional[float] = None
        movie_duration = float(project.movie_duration or 0.0)

        for idx, segment in enumerate(usable):
            duration = max(0.0, float(segment.narration_end) - float(segment.narration_start))
            if duration <= 0.0:
                continue

            is_anchor = idx in anchor_indices
            if is_anchor:
                source_start = float(segment.movie_start)
                jump_from_cursor = None if source_cursor is None else source_start - source_cursor
                boundary_is_cut = boundary_cuts.get(segment.id, True)
                if (
                    source_cursor is not None
                    and jump_from_cursor is not None
                    and abs(jump_from_cursor) > max(4.0, duration * 2.0)
                    and not boundary_is_cut
                ):
                    source_start = source_cursor
                    is_inferred = True
                else:
                    is_inferred = False
            elif source_cursor is not None:
                source_start = source_cursor
                is_inferred = True
            else:
                upcoming_idx = next_anchor[idx]
                if upcoming_idx is not None:
                    upcoming = usable[upcoming_idx]
                    lead = max(0.0, float(upcoming.narration_start) - float(segment.narration_start))
                    source_start = max(0.0, float(upcoming.movie_start) - lead)
                elif segment.movie_start is not None:
                    source_start = float(segment.movie_start)
                else:
                    source_start = 0.0
                is_inferred = True

            source_end = source_start + duration
            if movie_duration > 0.0 and source_end > movie_duration:
                source_end = movie_duration
                source_start = max(0.0, source_end - duration)

            ranges[segment.id] = (source_start, source_end, is_inferred)
            source_cursor = source_end

        return ranges

    def _detect_narration_boundary_cuts(
        self,
        project: Project,
        usable_segments: list[Segment],
    ) -> dict[str, bool]:
        """Return whether the boundary before each segment is a visual cut.

        Subtitles can change every segment, so the comparison ignores the bottom
        quarter of the frame and only uses a small grayscale preview.
        """
        if not project.narration_path or len(usable_segments) < 2:
            return {}

        try:
            import cv2
            import numpy as np
        except Exception:
            return {}

        capture = cv2.VideoCapture(project.narration_path)
        if not capture.isOpened():
            return {}

        def read_preview(time_seconds: float):
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            h, w = frame.shape[:2]
            frame = frame[: max(1, int(h * 0.72)), :]
            frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cuts: dict[str, bool] = {}
        try:
            previous_frame = None
            previous_segment: Optional[Segment] = None
            for segment in usable_segments:
                if previous_segment is None:
                    previous_segment = segment
                    previous_frame = read_preview(float(segment.narration_end) - 0.04)
                    continue

                current_frame = read_preview(float(segment.narration_start) + 0.04)
                is_cut = True
                if previous_frame is not None and current_frame is not None:
                    prev_hist = cv2.calcHist([previous_frame], [0], None, [32], [0, 256]).astype("float32")
                    curr_hist = cv2.calcHist([current_frame], [0], None, [32], [0, 256]).astype("float32")
                    prev_hist /= max(float(prev_hist.sum()), 1.0)
                    curr_hist /= max(float(curr_hist.sum()), 1.0)
                    hist_corr = float((cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
                    mean_diff = float(np.mean(np.abs(previous_frame.astype("float32") - current_frame.astype("float32")))) / 255.0
                    is_cut = not (hist_corr >= 0.88 and mean_diff <= 0.075)
                cuts[segment.id] = is_cut

                previous_segment = segment
                previous_frame = read_preview(float(segment.narration_end) - 0.04)
        finally:
            capture.release()

        return cuts

    def _selected_candidate_for_segment(self, segment: Segment):
        candidates = list(segment.match_candidates or [])
        if not candidates:
            return None
        if segment.selected_candidate_id:
            for candidate in candidates:
                if candidate.id == segment.selected_candidate_id:
                    return candidate
        if segment.movie_start is not None and segment.movie_end is not None:
            for candidate in candidates:
                if abs(float(candidate.start) - float(segment.movie_start)) <= 0.08:
                    return candidate
        return candidates[0]

    def _segment_visual_verify_score(self, segment: Segment) -> float:
        candidate = self._selected_candidate_for_segment(segment)
        sources = []
        if candidate is not None:
            sources.append(candidate.reason or "")
        sources.append(segment.evidence_summary or "")
        for source in sources:
            match = re.search(r"post_verify=([0-9.]+)", source)
            if match:
                return float(match.group(1))
        if candidate is not None:
            return float(candidate.verification_score or 0.0)
        return 0.0

    def _init_visual_restore_gpu(self, context) -> None:
        try:
            import torch
        except Exception as exc:
            logger.info("Visual restore CUDA acceleration unavailable: {}", exc)
            return

        if not torch.cuda.is_available():
            logger.info("Visual restore CUDA acceleration unavailable: torch.cuda.is_available() is false")
            return

        try:
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        device = torch.device("cuda:0")
        context["torch"] = torch
        context["visual_restore_gpu_device"] = device
        context["visual_restore_gpu_enabled"] = True
        logger.info("Visual restore CUDA acceleration enabled: {}", torch.cuda.get_device_name(0))

    def _prepare_visual_restore_gpu_index(self, context, frame_matcher) -> None:
        if not context.get("visual_restore_gpu_enabled"):
            return

        torch = context["torch"]
        np = context["np"]
        device = context["visual_restore_gpu_device"]
        tensor_index = {}
        for name, attr in (
            ("hist", "_idx_hist"),
            ("color", "_idx_color_hist"),
            ("spatial", "_idx_spatial_color_hist"),
            ("grad", "_idx_grad_hist"),
        ):
            arr = getattr(frame_matcher, attr, None)
            if arr is None:
                continue
            arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
            tensor_index[name] = torch.from_numpy(arr).to(device=device, dtype=torch.float16, non_blocking=True)

        if tensor_index:
            context["visual_restore_gpu_index"] = tensor_index
            logger.info(
                "Visual restore CUDA coarse index ready: {}",
                ", ".join(f"{name}{tuple(tensor.shape)}" for name, tensor in tensor_index.items()),
            )

    def _visual_restore_gpu_dense_scores(self, context, query_feature: dict, cache_key: Optional[str] = None):
        tensor_index = context.get("visual_restore_gpu_index")
        if not tensor_index:
            return None

        score_cache = context.setdefault("visual_restore_gpu_score_cache", {})
        if cache_key is not None and cache_key in score_cache:
            return score_cache[cache_key]

        torch = context["torch"]
        np = context["np"]
        device = context["visual_restore_gpu_device"]
        scores = None

        def add_score(index_name: str, query_name: str, weight: float) -> None:
            nonlocal scores
            idx_tensor = tensor_index.get(index_name)
            if idx_tensor is None:
                return
            query = np.asarray(query_feature.get(query_name), dtype=np.float32).reshape(-1)
            if query.size != int(idx_tensor.shape[1]):
                return
            norm = float(np.linalg.norm(query))
            if norm <= 1e-8:
                return
            query = np.ascontiguousarray(query / norm)
            query_tensor = torch.from_numpy(query).to(device=device, dtype=idx_tensor.dtype, non_blocking=True)
            sim = torch.clamp(idx_tensor @ query_tensor, 0.0, 1.0).float() * float(weight)
            scores = sim if scores is None else scores + sim

        add_score("color", "color_hist", 0.20)
        add_score("spatial", "spatial_color_hist", 0.24)
        add_score("grad", "grad_hist", 0.05)

        if scores is None:
            return None

        result = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        if cache_key is not None:
            if len(score_cache) > 1024:
                score_cache.clear()
            score_cache[cache_key] = result
        return result

    def _open_visual_restore_context(self, project: Project):
        if not project.movie_path or not project.narration_path:
            return None
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            logger.warning("Visual restore export disabled: {}", exc)
            return None

        narration_cap = cv2.VideoCapture(str(project.narration_path))
        movie_cap = cv2.VideoCapture(str(project.movie_path))
        if not narration_cap.isOpened() or not movie_cap.isOpened():
            narration_cap.release()
            movie_cap.release()
            logger.warning("Visual restore export disabled: cannot open source videos")
            return None

        movie_duration = float(project.movie_duration or 0.0)
        if movie_duration <= 0:
            fps = float(movie_cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(movie_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            movie_duration = frames / fps if fps > 0 else 0.0

        narration_duration = float(project.narration_duration or 0.0)
        if narration_duration <= 0:
            fps = float(narration_cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(narration_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            narration_duration = frames / fps if fps > 0 else 0.0

        context = {
            "cv2": cv2,
            "np": np,
            "narration_cap": narration_cap,
            "movie_cap": movie_cap,
            "orb": cv2.ORB_create(nfeatures=420, fastThreshold=14),
            "matcher": cv2.BFMatcher(cv2.NORM_HAMMING),
            "movie_duration": movie_duration,
            "narration_duration": narration_duration,
        }
        self._init_visual_restore_gpu(context)
        try:
            from core.video_processor.frame_matcher import FrameMatcher

            frame_matcher = FrameMatcher(
                enable_subtitle_masking=True,
                subtitle_mask_mode=getattr(project.subtitle_mask_mode, "value", project.subtitle_mask_mode),
                movie_subtitle_regions=[
                    region.model_dump() if hasattr(region, "model_dump") else dict(region)
                    for region in getattr(project, "movie_subtitle_regions", [])
                ],
                narration_subtitle_regions=[
                    region.model_dump() if hasattr(region, "model_dump") else dict(region)
                    for region in getattr(project, "narration_subtitle_regions", [])
                ],
            )
            cache_path = (
                Path(__file__).resolve().parents[2]
                / "temp"
                / "match_cache"
                / f"{Path(project.movie_path).stem}_frame.pkl"
            )
            if cache_path.exists():
                with open(cache_path, "rb") as handle:
                    payload = pickle.load(handle)
                if int(payload.get("cache_version", -1)) == int(frame_matcher.CACHE_VERSION):
                    frame_matcher._index = payload.get("index") or []
                    frame_matcher._times = [float(item) for item in payload.get("times") or []]
                    frame_matcher._movie_duration = float(payload.get("movie_duration") or movie_duration)
                    frame_matcher._sample_step_seconds = float(payload.get("sample_step_seconds") or 1.0)
                    frame_matcher._precompute_packed_hashes()
                    self._prepare_visual_restore_gpu_index(context, frame_matcher)
                    context["frame_matcher"] = frame_matcher
                    context["movie_index"] = frame_matcher._index
                    context["movie_times"] = frame_matcher._times
                    context["narration_masker"] = frame_matcher._get_subtitle_masker(
                        str(project.narration_path),
                        "narration",
                        True,
                    )
                    self._precompute_visual_restore_query_features(
                        context,
                        str(project.narration_path),
                    )
                    self._precompute_visual_restore_gray_series(
                        context,
                        str(project.narration_path),
                        "narration",
                        0.50,
                    )
                    self._precompute_visual_restore_gray_series(
                        context,
                        str(project.movie_path),
                        "movie",
                        1.00,
                    )
                    logger.info("Visual restore export loaded frame index: {} samples", len(frame_matcher._times))
        except Exception as exc:
            logger.warning("Visual restore frame index unavailable: {}", exc)
        return context

    def _close_visual_restore_context(self, context) -> None:
        if not context:
            return
        context["narration_cap"].release()
        context["movie_cap"].release()

    def _precompute_visual_restore_query_features(
        self,
        context,
        narration_path: str,
        step: float = 0.50,
    ) -> None:
        frame_matcher = context.get("frame_matcher")
        if frame_matcher is None:
            return
        duration = float(context.get("narration_duration") or 0.0)
        if duration <= 0.0:
            return

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        pipe_w = 256
        pipe_h = 144
        frame_bytes = pipe_w * pipe_h * 3
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            narration_path,
            "-an",
            "-vf",
            f"fps=1/{step:.4f},scale={pipe_w}:{pipe_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        proc = None
        feature_cache: dict[float, dict] = {}
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 64,
            )
            frame_index = 0
            while True:
                raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                if not raw or len(raw) < frame_bytes:
                    break
                timestamp = round(frame_index * step, 2)
                frame = context["np"].frombuffer(raw, dtype=context["np"].uint8).reshape(pipe_h, pipe_w, 3).copy()
                feature_cache[timestamp] = frame_matcher._frame_features_lite(
                    frame,
                    context.get("narration_masker"),
                    frame_time=float(timestamp),
                )
                frame_index += 1
            proc.wait(timeout=5)
        except Exception as exc:
            logger.warning("Visual restore narration feature pre-cache failed: {}", exc)
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            return

        if feature_cache:
            context["query_feature_cache"] = feature_cache
            context["query_feature_cache_step"] = float(step)
            context["query_feature_cache_precomputed"] = True
            logger.info("Visual restore precomputed narration query features: {}", len(feature_cache))

    def _precompute_visual_restore_gray_series(
        self,
        context,
        video_path: str,
        role: str,
        step: float,
    ) -> None:
        cv2 = context["cv2"]
        np = context["np"]
        cap = context["narration_cap"] if role == "narration" else context["movie_cap"]
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        pipe_w = 180
        pipe_h = 102
        if source_w > 0 and source_h > 0:
            pipe_h = max(2, int(source_h * pipe_w / source_w / 2) * 2)
        frame_bytes = pipe_w * pipe_h
        try:
            source_stat = Path(video_path).stat()
            cache_key = hashlib.md5(
                "|".join(
                    [
                        str(Path(video_path).resolve()).lower(),
                        str(int(source_stat.st_mtime)),
                        str(int(source_stat.st_size)),
                        role,
                        f"{step:.4f}",
                        str(pipe_w),
                        str(pipe_h),
                    ]
                ).encode("utf-8", errors="ignore")
            ).hexdigest()
            cache_dir = Path(__file__).resolve().parents[2] / "temp" / "visual_gray_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                frames_array = np.load(str(cache_path), mmap_mode="r")
                context.setdefault("visual_restore_gray_series", {})[role] = {
                    "step": float(step),
                    "frames": frames_array,
                }
                logger.info("Visual restore loaded {} cached gray frames for {}", len(frames_array), role)
                return
        except Exception:
            cache_path = None

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-an",
            "-vf",
            f"fps=1/{step:.4f},scale={pipe_w}:{pipe_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        proc = None
        frames = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 64,
            )
            while True:
                raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                if not raw or len(raw) < frame_bytes:
                    break
                gray = np.frombuffer(raw, dtype=np.uint8).reshape(pipe_h, pipe_w).copy()
                gray = gray[: max(1, int(pipe_h * 0.76)), :]
                frames.append(gray)
            proc.wait(timeout=5)
        except Exception as exc:
            logger.warning("Visual restore gray pre-cache failed for {}: {}", role, exc)
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            return

        if frames:
            try:
                if cache_path is not None:
                    np.save(str(cache_path), np.stack(frames, axis=0))
            except Exception as exc:
                logger.debug("Visual restore gray cache save skipped for {}: {}", role, exc)
            context.setdefault("visual_restore_gray_series", {})[role] = {
                "step": float(step),
                "frames": frames,
            }
            logger.info("Visual restore precomputed {} gray frames for {}", len(frames), role)

    def _read_visual_restore_gray(self, context, role: str, timestamp: float):
        cv2 = context["cv2"]
        series = (context.get("visual_restore_gray_series") or {}).get(role)
        if series:
            step = float(series.get("step") or 0.0)
            frames = series.get("frames")
            if step > 0.0 and frames is not None and len(frames) > 0:
                index = int(round(float(timestamp) / step))
                if 0 <= index < len(frames) and abs(index * step - float(timestamp)) <= step * 0.58:
                    return frames[index]
        cache = context.setdefault("visual_restore_gray_cache", {}).setdefault(role, {})
        key = round(float(timestamp), 2)
        if key in cache:
            return cache[key]
        cap = context["narration_cap"] if role == "narration" else context["movie_cap"]
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None
        # Ignore the subtitle band. It is useful in the narration video but
        # harmful for matching against the clean movie source.
        frame = frame[: max(1, int(height * 0.76)), :]
        height, width = frame.shape[:2]
        if width > 360:
            resized_height = max(32, int(height * 360.0 / width))
            frame = cv2.resize(frame, (360, resized_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cache[key] = gray
        if len(cache) > 1024:
            cache.clear()
        return gray

    def _visual_restore_query_feature(self, context, timestamp: float):
        frame_matcher = context.get("frame_matcher")
        if frame_matcher is None:
            return None
        cache = context.setdefault("query_feature_cache", {})
        cache_step = float(context.get("query_feature_cache_step") or 0.0)
        if cache_step > 0.0 and cache:
            nearest_key = round(round(float(timestamp) / cache_step) * cache_step, 2)
            if abs(nearest_key - float(timestamp)) <= cache_step * 0.55 and nearest_key in cache:
                return cache[nearest_key]
        key = round(float(timestamp), 2)
        if key in cache:
            return cache[key]
        cv2 = context["cv2"]
        cap = context["narration_cap"]
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        feature = frame_matcher._frame_features_lite(
            frame,
            context.get("narration_masker"),
            frame_time=float(timestamp),
        )
        cache[key] = feature
        if not context.get("query_feature_cache_precomputed") and len(cache) > 512:
            cache.clear()
        return feature

    def _visual_restore_sample_positions(self, duration: float) -> list[float]:
        if duration < 1.2:
            return [0.50]
        if duration < 2.5:
            return [0.33, 0.67]
        return [0.22, 0.50, 0.78]

    def _score_visual_restore_indexed_range(
        self,
        context,
        target_start: float,
        duration: float,
        source_start: float,
    ) -> float:
        frame_matcher = context.get("frame_matcher")
        movie_index = context.get("movie_index") or []
        movie_times = context.get("movie_times") or []
        if frame_matcher is None or not movie_index or not movie_times:
            return 0.0

        scores: list[float] = []
        for pos in self._visual_restore_sample_positions(duration):
            query_feature = self._visual_restore_query_feature(context, target_start + duration * pos)
            if query_feature is None:
                continue
            movie_time = source_start + duration * pos
            nearest = bisect.bisect_left(movie_times, movie_time)
            best = 0.0
            best_time: Optional[float] = None
            for idx in (nearest - 1, nearest, nearest + 1):
                if idx < 0 or idx >= len(movie_index):
                    continue
                if abs(float(movie_times[idx]) - movie_time) > 1.35:
                    continue
                feature_score = float(frame_matcher._feature_score(query_feature, movie_index[idx]))
                if feature_score > best:
                    best = feature_score
                    best_time = float(movie_times[idx])
            if best_time is not None:
                query_gray = self._read_visual_restore_gray(context, "narration", target_start + duration * pos)
                movie_gray = self._read_visual_restore_gray(context, "movie", best_time)
                pair_score = self._score_visual_restore_pair(context, query_gray, movie_gray)
                best = best * 0.35 + pair_score * 0.65
                if pair_score < 0.45:
                    best *= 0.72
            scores.append(best)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _visual_restore_local_best_time(
        self,
        context,
        target_time: float,
        source_hint: float,
        radius: Optional[float] = None,
    ) -> Optional[tuple[float, float]]:
        frame_matcher = context.get("frame_matcher")
        movie_index = context.get("movie_index") or []
        movie_times = context.get("movie_times") or []
        if frame_matcher is None or not movie_index or not movie_times:
            return None

        query_feature = self._visual_restore_query_feature(context, target_time)
        if query_feature is None:
            return None
        query_gray = self._read_visual_restore_gray(context, "narration", target_time)

        search_radius = float(radius or self.VISUAL_RESTORE_RETIME_SEARCH_RADIUS)
        left = bisect.bisect_left(movie_times, max(0.0, source_hint - search_radius))
        right = bisect.bisect_right(movie_times, source_hint + search_radius)
        if right <= left:
            return None

        best_time: Optional[float] = None
        best_score = 0.0
        for idx in range(left, right):
            feature_score = float(frame_matcher._feature_score(query_feature, movie_index[idx]))
            pair_score = 0.0
            if query_gray is not None:
                movie_gray = self._read_visual_restore_gray(context, "movie", float(movie_times[idx]))
                pair_score = self._score_visual_restore_pair(context, query_gray, movie_gray)
            score = feature_score * 0.25 + pair_score * 0.75 if pair_score > 0.0 else feature_score
            if score > best_score:
                best_score = score
                best_time = float(movie_times[idx])

        if best_time is None:
            return None
        return best_time, best_score

    def _retime_visual_restore_ranges(
        self,
        context,
        ranges: list[tuple[float, float, float, float, bool, float]],
        movie_duration: float,
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Fit source duration per visual clip so motion stays continuous.

        Earlier stages decide *which* movie moment a narration clip belongs to.
        This pass decides how much source time should be played inside the
        target span. That matters when the narration video has been sped up,
        slowed down, or cut with slightly different in/out points.
        """
        if not ranges:
            return []

        retimed: list[tuple[float, float, float, float, bool, float]] = []
        changes = 0
        for current in sorted(ranges, key=lambda item: item[0]):
            target_start, target_end, source_start, source_end, is_inferred, visual_score = current
            duration = max(0.0, target_end - target_start)
            source_duration = max(0.0, source_end - source_start)
            if duration < self.VISUAL_RESTORE_RETIME_MIN_DURATION or source_duration <= 0.0:
                retimed.append(current)
                continue

            edge_offset = min(0.45, max(0.18, duration * 0.16))
            start_target = target_start + edge_offset
            end_target = target_end - edge_offset
            start_hint = source_start + edge_offset
            end_hint = source_end - edge_offset
            start_anchor = self._visual_restore_local_best_time(context, start_target, start_hint)
            end_anchor = self._visual_restore_local_best_time(context, end_target, end_hint)

            if start_anchor is None or end_anchor is None:
                retimed.append(current)
                continue

            start_time, start_score = start_anchor
            end_time, end_score = end_anchor
            if start_score < self.VISUAL_RESTORE_RETIME_MIN_SCORE or end_score < self.VISUAL_RESTORE_RETIME_MIN_SCORE:
                retimed.append(current)
                continue

            fitted_source_start = max(0.0, start_time - edge_offset)
            fitted_source_end = max(fitted_source_start + 0.05, end_time + edge_offset)
            if movie_duration > 0.0:
                if fitted_source_end > movie_duration:
                    overflow = fitted_source_end - movie_duration
                    fitted_source_start = max(0.0, fitted_source_start - overflow)
                    fitted_source_end = movie_duration

            fitted_source_duration = fitted_source_end - fitted_source_start
            fitted_speed = fitted_source_duration / max(duration, 0.05)
            if not (
                self.VISUAL_RESTORE_RETIME_MIN_SPEED
                <= fitted_speed
                <= self.VISUAL_RESTORE_RETIME_MAX_SPEED
            ):
                retimed.append(current)
                continue

            old_score = self._score_visual_restore_indexed_range(
                context,
                target_start,
                duration,
                source_start,
            )
            new_score = self._score_visual_restore_indexed_range(
                context,
                target_start,
                duration,
                fitted_source_start,
            )
            # Keep the fitted source when it does not materially hurt visual
            # similarity; this favors motion continuity over frame-by-frame
            # overfitting to a single sampled point.
            if new_score >= old_score - 0.075:
                retimed.append(
                    (
                        target_start,
                        target_end,
                        fitted_source_start,
                        fitted_source_end,
                        is_inferred,
                        max(visual_score, min(start_score, end_score, new_score)),
                    )
                )
                if abs(fitted_source_start - source_start) > 0.08 or abs(fitted_source_end - source_end) > 0.08:
                    changes += 1
            else:
                retimed.append(current)

        if changes:
            logger.info("Visual restore retimed {} clips with fitted source duration", changes)
        return retimed

    def _is_protected_visual_restore_boundary(self, context, boundary: float) -> bool:
        protected_cuts = sorted(float(item) for item in (context.get("protected_visual_cuts") or []))
        if not protected_cuts:
            return False
        idx = bisect.bisect_left(protected_cuts, boundary - 0.08)
        return idx < len(protected_cuts) and protected_cuts[idx] <= boundary + 0.08

    def _is_detected_visual_restore_cut(self, context, boundary: float) -> bool:
        visual_cuts = sorted(float(item) for item in (context.get("detected_visual_cuts") or []))
        if not visual_cuts:
            return False
        tolerance = self.VISUAL_RESTORE_VISUAL_CUT_TOLERANCE
        idx = bisect.bisect_left(visual_cuts, boundary - tolerance)
        return idx < len(visual_cuts) and visual_cuts[idx] <= boundary + tolerance

    def _score_visual_restore_pair(self, context, query_gray, movie_gray) -> float:
        if query_gray is None or movie_gray is None:
            return 0.0
        cv2 = context["cv2"]
        np = context["np"]
        if query_gray.shape != movie_gray.shape:
            movie_gray = cv2.resize(
                movie_gray,
                (query_gray.shape[1], query_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        query_hist = cv2.calcHist([query_gray], [0], None, [32], [0, 256]).astype("float32")
        movie_hist = cv2.calcHist([movie_gray], [0], None, [32], [0, 256]).astype("float32")
        query_hist /= max(float(query_hist.sum()), 1.0)
        movie_hist /= max(float(movie_hist.sum()), 1.0)
        hist_score = float((cv2.compareHist(query_hist, movie_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)

        query_edges = cv2.Canny(query_gray, 60, 140)
        movie_edges = cv2.Canny(movie_gray, 60, 140)
        query_edge_mask = query_edges > 0
        movie_edge_mask = movie_edges > 0
        edge_intersection = int(np.logical_and(query_edge_mask, movie_edge_mask).sum())
        edge_union = int(np.logical_or(query_edge_mask, movie_edge_mask).sum())
        edge_iou = edge_intersection / max(1, edge_union)

        query_float = query_gray.astype(np.float32)
        movie_float = movie_gray.astype(np.float32)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        query_mean = cv2.GaussianBlur(query_float, (7, 7), 1.5)
        movie_mean = cv2.GaussianBlur(movie_float, (7, 7), 1.5)
        query_var = cv2.GaussianBlur(query_float * query_float, (7, 7), 1.5) - query_mean * query_mean
        movie_var = cv2.GaussianBlur(movie_float * movie_float, (7, 7), 1.5) - movie_mean * movie_mean
        covariance = cv2.GaussianBlur(query_float * movie_float, (7, 7), 1.5) - query_mean * movie_mean
        ssim_map = ((2 * query_mean * movie_mean + c1) * (2 * covariance + c2)) / (
            (query_mean * query_mean + movie_mean * movie_mean + c1) * (query_var + movie_var + c2) + 1e-6
        )
        ssim_score = float(np.clip(np.mean(ssim_map), 0.0, 1.0))

        query_centered = query_float - float(query_float.mean())
        movie_centered = movie_float - float(movie_float.mean())
        ncc_denominator = float(
            np.sqrt(np.mean(query_centered * query_centered) * np.mean(movie_centered * movie_centered))
        )
        if ncc_denominator <= 1e-6:
            ncc_score = 0.0
        else:
            ncc_score = float(
                np.clip((np.mean(query_centered * movie_centered) / ncc_denominator) * 0.5 + 0.5, 0.0, 1.0)
            )

        structural_score = (
            ssim_score * 0.42
            + ncc_score * 0.30
            + edge_iou * 0.18
            + hist_score * 0.10
        )

        orb = context["orb"]
        matcher = context["matcher"]
        kp1, des1 = orb.detectAndCompute(query_gray, None)
        kp2, des2 = orb.detectAndCompute(movie_gray, None)
        inliers = 0
        inlier_ratio = 0.0
        warp_score = 0.0
        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            matches = matcher.knnMatch(des1, des2, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.78 * n.distance:
                    good.append(m)
            if len(good) >= 8:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())
                    inlier_ratio = inliers / max(1, len(good))
                    if homography is not None and inliers >= 8:
                        try:
                            warped = cv2.warpPerspective(
                                movie_gray,
                                homography,
                                (query_gray.shape[1], query_gray.shape[0]),
                            )
                            diff = float(
                                np.mean(np.abs(query_gray.astype(np.float32) - warped.astype(np.float32)))
                            ) / 255.0
                            warp_score = max(0.0, 1.0 - diff)
                        except cv2.error:
                            warp_score = 0.0
            elif good:
                inliers = len(good)
                inlier_ratio = min(1.0, len(good) / max(12.0, min(len(kp1), len(kp2))))

        geom_score = min(1.0, inliers / 36.0) * 0.28 + inlier_ratio * 0.32 + warp_score * 0.40
        if inliers < 5:
            return float(structural_score)
        return float(max(structural_score * 0.72 + geom_score * 0.28, structural_score))

    def _detect_narration_visual_cuts_from_context(self, project: Project, context) -> list[float]:
        duration = float(context.get("narration_duration") or project.narration_duration or 0.0)
        if duration <= 0:
            return []

        cuts = [0.0]
        protected_visual_cuts: list[float] = []
        detected_visual_cuts: list[float] = []
        previous = None
        previous_time = 0.0
        last_cut = 0.0
        step = self.VISUAL_RESTORE_CUT_SAMPLE_STEP
        cv2 = context["cv2"]
        np = context["np"]

        def consider_frame(gray, timestamp: float) -> None:
            nonlocal previous, previous_time, last_cut
            if previous is not None:
                if previous.shape != gray.shape:
                    gray_cmp = cv2.resize(gray, (previous.shape[1], previous.shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    gray_cmp = gray
                diff = float(np.mean(np.abs(previous.astype(np.float32) - gray_cmp.astype(np.float32)))) / 255.0
                prev_hist = cv2.calcHist([previous], [0], None, [32], [0, 256]).astype("float32")
                curr_hist = cv2.calcHist([gray_cmp], [0], None, [32], [0, 256]).astype("float32")
                prev_hist /= max(float(prev_hist.sum()), 1.0)
                curr_hist /= max(float(curr_hist.sum()), 1.0)
                hist_corr = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))
                previous_edges = cv2.Canny(previous, 60, 140) > 0
                current_edges = cv2.Canny(gray_cmp, 60, 140) > 0
                edge_intersection = int(np.logical_and(previous_edges, current_edges).sum())
                edge_union = int(np.logical_or(previous_edges, current_edges).sum())
                edge_iou = edge_intersection / max(1, edge_union)
                is_cut = (
                    diff >= 0.22
                    or (diff >= 0.15 and hist_corr <= 0.82)
                    or (diff >= 0.10 and edge_iou <= 0.12)
                )
                if is_cut and timestamp - last_cut >= self.VISUAL_RESTORE_MIN_CUT_GAP:
                    cut_time = max(0.0, previous_time)
                    cuts.append(cut_time)
                    detected_visual_cuts.append(cut_time)
                    last_cut = timestamp
            previous = gray
            previous_time = timestamp

        pipe_w = 180
        source_w = int(context["narration_cap"].get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_h = int(context["narration_cap"].get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        pipe_h = 102
        if source_w > 0 and source_h > 0:
            pipe_h = max(2, int(source_h * pipe_w / source_w / 2) * 2)
        frame_bytes = pipe_w * pipe_h
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(project.narration_path),
            "-an",
            "-vf",
            f"fps=1/{step:.4f},scale={pipe_w}:{pipe_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 32,
            )
            frame_index = 0
            while True:
                raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                if not raw or len(raw) < frame_bytes:
                    break
                gray = np.frombuffer(raw, dtype=np.uint8).reshape(pipe_h, pipe_w)
                gray = gray[: max(1, int(pipe_h * 0.76)), :]
                consider_frame(gray, frame_index * step)
                frame_index += 1
            proc.wait(timeout=2)
        except Exception as exc:
            logger.warning("Fast visual cut scan failed, fallback to sparse OpenCV scan: {}", exc)
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            cuts = [0.0]
            detected_visual_cuts = []
            previous = None
            previous_time = 0.0
            last_cut = 0.0
            timestamp = 0.0
            fallback_step = max(1.0, step)
            while timestamp <= duration:
                gray = self._read_visual_restore_gray(context, "narration", timestamp)
                if gray is not None:
                    consider_frame(gray, timestamp)
                timestamp += fallback_step

        usable_segments = [
            segment for segment in (getattr(project, "segments", []) or [])
            if segment.use_segment and segment.segment_type != SegmentType.NON_MOVIE
        ]
        usable_segments.sort(key=lambda item: float(item.narration_start))
        for segment in usable_segments:
            if not segment.use_segment or segment.segment_type == SegmentType.NON_MOVIE:
                continue
            for boundary in (float(segment.narration_start), float(segment.narration_end)):
                if 0.0 < boundary < duration:
                    cuts.append(boundary)

        def primary_source_range(segment: Segment) -> tuple[Optional[float], Optional[float], bool]:
            if segment.movie_start is not None and segment.movie_end is not None:
                return float(segment.movie_start), float(segment.movie_end), False
            candidates = sorted(
                segment.match_candidates or [],
                key=lambda item: (item.rank or 999, -(item.score or 0.0)),
            )
            for candidate in candidates[:2]:
                candidate_score = max(
                    float(candidate.score or 0.0),
                    float(candidate.confidence or 0.0),
                    float(candidate.visual_confidence or 0.0),
                    float(candidate.verification_score or 0.0),
                )
                if candidate_score >= 0.55:
                    return float(candidate.start), float(candidate.end), True
            return None, None, True

        for previous_segment, next_segment in zip(usable_segments, usable_segments[1:]):
            _prev_start, previous_source_end, prev_from_candidate = primary_source_range(previous_segment)
            next_source_start, _next_end, next_from_candidate = primary_source_range(next_segment)
            if previous_source_end is None or next_source_start is None:
                continue
            boundary = float(next_segment.narration_start)
            source_jump = next_source_start - previous_source_end
            jump_threshold = (
                self.VISUAL_RESTORE_CANDIDATE_PROTECTED_SOURCE_JUMP
                if prev_from_candidate or next_from_candidate
                else self.VISUAL_RESTORE_PROTECTED_SOURCE_JUMP
            )
            if 0.0 < boundary < duration and abs(source_jump) >= jump_threshold:
                protected_visual_cuts.append(boundary)
        cuts = sorted(cuts)

        if duration - cuts[-1] >= 0.18:
            cuts.append(duration)
        elif len(cuts) == 1:
            cuts.append(duration)
        else:
            cuts[-1] = duration

        normalized = [cuts[0]]
        for cut in cuts[1:]:
            if cut - normalized[-1] < 0.18:
                continue
            else:
                normalized.append(cut)
        context["protected_visual_cuts"] = protected_visual_cuts
        context["detected_visual_cuts"] = [
            cut for cut in sorted(detected_visual_cuts)
            if 0.0 < cut < duration
        ]
        return normalized

    def _visual_restore_candidate_seeds(
        self,
        project: Project,
        segments: list[Segment],
        target_start: float,
        target_end: float,
        duration: float,
        last_source_end: Optional[float],
    ) -> list[tuple[float, float]]:
        center = (target_start + target_end) * 0.5
        related: list[tuple[float, float, Segment]] = []
        for segment in segments:
            if not segment.use_segment or segment.segment_type == SegmentType.NON_MOVIE:
                continue
            if not segment.match_candidates and segment.movie_start is None:
                continue
            overlap = max(
                0.0,
                min(target_end, float(segment.narration_end) + 0.45)
                - max(target_start, float(segment.narration_start) - 0.45),
            )
            segment_center = (float(segment.narration_start) + float(segment.narration_end)) * 0.5
            distance = abs(center - segment_center)
            if overlap <= 0.0 and distance > max(1.6, duration * 1.2):
                continue
            related.append((overlap, -distance, segment))

        related.sort(reverse=True, key=lambda item: (item[0], item[1]))
        seeds: list[tuple[float, float]] = []
        movie_duration = float(project.movie_duration or 0.0)

        def add_seed(source_start: float, score: float) -> None:
            if movie_duration > 0.0:
                source_start = min(max(0.0, source_start), max(0.0, movie_duration - duration))
            else:
                source_start = max(0.0, source_start)
            seeds.append((source_start, max(0.0, min(1.0, float(score)))))

        for overlap, _neg_distance, segment in related[:4]:
            segment_offset = target_start - float(segment.narration_start)
            candidates = []
            selected = self._selected_candidate_for_segment(segment)
            if selected is not None:
                candidates.append(selected)
            for candidate in sorted(segment.match_candidates or [], key=lambda item: (item.rank or 999, -item.score)):
                if selected is not None and candidate.id == selected.id:
                    continue
                candidates.append(candidate)
                if len(candidates) >= 2:
                    break
            for candidate in candidates:
                verify_score = float(candidate.verification_score or 0.0)
                reason_match = re.search(r"post_verify=([0-9.]+)", candidate.reason or "")
                if reason_match:
                    verify_score = max(verify_score, float(reason_match.group(1)))
                base_score = max(
                    float(candidate.score or 0.0),
                    float(candidate.confidence or 0.0),
                    float(candidate.visual_confidence or 0.0),
                    verify_score,
                )
                overlap_ratio = max(0.0, min(1.0, overlap / max(0.2, duration)))
                relation_score = base_score * 0.62 + overlap_ratio * 0.38
                add_seed(float(candidate.start) + segment_offset, relation_score)
            if segment.movie_start is not None:
                overlap_ratio = max(0.0, min(1.0, overlap / max(0.2, duration)))
                relation_score = float(segment.match_confidence or 0.0) * 0.62 + overlap_ratio * 0.38
                add_seed(float(segment.movie_start) + segment_offset, relation_score)

        if last_source_end is not None:
            add_seed(last_source_end, 0.45)

        deduped: dict[float, tuple[float, float]] = {}
        for source_start, score in seeds:
            key = round(source_start, 1)
            if key not in deduped or score > deduped[key][1]:
                deduped[key] = (source_start, score)
        return sorted(deduped.values(), key=lambda item: item[1], reverse=True)[: self.VISUAL_RESTORE_MAX_SEEDS]

    def _visual_restore_global_index_search(
        self,
        context,
        target_start: float,
        duration: float,
        movie_duration: float,
    ) -> Optional[tuple[float, float]]:
        frame_matcher = context.get("frame_matcher")
        movie_index = context.get("movie_index") or []
        movie_times = context.get("movie_times") or []
        if frame_matcher is None or not movie_index or not movie_times:
            return None

        offset = duration * 0.5
        query_feature = self._visual_restore_query_feature(context, target_start + offset)
        if query_feature is None:
            return None

        coarse_hits: list[tuple[float, float]] = []
        np = context["np"]
        try:
            idx_ahash = getattr(frame_matcher, "_idx_ahash_packed", None)
            idx_phash = getattr(frame_matcher, "_idx_phash_packed", None)
            idx_edge = getattr(frame_matcher, "_idx_edge_packed", None)
            idx_color = getattr(frame_matcher, "_idx_color_hist", None)
            idx_spatial = getattr(frame_matcher, "_idx_spatial_color_hist", None)
            idx_grad = getattr(frame_matcher, "_idx_grad_hist", None)
            query_full = (query_feature.get("variants") or {}).get("full")
            if idx_ahash is None or idx_phash is None or query_full is None:
                raise ValueError("packed index unavailable")

            popcount_table = frame_matcher.__class__._POPCOUNT_TABLE
            q_ahash = np.packbits(query_full["hash"]).astype(np.uint8)
            q_phash = np.packbits(query_full["phash"]).astype(np.uint8)
            ahash_score = 1.0 - popcount_table[idx_ahash ^ q_ahash].sum(axis=1, dtype=np.int32) / 256.0
            phash_score = 1.0 - popcount_table[idx_phash ^ q_phash].sum(axis=1, dtype=np.int32) / 64.0
            coarse_scores = ahash_score * 0.20 + phash_score * 0.28

            gpu_dense_scores = self._visual_restore_gpu_dense_scores(
                context,
                query_feature,
                cache_key=f"range:{round(target_start + offset, 2):.2f}",
            )
            if gpu_dense_scores is not None:
                coarse_scores += gpu_dense_scores
            else:
                if idx_color is not None:
                    q_color = np.asarray(query_feature.get("color_hist"), dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_color))
                    if q_norm > 0 and q_color.shape[-1] == idx_color.shape[1]:
                        coarse_scores += np.clip((idx_color @ (q_color / q_norm)), 0.0, 1.0) * 0.20
                if idx_spatial is not None:
                    q_spatial = np.asarray(query_feature.get("spatial_color_hist"), dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_spatial))
                    if q_norm > 0 and q_spatial.shape[-1] == idx_spatial.shape[1]:
                        coarse_scores += np.clip((idx_spatial @ (q_spatial / q_norm)), 0.0, 1.0) * 0.24
                if idx_grad is not None:
                    q_grad = np.asarray(query_feature.get("grad_hist"), dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_grad))
                    if q_norm > 0 and q_grad.shape[-1] == idx_grad.shape[1]:
                        coarse_scores += np.clip((idx_grad @ (q_grad / q_norm)), 0.0, 1.0) * 0.05
            if idx_edge is not None:
                q_edge = np.packbits(np.asarray(query_feature.get("edge_hash"), dtype=np.uint8))
                edge_score = 1.0 - popcount_table[idx_edge ^ q_edge].sum(axis=1, dtype=np.int32) / 144.0
                coarse_scores += edge_score * 0.03

            top_n = min(len(movie_times), max(self.VISUAL_RESTORE_GLOBAL_CANDIDATES * 2, 256))
            top_indices = np.argpartition(coarse_scores, -top_n)[-top_n:]
            for idx in top_indices[np.argsort(coarse_scores[top_indices])[::-1]]:
                coarse_score = float(coarse_scores[idx])
                if coarse_score < self.VISUAL_RESTORE_GLOBAL_COARSE_FLOOR:
                    break
                source_start = float(movie_times[int(idx)]) - offset
                if movie_duration > 0.0:
                    source_start = min(max(0.0, source_start), max(0.0, movie_duration - duration))
                else:
                    source_start = max(0.0, source_start)
                coarse_hits.append((coarse_score, source_start))
        except Exception:
            for candidate, movie_time in zip(movie_index, movie_times):
                coarse_score = float(frame_matcher._feature_score(query_feature, candidate))
                if coarse_score < self.VISUAL_RESTORE_GLOBAL_COARSE_FLOOR:
                    continue
                source_start = float(movie_time) - offset
                if movie_duration > 0.0:
                    source_start = min(max(0.0, source_start), max(0.0, movie_duration - duration))
                else:
                    source_start = max(0.0, source_start)
                coarse_hits.append((coarse_score, source_start))

        if not coarse_hits:
            return None

        deduped: list[tuple[float, float]] = []
        seen: list[float] = []
        for coarse_score, source_start in sorted(coarse_hits, reverse=True):
            if any(abs(source_start - previous_start) < 0.35 for previous_start in seen):
                continue
            seen.append(source_start)
            deduped.append((coarse_score, source_start))
            if len(deduped) >= self.VISUAL_RESTORE_GLOBAL_CANDIDATES:
                break

        best_source_start: Optional[float] = None
        best_range_score = 0.0
        for _coarse_score, source_start in deduped:
            range_score = self._score_visual_restore_indexed_range(
                context,
                target_start,
                duration,
                source_start,
            )
            if range_score > best_range_score:
                best_range_score = range_score
                best_source_start = source_start

        if best_source_start is None:
            return None
        return best_source_start, best_range_score

    def _visual_restore_global_frame_hits(
        self,
        context,
        target_time: float,
        limit: int = 24,
    ) -> list[tuple[float, float]]:
        frame_matcher = context.get("frame_matcher")
        movie_index = context.get("movie_index") or []
        movie_times = context.get("movie_times") or []
        if frame_matcher is None or not movie_index or not movie_times:
            return []

        query_feature = self._visual_restore_query_feature(context, target_time)
        if query_feature is None:
            return []
        query_gray = self._read_visual_restore_gray(context, "narration", target_time)

        np = context["np"]
        scored: list[tuple[float, float]] = []
        try:
            idx_ahash = getattr(frame_matcher, "_idx_ahash_packed", None)
            idx_phash = getattr(frame_matcher, "_idx_phash_packed", None)
            idx_edge = getattr(frame_matcher, "_idx_edge_packed", None)
            idx_color = getattr(frame_matcher, "_idx_color_hist", None)
            idx_spatial = getattr(frame_matcher, "_idx_spatial_color_hist", None)
            idx_grad = getattr(frame_matcher, "_idx_grad_hist", None)
            query_full = (query_feature.get("variants") or {}).get("full")
            if idx_ahash is None or idx_phash is None or query_full is None:
                raise ValueError("packed index unavailable")

            popcount_table = frame_matcher.__class__._POPCOUNT_TABLE
            q_ahash = np.packbits(query_full["hash"]).astype(np.uint8)
            q_phash = np.packbits(query_full["phash"]).astype(np.uint8)
            ahash_score = 1.0 - popcount_table[idx_ahash ^ q_ahash].sum(axis=1, dtype=np.int32) / 256.0
            phash_score = 1.0 - popcount_table[idx_phash ^ q_phash].sum(axis=1, dtype=np.int32) / 64.0
            coarse_scores = ahash_score * 0.20 + phash_score * 0.28

            gpu_dense_scores = self._visual_restore_gpu_dense_scores(
                context,
                query_feature,
                cache_key=f"frame:{round(target_time, 2):.2f}",
            )
            if gpu_dense_scores is not None:
                coarse_scores += gpu_dense_scores
            else:
                if idx_color is not None:
                    q_color = np.asarray(query_feature.get("color_hist"), dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_color))
                    if q_norm > 0 and q_color.shape[-1] == idx_color.shape[1]:
                        coarse_scores += np.clip((idx_color @ (q_color / q_norm)), 0.0, 1.0) * 0.20
                if idx_spatial is not None:
                    q_spatial = np.asarray(query_feature.get("spatial_color_hist"), dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_spatial))
                    if q_norm > 0 and q_spatial.shape[-1] == idx_spatial.shape[1]:
                        coarse_scores += np.clip((idx_spatial @ (q_spatial / q_norm)), 0.0, 1.0) * 0.24
                if idx_grad is not None:
                    q_grad = np.asarray(query_feature.get("grad_hist"), dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_grad))
                    if q_norm > 0 and q_grad.shape[-1] == idx_grad.shape[1]:
                        coarse_scores += np.clip((idx_grad @ (q_grad / q_norm)), 0.0, 1.0) * 0.05
            if idx_edge is not None:
                q_edge = np.packbits(np.asarray(query_feature.get("edge_hash"), dtype=np.uint8))
                edge_score = 1.0 - popcount_table[idx_edge ^ q_edge].sum(axis=1, dtype=np.int32) / 144.0
                coarse_scores += edge_score * 0.03

            top_n = min(len(movie_times), max(limit * 4, 48))
            top_indices = np.argpartition(coarse_scores, -top_n)[-top_n:]
            for idx in top_indices[np.argsort(coarse_scores[top_indices])[::-1]]:
                if float(coarse_scores[idx]) < self.VISUAL_RESTORE_GLOBAL_COARSE_FLOOR:
                    break
                exact_score = float(frame_matcher._feature_score(query_feature, movie_index[int(idx)]))
                movie_time = float(movie_times[int(idx)])
                pair_score = 0.0
                if query_gray is not None:
                    movie_gray = self._read_visual_restore_gray(context, "movie", movie_time)
                    pair_score = self._score_visual_restore_pair(context, query_gray, movie_gray)
                final_score = exact_score * 0.25 + pair_score * 0.75 if pair_score > 0.0 else exact_score
                if pair_score > 0.0 and pair_score < 0.45:
                    final_score *= 0.62
                scored.append((final_score, movie_time))
        except Exception:
            for candidate, movie_time in zip(movie_index, movie_times):
                score = float(frame_matcher._feature_score(query_feature, candidate))
                if score >= self.VISUAL_RESTORE_GLOBAL_COARSE_FLOOR:
                    scored.append((score, float(movie_time)))

        deduped: list[tuple[float, float]] = []
        seen: list[float] = []
        for score, movie_time in sorted(scored, reverse=True):
            if any(abs(movie_time - previous_time) < 0.35 for previous_time in seen):
                continue
            seen.append(movie_time)
            deduped.append((movie_time, score))
            if len(deduped) >= limit:
                break
        return deduped

    def _select_visual_restore_shot_source(
        self,
        context,
        target_start: float,
        target_end: float,
        last_source_end: Optional[float],
    ) -> Optional[tuple[float, float, bool]]:
        duration = max(0.05, target_end - target_start)
        movie_duration = float(context.get("movie_duration") or 0.0)
        positions = self._visual_restore_sample_positions(duration)
        if duration >= 5.0:
            positions = [0.15, 0.34, 0.50, 0.66, 0.85]
        elif duration >= 3.0:
            positions = [0.20, 0.50, 0.80]

        candidates: dict[float, float] = {}
        for pos in positions:
            query_time = target_start + duration * pos
            for movie_time, hit_score in self._visual_restore_global_frame_hits(context, query_time, limit=18):
                source_start = movie_time - duration * pos
                if movie_duration > 0.0:
                    source_start = min(max(0.0, source_start), max(0.0, movie_duration - duration))
                else:
                    source_start = max(0.0, source_start)
                key = round(source_start * 2.0) / 2.0
                candidates[key] = max(candidates.get(key, 0.0), hit_score)

        if last_source_end is not None:
            candidates[round(last_source_end * 2.0) / 2.0] = max(
                candidates.get(round(last_source_end * 2.0) / 2.0, 0.0),
                0.45,
            )

        if not candidates:
            return None

        best_source_start: Optional[float] = None
        best_score = -1.0
        best_is_inferred = False
        for source_start, seed_score in candidates.items():
            range_score = self._score_visual_restore_indexed_range(
                context,
                target_start,
                duration,
                source_start,
            )
            continuity_bonus = 0.0
            if last_source_end is not None:
                source_jump = abs(source_start - last_source_end)
                if source_jump <= 1.2:
                    continuity_bonus = 0.035
                elif source_jump <= 4.0:
                    continuity_bonus = 0.018
            total_score = range_score * 0.92 + seed_score * 0.08 + continuity_bonus
            if total_score > best_score:
                best_score = total_score
                best_source_start = source_start
                best_is_inferred = seed_score <= 0.46

        if best_source_start is None:
            return None
        return best_source_start, max(0.0, min(1.0, best_score)), best_is_inferred

    def _select_visual_restore_source(
        self,
        context,
        project: Project,
        segments: list[Segment],
        target_start: float,
        target_end: float,
        last_source_end: Optional[float],
    ) -> Optional[tuple[float, float, bool]]:
        duration = max(0.05, target_end - target_start)
        protected_start = self._is_protected_visual_restore_boundary(context, target_start)
        has_visual_cut = target_start <= 0.08 or self._is_detected_visual_restore_cut(context, target_start)
        prefer_continuity = last_source_end is not None and not protected_start and not has_visual_cut
        seeds = self._visual_restore_candidate_seeds(
            project,
            segments,
            target_start,
            target_end,
            duration,
            None if protected_start else last_source_end,
        )
        if not seeds:
            return None
        anchor_source_start: Optional[float] = None
        anchor_seed_score = 0.0
        for seed_start, seed_score in seeds:
            if seed_score >= self.VISUAL_RESTORE_ANCHOR_LOCK_SCORE:
                anchor_source_start = seed_start
                anchor_seed_score = seed_score
                break

        frame_matcher = context.get("frame_matcher")
        movie_index = context.get("movie_index") or []
        movie_times = context.get("movie_times") or []
        query_feature = self._visual_restore_query_feature(context, target_start + duration * 0.5)
        if frame_matcher is not None and query_feature is not None and movie_index and movie_times:
            offset = duration * 0.5
            movie_duration = float(context.get("movie_duration") or project.movie_duration or 0.0)
            tested: set[int] = set()
            search_radius = max(8.0, min(24.0, duration * 5.0))
            coarse_candidates: list[tuple[float, float, float, float]] = []
            for seed_start, seed_score in seeds:
                center_hint = seed_start + offset
                left = bisect.bisect_left(movie_times, max(0.0, center_hint - search_radius))
                right = bisect.bisect_right(movie_times, center_hint + search_radius)
                if right <= left:
                    continue
                for idx in range(left, right):
                    if idx in tested:
                        continue
                    tested.add(idx)
                    candidate = movie_index[idx]
                    movie_time = float(candidate.get("time", movie_times[idx]))
                    source_start = movie_time - offset
                    if movie_duration > 0.0:
                        source_start = min(max(0.0, source_start), max(0.0, movie_duration - duration))
                    else:
                        source_start = max(0.0, source_start)
                    feature_score = float(frame_matcher._feature_score(query_feature, candidate))
                    continuity_score = 0.0
                    if last_source_end is not None and not protected_start:
                        jump = abs(source_start - last_source_end)
                        continuity_score = max(0.0, 1.0 - min(1.0, jump / max(12.0, duration * 5.0)))
                    if prefer_continuity:
                        total_score = feature_score * 0.65 + seed_score * 0.05 + continuity_score * 0.30
                    else:
                        total_score = feature_score * 0.78 + seed_score * 0.12 + continuity_score * 0.10
                    coarse_candidates.append((total_score, source_start, seed_score, continuity_score))
            if coarse_candidates:
                best_source_start = None
                best_range_score = 0.0
                best_total_score = -1.0
                best_from_global = False
                candidate_limit = 3 if duration <= 1.5 else (4 if duration <= 4.5 else 8)
                for _coarse_score, source_start, seed_score, continuity_score in sorted(coarse_candidates, reverse=True)[:candidate_limit]:
                    range_score = self._score_visual_restore_indexed_range(context, target_start, duration, source_start)
                    total_score = range_score * 0.82 + seed_score * 0.08 + continuity_score * 0.10
                    if total_score > best_total_score:
                        best_total_score = total_score
                        best_range_score = range_score
                        best_source_start = source_start

                if best_source_start is not None and last_source_end is not None and not protected_start:
                    source_jump = best_source_start - last_source_end
                    if prefer_continuity and 1.2 < abs(source_jump) <= self.VISUAL_RESTORE_NONCUT_MAX_JUMP:
                        continuity_score = self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            last_source_end,
                        )
                        if continuity_score >= best_range_score - self.VISUAL_RESTORE_NONCUT_CONTINUITY_MARGIN:
                            best_source_start = last_source_end
                            best_range_score = continuity_score
                    elif 1.2 < abs(source_jump) <= max(8.0, duration * 2.0):
                        continuity_score = self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            last_source_end,
                        )
                        # If the continuous source is nearly as good, prefer it:
                        # this removes small source-time hops that look like dropped
                        # frames while preserving real hard cuts.
                        if continuity_score >= best_range_score - 0.055:
                            best_source_start = last_source_end
                            best_range_score = continuity_score
                            best_from_global = False
                if best_source_start is not None:
                    local_jump = (
                        abs(best_source_start - last_source_end)
                        if last_source_end is not None
                        else 0.0
                    )
                    local_already_jumped = (
                        last_source_end is not None
                        and local_jump > self.VISUAL_RESTORE_NONCUT_MAX_JUMP
                    )
                    should_try_global = (
                        (local_already_jumped and best_range_score < 0.76)
                        or best_range_score < 0.62
                        or (protected_start and best_range_score < 0.70)
                    )
                    if should_try_global:
                        global_candidate = self._visual_restore_global_index_search(
                            context,
                            target_start,
                            duration,
                            movie_duration,
                        )
                        if global_candidate is not None:
                            global_source_start, global_range_score = global_candidate
                            required_gain = self.VISUAL_RESTORE_GLOBAL_FALLBACK_GAIN
                            global_changes_scene = (
                                abs(global_source_start - best_source_start)
                                > self.VISUAL_RESTORE_NONCUT_MAX_JUMP
                            )
                            if local_already_jumped and global_changes_scene:
                                required_gain = 0.0
                            elif local_already_jumped or best_range_score < 0.76:
                                required_gain *= 0.40
                            elif best_range_score < 0.66:
                                required_gain *= 0.45
                            minimum_global_score = 0.70 if best_range_score < 0.62 else 0.74
                            if (
                                global_range_score >= minimum_global_score
                                and global_range_score >= best_range_score + required_gain
                            ):
                                best_source_start = global_source_start
                                best_range_score = global_range_score
                                best_total_score = max(best_total_score, global_range_score)
                                best_from_global = True
                                if global_changes_scene:
                                    context.setdefault("verified_visual_jumps", []).append(float(target_start))
                if (
                    best_source_start is not None
                    and anchor_source_start is not None
                    and abs(best_source_start - anchor_source_start) > self.VISUAL_RESTORE_ANCHOR_LOCK_RADIUS
                ):
                    anchor_range_score = self._score_visual_restore_indexed_range(
                        context,
                        target_start,
                        duration,
                        anchor_source_start,
                    )
                    # Similar frames in this movie can pull retrieval hundreds of
                    # seconds away. A strong segment anchor is safer than a visually
                    # close but temporally wrong free-search hit.
                    if best_from_global or has_visual_cut or protected_start:
                        anchor_margin = 0.02
                    else:
                        anchor_margin = self.VISUAL_RESTORE_ANCHOR_ESCAPE_MARGIN
                    if last_source_end is not None:
                        anchor_jump = abs(anchor_source_start - last_source_end)
                        best_jump = abs(best_source_start - last_source_end)
                        if anchor_jump > self.VISUAL_RESTORE_NONCUT_MAX_JUMP and best_jump <= self.VISUAL_RESTORE_NONCUT_MAX_JUMP:
                            anchor_margin = min(anchor_margin, 0.01)
                    keep_anchor = anchor_range_score >= best_range_score - anchor_margin
                    if keep_anchor:
                        best_source_start = anchor_source_start
                        best_range_score = max(anchor_range_score, anchor_seed_score * 0.8)
            if best_source_start is not None:
                return best_source_start, best_range_score, False

        best_source_start = None
        best_total_score = -1.0
        movie_duration = float(context.get("movie_duration") or project.movie_duration or 0.0)
        for seed_start, seed_score in seeds:
            source_start = seed_start
            if movie_duration > 0.0:
                source_start = min(max(0.0, source_start), max(0.0, movie_duration - duration))
            else:
                source_start = max(0.0, source_start)
            continuity_score = 0.0
            if last_source_end is not None and not protected_start:
                jump = abs(source_start - last_source_end)
                continuity_score = max(0.0, 1.0 - min(1.0, jump / max(12.0, duration * 5.0)))
            if prefer_continuity:
                total_score = seed_score * 0.58 + continuity_score * 0.42
            else:
                total_score = seed_score * 0.86 + continuity_score * 0.14
            if total_score > best_total_score:
                best_total_score = total_score
                best_source_start = source_start
        if (
            prefer_continuity
            and best_source_start is not None
            and last_source_end is not None
            and 1.2 < abs(best_source_start - last_source_end) <= self.VISUAL_RESTORE_NONCUT_MAX_JUMP
        ):
            continuity_score = self._score_visual_restore_indexed_range(
                context,
                target_start,
                duration,
                last_source_end,
            )
            best_score_for_compare = max(0.0, min(1.0, best_total_score))
            if continuity_score >= best_score_for_compare - self.VISUAL_RESTORE_NONCUT_CONTINUITY_MARGIN:
                best_source_start = last_source_end
                best_total_score = max(best_total_score, continuity_score)
        if (
            best_source_start is not None
            and anchor_source_start is not None
            and abs(best_source_start - anchor_source_start) > self.VISUAL_RESTORE_ANCHOR_LOCK_RADIUS
            and anchor_seed_score >= self.VISUAL_RESTORE_ANCHOR_LOCK_SCORE
        ):
            best_source_start = anchor_source_start
            best_total_score = max(best_total_score, anchor_seed_score)
        if best_source_start is None:
            return None
        return best_source_start, max(0.0, min(1.0, best_total_score)), False

    def _merge_visual_restore_ranges(
        self,
        previous: tuple[float, float, float, float, bool, float],
        current: tuple[float, float, float, float, bool, float],
        movie_duration: float,
    ) -> tuple[float, float, float, float, bool, float]:
        target_start, _target_end, source_start, _source_end, prev_inferred, prev_score = previous
        _cur_target_start, cur_target_end, _cur_source_start, _cur_source_end, cur_inferred, cur_score = current
        total_duration = max(0.0, cur_target_end - target_start)
        merged_source_start = source_start
        merged_source_end = merged_source_start + total_duration
        if movie_duration > 0.0 and merged_source_end > movie_duration:
            merged_source_end = movie_duration
            merged_source_start = max(0.0, merged_source_end - total_duration)
        return (
            target_start,
            cur_target_end,
            merged_source_start,
            merged_source_end,
            prev_inferred or cur_inferred,
            min(prev_score, cur_score),
        )

    def _smooth_visual_restore_ranges(
        self,
        ranges: list[tuple[float, float, float, float, bool, float]],
        movie_duration: float,
        protected_cuts: Optional[list[float]] = None,
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Merge micro-cuts that cause Jianying preview/render stutter.

        The matcher may produce many sub-second cuts because it combines speech
        boundaries with visual cuts. When the movie source is already nearly
        continuous, exporting those as separate clips only creates playback
        jitter. Keep hard jumps, but merge continuous or near-continuous shards.
        """
        if not ranges:
            return []
        protected_cuts = sorted(float(item) for item in (protected_cuts or []))

        def crosses_protected_cut(boundary: float) -> bool:
            if not protected_cuts:
                return False
            idx = bisect.bisect_left(protected_cuts, boundary - 0.08)
            return idx < len(protected_cuts) and protected_cuts[idx] <= boundary + 0.08

        smoothed: list[tuple[float, float, float, float, bool, float]] = []
        for current in sorted(ranges, key=lambda item: item[0]):
            target_start, target_end, source_start, _source_end, _is_inferred, _visual_score = current
            duration = max(0.0, target_end - target_start)
            if smoothed:
                prev = smoothed[-1]
                target_gap = target_start - prev[1]
                source_jump = source_start - prev[3]
                target_contiguous = abs(target_gap) <= 0.05
                merge_continuous = (
                    target_contiguous
                    and not crosses_protected_cut(target_start)
                    and abs(source_jump) <= self.VISUAL_RESTORE_CONTINUITY_TOLERANCE
                )
                merge_short = (
                    target_contiguous
                    and not crosses_protected_cut(target_start)
                    and duration < self.VISUAL_RESTORE_MIN_SMOOTH_SEGMENT
                    and abs(source_jump) <= self.VISUAL_RESTORE_SHORT_MERGE_TOLERANCE
                )
                if merge_continuous or merge_short:
                    smoothed[-1] = self._merge_visual_restore_ranges(prev, current, movie_duration)
                    continue
            smoothed.append(current)

        final: list[tuple[float, float, float, float, bool, float]] = []
        for current in smoothed:
            duration = max(0.0, current[1] - current[0])
            if final and duration < self.VISUAL_RESTORE_TINY_SEGMENT:
                prev = final[-1]
                target_gap = current[0] - prev[1]
                source_jump = current[2] - prev[3]
                if (
                    abs(target_gap) <= 0.05
                    and not crosses_protected_cut(current[0])
                    and abs(source_jump) <= self.VISUAL_RESTORE_SHORT_MERGE_TOLERANCE * 1.5
                ):
                    final[-1] = self._merge_visual_restore_ranges(prev, current, movie_duration)
                    continue
            final.append(current)

        anti_flicker: list[tuple[float, float, float, float, bool, float]] = []
        flicker_replacements = 0
        index = 0
        while index < len(final):
            current = final[index]
            if anti_flicker and index + 1 < len(final):
                prev = anti_flicker[-1]
                next_range = final[index + 1]
                duration = max(0.0, current[1] - current[0])
                target_gap_prev = current[0] - prev[1]
                target_gap_next = next_range[0] - current[1]
                direct_source_gap = next_range[2] - prev[3]
                jump_in = current[2] - prev[3]
                jump_out = next_range[2] - current[3]
                is_isolated_flicker = (
                    duration <= self.VISUAL_RESTORE_FLICKER_MAX_DURATION
                    and abs(target_gap_prev) <= 0.05
                    and abs(target_gap_next) <= 0.05
                    and not crosses_protected_cut(current[0])
                    and not crosses_protected_cut(current[1])
                    and abs(direct_source_gap) <= self.VISUAL_RESTORE_FLICKER_DIRECT_TOLERANCE
                    and (
                        abs(jump_in) >= self.VISUAL_RESTORE_FLICKER_JUMP_MIN
                        or abs(jump_out) >= self.VISUAL_RESTORE_FLICKER_JUMP_MIN
                    )
                )
                if is_isolated_flicker:
                    replacement = (
                        current[0],
                        current[1],
                        prev[3],
                        prev[3] + duration,
                        True,
                        min(prev[5], current[5]),
                    )
                    anti_flicker[-1] = self._merge_visual_restore_ranges(prev, replacement, movie_duration)
                    flicker_replacements += 1
                    index += 1
                    continue
            anti_flicker.append(current)
            index += 1

        compacted: list[tuple[float, float, float, float, bool, float]] = []
        for current in anti_flicker:
            if compacted:
                prev = compacted[-1]
                target_gap = current[0] - prev[1]
                source_jump = current[2] - prev[3]
                if (
                    abs(target_gap) <= 0.05
                    and not crosses_protected_cut(current[0])
                    and abs(source_jump) <= self.VISUAL_RESTORE_CONTINUITY_TOLERANCE
                ):
                    compacted[-1] = self._merge_visual_restore_ranges(prev, current, movie_duration)
                    continue
            compacted.append(current)

        if flicker_replacements:
            logger.info("Visual restore anti-flicker replaced {} isolated source jumps", flicker_replacements)

        anti_repeat: list[tuple[float, float, float, float, bool, float]] = []
        repeat_replacements = 0
        for current in compacted:
            if anti_repeat:
                prev = anti_repeat[-1]
                target_gap = current[0] - prev[1]
                source_jump = current[2] - prev[3]
                duration = max(0.0, current[1] - current[0])
                overlap = max(0.0, -source_jump)
                is_replay_overlap = (
                    abs(target_gap) <= 0.05
                    and source_jump < -self.VISUAL_RESTORE_REPEAT_BACKTRACK_TOLERANCE
                    and overlap >= self.VISUAL_RESTORE_REPEAT_MIN_OVERLAP_SECONDS
                    and overlap / max(duration, 0.1) >= self.VISUAL_RESTORE_REPEAT_MIN_OVERLAP_RATIO
                )
                if is_replay_overlap:
                    protected_boundary = crosses_protected_cut(current[0])
                    if protected_boundary:
                        anti_repeat.append(current)
                        continue
                    if not protected_boundary or duration <= self.VISUAL_RESTORE_FLICKER_MAX_DURATION:
                        replacement = (
                            current[0],
                            current[1],
                            prev[3],
                            prev[3] + duration,
                            True,
                            min(prev[5], current[5]),
                        )
                        anti_repeat.append(replacement)
                        repeat_replacements += 1
                        continue
                    if protected_boundary:
                        continuous_end = prev[3] + duration
                        local_repeat_limit = max(15.0, duration * 2.0)
                        if overlap <= local_repeat_limit and (movie_duration <= 0.0 or continuous_end <= movie_duration):
                            replacement = (
                                current[0],
                                current[1],
                                prev[3],
                                continuous_end,
                                True,
                                min(prev[5], current[5]),
                            )
                            anti_repeat.append(replacement)
                            repeat_replacements += 1
                            continue
                    prev_duration = max(0.0, prev[1] - prev[0])
                    if protected_boundary and prev_duration <= self.VISUAL_RESTORE_FLICKER_MAX_DURATION:
                        prior = anti_repeat[-2] if len(anti_repeat) >= 2 else None
                        if prior is not None:
                            prior_duration = max(0.0, prior[1] - prior[0])
                            total_precut_duration = max(0.0, prev[1] - prior[0])
                            prior_source_gap = prev[2] - prior[3]
                            adjusted_source_start = max(0.0, current[2] - total_precut_duration)
                            previous_block = anti_repeat[-3] if len(anti_repeat) >= 3 else None
                            previous_block_ok = (
                                previous_block is None
                                or adjusted_source_start >= previous_block[3] - self.VISUAL_RESTORE_CONTINUITY_TOLERANCE
                            )
                            if (
                                abs(prev[0] - prior[1]) <= 0.05
                                and abs(prior_source_gap) <= self.VISUAL_RESTORE_CONTINUITY_TOLERANCE
                                and total_precut_duration <= self.VISUAL_RESTORE_PRECUT_RETIMER_MAX_DURATION
                                and previous_block_ok
                            ):
                                split_source = adjusted_source_start + prior_duration
                                anti_repeat[-2] = (
                                    prior[0],
                                    prior[1],
                                    adjusted_source_start,
                                    split_source,
                                    True,
                                    min(prior[5], current[5]),
                                )
                                anti_repeat[-1] = (
                                    prev[0],
                                    prev[1],
                                    split_source,
                                    current[2],
                                    True,
                                    min(prev[5], current[5]),
                                )
                                repeat_replacements += 1
                                anti_repeat.append(current)
                                continue
                        adjusted_source_end = max(0.0, current[2])
                        adjusted_source_start = max(0.0, adjusted_source_end - prev_duration)
                        anti_repeat[-1] = (
                            prev[0],
                            prev[1],
                            adjusted_source_start,
                            adjusted_source_end,
                            True,
                            min(prev[5], current[5]),
                        )
                        repeat_replacements += 1
            anti_repeat.append(current)

        if repeat_replacements:
            logger.info("Visual restore anti-repeat removed {} source backtracks", repeat_replacements)
        return anti_repeat

    def _repair_visual_restore_isolated_excursions(
        self,
        context,
        ranges: list[tuple[float, float, float, float, bool, float]],
        movie_duration: float,
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Remove short far-away excursions that immediately jump back.

        These are usually caused by visually similar scenes winning retrieval for
        one tiny clip. They look especially bad in playback: the picture jumps
        away for a second, then cuts back to the original action.
        """
        if len(ranges) < 3:
            return ranges

        ordered = sorted(ranges, key=lambda item: item[0])
        repaired: list[tuple[float, float, float, float, bool, float]] = []
        replacements = 0
        index = 0
        while index < len(ordered):
            current = ordered[index]
            if repaired and index + 1 < len(ordered):
                prev = repaired[-1]
                next_range = ordered[index + 1]
                target_start, target_end, source_start, source_end, is_inferred, visual_score = current
                duration = max(0.0, target_end - target_start)
                jump_in = source_start - prev[3]
                jump_out = next_range[2] - source_end
                returns_to_previous_timeline = abs(next_range[2] - prev[3]) <= self.VISUAL_RESTORE_EXCURSION_RETURN_TOLERANCE
                is_short_excursion = (
                    duration <= self.VISUAL_RESTORE_EXCURSION_MAX_DURATION
                    and abs(current[0] - prev[1]) <= 0.05
                    and abs(next_range[0] - current[1]) <= 0.05
                    and abs(jump_in) >= self.VISUAL_RESTORE_EXCURSION_MIN_JUMP
                    and abs(jump_out) >= self.VISUAL_RESTORE_EXCURSION_MIN_JUMP
                    and jump_in * jump_out < 0.0
                    and returns_to_previous_timeline
                )
                if is_short_excursion:
                    bridge_options: list[tuple[float, float]] = []
                    if movie_duration <= 0.0 or prev[3] + duration <= movie_duration:
                        bridge_options.append((prev[3], self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            prev[3],
                        )))
                    source_to_next = max(0.0, next_range[2] - duration)
                    if movie_duration <= 0.0 or source_to_next + duration <= movie_duration:
                        bridge_options.append((source_to_next, self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            source_to_next,
                        )))
                    if bridge_options:
                        bridge_start, bridge_score = max(bridge_options, key=lambda item: item[1])
                    else:
                        bridge_start, bridge_score = prev[3], 0.0
                    repaired.append(
                        (
                            target_start,
                            target_end,
                            bridge_start,
                            bridge_start + duration,
                            True,
                            max(bridge_score, min(visual_score, bridge_score)),
                        )
                    )
                    replacements += 1
                    index += 1
                    continue
            repaired.append(current)
            index += 1

        if replacements:
            logger.info("Visual restore removed {} isolated far-away excursions", replacements)
        return repaired

    def _repair_visual_restore_noncut_jumps(
        self,
        context,
        ranges: list[tuple[float, float, float, float, bool, float]],
        movie_duration: float,
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Retimestamp jumps that happen on speech-only boundaries.

        Speech segmentation often creates boundaries where the underlying video
        frame has not cut. If a candidate jumps forward/backward at such a
        boundary, re-score the same target span at the previous source end and
        keep the continuous source when it is competitive.
        """
        if not ranges:
            return []

        repaired: list[tuple[float, float, float, float, bool, float]] = []
        replacements = 0
        for current in sorted(ranges, key=lambda item: item[0]):
            if repaired:
                prev = repaired[-1]
                target_start, target_end, source_start, _source_end, is_inferred, visual_score = current
                duration = max(0.0, target_end - target_start)
                source_jump = source_start - prev[3]
                can_repair = (
                    duration >= 0.18
                    and abs(target_start - prev[1]) <= 0.05
                    and not self._is_protected_visual_restore_boundary(context, target_start)
                    and not self._is_detected_visual_restore_cut(context, target_start)
                    and 0.35 < abs(source_jump) <= self.VISUAL_RESTORE_NONCUT_MAX_JUMP
                )
                if can_repair:
                    continuous_start = prev[3]
                    continuous_end = continuous_start + duration
                    if movie_duration <= 0.0 or continuous_end <= movie_duration:
                        current_score = self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            source_start,
                        )
                        continuous_score = self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            continuous_start,
                        )
                        if continuous_score >= current_score - self.VISUAL_RESTORE_NONCUT_CONTINUITY_MARGIN:
                            current = (
                                target_start,
                                target_end,
                                continuous_start,
                                continuous_end,
                                True,
                                max(continuous_score, min(visual_score, current_score)),
                            )
                            replacements += 1

                prev = repaired[-1]
                target_gap = current[0] - prev[1]
                source_gap = current[2] - prev[3]
                if (
                    abs(target_gap) <= 0.05
                    and abs(source_gap) <= self.VISUAL_RESTORE_CONTINUITY_TOLERANCE
                    and not self._is_protected_visual_restore_boundary(context, current[0])
                    and not self._is_detected_visual_restore_cut(context, current[0])
                ):
                    repaired[-1] = self._merge_visual_restore_ranges(prev, current, movie_duration)
                    continue
            repaired.append(current)

        if replacements:
            logger.info("Visual restore repaired {} speech-boundary source jumps", replacements)
        return repaired

    def _repair_visual_restore_local_backtracks(
        self,
        context,
        ranges: list[tuple[float, float, float, float, bool, float]],
        movie_duration: float,
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Remove local source-time rewinds when continuous playback is competitive."""
        if not ranges:
            return []

        repaired: list[tuple[float, float, float, float, bool, float]] = []
        replacements = 0
        for current in sorted(ranges, key=lambda item: item[0]):
            if repaired:
                prev = repaired[-1]
                target_start, target_end, source_start, _source_end, is_inferred, visual_score = current
                duration = max(0.0, target_end - target_start)
                source_jump = source_start - prev[3]
                backtrack = max(0.0, -source_jump)
                can_repair = (
                    duration >= 0.18
                    and abs(target_start - prev[1]) <= 0.05
                    and not self._is_detected_visual_restore_cut(context, target_start)
                    and not self._is_protected_visual_restore_boundary(context, target_start)
                    and self.VISUAL_RESTORE_REPEAT_BACKTRACK_TOLERANCE < backtrack <= self.VISUAL_RESTORE_LOCAL_BACKTRACK_MAX
                )
                if can_repair:
                    continuous_start = prev[3]
                    continuous_end = continuous_start + duration
                    if movie_duration <= 0.0 or continuous_end <= movie_duration:
                        current_score = self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            source_start,
                        )
                        continuous_score = self._score_visual_restore_indexed_range(
                            context,
                            target_start,
                            duration,
                            continuous_start,
                        )
                        margin = self.VISUAL_RESTORE_LOCAL_BACKTRACK_MARGIN
                        if duration <= self.VISUAL_RESTORE_FLICKER_MAX_DURATION:
                            margin += 0.05
                        if continuous_score >= current_score - margin:
                            current = (
                                target_start,
                                target_end,
                                continuous_start,
                                continuous_end,
                                True,
                                max(continuous_score, min(visual_score, current_score)),
                            )
                            replacements += 1

                prev = repaired[-1]
                target_gap = current[0] - prev[1]
                source_gap = current[2] - prev[3]
                if (
                    abs(target_gap) <= 0.05
                    and abs(source_gap) <= self.VISUAL_RESTORE_CONTINUITY_TOLERANCE
                    and not self._is_detected_visual_restore_cut(context, current[0])
                ):
                    repaired[-1] = self._merge_visual_restore_ranges(prev, current, movie_duration)
                    continue
            repaired.append(current)

        if replacements:
            logger.info("Visual restore repaired {} local source backtracks", replacements)
        return repaired

    def _build_visual_restore_ranges(
        self,
        project: Project,
        segments: list[Segment],
    ) -> list[tuple[float, float, float, float, bool, float]]:
        context = self._open_visual_restore_context(project)
        if not context:
            return []
        try:
            self._detect_narration_visual_cuts_from_context(project, context)
            narration_duration = float(
                context.get("narration_duration")
                or project.narration_duration
                or 0.0
            )
            detected_cuts = [
                float(cut)
                for cut in (context.get("detected_visual_cuts") or [])
                if 0.0 < float(cut) < narration_duration
            ]
            cuts = [0.0]
            for cut in sorted(detected_cuts):
                if cut - cuts[-1] >= 0.22:
                    cuts.append(cut)
            if narration_duration > 0.0 and self.VISUAL_RESTORE_PROBE_SEGMENT > 0.0:
                probed_cuts = [cuts[0]]
                for left, right in zip(cuts, cuts[1:] + [narration_duration]):
                    next_probe = left + self.VISUAL_RESTORE_PROBE_SEGMENT
                    while next_probe < right - 0.35:
                        if next_probe - probed_cuts[-1] >= 0.35:
                            probed_cuts.append(next_probe)
                        next_probe += self.VISUAL_RESTORE_PROBE_SEGMENT
                    if right - probed_cuts[-1] >= 0.18:
                        probed_cuts.append(right)
                cuts = sorted(set(round(float(cut), 3) for cut in probed_cuts))
            if narration_duration > 0.0:
                if narration_duration - cuts[-1] >= 0.12:
                    cuts.append(narration_duration)
                elif len(cuts) == 1:
                    cuts.append(narration_duration)
                else:
                    cuts[-1] = narration_duration
            if len(cuts) < 2:
                return []

            ranges: list[tuple[float, float, float, float, bool, float]] = []
            last_source_end: Optional[float] = None
            movie_duration = float(context.get("movie_duration") or project.movie_duration or 0.0)
            for target_start, target_end in zip(cuts, cuts[1:]):
                duration = max(0.0, target_end - target_start)
                if duration < 0.12:
                    continue
                selected = self._select_visual_restore_shot_source(
                    context,
                    target_start,
                    target_end,
                    last_source_end,
                )
                if selected is None:
                    if last_source_end is None:
                        continue
                    source_start = last_source_end
                    visual_score = 0.0
                    is_inferred = True
                else:
                    source_start, visual_score, is_inferred = selected
                source_end = source_start + duration
                if movie_duration > 0.0 and source_end > movie_duration:
                    source_end = movie_duration
                    source_start = max(0.0, source_end - duration)
                ranges.append((target_start, target_end, source_start, source_end, is_inferred, visual_score))
                last_source_end = source_end

            protected_boundaries = sorted(
                (context.get("detected_visual_cuts") or [])
                + (context.get("verified_visual_jumps") or [])
            )
            context["protected_visual_cuts"] = protected_boundaries
            ranges = self._repair_visual_restore_isolated_excursions(
                context,
                ranges,
                movie_duration,
            )
            smoothed_ranges = self._smooth_visual_restore_ranges(
                ranges,
                movie_duration,
                protected_boundaries,
            )
            smoothed_ranges = self._retime_visual_restore_ranges(
                context,
                smoothed_ranges,
                movie_duration,
            )
            if len(smoothed_ranges) != len(ranges):
                logger.info(
                    "Visual restore smoothing merged {} micro segments ({} -> {})",
                    len(ranges) - len(smoothed_ranges),
                    len(ranges),
                    len(smoothed_ranges),
                )
            return smoothed_ranges
        finally:
            self._close_visual_restore_context(context)

    def _merge_segment_restore_range(
        self,
        previous: tuple[float, float, float, float, bool, float],
        current: tuple[float, float, float, float, bool, float],
    ) -> tuple[float, float, float, float, bool, float]:
        return (
            previous[0],
            current[1],
            previous[2],
            current[3],
            previous[4] or current[4],
            min(previous[5], current[5]),
        )

    def _repair_segment_locked_restore_ranges(
        self,
        project: Project,
        segments: list[Segment],
        ranges: list[tuple[float, float, float, float, bool, float]],
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Re-check risky persisted matches before writing the Jianying draft.

        The project match timeline remains the source of truth. This pass only
        edits clips that are likely to be "same scene, wrong motion phase" or
        "short/effected clip matched to a visually similar frame". It uses the
        export-time visual restore context, where movie/query features and gray
        frames are cached, so it is much cheaper than the failed wide OpenCV seek
        pass inside the main matcher.
        """
        if not ranges:
            return ranges

        context = self._open_visual_restore_context(project)
        if not context or not context.get("frame_matcher"):
            self._close_visual_restore_context(context)
            return ranges

        try:
            self._detect_narration_visual_cuts_from_context(project, context)
            movie_duration = float(context.get("movie_duration") or project.movie_duration or 0.0)
            repaired: list[tuple[float, float, float, float, bool, float]] = []
            changes = 0
            checked = 0
            max_checks = 72
            last_source_end: Optional[float] = None
            last_target_end: Optional[float] = None

            for current in sorted(ranges, key=lambda item: item[0]):
                target_start, target_end, source_start, source_end, is_inferred, visual_score = current
                duration = max(0.05, target_end - target_start)
                if duration <= 0.0:
                    continue

                current_score = self._score_visual_restore_indexed_range(
                    context,
                    target_start,
                    duration,
                    source_start,
                )
                target_gap = 0.0 if last_target_end is None else target_start - last_target_end
                expected_source_start = (
                    None
                    if last_source_end is None
                    else last_source_end + max(0.0, target_gap)
                )
                source_jump = (
                    0.0
                    if expected_source_start is None
                    else source_start - expected_source_start
                )
                boundary_is_cut = (
                    target_start <= 0.08
                    or self._is_detected_visual_restore_cut(context, target_start)
                    or self._is_protected_visual_restore_boundary(context, target_start)
                )
                visual_low = current_score < 0.70 or visual_score < 0.78
                short_uncertain = duration <= 1.25 and (current_score < 0.82 or visual_score < 0.86)
                phase_jump = (
                    expected_source_start is not None
                    and not boundary_is_cut
                    and abs(source_jump) > max(1.20, duration * 1.35)
                    and current_score < 0.88
                )
                risky = visual_low or short_uncertain or phase_jump or is_inferred

                if risky and checked < max_checks:
                    checked += 1
                    selected = self._select_visual_restore_source(
                        context,
                        project,
                        segments,
                        target_start,
                        target_end,
                        None if boundary_is_cut else last_source_end,
                    )
                    if selected is not None:
                        candidate_start, candidate_score, candidate_inferred = selected
                        candidate_score = max(
                            float(candidate_score),
                            self._score_visual_restore_indexed_range(
                                context,
                                target_start,
                                duration,
                                candidate_start,
                            ),
                        )
                        candidate_end = candidate_start + duration
                        if movie_duration > 0.0 and candidate_end > movie_duration:
                            candidate_end = movie_duration
                            candidate_start = max(0.0, candidate_end - duration)

                        replacement_jump = (
                            0.0
                            if expected_source_start is None
                            else candidate_start - expected_source_start
                        )
                        jump_improves = (
                            expected_source_start is not None
                            and abs(replacement_jump) + 0.40 < abs(source_jump)
                        )
                        required_gain = 0.035
                        if current_score < 0.62:
                            required_gain = 0.010
                        elif phase_jump or jump_improves:
                            required_gain = 0.018
                        min_score = 0.66 if current_score < 0.62 else 0.72
                        accept = (
                            candidate_score >= min_score
                            and candidate_score >= current_score + required_gain
                        )
                        if accept:
                            source_start = candidate_start
                            source_end = candidate_end
                            is_inferred = bool(is_inferred or candidate_inferred)
                            visual_score = max(float(visual_score), float(candidate_score))
                            changes += 1

                repaired.append((target_start, target_end, source_start, source_end, is_inferred, visual_score))
                last_target_end = target_end
                last_source_end = source_end

            if changes:
                protected_boundaries = sorted(
                    (context.get("detected_visual_cuts") or [])
                    + (context.get("verified_visual_jumps") or [])
                )
                repaired = self._smooth_visual_restore_ranges(repaired, movie_duration, protected_boundaries)
                repaired = self._retime_visual_restore_ranges(context, repaired, movie_duration)
                logger.info(
                    "Segment-locked visual repair changed {} risky ranges (checked {})",
                    changes,
                    checked,
                )
            return repaired
        finally:
            self._close_visual_restore_context(context)

    def _build_segment_locked_restore_ranges(
        self,
        project: Project,
        segments: list[Segment],
    ) -> list[tuple[float, float, float, float, bool, float]]:
        """Use the persisted match timeline, then lock non-cut shards together.

        This is intentionally preferred over export-time global re-search: the
        user-visible matching result must be the source of truth for the draft.
        Non-linear narration edits are allowed, so direct visual chunk matches
        must not be rewritten into a monotonic movie timeline.
        """
        narration_duration_limit = float(project.narration_duration or 0.0)
        usable_segments = [
            segment
            for segment in sorted(segments, key=lambda item: (float(item.narration_start), float(item.narration_end)))
            if segment.use_segment and segment.segment_type != SegmentType.NON_MOVIE
            and (narration_duration_limit <= 0.0 or float(segment.narration_start) < narration_duration_limit)
        ]
        if not usable_segments:
            return []

        use_direct_visual_chunks = True
        plan = None
        movie_duration = float(project.movie_duration or 0.0)
        ranges: list[tuple[float, float, float, float, bool, float]] = []
        last_target_end: Optional[float] = None
        last_source_end: Optional[float] = None
        last_allows_speed_change: Optional[bool] = None
        last_merge_safe: Optional[bool] = None

        for segment in usable_segments:
            target_start = max(0.0, float(segment.narration_start))
            target_end = max(target_start, float(segment.narration_end))
            if narration_duration_limit > 0.0:
                target_end = min(target_end, narration_duration_limit)
            target_duration = target_end - target_start
            if target_duration <= 0.0:
                continue

            planned = plan.segment_ranges.get(segment.id) if plan is not None else None
            if planned is not None:
                source_start, source_end, is_inferred, visual_score = planned
            elif segment.movie_start is not None and segment.movie_end is not None:
                source_start = float(segment.movie_start)
                source_end = float(segment.movie_end)
                is_inferred = bool(segment.review_required or segment.match_type == "inferred")
                visual_score = float(segment.visual_confidence or segment.match_confidence or 0.0)
            elif last_source_end is not None:
                source_start = last_source_end
                source_end = source_start + target_duration
                is_inferred = True
                visual_score = 0.0
            else:
                continue

            allows_speed_change = self._segment_allows_movie_speed_change(segment)
            merge_safe = (
                not bool(segment.review_required)
                and not bool(is_inferred)
                and str(getattr(segment, "match_type", "")) == "exact"
                and float(visual_score) >= 0.82
                and float(segment.match_confidence or 0.0) >= 0.82
            )
            if not allows_speed_change:
                source_end = source_start + target_duration
            if source_end <= source_start:
                source_end = source_start + target_duration
            if movie_duration > 0.0 and source_end > movie_duration:
                overflow = source_end - movie_duration
                source_start = max(0.0, source_start - overflow)
                source_end = movie_duration

            if (
                ranges
                and last_target_end is not None
                and last_source_end is not None
                and 0.05 < target_start - last_target_end <= 0.18
            ):
                gap_duration = target_start - last_target_end
                gap_source_start = last_source_end
                gap_source_end = gap_source_start + gap_duration
                source_gap = source_start - last_source_end
                previous_is_strong_anchor = not bool(ranges[-1][4]) and float(ranges[-1][5]) >= 0.78
                current_is_strong_anchor = (
                    not bool(is_inferred)
                    and float(visual_score) >= 0.78
                    and float(segment.match_confidence or 0.0) >= 0.80
                    and not bool(segment.review_required)
                )
                is_safe_continuity_gap = (
                    gap_duration <= 0.22
                    or (
                        source_gap >= -0.05
                        and abs(source_gap - gap_duration) <= max(0.12, gap_duration * 0.50)
                    )
                )
                if (
                    (gap_duration <= 0.22 or (previous_is_strong_anchor and current_is_strong_anchor))
                    and is_safe_continuity_gap
                    and movie_duration > 0.0
                    and gap_source_end <= movie_duration
                ):
                    ranges.append((last_target_end, target_start, gap_source_start, gap_source_end, True, 0.0))
                    last_source_end = gap_source_end

            current = (target_start, target_end, source_start, source_end, bool(is_inferred), float(visual_score))
            if ranges:
                previous = ranges[-1]
                target_gap = current[0] - previous[1]
                source_gap = current[2] - previous[3]
                if (
                    abs(target_gap) <= 0.05
                    and abs(source_gap) <= 0.18
                    and allows_speed_change == last_allows_speed_change
                    and bool(last_merge_safe)
                    and merge_safe
                ):
                    ranges[-1] = self._merge_segment_restore_range(previous, current)
                else:
                    ranges.append(current)
            else:
                ranges.append(current)

            last_target_end = target_end
            last_source_end = source_end
            last_allows_speed_change = allows_speed_change
            last_merge_safe = merge_safe

        if ranges:
            logger.info(
                "Segment-locked restore ranges: {} ranges from {} segments, mode={}, shot groups={}, repaired={}",
                len(ranges),
                len(usable_segments),
                    "persisted_matches",
                    0,
                    0,
                )
            # Export-time global repair is intentionally disabled by default.
            # It reduced obvious backtracks but also merged/retimed real cuts in
            # testing, dropping the first-120s visual audit from ~0.947 to ~0.853.
            # Keep persisted matches as the draft source of truth until a safer
            # opt-in repair mode is added.
        return ranges

    def _segment_allows_movie_speed_change(self, segment: Segment) -> bool:
        if not bool(getattr(segment, "speed_changed", False)):
            return False
        ratio = float(getattr(segment, "source_speed_ratio", 1.0) or 1.0)
        confidence = float(getattr(segment, "speed_change_confidence", 0.0) or 0.0)
        return 0.50 <= ratio <= 2.00 and abs(ratio - 1.0) >= 0.08 and confidence >= 0.70

    def _range_allows_movie_speed_change(
        self,
        segments: list[Segment],
        target_start: float,
        target_end: float,
    ) -> bool:
        overlapping: list[Segment] = []
        for segment in segments:
            if not segment.use_segment or segment.segment_type == SegmentType.NON_MOVIE:
                continue
            overlap = min(float(segment.narration_end), target_end) - max(float(segment.narration_start), target_start)
            if overlap > 0.02:
                overlapping.append(segment)
        return bool(overlapping) and all(self._segment_allows_movie_speed_change(segment) for segment in overlapping)

    def _generate_tracks(self, project: Project) -> list:
        """Generate restore tracks anchored to narration timestamps."""
        tracks = []

        main_video_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "",
            "segments": [],
            "type": "video"
        }

        main_audio_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "",
            "segments": [],
            "type": "audio"
        }

        narration_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "解说视频",
            "segments": [],
            "type": "video"
        }

        cut_movie_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "原电影片段",
            "segments": [],
            "type": "video"
        }

        low_confidence_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": False,
            "name": "待检查(低置信度)",
            "segments": [],
            "type": "video"
        }

        inferred_fill_count = 0
        using_original_audio = self.audio_source == "original" or any(
            segment.use_segment and not segment.tts_audio_path
            for segment in project.segments
        )
        if using_original_audio and project.narration_path and project.movie_path:
            visual_restore_ranges = self._build_segment_locked_restore_ranges(project, project.segments)
            restore_label = "已匹配时间轴"
            if not visual_restore_ranges:
                visual_restore_ranges = self._build_visual_restore_ranges(project, project.segments)
                restore_label = "解说画面重查"
            if visual_restore_ranges:
                low_visual_count = 0
                inferred_visual_count = 0
                for target_start, target_end, source_start, source_end, is_inferred, visual_score in visual_restore_ranges:
                    target_duration = max(0.0, target_end - target_start)
                    if target_duration <= 0:
                        continue
                    video_seg = self._create_video_segment(
                        f"movie_{project.id}",
                        project.movie_path,
                        int(max(0.0, target_start) * self.TIME_SCALE),
                        source_start,
                        source_end,
                        True,
                        target_duration=target_duration,
                        allow_speed_change=self._range_allows_movie_speed_change(
                            project.segments,
                            target_start,
                            target_end,
                        ),
                    )
                    main_video_track["segments"].append(video_seg)
                    if is_inferred:
                        inferred_visual_count += 1
                    if is_inferred or visual_score < 0.36:
                        low_visual_count += 1

                narration_duration = float(project.narration_duration or 0.0)
                if narration_duration <= 0.0 and visual_restore_ranges:
                    narration_duration = max(item[1] for item in visual_restore_ranges)
                if narration_duration > 0.0:
                    main_audio_track["segments"].append(
                        self._create_original_audio_segment(
                            f"narration_audio_{project.id}",
                            project.narration_path,
                            0,
                            0.0,
                            narration_duration,
                        )
                    )

                logger.info(
                    "导出剪映草稿: 使用{}重建视频轨，{} 个画面段，{} 个低分段，{} 个连续补齐段",
                    restore_label,
                    len(main_video_track["segments"]),
                    low_visual_count,
                    inferred_visual_count,
                )
                tracks.extend([main_video_track, main_audio_track])
                return tracks

        # Match-first export: use each segment's own visual match. Continuous
        # smoothing made playback steadier but moved many shots away from the
        # narration reference, which is worse for this tool's core purpose.
        continuous_source_ranges: dict[str, tuple[float, float, bool]] = {}
        last_video_timeline_end_us = 0
        last_video_source_end: Optional[float] = None

        for index, segment in enumerate(project.segments):
            if not segment.use_segment:
                continue
            if segment.segment_type == SegmentType.NON_MOVIE:
                continue

            timeline_duration = max(0.0, segment.narration_end - segment.narration_start)
            if timeline_duration <= 0:
                continue

            timeline_start_us = int(max(segment.narration_start, 0.0) * self.TIME_SCALE)
            should_use_original = (
                self.audio_source == "original"
                or segment.segment_type == SegmentType.NO_NARRATION
                or not segment.tts_audio_path
            )

            resolved_source = continuous_source_ranges.get(segment.id)
            if resolved_source is None:
                resolved_source = self._resolve_restore_source_range(
                    project,
                    project.segments,
                    index,
                    segment,
                    timeline_duration,
            )
            if resolved_source is not None:
                source_start, source_end, is_inferred_fill = resolved_source
                verify_score = self._segment_visual_verify_score(segment)
                source_jump = None if last_video_source_end is None else source_start - last_video_source_end
                if (
                    source_jump is not None
                    and abs(source_jump) > max(8.0, timeline_duration * 4.0)
                    and segment.review_required
                    and verify_score < 0.65
                    and last_video_source_end is not None
                ):
                    source_start = last_video_source_end
                    source_end = source_start + timeline_duration
                    movie_duration = float(project.movie_duration or 0.0)
                    if movie_duration > 0.0 and source_end > movie_duration:
                        source_end = movie_duration
                        source_start = max(0.0, source_end - timeline_duration)
                    is_inferred_fill = True

                if (
                    main_video_track["segments"]
                    and timeline_start_us > last_video_timeline_end_us + 50_000
                    and last_video_source_end is not None
                ):
                    gap_duration = (timeline_start_us - last_video_timeline_end_us) / self.TIME_SCALE
                    gap_source_start = last_video_source_end
                    gap_source_end = gap_source_start + gap_duration
                    movie_duration = float(project.movie_duration or 0.0)
                    should_fill_gap = (
                        gap_duration <= 0.18
                        and not bool(is_inferred_fill)
                        and float(segment.match_confidence or 0.0) >= 0.80
                        and verify_score >= 0.78
                    )
                    if should_fill_gap:
                        if movie_duration > 0.0 and gap_source_end > movie_duration:
                            gap_source_end = movie_duration
                            gap_source_start = max(0.0, gap_source_end - gap_duration)
                        main_video_track["segments"].append(
                            self._create_video_segment(
                                f"movie_{project.id}",
                                project.movie_path,
                                last_video_timeline_end_us,
                                gap_source_start,
                                gap_source_end,
                                True,
                                target_duration=gap_duration,
                            )
                        )
                        last_video_source_end = gap_source_end

                allow_speed_change = self._segment_allows_movie_speed_change(segment)
                video_seg = self._create_video_segment(
                    f"movie_{project.id}",
                    project.movie_path,
                    timeline_start_us,
                    source_start,
                    source_end,
                    segment.mute_movie_audio,
                    target_duration=timeline_duration,
                    allow_speed_change=allow_speed_change,
                )
                main_video_track["segments"].append(video_seg)
                if is_inferred_fill:
                    inferred_fill_count += 1
                if is_inferred_fill or segment.match_confidence < self.LOW_CONFIDENCE_THRESHOLD or segment.review_required:
                    review_seg = video_seg.copy()
                    review_seg["visible"] = False
                    low_confidence_track["segments"].append(review_seg)
                last_video_timeline_end_us = timeline_start_us + video_seg["target_timerange"]["duration"]
                last_video_source_end = source_end if allow_speed_change else source_start + timeline_duration
                movie_duration = float(project.movie_duration or 0.0)
                if movie_duration > 0.0:
                    last_video_source_end = min(last_video_source_end, movie_duration)

            if should_use_original:
                main_audio_track["segments"].append(
                    self._create_original_audio_segment(
                        f"narration_audio_{project.id}",
                        project.narration_path,
                        timeline_start_us,
                        segment.narration_start,
                        segment.narration_end,
                    )
                )
            elif segment.tts_audio_path and segment.tts_duration:
                main_audio_track["segments"].append(
                    self._create_audio_segment(
                        f"tts_{segment.id}",
                        segment.tts_audio_path,
                        timeline_start_us,
                        segment.tts_duration,
                    )
                )

        if project.narration_path and project.narration_duration:
            narration_track["segments"].append(
                self._create_video_segment(
                    f"narration_{project.id}",
                    project.narration_path,
                    0,
                    0,
                    project.narration_duration,
                    True,
                    visible=False,
                )
            )

        if low_confidence_track["segments"]:
            logger.info(
                f"导出剪映草稿: {len(low_confidence_track['segments'])} 个低置信度片段同步放入待检查轨道"
            )
        if inferred_fill_count:
            logger.info(f"导出剪映草稿: {inferred_fill_count} 个未命中片段已用邻居时间轴补齐，避免视频断轨")

        # Keep the restore draft clean: one movie-video track plus one narration-audio track.
        # Extra hidden reference tracks make Jianying look like the main picture is narration.
        tracks.extend([main_video_track, main_audio_track])
        return tracks

    def _force_monotonic_video_source(self, segments: list[dict]) -> None:
        last_source_end: Optional[int] = None
        adjusted = 0
        for segment in segments:
            source_range = segment.get("source_timerange") or {}
            source_start = int(source_range.get("start") or 0)
            source_duration = int(source_range.get("duration") or 0)
            if last_source_end is not None and source_start < last_source_end:
                source_range["start"] = last_source_end
                source_start = last_source_end
                adjusted += 1
            last_source_end = source_start + max(0, source_duration)
        if adjusted:
            logger.info("Export source monotonic guard adjusted {} video segments", adjusted)

    def _create_video_segment(
        self,
        material_id: str,
        path: str,
        timeline_start: int,
        source_start: float,
        source_end: float,
        mute: bool,
        visible: bool = True,
        target_duration: float = None,
        allow_speed_change: bool = False,
    ) -> dict:
        """鍒涘缓瑙嗛鐗囨
        
        Args:
            target_duration: 鐩爣鏃堕暱锛堢锛夛紝濡傛灉涓嶄紶鍒欓粯璁ょ瓑浜庢簮鏃堕暱
        """
        source_duration = source_end - source_start
        source_duration_us = int(source_duration * self.TIME_SCALE)

        if target_duration:
            target_duration_us = int(target_duration * self.TIME_SCALE)
            if not allow_speed_change:
                speed = 1.0
                source_duration_us = target_duration_us
            else:
                raw_speed = source_duration / target_duration if target_duration > 0 else 1.0

                if raw_speed > self.max_playback_speed:
                    speed = self.max_playback_speed
                    actual_source_duration = speed * target_duration
                    source_duration_us = int(actual_source_duration * self.TIME_SCALE)
                    logger.debug(
                        f"Limit playback speed: {raw_speed:.2f}x -> {speed:.2f}x, "
                        f"source duration {source_duration:.1f}s -> {actual_source_duration:.1f}s"
                    )
                elif raw_speed < self.min_playback_speed:
                    speed = raw_speed
                    logger.debug(
                        "Source clip is shorter than the target narration span: "
                        f"{raw_speed:.2f}x below minimum {self.min_playback_speed:.2f}x, "
                        f"keep matched speed ({source_duration:.1f}s source / {target_duration:.1f}s target)"
                    )
                else:
                    speed = raw_speed
        else:
            target_duration_us = source_duration_us
            speed = 1.0

        return {
            "caption_info": None,
            "cartoon": False,
            "clip": {"alpha": 1.0, "flip": {"horizontal": False, "vertical": False}, "rotation": 0.0, "scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": 0.0}},
            "common_keyframes": [],
            "enable_adjust": True,
            "enable_color_correct_adjust": False,
            "enable_color_curves": True,
            "enable_color_match_adjust": False,
            "enable_color_wheels": True,
            "enable_lut": True,
            "enable_smart_color_adjust": False,
            "extra_material_refs": [],
            "group_id": "",
            "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
            "id": str(uuid.uuid4()),
            "intensifies_audio": False,
            "is_placeholder": False,
            "is_tone_modify": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "material_id": material_id,
            "render_index": 0,
            "responsive_layout": {"enable": False, "horizontal_pos_layout": 0, "size_layout": 0, "target_follow": "", "vertical_pos_layout": 0},
            "reverse": False,
            "source_timerange": {"duration": source_duration_us, "start": int(source_start * self.TIME_SCALE)},
            "speed": speed,
            "target_timerange": {"duration": target_duration_us, "start": timeline_start},
            "template_id": "",
            "template_scene": "default",
            "track_attribute": 0,
            "track_render_index": 0,
            "uniform_scale": {"on": True, "value": 1.0},
            "visible": visible,
            "volume": 0.0 if mute else 1.0
        }

    def _create_audio_segment(
        self,
        material_id: str,
        path: str,
        timeline_start: int,
        duration: float
    ) -> dict:
        """鍒涘缓闊抽鐗囨"""
        duration_us = int(duration * self.TIME_SCALE)

        return {
            "caption_info": None,
            "category_id": "",
            "category_name": "",
            "channel_index": None,
            "check_flag": 0,
            "common_keyframes": [],
            "effect_id": "",
            "effects_adjust": None,
            "enable_audio_loop": False,
            "formula_id": "",
            "hdr_settings": None,
            "id": str(uuid.uuid4()),
            "intensifies_audio": False,
            "is_tone_modify": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "local_material_id": "",
            "material_id": material_id,
            "music_id": "",
            "render_index": 0,
            "responsive_layout": {"enable": False, "horizontal_pos_layout": 0, "size_layout": 0, "target_follow": "", "vertical_pos_layout": 0},
            "reverse": False,
            "source_timerange": {"duration": duration_us, "start": 0},
            "speed": 1.0,
            "target_timerange": {"duration": duration_us, "start": timeline_start},
            "template_id": "",
            "template_scene": "default",
            "tone_category_id": "",
            "tone_category_name": "",
            "tone_effect_id": "",
            "tone_effect_name": "",
            "tone_platform": "",
            "tone_second_category_id": "",
            "tone_second_category_name": "",
            "tone_speaker": "",
            "tone_type": "",
            "track_attribute": 0,
            "track_render_index": 0,
            "visible": True,
            "volume": 1.0
        }

    def _create_original_audio_segment(
        self,
        material_id: str,
        path: str,
        timeline_start: int,
        source_start: float,
        source_end: float
    ) -> dict:
        """鍒涘缓鍘熷闊抽鐗囨锛堜粠瑙嗛涓彁鍙栨寚瀹氭椂闂存鐨勯煶棰戯級"""
        duration_us = int((source_end - source_start) * self.TIME_SCALE)

        return {
            "caption_info": None,
            "category_id": "",
            "category_name": "",
            "channel_index": None,
            "check_flag": 0,
            "common_keyframes": [],
            "effect_id": "",
            "effects_adjust": None,
            "enable_audio_loop": False,
            "formula_id": "",
            "hdr_settings": None,
            "id": str(uuid.uuid4()),
            "intensifies_audio": False,
            "is_tone_modify": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "local_material_id": "",
            "material_id": material_id,
            "music_id": "",
            "render_index": 0,
            "responsive_layout": {"enable": False, "horizontal_pos_layout": 0, "size_layout": 0, "target_follow": "", "vertical_pos_layout": 0},
            "reverse": False,
            "source_timerange": {
                "duration": duration_us,
                "start": int(source_start * self.TIME_SCALE)
            },
            "speed": 1.0,
            "target_timerange": {"duration": duration_us, "start": timeline_start},
            "template_id": "",
            "template_scene": "default",
            "tone_category_id": "",
            "tone_category_name": "",
            "tone_effect_id": "",
            "tone_effect_name": "",
            "tone_platform": "",
            "tone_second_category_id": "",
            "tone_second_category_name": "",
            "tone_speaker": "",
            "tone_type": "",
            "track_attribute": 0,
            "track_render_index": 0,
            "visible": True,
            "volume": 1.0
        }

    def _calculate_duration(self, project: Project) -> int:
        """Keep restore-draft duration anchored to the narration timeline."""
        if project.narration_duration and project.narration_duration > 0:
            return int(float(project.narration_duration) * self.TIME_SCALE)

        segment_end = 0.0
        for segment in project.segments:
            if not segment.use_segment:
                continue
            if segment.segment_type == SegmentType.NON_MOVIE:
                continue
            segment_end = max(segment_end, segment.narration_end)

        return int(segment_end * self.TIME_SCALE)
    def _generate_meta_content(
        self,
        project: Project,
        draft_id: str,
        draft_name: str,
        export_mode: ExportMode,
        duration_us: int,
    ) -> dict:
        """Generate draft metadata."""
        return {
            "draft_cloud_capcut_purchase_info": None,
            "draft_cloud_last_action_download": False,
            "draft_cloud_materials": [],
            "draft_cloud_purchase_info": None,
            "draft_cloud_template_id": "",
            "draft_cloud_tutorial_info": None,
            "draft_cloud_videocut_purchase_info": None,
            "draft_cover": "",
            "draft_deeplink_url": "",
            "draft_enterprise_info": None,
            "draft_fold_path": "",
            "draft_id": draft_id,
            "draft_is_ai_shorts": False,
            "draft_is_article_video_draft": False,
            "draft_is_from_deeplink": False,
            "draft_is_invisible": False,
            "draft_materials": [],
            "draft_materials_copied": False,
            "draft_name": draft_name,
            "draft_new_version": "",
            "draft_removable_storage_device": "",
            "draft_root_path": "",
            "draft_segment_extra_info": None,
            "draft_timeline_materials_size": 0,
            "draft_type": "",
            "tm_draft_cloud_completed": False,
            "tm_draft_cloud_modified": False,
            "tm_draft_create": int(datetime.now().timestamp() * 1000),
            "tm_draft_modified": int(datetime.now().timestamp() * 1000),
            "tm_draft_removed": 0,
            "tm_duration": duration_us // 1000
        }

    async def _copy_materials(self, project: Project, draft_dir: Path):
        """Copy material files into the draft directory."""
        materials_dir = draft_dir / "materials"
        materials_dir.mkdir(exist_ok=True)

        # 澶嶅埗TTS闊抽
        for segment in project.segments:
            if segment.tts_audio_path and Path(segment.tts_audio_path).exists():
                src = Path(segment.tts_audio_path)
                dst = materials_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

    async def export_from_matches(
        self,
        results: list,
        movie_path: str,
        narration_path: str,
        project_name: str
    ) -> Path:
        """浠庡尮閰嶇粨鏋滃鍑哄壀鏄犺崏绋?
        Args:
            results: MatchResult 鍒楄〃
            movie_path: 鍘熺數褰辫矾寰?            narration_path: 瑙ｈ瑙嗛璺緞
            project_name: 椤圭洰鍚嶇О

        Returns:
            鑽夌鐩綍璺緞
        """
        # 鍒涘缓鑽夌鐩綍
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        draft_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        draft_name, draft_dir, staged_dir = self._create_staged_draft_dir(draft_name)

        logger.info(f"鍒涘缓鍓槧鑽夌: {draft_dir}")

        # 鑾峰彇瑙嗛鏃堕暱
        import cv2
        movie_cap = cv2.VideoCapture(movie_path)
        movie_duration = movie_cap.get(cv2.CAP_PROP_FRAME_COUNT) / movie_cap.get(cv2.CAP_PROP_FPS)
        movie_cap.release()

        narration_cap = cv2.VideoCapture(narration_path)
        narration_duration = narration_cap.get(cv2.CAP_PROP_FRAME_COUNT) / narration_cap.get(cv2.CAP_PROP_FPS)
        narration_cap.release()

        # 鐢熸垚鑽夌鍐呭
        draft_content = self._generate_draft_from_matches(
            results, movie_path, narration_path,
            movie_duration, narration_duration, draft_id, project_name
        )

        # 鍐欏叆鑽夌鏂囦欢
        draft_file = staged_dir / "draft_content.json"
        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

        # Generate draft metadata.
        meta_content = self._generate_meta_from_matches(
            draft_id, draft_name, draft_content["duration"]
        )
        self._apply_draft_meta_paths(meta_content, draft_dir)
        meta_file = staged_dir / "draft_meta_info.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, ensure_ascii=False, indent=2)

        self._publish_staged_draft(staged_dir, draft_dir)
        self._verify_draft_integrity(draft_dir)
        logger.info(f"鍓槧鑽夌瀵煎嚭瀹屾垚: {draft_dir}")
        return draft_dir

    def _generate_draft_from_matches(
        self,
        results: list,
        movie_path: str,
        narration_path: str,
        movie_duration: float,
        narration_duration: float,
        draft_id: str,
        project_name: str
    ) -> dict:
        """Generate draft content directly from raw match results."""
        # Calculate output aspect ratio.
        ratio = f"{self.width}:{self.height}"
        if self.width == 1920 and self.height == 1080:
            ratio = "16:9"
        elif self.width == 1280 and self.height == 720:
            ratio = "16:9"
        elif self.width == 1080 and self.height == 1920:
            ratio = "9:16"

        # 鍩虹缁撴瀯
        content = {
            "canvas_config": {
                "height": self.height,
                "ratio": ratio,
                "width": self.width
            },
            "color_space": 0,
            "config": {
                "adjust_max_index": 1,
                "attachment_info": [],
                "combination_max_index": 1,
                "export_range": None,
                "extract_audio_last_index": 1,
                "lyrics_recognition_id": "",
                "lyrics_sync": True,
                "lyrics_taskinfo": [],
                "maintrack_adsorb": True,
                "material_save_mode": 0,
                "original_sound_last_index": 1,
                "record_audio_last_index": 1,
                "sticker_max_index": 1,
                "subtitle_recognition_id": "",
                "subtitle_sync": True,
                "subtitle_taskinfo": [],
                "system_font_list": [],
                "video_mute": False,
                "zoom_info_params": None
            },
            "cover": None,
            "create_time": int(datetime.now().timestamp() * 1000000),
            "duration": 0,
            "extra_info": None,
            "fps": float(self.output_fps) if self.output_fps > 0 else 30.0,
            "free_render_index_mode_on": False,
            "group_container": None,
            "id": draft_id,
            "keyframe_graph_list": [],
            "keyframes": {"adjusts": [], "audios": [], "effects": [], "filters": [], "handwrites": [], "stickers": [], "texts": [], "videos": []},
            "last_modified_platform": {"app_id": 3704, "app_source": "lv", "app_version": "5.9.0", "device_id": "", "hard_disk_id": "", "mac_address": "", "os": "windows", "os_version": "10.0.19045"},
            "materials": self._generate_materials_from_matches(
                movie_path, narration_path, movie_duration, narration_duration
            ),
            "mutable_config": None,
            "name": project_name,
            "new_version": "110.0.0",
            "platform": {"app_id": 3704, "app_source": "lv", "app_version": "5.9.0", "device_id": "", "hard_disk_id": "", "mac_address": "", "os": "windows", "os_version": "10.0.19045"},
            "relationships": [],
            "render_index_track_mode_on": False,
            "retouch_cover": None,
            "source": "default",
            "static_cover_image_path": "",
            "tracks": self._generate_tracks_from_matches(results, movie_path, narration_path),
            "update_time": int(datetime.now().timestamp() * 1000000),
            "version": 360000
        }

        # Keep raw-match export anchored to the narration timeline.
        timeline_end = 0.0
        for result in results:
            end = getattr(result, "narration_end", None)
            if end is None:
                start = float(getattr(result, "narration_start", 0.0) or 0.0)
                duration = float(getattr(result, "narration_duration", 0.0) or 0.0)
                end = start + duration
            timeline_end = max(timeline_end, float(end))

        content["duration"] = int(timeline_end * self.TIME_SCALE)

        return content

    def _generate_materials_from_matches(
        self,
        movie_path: str,
        narration_path: str,
        movie_duration: float,
        narration_duration: float
    ) -> dict:
        """Generate materials from raw match results."""
        materials = {
            "audios": [],
            "audio_fades": [],
            "beats": [],
            "canvases": [],
            "chromas": [],
            "color_curves": [],
            "digital_humans": [],
            "drafts": [],
            "effects": [],
            "flowers": [],
            "green_screens": [],
            "handwrites": [],
            "hsl": [],
            "images": [],
            "log_color_wheels": [],
            "loudnesses": [],
            "manual_deformations": [],
            "masks": [],
            "material_animations": [],
            "material_colors": [],
            "multi_language_refs": [],
            "placeholders": [],
            "plugin_effects": [],
            "primary_color_wheels": [],
            "realtime_denoises": [],
            "shapes": [],
            "smart_crops": [],
            "smart_relights": [],
            "sound_channel_mappings": [],
            "speeds": [],
            "stickers": [],
            "tail_leaders": [],
            "text_templates": [],
            "texts": [],
            "time_marks": [],
            "transitions": [],
            "video_effects": [],
            "video_trackings": [],
            "videos": [],
            "vocal_beautifys": [],
            "vocal_separations": []
        }

        # Add the source movie material.
        movie_material = self._create_video_material(
            movie_path,
            "movie_material",
            movie_duration
        )
        materials["videos"].append(movie_material)

        # 娣诲姞瑙ｈ瑙嗛绱犳潗锛堢敤浜庢彁鍙栭煶棰戯級
        narration_material = self._create_video_material(
            narration_path,
            "narration_material",
            narration_duration
        )
        materials["videos"].append(narration_material)

        # Add narration audio extracted from the narration video.
        narration_audio = {
            "app_id": 0,
            "category_id": "",
            "category_name": "",
            "check_flag": 1,
            "copyright_limit_type": "none",
            "duration": int(narration_duration * self.TIME_SCALE),
            "effect_id": "",
            "formula_id": "",
            "id": "narration_audio_material",
            "intensifies_path": "",
            "is_ai_clone_tone": False,
            "is_text_edit_overdub": False,
            "is_ugc": False,
            "local_material_id": "",
            "music_id": "",
            "name": Path(narration_path).name,
            "path": narration_path,
            "query": "",
            "request_id": "",
            "resource_id": "",
            "search_id": "",
            "source_from": "",
            "source_platform": 0,
            "team_id": "",
            "text_id": "",
            "tone_category_id": "",
            "tone_category_name": "",
            "tone_effect_id": "",
            "tone_effect_name": "",
            "tone_platform": "",
            "tone_second_category_id": "",
            "tone_second_category_name": "",
            "tone_speaker": "",
            "tone_type": "",
            "type": "video_original_sound",
            "video_id": "",
            "wave_points": []
        }
        materials["audios"].append(narration_audio)

        return materials

    def _generate_tracks_from_matches(
        self,
        results: list,
        movie_path: str,
        narration_path: str
    ) -> list:
        """Generate tracks from raw match results."""
        tracks = []

        # 杞ㄩ亾1: 涓昏棰戣建閬擄紙鍘熺數褰卞尮閰嶇墖娈碉級
        main_video_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "原电影画面",
            "segments": [],
            "type": "video"
        }

        # 杞ㄩ亾2: 瑙ｈ闊抽杞ㄩ亾
        narration_audio_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "瑙ｈ闊抽",
            "segments": [],
            "type": "audio"
        }

        # 杞ㄩ亾3: 鍙傝€冭建閬擄紙瀹屾暣瑙ｈ瑙嗛锛岄潤闊筹級
        reference_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "解说视频(参考)",
            "segments": [],
            "type": "video"
        }

        # Place each match on the original narration timeline instead of
        # compacting everything sequentially.
        for result in results:
            segment_start = float(getattr(result, "narration_start", 0.0) or 0.0)
            segment_end = getattr(result, "narration_end", None)
            if segment_end is None:
                segment_end = segment_start + float(getattr(result, "narration_duration", 0.0) or 0.0)
            segment_duration = max(0.0, float(segment_end) - segment_start)
            duration_us = int(segment_duration * self.TIME_SCALE)
            timeline_start_us = int(segment_start * self.TIME_SCALE)

            # 瑙嗛鐗囨锛堜粠鍘熺數褰卞壀鍒囷級
            video_seg = self._create_video_segment(
                "movie_material",
                movie_path,
                timeline_start_us,
                result.movie_start,
                result.movie_end,
                mute=True,
                target_duration=segment_duration,
            )
            main_video_track["segments"].append(video_seg)

            # Build an audio segment from the narration source.
            audio_seg = {
                "caption_info": None,
                "cartoon": False,
                "clip": None,
                "common_keyframes": [],
                "enable_adjust": False,
                "enable_color_correct_adjust": False,
                "enable_color_curves": True,
                "enable_color_match_adjust": False,
                "enable_color_wheels": True,
                "enable_lut": False,
                "enable_smart_color_adjust": False,
                "extra_material_refs": [],
                "group_id": "",
                "hdr_settings": None,
                "id": str(uuid.uuid4()),
                "intensifies_audio": False,
                "is_placeholder": False,
                "is_tone_modify": False,
                "keyframe_refs": [],
                "last_nonzero_volume": 1.0,
                "material_id": "narration_audio_material",
                "render_index": 0,
                "responsive_layout": {"enable": False, "horizontal_pos_layout": 0, "size_layout": 0, "target_follow": "", "vertical_pos_layout": 0},
                "reverse": False,
                "source_timerange": {
                    "duration": duration_us,
                    "start": timeline_start_us
                },
                "speed": 1.0,
                "target_timerange": {
                    "duration": duration_us,
                    "start": timeline_start_us
                },
                "template_id": "",
                "template_scene": "default",
                "track_attribute": 0,
                "track_render_index": 0,
                "uniform_scale": None,
                "visible": True,
                "volume": 1.0
            }
            narration_audio_track["segments"].append(audio_seg)

        tracks.extend([main_video_track, narration_audio_track, reference_track])
        return tracks

    def _generate_meta_from_matches(
        self,
        draft_id: str,
        draft_name: str,
        duration: int
    ) -> dict:
        """Generate draft metadata from raw match results."""
        return {
            "draft_cloud_capcut_purchase_info": None,
            "draft_cloud_last_action_download": False,
            "draft_cloud_materials": [],
            "draft_cloud_purchase_info": None,
            "draft_cloud_template_id": "",
            "draft_cloud_tutorial_info": None,
            "draft_cloud_videocut_purchase_info": None,
            "draft_cover": "",
            "draft_deeplink_url": "",
            "draft_enterprise_info": None,
            "draft_fold_path": "",
            "draft_id": draft_id,
            "draft_is_ai_shorts": False,
            "draft_is_article_video_draft": False,
            "draft_is_from_deeplink": False,
            "draft_is_invisible": False,
            "draft_materials": [],
            "draft_materials_copied": False,
            "draft_name": draft_name,
            "draft_new_version": "",
            "draft_removable_storage_device": "",
            "draft_root_path": "",
            "draft_segment_extra_info": None,
            "draft_timeline_materials_size": 0,
            "draft_type": "",
            "tm_draft_cloud_completed": False,
            "tm_draft_cloud_modified": False,
            "tm_draft_create": int(datetime.now().timestamp() * 1000),
            "tm_draft_modified": int(datetime.now().timestamp() * 1000),
            "tm_draft_removed": 0,
            "tm_duration": duration // 1000
        }

