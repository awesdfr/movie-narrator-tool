"""剪映草稿导出模块

生成剪映Pro可识别的草稿文件
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings
from models.project import Project
from models.segment import Segment, SegmentType, compute_segment_duration


class JianyingExporter:
    """剪映草稿导出器

    生成剪映Pro Drafts格式的草稿文件

    轨道结构:
    - 轨道1: 成品视频 + TTS音频
    - 轨道2: 原解说视频（完整）
    - 轨道3: 被剪掉的原电影片段
    """

    # 时间单位转换（剪映使用微秒）
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
        """初始化

        Args:
            drafts_dir: 剪映草稿目录
            output_fps: 输出帧率，0表示保持原始
            output_resolution: 输出分辨率
            audio_source: 音频来源 "original"(原始解说) 或 "tts"(TTS生成)
            min_playback_speed: 最小播放速度（慢放下限）
            max_playback_speed: 最大播放速度（快放上限）
        """
        self.drafts_dir = Path(drafts_dir or settings.jianying_drafts_dir)
        self.output_fps = output_fps
        self.output_resolution = output_resolution
        self.audio_source = audio_source
        self.min_playback_speed = min_playback_speed
        self.max_playback_speed = max_playback_speed

        # 解析分辨率
        if output_resolution and output_resolution != "original":
            parts = output_resolution.split("x")
            self.width = int(parts[0])
            self.height = int(parts[1])
        else:
            self.width = 1920
            self.height = 1080

    async def export(self, project: Project) -> Path:
        """导出项目到剪映草稿

        Args:
            project: 项目数据

        Returns:
            草稿目录路径
        """
        # 创建草稿目录
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        draft_name = f"{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        draft_dir = self.drafts_dir / draft_name

        draft_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"创建剪映草稿: {draft_dir}")

        # 生成草稿内容
        draft_content = self._generate_draft_content(project, draft_id)

        # 写入草稿文件
        draft_file = draft_dir / "draft_content.json"
        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

        # 复制素材文件到草稿目录
        await self._copy_materials(project, draft_dir)

        # 生成草稿元信息
        meta_content = self._generate_meta_content(project, draft_id, draft_name)
        meta_file = draft_dir / "draft_meta_info.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, ensure_ascii=False, indent=2)

        logger.info(f"剪映草稿导出完成: {draft_dir}")
        return draft_dir

    def _generate_draft_content(self, project: Project, draft_id: str) -> dict:
        """生成草稿内容JSON"""
        # 计算宽高比
        ratio = f"{self.width}:{self.height}"
        if self.width == 1920 and self.height == 1080:
            ratio = "16:9"
        elif self.width == 1280 and self.height == 720:
            ratio = "16:9"
        elif self.width == 1080 and self.height == 1920:
            ratio = "9:16"

        # 基础结构
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
            "tracks": self._generate_tracks(project),
            "update_time": int(datetime.now().timestamp() * 1000000),
            "version": 360000
        }

        # 计算总时长
        content["duration"] = self._calculate_duration(project)

        return content

    def _generate_materials(self, project: Project) -> dict:
        """生成素材列表"""
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

        # 添加原电影视频素材
        if project.movie_path:
            movie_material = self._create_video_material(
                project.movie_path,
                f"movie_{project.id}",
                project.movie_duration or 0
            )
            materials["videos"].append(movie_material)

        # 添加解说视频素材
        if project.narration_path:
            narration_material = self._create_video_material(
                project.narration_path,
                f"narration_{project.id}",
                project.narration_duration or 0
            )
            materials["videos"].append(narration_material)

            # 添加原始解说音频素材（用于提取音频）
            if self.audio_source == "original":
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

        # 添加TTS音频素材（仅在使用TTS模式时）
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
        """创建视频素材"""
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
        """创建音频素材"""
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

    # 低置信度阈值，低于此值的片段放到"待检查"轨道
    LOW_CONFIDENCE_THRESHOLD = 0.78

    def _generate_tracks(self, project: Project) -> list:
        """生成轨道列表"""
        tracks = []

        # 轨道1: 主视频轨道（原电影匹配片段，高置信度）
        main_video_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "",
            "segments": [],
            "type": "video"
        }

        # 轨道2: 主音频轨道（TTS音频）
        main_audio_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "",
            "segments": [],
            "type": "audio"
        }

        # 轨道3: 原解说视频（完整）
        narration_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "解说视频",
            "segments": [],
            "type": "video"
        }

        # 轨道4: 被剪掉的原电影片段
        cut_movie_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "原电影片段",
            "segments": [],
            "type": "video"
        }

        # 轨道5: 待检查片段（低置信度，需要手动调整）
        low_confidence_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": False,
            "name": "待检查(低置信度)",
            "segments": [],
            "type": "video"
        }

        # 填充主视频轨道和音频轨道
        current_time = 0
        for i, segment in enumerate(project.segments):
            if not segment.use_segment:
                continue

            if segment.segment_type == SegmentType.NON_MOVIE:
                continue

            # 判断实际使用的音频来源
            should_use_original = (self.audio_source == "original" or
                                 segment.segment_type == SegmentType.NO_NARRATION or
                                 not segment.tts_audio_path)

            # 根据实际音频来源决定片段时长
            # 使用原始音频时，时间线以原始音频时长为准，确保音画同步
            if should_use_original:
                duration = segment.narration_end - segment.narration_start
            else:
                duration = compute_segment_duration(segment, self.audio_source)

            # 没有匹配到原电影画面的片段直接跳过，不使用解说视频凑数
            if segment.movie_start is None or segment.movie_end is None:
                continue

            # 使用原电影匹配片段
            # 智能延长逻辑：
            # 1. 计算音频需要的结束时间 (desired_end)
            # 2. 查找下一个片段的开始时间作为限制 (limit_end)，防止画面重复
            # 3. 如果有空隙，优先延长视频
            # 4. 如果没空隙或延长后仍不够，则使用变速
            desired_end = segment.movie_start + duration

            # 查找下一个使用同一电影源的片段的开始时间
            next_movie_start = None
            for next_seg in project.segments[i+1:]:
                if next_seg.use_segment and next_seg.movie_start is not None:
                    next_movie_start = next_seg.movie_start
                    break

            limit_end = next_movie_start if next_movie_start is not None else (project.movie_duration or float('inf'))
            base_end = segment.movie_end

            if desired_end > base_end:
                if limit_end > base_end:
                    extension_end = min(desired_end, limit_end)
                    final_movie_end = max(extension_end, base_end)
                else:
                    final_movie_end = base_end
            else:
                final_movie_end = base_end

            video_seg = self._create_video_segment(
                f"movie_{project.id}",
                project.movie_path,
                current_time,
                segment.movie_start,
                final_movie_end,
                segment.mute_movie_audio,
                target_duration=duration
            )

            # 根据置信度决定放到哪个轨道
            if segment.match_confidence < self.LOW_CONFIDENCE_THRESHOLD:
                # 低置信度：放到"待检查"轨道
                low_confidence_track["segments"].append(video_seg)
                logger.debug(f"片段 {segment.id} 置信度 {segment.match_confidence:.2f} 低于阈值，放入待检查轨道")
            else:
                # 高置信度：放到主轨道
                main_video_track["segments"].append(video_seg)

            # 音频片段
            if should_use_original:
                # 使用原始解说音频
                audio_seg = self._create_original_audio_segment(
                    f"narration_audio_{project.id}",
                    project.narration_path,
                    current_time,
                    segment.narration_start,
                    segment.narration_end
                )
                main_audio_track["segments"].append(audio_seg)
            elif segment.tts_audio_path and segment.tts_duration:
                # 使用TTS音频
                audio_seg = self._create_audio_segment(
                    f"tts_{segment.id}",
                    segment.tts_audio_path,
                    current_time,
                    segment.tts_duration
                )
                main_audio_track["segments"].append(audio_seg)

            # 更新时间线
            current_time += int(duration * self.TIME_SCALE)

        # 添加完整解说视频到轨道3
        if project.narration_path and project.narration_duration:
            narration_seg = self._create_video_segment(
                f"narration_{project.id}",
                project.narration_path,
                0,
                0,
                project.narration_duration,
                True,  # 静音背景轨道，防止音画重复
                visible=False  # 隐藏背景轨道，防止画面重叠
            )
            narration_track["segments"].append(narration_seg)

        # 统计待检查片段数量
        low_conf_count = len(low_confidence_track["segments"])
        if low_conf_count > 0:
            logger.info(f"导出剪映草稿: {low_conf_count} 个低置信度片段放入'待检查'轨道")

        tracks.extend([main_video_track, main_audio_track, narration_track, cut_movie_track, low_confidence_track])
        return tracks

    def _create_video_segment(
        self,
        material_id: str,
        path: str,
        timeline_start: int,
        source_start: float,
        source_end: float,
        mute: bool,
        visible: bool = True,
        target_duration: float = None
    ) -> dict:
        """创建视频片段
        
        Args:
            target_duration: 目标时长（秒），如果不传则默认等于源时长
        """
        source_duration = source_end - source_start
        source_duration_us = int(source_duration * self.TIME_SCALE)

        if target_duration:
            target_duration_us = int(target_duration * self.TIME_SCALE)
            # 计算变速倍率: 源时长 / 目标时长
            # 例如: 5秒视频要播10秒，速度就是 0.5
            raw_speed = source_duration / target_duration if target_duration > 0 else 1.0

            if raw_speed > self.max_playback_speed:
                # 快放截断：源片段太长，截取前面一部分（安全）
                speed = self.max_playback_speed
                actual_source_duration = speed * target_duration
                source_duration_us = int(actual_source_duration * self.TIME_SCALE)
                logger.debug(
                    f"快放截断: {raw_speed:.2f}x → {speed:.2f}x, "
                    f"源时长 {source_duration:.1f}s → {actual_source_duration:.1f}s"
                )
            elif raw_speed < self.min_playback_speed:
                # 慢放超限：源片段太短，但不能凭空创造源素材
                # 保持原始速度，避免 source_duration 超出实际可用长度
                speed = raw_speed
                logger.debug(
                    f"慢放速度 {raw_speed:.2f}x 低于下限 {self.min_playback_speed}x，"
                    f"保持原速度（源片段不足以截断）"
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
        """创建音频片段"""
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
        """创建原始音频片段（从视频中提取指定时间段的音频）"""
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
        """计算项目总时长（仅包含有电影匹配的片段）"""
        total_duration = 0
        for segment in project.segments:
            if not segment.use_segment:
                continue
            if segment.segment_type == SegmentType.NON_MOVIE:
                continue
            # 与 _generate_tracks 一致：跳过没有电影匹配的片段
            if segment.movie_start is None or segment.movie_end is None:
                continue

            should_use_original = (self.audio_source == "original" or
                                 segment.segment_type == SegmentType.NO_NARRATION or
                                 not segment.tts_audio_path)
            if should_use_original:
                total_duration += segment.narration_end - segment.narration_start
            else:
                total_duration += compute_segment_duration(segment, self.audio_source)

        return int(total_duration * self.TIME_SCALE)

    def _generate_meta_content(self, project: Project, draft_id: str, draft_name: str) -> dict:
        """生成草稿元信息"""
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
            "tm_duration": self._calculate_duration(project) // 1000
        }

    async def _copy_materials(self, project: Project, draft_dir: Path):
        """复制素材文件到草稿目录"""
        materials_dir = draft_dir / "materials"
        materials_dir.mkdir(exist_ok=True)

        # 复制TTS音频
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
        """从匹配结果导出剪映草稿

        Args:
            results: MatchResult 列表
            movie_path: 原电影路径
            narration_path: 解说视频路径
            project_name: 项目名称

        Returns:
            草稿目录路径
        """
        # 创建草稿目录
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        draft_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        draft_dir = self.drafts_dir / draft_name

        draft_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"创建剪映草稿: {draft_dir}")

        # 获取视频时长
        import cv2
        movie_cap = cv2.VideoCapture(movie_path)
        movie_duration = movie_cap.get(cv2.CAP_PROP_FRAME_COUNT) / movie_cap.get(cv2.CAP_PROP_FPS)
        movie_cap.release()

        narration_cap = cv2.VideoCapture(narration_path)
        narration_duration = narration_cap.get(cv2.CAP_PROP_FRAME_COUNT) / narration_cap.get(cv2.CAP_PROP_FPS)
        narration_cap.release()

        # 生成草稿内容
        draft_content = self._generate_draft_from_matches(
            results, movie_path, narration_path,
            movie_duration, narration_duration, draft_id, project_name
        )

        # 写入草稿文件
        draft_file = draft_dir / "draft_content.json"
        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

        # 生成草稿元信息
        meta_content = self._generate_meta_from_matches(
            draft_id, draft_name, draft_content["duration"]
        )
        meta_file = draft_dir / "draft_meta_info.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, ensure_ascii=False, indent=2)

        logger.info(f"剪映草稿导出完成: {draft_dir}")
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
        """从匹配结果生成草稿内容"""
        # 计算宽高比
        ratio = f"{self.width}:{self.height}"
        if self.width == 1920 and self.height == 1080:
            ratio = "16:9"
        elif self.width == 1280 and self.height == 720:
            ratio = "16:9"
        elif self.width == 1080 and self.height == 1920:
            ratio = "9:16"

        # 基础结构
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

        # 计算总时长
        if results:
            total_duration = sum(r.narration_duration for r in results)
            content["duration"] = int(total_duration * self.TIME_SCALE)
        else:
            content["duration"] = 0

        return content

    def _generate_materials_from_matches(
        self,
        movie_path: str,
        narration_path: str,
        movie_duration: float,
        narration_duration: float
    ) -> dict:
        """从匹配结果生成素材列表"""
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

        # 添加原电影视频素材
        movie_material = self._create_video_material(
            movie_path,
            "movie_material",
            movie_duration
        )
        materials["videos"].append(movie_material)

        # 添加解说视频素材（用于提取音频）
        narration_material = self._create_video_material(
            narration_path,
            "narration_material",
            narration_duration
        )
        materials["videos"].append(narration_material)

        # 添加解说视频的原声音频素材
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
        """从匹配结果生成轨道"""
        tracks = []

        # 轨道1: 主视频轨道（原电影匹配片段）
        main_video_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "原电影画面",
            "segments": [],
            "type": "video"
        }

        # 轨道2: 解说音频轨道
        narration_audio_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "解说音频",
            "segments": [],
            "type": "audio"
        }

        # 轨道3: 参考轨道（完整解说视频，静音）
        reference_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()),
            "is_default_name": True,
            "name": "解说视频(参考)",
            "segments": [],
            "type": "video"
        }

        # 填充轨道
        timeline_pos = 0
        for result in results:
            segment_duration = result.narration_duration
            duration_us = int(segment_duration * self.TIME_SCALE)

            # 视频片段（从原电影剪切）
            video_seg = self._create_video_segment(
                "movie_material",
                movie_path,
                timeline_pos,
                result.movie_start,
                result.movie_end,
                mute=True  # 静音原电影音频
            )
            main_video_track["segments"].append(video_seg)

            # 音频片段（从解说视频剪切）
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
                    "start": int(result.narration_start * self.TIME_SCALE)
                },
                "speed": 1.0,
                "target_timerange": {
                    "duration": duration_us,
                    "start": timeline_pos
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

            timeline_pos += duration_us

        tracks.extend([main_video_track, narration_audio_track, reference_track])
        return tracks

    def _generate_meta_from_matches(
        self,
        draft_id: str,
        draft_name: str,
        duration: int
    ) -> dict:
        """生成草稿元信息"""
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
