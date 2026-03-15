"""字幕导出模块

支持SRT和ASS格式
"""
from pathlib import Path
from typing import Optional
from loguru import logger

from models.project import Project
from models.segment import SegmentType, compute_segment_duration


class SubtitleExporter:
    """字幕导出器"""

    def __init__(self, audio_source: str = "tts"):
        """初始化

        Args:
            audio_source: 音频来源 "original" 或 "tts"，影响字幕时长计算
        """
        self.audio_source = audio_source

    async def export(
        self,
        project: Project,
        format: str = "srt",
        output_dir: Optional[Path] = None,
        use_polished: bool = True
    ) -> Path:
        """导出字幕文件

        Args:
            project: 项目数据
            format: 字幕格式 (srt/ass)
            output_dir: 输出目录
            use_polished: 是否使用润色后的文案

        Returns:
            字幕文件路径
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        output_path = output_dir / f"{project.name}.{format}"

        if format == "srt":
            content = self._generate_srt(project, use_polished)
        elif format == "ass":
            content = self._generate_ass(project, use_polished)
        else:
            raise ValueError(f"不支持的字幕格式: {format}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"字幕导出完成: {output_path}")
        return output_path

    def _compute_timeline_duration(self, segment) -> float:
        """计算片段在时间线上的实际时长，与剪映导出器保持一致"""
        should_use_original = (self.audio_source == "original" or
                             segment.segment_type == SegmentType.NO_NARRATION or
                             not segment.tts_audio_path)
        if should_use_original:
            return segment.narration_end - segment.narration_start
        return compute_segment_duration(segment, self.audio_source)

    def _generate_srt(self, project: Project, use_polished: bool) -> str:
        """生成SRT格式字幕"""
        lines = []
        index = 1
        current_time = 0.0

        for segment in project.segments:
            if not segment.use_segment:
                continue
            if segment.segment_type == SegmentType.NON_MOVIE:
                continue
            # 与剪映导出器一致：跳过没有电影匹配的片段
            if segment.movie_start is None or segment.movie_end is None:
                continue

            text = segment.polished_text if use_polished and segment.polished_text else segment.original_text
            if not text:
                continue

            duration = self._compute_timeline_duration(segment)
            start_time = current_time
            end_time = current_time + duration

            lines.append(str(index))
            lines.append(f"{self._format_srt_time(start_time)} --> {self._format_srt_time(end_time)}")
            lines.append(text)
            lines.append("")

            current_time = end_time
            index += 1

        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """格式化SRT时间戳

        格式: HH:MM:SS,mmm
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _generate_ass(self, project: Project, use_polished: bool) -> str:
        """生成ASS格式字幕"""
        # ASS头部
        header = """[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,微软雅黑,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(title=project.name)

        events = []
        current_time = 0.0

        for segment in project.segments:
            if not segment.use_segment:
                continue
            if segment.segment_type == SegmentType.NON_MOVIE:
                continue
            if segment.movie_start is None or segment.movie_end is None:
                continue

            text = segment.polished_text if use_polished and segment.polished_text else segment.original_text
            if not text:
                continue

            duration = self._compute_timeline_duration(segment)
            start_time = current_time
            end_time = current_time + duration

            # 处理换行
            text = text.replace("\n", "\\N")

            event = f"Dialogue: 0,{self._format_ass_time(start_time)},{self._format_ass_time(end_time)},Default,,0,0,0,,{text}"
            events.append(event)

            current_time = end_time

        return header + "\n".join(events)

    def _format_ass_time(self, seconds: float) -> str:
        """格式化ASS时间戳

        格式: H:MM:SS.cc
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds - int(seconds)) * 100)

        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

    async def export_with_timestamps(
        self,
        project: Project,
        format: str = "srt",
        output_dir: Optional[Path] = None,
        use_movie_time: bool = False
    ) -> Path:
        """导出带原始时间戳的字幕

        Args:
            project: 项目数据
            format: 字幕格式
            output_dir: 输出目录
            use_movie_time: 使用原电影时间而非TTS时间

        Returns:
            字幕文件路径
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        suffix = "_movie_time" if use_movie_time else "_tts_time"
        output_path = output_dir / f"{project.name}{suffix}.{format}"

        lines = []
        index = 1

        for segment in project.segments:
            if not segment.use_segment:
                continue
            if segment.segment_type == SegmentType.NON_MOVIE:
                continue

            text = segment.polished_text if segment.use_polished_text else segment.original_text
            if not text:
                continue

            if use_movie_time:
                start = segment.movie_start or 0
                end = segment.movie_end or start + 3
            else:
                start = segment.narration_start
                end = segment.narration_end

            if format == "srt":
                lines.append(str(index))
                lines.append(f"{self._format_srt_time(start)} --> {self._format_srt_time(end)}")
                lines.append(text)
                lines.append("")
            else:
                text = text.replace("\n", "\\N")
                lines.append(f"Dialogue: 0,{self._format_ass_time(start)},{self._format_ass_time(end)},Default,,0,0,0,,{text}")

            index += 1

        if format == "srt":
            content = "\n".join(lines)
        else:
            header = self._get_ass_header(project.name)
            content = header + "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def _get_ass_header(self, title: str) -> str:
        """获取ASS文件头"""
        return f"""[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,微软雅黑,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
