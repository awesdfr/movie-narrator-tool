"""DaVinci Resolve XML exporter."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from loguru import logger

from models.project import Project
from models.segment import SegmentType


class DaVinciXMLExporter:
    """Export matched movie segments as an xmeml timeline."""

    def __init__(self, default_fps: int = 30):
        self.default_fps = default_fps

    async def export(self, project: Project, output_dir: Path | None = None) -> Path:
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        usable_segments = [
            segment for segment in project.segments
            if segment.use_segment
            and segment.segment_type != SegmentType.NON_MOVIE
            and segment.movie_start is not None
            and segment.movie_end is not None
            and segment.movie_end > segment.movie_start
        ]
        if not usable_segments:
            raise ValueError("没有可导出的已匹配电影片段")

        fps = self._resolve_fps(project.movie_fps)
        width, height = self._resolve_resolution(project.movie_resolution)
        xml_text = await asyncio.to_thread(self._build_xml, project, usable_segments, fps, width, height)

        output_path = output_dir / f"{self._slugify(project.name)}_davinci_timeline.xml"
        output_path.write_text(xml_text, encoding="utf-8")
        logger.info(f"DaVinci XML exported: {output_path}")
        return output_path

    def _build_xml(self, project: Project, usable_segments: list, fps: int, width: int, height: int) -> str:
        timeline_duration = sum(self._seconds_to_frames(segment.movie_end - segment.movie_start, fps) for segment in usable_segments)
        movie_duration_frames = self._seconds_to_frames(project.movie_duration or 0.0, fps)
        movie_uri = Path(project.movie_path).resolve().as_uri() if project.movie_path else ""

        root = Element("xmeml", version="5")
        sequence = SubElement(root, "sequence")
        SubElement(sequence, "name").text = f"{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        SubElement(sequence, "duration").text = str(timeline_duration)
        self._append_rate(sequence, fps)

        media = SubElement(sequence, "media")
        video = SubElement(media, "video")
        video_format = SubElement(video, "format")
        sample = SubElement(video_format, "samplecharacteristics")
        self._append_rate(sample, fps)
        SubElement(sample, "width").text = str(width)
        SubElement(sample, "height").text = str(height)
        video_track = SubElement(video, "track")
        audio = SubElement(media, "audio")
        audio_track = SubElement(audio, "track")

        current_frame = 0
        for index, segment in enumerate(usable_segments, start=1):
            clip_frames = self._seconds_to_frames(segment.movie_end - segment.movie_start, fps)
            source_in = self._seconds_to_frames(segment.movie_start, fps)
            source_out = source_in + clip_frames
            current_end = current_frame + clip_frames
            file_name = Path(project.movie_path).name if project.movie_path else f"movie_{index}.mp4"

            video_item = SubElement(video_track, "clipitem", id=f"video_clip_{index}")
            self._populate_clip_item(
                video_item,
                name=f"Segment_{segment.index + 1:03d}",
                fps=fps,
                start=current_frame,
                end=current_end,
                source_in=source_in,
                source_out=source_out,
                clip_frames=clip_frames,
                file_id=f"video_file_{index}",
                file_name=file_name,
                file_pathurl=movie_uri,
                media_type="video",
                media_duration=movie_duration_frames,
                width=width,
                height=height,
            )

            audio_item = SubElement(audio_track, "clipitem", id=f"audio_clip_{index}")
            self._populate_clip_item(
                audio_item,
                name=f"Segment_{segment.index + 1:03d}_audio",
                fps=fps,
                start=current_frame,
                end=current_end,
                source_in=source_in,
                source_out=source_out,
                clip_frames=clip_frames,
                file_id=f"audio_file_{index}",
                file_name=file_name,
                file_pathurl=movie_uri,
                media_type="audio",
                media_duration=movie_duration_frames,
                width=width,
                height=height,
            )

            current_frame = current_end

        pretty_xml = minidom.parseString(tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
        return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + "\n".join(pretty_xml.splitlines()[1:])

    def _populate_clip_item(
        self,
        parent: Element,
        *,
        name: str,
        fps: int,
        start: int,
        end: int,
        source_in: int,
        source_out: int,
        clip_frames: int,
        file_id: str,
        file_name: str,
        file_pathurl: str,
        media_type: str,
        media_duration: int,
        width: int,
        height: int,
    ) -> None:
        SubElement(parent, "name").text = name
        SubElement(parent, "duration").text = str(clip_frames)
        self._append_rate(parent, fps)
        SubElement(parent, "start").text = str(start)
        SubElement(parent, "end").text = str(end)
        SubElement(parent, "in").text = str(source_in)
        SubElement(parent, "out").text = str(source_out)

        file_node = SubElement(parent, "file", id=file_id)
        SubElement(file_node, "name").text = file_name
        SubElement(file_node, "pathurl").text = file_pathurl
        SubElement(file_node, "duration").text = str(media_duration)
        self._append_rate(file_node, fps)

        media = SubElement(file_node, "media")
        media_node = SubElement(media, media_type)
        sample = SubElement(media_node, "samplecharacteristics")
        self._append_rate(sample, fps)
        if media_type == "video":
            SubElement(sample, "width").text = str(width)
            SubElement(sample, "height").text = str(height)

        source_track = SubElement(parent, "sourcetrack")
        SubElement(source_track, "mediatype").text = media_type
        if media_type == "audio":
            SubElement(source_track, "trackindex").text = "1"

    def _append_rate(self, parent: Element, fps: int) -> None:
        rate = SubElement(parent, "rate")
        SubElement(rate, "timebase").text = str(fps)
        SubElement(rate, "ntsc").text = "FALSE"

    def _seconds_to_frames(self, seconds: float, fps: int) -> int:
        return max(1, int(round(max(0.0, float(seconds)) * fps)))

    def _resolve_fps(self, value: float | None) -> int:
        if not value or value <= 0:
            return self.default_fps
        return max(1, int(round(value)))

    def _resolve_resolution(self, value: tuple[int, int] | None) -> tuple[int, int]:
        if value and value[0] > 0 and value[1] > 0:
            return int(value[0]), int(value[1])
        return 1920, 1080

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "project")
        slug = slug.strip("._")
        return slug or "project"
