"""SRT subtitle parser."""
import re
from pathlib import Path

from loguru import logger


class SubtitleParser:
    """Parse SRT files into the same shape as Whisper output."""

    SRT_TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})")
    SRT_TIMELINE_PATTERN = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
    )

    def parse_srt(self, srt_path: str) -> list[dict]:
        path = Path(srt_path)
        if not path.exists():
            raise FileNotFoundError(f"SRT file does not exist: {srt_path}")

        content = self._read_file_with_encoding(path)
        segments = self._parse_srt_content(content)
        if not segments:
            raise ValueError(f"No subtitle segments parsed from: {srt_path}")

        logger.info(
            "SRT parsed: {} segments, {:.1f}s -> {:.1f}s".format(
                len(segments), segments[0]["start"], segments[-1]["end"]
            )
        )
        return segments

    def _read_file_with_encoding(self, path: Path) -> str:
        for encoding in ["utf-8-sig", "utf-8", "gbk", "gb2312", "big5", "latin-1"]:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Unable to detect SRT encoding: {path}")

    def _parse_srt_content(self, content: str) -> list[dict]:
        segments = []
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if len(lines) < 2:
                continue

            time_line = next((line for line in lines if self.SRT_TIMELINE_PATTERN.search(line)), None)
            if not time_line:
                continue

            match = self.SRT_TIMELINE_PATTERN.search(time_line)
            if not match:
                continue

            start = self._parse_srt_time(match.group(1))
            end = self._parse_srt_time(match.group(2))
            text_start_idx = lines.index(time_line) + 1
            text = "\n".join(lines[text_start_idx:]).strip()
            text = re.sub(r"<[^>]+>", "", text)
            if not text or end <= start:
                continue

            segments.append({"start": start, "end": end, "text": text, "words": []})

        segments.sort(key=lambda item: item["start"])
        return segments

    def _parse_srt_time(self, time_str: str) -> float:
        match = self.SRT_TIME_PATTERN.match(time_str.strip())
        if not match:
            raise ValueError(f"Invalid SRT timestamp: {time_str}")
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int(match.group(4))
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
