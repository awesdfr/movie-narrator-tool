"""时间工具函数"""
import re
from typing import Optional


def format_duration(seconds: float, include_ms: bool = False) -> str:
    """格式化时长

    Args:
        seconds: 秒数
        include_ms: 是否包含毫秒

    Returns:
        格式化字符串 (HH:MM:SS 或 HH:MM:SS.mmm)
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if include_ms:
        ms = int((seconds - int(seconds)) * 1000)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
        return f"{minutes:02d}:{secs:02d}.{ms:03d}"
    else:
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def parse_duration(time_str: str) -> Optional[float]:
    """解析时长字符串

    支持格式:
    - HH:MM:SS
    - MM:SS
    - HH:MM:SS.mmm
    - MM:SS.mmm
    - 纯秒数

    Args:
        time_str: 时长字符串

    Returns:
        秒数，解析失败返回None
    """
    time_str = time_str.strip()

    # 纯数字（秒）
    try:
        return float(time_str)
    except ValueError:
        pass

    # HH:MM:SS.mmm 或 MM:SS.mmm
    patterns = [
        r'^(\d+):(\d{2}):(\d{2})\.(\d{1,3})$',  # HH:MM:SS.mmm
        r'^(\d+):(\d{2}):(\d{2})$',              # HH:MM:SS
        r'^(\d+):(\d{2})\.(\d{1,3})$',           # MM:SS.mmm
        r'^(\d+):(\d{2})$',                       # MM:SS
    ]

    for pattern in patterns:
        match = re.match(pattern, time_str)
        if match:
            groups = match.groups()
            if len(groups) == 4:  # HH:MM:SS.mmm
                hours, minutes, secs, ms = groups
                ms = int(ms.ljust(3, '0'))
                return int(hours) * 3600 + int(minutes) * 60 + int(secs) + ms / 1000
            elif len(groups) == 3:
                if ':' in time_str and '.' in time_str:  # MM:SS.mmm
                    minutes, secs, ms = groups
                    ms = int(ms.ljust(3, '0'))
                    return int(minutes) * 60 + int(secs) + ms / 1000
                else:  # HH:MM:SS
                    hours, minutes, secs = groups
                    return int(hours) * 3600 + int(minutes) * 60 + int(secs)
            elif len(groups) == 2:  # MM:SS
                minutes, secs = groups
                return int(minutes) * 60 + int(secs)

    return None


def format_timestamp(seconds: float) -> str:
    """格式化为SRT时间戳格式

    Args:
        seconds: 秒数

    Returns:
        HH:MM:SS,mmm 格式
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def seconds_to_frames(seconds: float, fps: float) -> int:
    """秒数转帧数

    Args:
        seconds: 秒数
        fps: 帧率

    Returns:
        帧数
    """
    return int(seconds * fps)


def frames_to_seconds(frames: int, fps: float) -> float:
    """帧数转秒数

    Args:
        frames: 帧数
        fps: 帧率

    Returns:
        秒数
    """
    return frames / fps if fps > 0 else 0


def get_time_overlap(
    start1: float, end1: float,
    start2: float, end2: float
) -> float:
    """计算两个时间段的重叠时长

    Args:
        start1, end1: 第一个时间段
        start2, end2: 第二个时间段

    Returns:
        重叠时长（秒）
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_end > overlap_start:
        return overlap_end - overlap_start
    return 0


def merge_time_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """合并重叠的时间段

    Args:
        ranges: [(start, end), ...] 时间段列表

    Returns:
        合并后的时间段列表
    """
    if not ranges:
        return []

    # 按起始时间排序
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    merged = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # 重叠或相邻
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged
