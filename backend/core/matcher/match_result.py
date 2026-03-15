"""匹配结果模型"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchResult:
    """视频匹配结果

    记录解说视频片段与原电影的匹配信息
    """

    # 解说视频时间范围
    narration_start: float
    narration_end: float

    # 原电影匹配时间范围
    movie_start: float
    movie_end: float

    # 置信度
    frame_confidence: float = 0.0
    audio_confidence: float = 0.0
    combined_confidence: float = 0.0

    # 匹配来源
    match_source: str = "hybrid"  # "frame", "audio", "hybrid"

    # 可选元数据
    segment_index: int = 0
    notes: str = ""

    @property
    def narration_duration(self) -> float:
        """解说片段时长"""
        return self.narration_end - self.narration_start

    @property
    def movie_duration(self) -> float:
        """原电影匹配片段时长"""
        return self.movie_end - self.movie_start

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "narration_start": self.narration_start,
            "narration_end": self.narration_end,
            "movie_start": self.movie_start,
            "movie_end": self.movie_end,
            "frame_confidence": self.frame_confidence,
            "audio_confidence": self.audio_confidence,
            "combined_confidence": self.combined_confidence,
            "match_source": self.match_source,
            "segment_index": self.segment_index,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        """从字典创建"""
        return cls(
            narration_start=data["narration_start"],
            narration_end=data["narration_end"],
            movie_start=data["movie_start"],
            movie_end=data["movie_end"],
            frame_confidence=data.get("frame_confidence", 0.0),
            audio_confidence=data.get("audio_confidence", 0.0),
            combined_confidence=data.get("combined_confidence", 0.0),
            match_source=data.get("match_source", "hybrid"),
            segment_index=data.get("segment_index", 0),
            notes=data.get("notes", ""),
        )


@dataclass
class MatchConfig:
    """匹配配置"""

    # 权重配置
    frame_weight: float = 0.6
    audio_weight: float = 0.4

    # 阈值配置
    min_confidence: float = 0.5
    frame_threshold: float = 0.7
    audio_threshold: float = 0.6

    # 采样配置
    frame_sample_interval: int = 5  # 帧采样间隔
    audio_window_sec: float = 2.0   # 音频窗口大小（秒）
    audio_step_sec: float = 0.5     # 音频滑动步长（秒）

    # 场景切分配置
    scene_threshold: float = 30.0   # 场景切换阈值
    min_segment_duration: float = 1.0  # 最小片段时长（秒）
