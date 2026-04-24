"""分辨率自适应采样模块

根据视频分辨率动态调整采样策略和匹配阈值。
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SamplingConfig:
    """采样配置"""
    sample_fps: float               # 采样帧率（帧/秒）
    frame_threshold: float          # 帧匹配阈值（汉明距离）
    audio_threshold: float          # 音频匹配阈值
    quality_factor: float           # 质量因子 (0-1)
    complexity_level: int           # 复杂度等级 (1-3)


class ResolutionAdaptiveSampler:
    """分辨率自适应采样器
    
    特点：
    1. 分辨率感知：自动识别视频分辨率等级
    2. 动态采样：根据分辨率调整采样密度
    3. 阈值自适应：基于分辨率和质量调整匹配阈值
    4. 复杂度控制：平衡精度和性能
    """
    
    # 分辨率等级定义
    RESOLUTION_LEVELS = {
        'ultra_low': {'pixels': 320 * 240, 'scale': 0.5},       # <77K
        'very_low': {'pixels': 480 * 360, 'scale': 0.75},       # <173K
        'low': {'pixels': 640 * 480, 'scale': 1.0},             # <307K
        'medium': {'pixels': 1024 * 768, 'scale': 1.5},         # <786K
        'high': {'pixels': 1280 * 720, 'scale': 2.0},           # <921K
        'very_high': {'pixels': 1920 * 1080, 'scale': 3.0},     # <2M
        'ultra_high': {'pixels': 3840 * 2160, 'scale': 6.0},    # <8M
    }
    
    # 采样配置（按分辨率）
    SAMPLING_PROFILES = {
        'ultra_low': {
            'sample_fps': 2.0,
            'frame_threshold': 12,
            'audio_threshold': 0.4,
            'quality_factor': 0.5,
            'complexity_level': 1
        },
        'very_low': {
            'sample_fps': 2.5,
            'frame_threshold': 11,
            'audio_threshold': 0.45,
            'quality_factor': 0.65,
            'complexity_level': 1
        },
        'low': {
            'sample_fps': 3.0,
            'frame_threshold': 10,
            'audio_threshold': 0.5,
            'quality_factor': 0.75,
            'complexity_level': 2
        },
        'medium': {
            'sample_fps': 3.5,
            'frame_threshold': 9,
            'audio_threshold': 0.55,
            'quality_factor': 0.85,
            'complexity_level': 2
        },
        'high': {
            'sample_fps': 4.0,
            'frame_threshold': 8,
            'audio_threshold': 0.6,
            'quality_factor': 0.9,
            'complexity_level': 2
        },
        'very_high': {
            'sample_fps': 5.0,
            'frame_threshold': 7,
            'audio_threshold': 0.65,
            'quality_factor': 0.95,
            'complexity_level': 3
        },
        'ultra_high': {
            'sample_fps': 6.0,
            'frame_threshold': 6,
            'audio_threshold': 0.7,
            'quality_factor': 1.0,
            'complexity_level': 3
        }
    }
    
    def __init__(self):
        """初始化"""
        pass
    
    def detect_resolution_level(self, width: int, height: int) -> str:
        """检测分辨率等级
        
        Args:
            width: 视频宽度（像素）
            height: 视频高度（像素）
            
        Returns:
            分辨率等级字符串
        """
        pixels = width * height
        
        # 找到最接近的分辨率等级
        best_level = 'medium'
        best_distance = abs(pixels - self.RESOLUTION_LEVELS['medium']['pixels'])
        
        for level, info in self.RESOLUTION_LEVELS.items():
            distance = abs(pixels - info['pixels'])
            if distance < best_distance:
                best_distance = distance
                best_level = level
        
        logger.info(f"检测分辨率: {width}x{height} ({pixels:,} pixels) -> {best_level}")
        return best_level
    
    def get_sampling_config(
        self,
        width: int,
        height: int,
        quality_override: Optional[float] = None
    ) -> SamplingConfig:
        """获取采样配置
        
        Args:
            width: 视频宽度
            height: 视频高度
            quality_override: 质量覆盖值 (0-1)，None 表示自动
            
        Returns:
            SamplingConfig 对象
        """
        level = self.detect_resolution_level(width, height)
        profile = self.SAMPLING_PROFILES.get(level, self.SAMPLING_PROFILES['medium']).copy()
        
        # 应用质量覆盖
        if quality_override is not None:
            profile['quality_factor'] = quality_override
            # 根据质量调整阈值
            if quality_override < 0.5:
                profile['frame_threshold'] = min(12, profile['frame_threshold'] + 2)
                profile['audio_threshold'] = min(0.7, profile['audio_threshold'] + 0.1)
            elif quality_override > 0.9:
                profile['frame_threshold'] = max(6, profile['frame_threshold'] - 1)
                profile['audio_threshold'] = max(0.3, profile['audio_threshold'] - 0.1)
        
        return SamplingConfig(
            sample_fps=profile['sample_fps'],
            frame_threshold=profile['frame_threshold'],
            audio_threshold=profile['audio_threshold'],
            quality_factor=profile['quality_factor'],
            complexity_level=profile['complexity_level']
        )
    
    def compute_sample_step(
        self,
        video_fps: float,
        resolution_level: str,
        override_sample_fps: Optional[float] = None
    ) -> int:
        """计算采样步长（帧数）
        
        Args:
            video_fps: 视频帧率
            resolution_level: 分辨率等级
            override_sample_fps: 覆盖采样帧率
            
        Returns:
            采样步长（帧数）
        """
        if override_sample_fps is not None:
            sample_fps = override_sample_fps
        else:
            sample_fps = self.SAMPLING_PROFILES[resolution_level]['sample_fps']
        
        if video_fps <= 0:
            return 1
        
        # 采样步长 = 视频帧率 / 采样帧率
        step = max(1, int(video_fps / sample_fps))
        
        logger.debug(f"采样步长: {step} frames (video_fps={video_fps}, sample_fps={sample_fps})")
        return step
    
    def adjust_thresholds(
        self,
        base_frame_threshold: int,
        base_audio_threshold: float,
        width: int,
        height: int,
        quality_score: Optional[float] = None
    ) -> Tuple[int, float]:
        """调整匹配阈值
        
        Args:
            base_frame_threshold: 基础帧匹配阈值
            base_audio_threshold: 基础音频阈值
            width: 视频宽度
            height: 视频高度
            quality_score: 图像质量分数 (0-1)
            
        Returns:
            (调整后的帧阈值, 调整后的音频阈值)
        """
        level = self.detect_resolution_level(width, height)
        profile = self.SAMPLING_PROFILES[level]
        
        # 基于分辨率调整
        frame_threshold = profile['frame_threshold']
        audio_threshold = profile['audio_threshold']
        
        # 基于质量进一步调整
        if quality_score is not None:
            if quality_score < 0.3:
                # 低质量：放松阈值
                frame_threshold += 3
                audio_threshold += 0.15
            elif quality_score < 0.6:
                # 中低质量：略放松
                frame_threshold += 1
                audio_threshold += 0.05
            elif quality_score > 0.85:
                # 高质量：收紧阈值
                frame_threshold = max(6, frame_threshold - 1)
                audio_threshold = max(0.3, audio_threshold - 0.05)
        
        return int(frame_threshold), float(audio_threshold)
    
    def estimate_processing_time(
        self,
        width: int,
        height: int,
        duration_sec: float
    ) -> float:
        """估计处理时间
        
        Args:
            width: 视频宽度
            height: 视频高度
            duration_sec: 视频时长（秒）
            
        Returns:
            估计处理时间（秒）
        """
        pixels = width * height
        
        # 每百万像素每秒的基础处理时间（毫秒）
        base_time_per_megapixel_sec = {
            'ultra_low': 10,
            'very_low': 15,
            'low': 20,
            'medium': 30,
            'high': 50,
            'very_high': 80,
            'ultra_high': 150
        }
        
        level = self.detect_resolution_level(width, height)
        base_ms = base_time_per_megapixel_sec.get(level, 30)
        
        # 计算总处理时间
        megapixels = pixels / 1e6
        total_ms = base_ms * megapixels * duration_sec
        
        return total_ms / 1000.0  # 转换为秒
    
    def recommend_settings(
        self,
        width: int,
        height: int,
        duration_sec: float,
        available_memory_gb: float = 8.0,
        available_time_sec: Optional[float] = None
    ) -> dict:
        """推荐最优设置
        
        Args:
            width: 视频宽度
            height: 视频高度
            duration_sec: 视频时长
            available_memory_gb: 可用内存（GB）
            available_time_sec: 可用处理时间（秒）
            
        Returns:
            推荐设置字典
        """
        level = self.detect_resolution_level(width, height)
        config = self.get_sampling_config(width, height)
        
        # 基于内存调整
        pixels_per_frame = width * height
        estimated_frames = int(duration_sec * config.sample_fps)
        bytes_per_frame = pixels_per_frame * 3  # BGR 格式
        estimated_memory_gb = (estimated_frames * bytes_per_frame) / (1024**3)
        
        memory_ratio = estimated_memory_gb / available_memory_gb
        if memory_ratio > 0.8:
            # 内存紧张，降低采样率
            config.sample_fps *= 0.7
            config.complexity_level = max(1, config.complexity_level - 1)
            logger.warning(f"内存紧张({memory_ratio:.1%})，降低采样率")
        
        # 基于时间调整
        estimated_time = self.estimate_processing_time(width, height, duration_sec)
        if available_time_sec is not None and estimated_time > available_time_sec:
            # 时间不足，进一步优化
            time_ratio = estimated_time / available_time_sec
            config.sample_fps /= time_ratio
            config.complexity_level = 1
            logger.warning(f"时间不足({time_ratio:.1f}x)，严格优化处理")
        
        return {
            'resolution_level': level,
            'sampling_config': config,
            'estimated_memory_gb': estimated_memory_gb,
            'estimated_processing_time_sec': estimated_time,
            'memory_utilization_ratio': memory_ratio if available_memory_gb > 0 else 0
        }
