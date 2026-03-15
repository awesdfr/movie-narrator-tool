"""算力自适应策略框架

根据运行环境自动选择最优策略：
1. 低算力模式：边缘端/低配机器，仅使用核心功能
2. 中算力模式：普通服务器，平衡精度和效率
3. 高算力模式：高性能GPU服务器，启用全部功能

支持动态切换和手动覆盖。
"""
import os
import platform
from typing import Optional, Dict, Any

# 尝试导入 psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ComputeLevel(Enum):
    """算力级别"""
    LOW = "low"           # 低算力（边缘端、CPU-only）
    MEDIUM = "medium"     # 中算力（普通GPU、低显存）
    HIGH = "high"         # 高算力（高性能GPU）
    AUTO = "auto"         # 自动检测


@dataclass
class StrategyConfig:
    """策略配置"""
    # 采样配置
    sample_fps: float = 4.0               # 采样帧率
    use_dynamic_sampling: bool = True     # 动态采样
    max_sample_fps: float = 15.0          # 最大采样率

    # 哈希配置
    use_multi_scale_hash: bool = True     # 多尺度哈希
    hash_scales: list = field(default_factory=lambda: [24, 32, 48])

    # 匹配配置
    use_sequence_alignment: bool = True   # 序列比对
    phash_threshold: int = 8              # pHash 阈值

    # 预处理配置
    use_distortion_normalize: bool = True  # 失真归一化
    use_prefilter: bool = True             # 预筛选

    # 验证配置
    use_optical_flow: bool = True          # 光流验证
    use_confidence_verify: bool = True     # 置信度验证

    # GPU 配置
    use_gpu: bool = True                   # 使用 GPU
    gpu_batch_size: int = 64               # GPU 批处理大小
    max_gpu_memory_mb: int = 0             # 最大显存（0 = 自动 80%）

    # 并发配置
    max_concurrent_tasks: int = 4          # 最大并发任务


# 预定义策略
LOW_COMPUTE_STRATEGY = StrategyConfig(
    sample_fps=2.0,
    use_dynamic_sampling=False,
    max_sample_fps=4.0,
    use_multi_scale_hash=False,
    hash_scales=[32],
    use_sequence_alignment=False,
    phash_threshold=12,
    use_distortion_normalize=False,
    use_prefilter=True,  # 保留预筛选，减少计算量
    use_optical_flow=False,
    use_confidence_verify=False,
    use_gpu=False,
    gpu_batch_size=16,
    max_gpu_memory_mb=512,
    max_concurrent_tasks=2
)

MEDIUM_COMPUTE_STRATEGY = StrategyConfig(
    sample_fps=4.0,
    use_dynamic_sampling=True,
    max_sample_fps=8.0,
    use_multi_scale_hash=True,
    hash_scales=[32, 48],
    use_sequence_alignment=True,
    phash_threshold=8,
    use_distortion_normalize=True,
    use_prefilter=True,
    use_optical_flow=False,  # 光流计算较重，中配不启用
    use_confidence_verify=True,
    use_gpu=True,
    gpu_batch_size=64,
    max_gpu_memory_mb=0,  # 自动
    max_concurrent_tasks=4
)

HIGH_COMPUTE_STRATEGY = StrategyConfig(
    sample_fps=6.0,
    use_dynamic_sampling=True,
    max_sample_fps=15.0,
    use_multi_scale_hash=True,
    hash_scales=[24, 32, 48],
    use_sequence_alignment=True,
    phash_threshold=6,
    use_distortion_normalize=True,
    use_prefilter=True,
    use_optical_flow=True,
    use_confidence_verify=True,
    use_gpu=True,
    gpu_batch_size=128,
    max_gpu_memory_mb=0,  # 0 = 自动使用 80% 显存
    max_concurrent_tasks=8
)


@dataclass
class SystemInfo:
    """系统信息"""
    platform: str
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_memory_mb: int
    cuda_version: str


class AdaptiveStrategyManager:
    """算力自适应策略管理器

    功能：
    1. 自动检测系统硬件能力
    2. 选择最优匹配策略
    3. 支持动态调整和手动覆盖
    4. 提供策略配置接口
    """

    def __init__(self, compute_level: ComputeLevel = ComputeLevel.AUTO):
        """初始化

        Args:
            compute_level: 算力级别，AUTO 表示自动检测
        """
        self._system_info: Optional[SystemInfo] = None
        self._current_level: ComputeLevel = compute_level
        self._current_strategy: Optional[StrategyConfig] = None
        self._custom_overrides: Dict[str, Any] = {}

        self._detect_system()
        self._select_strategy()

    def _detect_system(self):
        """检测系统硬件信息"""
        # 基础信息
        if PSUTIL_AVAILABLE:
            cpu_count = psutil.cpu_count(logical=False) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            cpu_count = os.cpu_count() or 1
            memory_gb = 8.0  # 默认假设 8GB

        # GPU 信息
        gpu_available = False
        gpu_name = "N/A"
        gpu_memory_mb = 0
        cuda_version = "N/A"

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                gpu_memory_mb = props.total_memory // (1024 * 1024)
                cuda_version = torch.version.cuda or "N/A"
        except ImportError:
            pass

        self._system_info = SystemInfo(
            platform=platform.system(),
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            cuda_version=cuda_version
        )

        logger.info(
            f"系统检测: CPU={cpu_count}核, 内存={memory_gb:.1f}GB, "
            f"GPU={gpu_name if gpu_available else '无'}, "
            f"显存={gpu_memory_mb}MB"
        )

    def _select_strategy(self):
        """根据系统能力选择策略"""
        if self._current_level != ComputeLevel.AUTO:
            # 手动指定级别
            if self._current_level == ComputeLevel.LOW:
                self._current_strategy = StrategyConfig(**LOW_COMPUTE_STRATEGY.__dict__)
            elif self._current_level == ComputeLevel.MEDIUM:
                self._current_strategy = StrategyConfig(**MEDIUM_COMPUTE_STRATEGY.__dict__)
            else:
                self._current_strategy = StrategyConfig(**HIGH_COMPUTE_STRATEGY.__dict__)
            logger.info(f"使用手动指定策略: {self._current_level.value}")
            return

        # 自动检测
        info = self._system_info

        # 判断条件
        if not info.gpu_available or info.gpu_memory_mb < 2048:
            # 无 GPU 或显存 < 2GB
            if info.cpu_count >= 8 and info.memory_gb >= 16:
                # 高配 CPU
                self._current_level = ComputeLevel.MEDIUM
                self._current_strategy = StrategyConfig(**MEDIUM_COMPUTE_STRATEGY.__dict__)
                self._current_strategy.use_gpu = False
            else:
                self._current_level = ComputeLevel.LOW
                self._current_strategy = StrategyConfig(**LOW_COMPUTE_STRATEGY.__dict__)

        elif info.gpu_memory_mb >= 8192:
            # 高显存 GPU (>= 8GB)
            self._current_level = ComputeLevel.HIGH
            self._current_strategy = StrategyConfig(**HIGH_COMPUTE_STRATEGY.__dict__)

        else:
            # 中等 GPU (2-8GB)
            self._current_level = ComputeLevel.MEDIUM
            self._current_strategy = StrategyConfig(**MEDIUM_COMPUTE_STRATEGY.__dict__)

        logger.info(f"自动选择策略: {self._current_level.value}")

    @property
    def current_level(self) -> ComputeLevel:
        """当前算力级别"""
        return self._current_level

    @property
    def strategy(self) -> StrategyConfig:
        """当前策略配置"""
        return self._current_strategy

    @property
    def system_info(self) -> SystemInfo:
        """系统信息"""
        return self._system_info

    def set_level(self, level: ComputeLevel):
        """手动设置算力级别

        Args:
            level: 算力级别
        """
        self._current_level = level
        self._select_strategy()
        # 应用自定义覆盖
        self._apply_overrides()

    def override(self, **kwargs):
        """覆盖特定配置项

        Args:
            **kwargs: 要覆盖的配置项
        """
        for key, value in kwargs.items():
            if hasattr(self._current_strategy, key):
                setattr(self._current_strategy, key, value)
                self._custom_overrides[key] = value
                logger.debug(f"覆盖配置: {key}={value}")
            else:
                logger.warning(f"未知配置项: {key}")

    def _apply_overrides(self):
        """应用自定义覆盖"""
        for key, value in self._custom_overrides.items():
            if hasattr(self._current_strategy, key):
                setattr(self._current_strategy, key, value)

    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            'level': self._current_level.value,
            'sample_fps': self._current_strategy.sample_fps,
            'use_dynamic_sampling': self._current_strategy.use_dynamic_sampling,
            'use_multi_scale_hash': self._current_strategy.use_multi_scale_hash,
            'use_sequence_alignment': self._current_strategy.use_sequence_alignment,
            'phash_threshold': self._current_strategy.phash_threshold,
            'use_distortion_normalize': self._current_strategy.use_distortion_normalize,
            'use_optical_flow': self._current_strategy.use_optical_flow,
            'use_gpu': self._current_strategy.use_gpu,
            'gpu_batch_size': self._current_strategy.gpu_batch_size,
        }

    def should_use_feature(self, feature_name: str) -> bool:
        """检查是否应该使用某个功能

        Args:
            feature_name: 功能名称

        Returns:
            是否启用该功能
        """
        feature_map = {
            'dynamic_sampling': self._current_strategy.use_dynamic_sampling,
            'multi_scale_hash': self._current_strategy.use_multi_scale_hash,
            'sequence_alignment': self._current_strategy.use_sequence_alignment,
            'distortion_normalize': self._current_strategy.use_distortion_normalize,
            'prefilter': self._current_strategy.use_prefilter,
            'optical_flow': self._current_strategy.use_optical_flow,
            'confidence_verify': self._current_strategy.use_confidence_verify,
            'gpu': self._current_strategy.use_gpu,
        }
        return feature_map.get(feature_name, False)


# 全局策略管理器（延迟初始化）
_global_strategy_manager: Optional[AdaptiveStrategyManager] = None


def get_strategy_manager() -> AdaptiveStrategyManager:
    """获取全局策略管理器"""
    global _global_strategy_manager
    if _global_strategy_manager is None:
        _global_strategy_manager = AdaptiveStrategyManager()
    return _global_strategy_manager


def get_current_strategy() -> StrategyConfig:
    """获取当前策略配置"""
    return get_strategy_manager().strategy


def set_compute_level(level: ComputeLevel):
    """设置算力级别"""
    get_strategy_manager().set_level(level)


# 便捷函数
def is_feature_enabled(feature_name: str) -> bool:
    """检查功能是否启用"""
    return get_strategy_manager().should_use_feature(feature_name)
