"""重复/循环帧过滤模块

检测并过滤：
1. 重复帧：连续帧哈希相似度 >= 0.98
2. 循环帧：周期性重复的帧序列

保留首帧，压缩冗余帧，减少后续计算量。
"""
import cv2
import numpy as np
from PIL import Image
import imagehash
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FilteredFrame:
    """过滤后的帧信息"""
    frame: np.ndarray       # 帧图像 (BGR)
    frame_idx: int          # 原始帧号
    time_sec: float         # 时间戳（秒）
    phash: str              # pHash 值
    is_duplicate: bool      # 是否为重复帧
    is_cycle: bool          # 是否为循环帧
    group_id: int           # 所属分组 ID（重复帧属于同一组）


@dataclass
class FilterConfig:
    """过滤配置"""
    duplicate_threshold: float = 0.98     # 重复帧相似度阈值
    cycle_window_size: int = 30           # 循环检测滑动窗口大小
    cycle_min_period: int = 3             # 最小循环周期（帧数）
    cycle_max_period: int = 15            # 最大循环周期（帧数）
    cycle_similarity_threshold: float = 0.95  # 循环匹配相似度阈值


class CycleFrameFilter:
    """重复/循环帧过滤器

    核心功能：
    1. 检测连续重复帧（静态画面），仅保留首帧
    2. 检测循环帧序列（如加载动画、循环视频），压缩为单个序列
    3. 维护帧分组信息，支持后续处理时还原
    """

    def __init__(self, config: FilterConfig = None):
        """初始化

        Args:
            config: 过滤配置
        """
        self.config = config or FilterConfig()
        self._hash_cache = {}  # 缓存已计算的哈希
        self._group_counter = 0

    def filter_frames(
        self,
        frames: list[tuple[np.ndarray, int, float]]
    ) -> list[FilteredFrame]:
        """过滤帧序列

        Args:
            frames: [(frame, frame_idx, time_sec), ...] 帧列表

        Returns:
            FilteredFrame 列表（已去除重复和循环帧）
        """
        if not frames:
            return []

        logger.debug(f"开始过滤 {len(frames)} 帧")

        # 第一遍：计算所有帧的 pHash
        hashes = []
        for frame, frame_idx, time_sec in frames:
            phash = self._compute_phash(frame)
            hashes.append((frame, frame_idx, time_sec, phash))

        # 第二遍：检测重复帧
        filtered = []
        duplicate_groups = {}  # phash -> group_id
        self._group_counter = 0

        i = 0
        while i < len(hashes):
            frame, frame_idx, time_sec, phash = hashes[i]

            # 检查是否与前一帧重复
            is_duplicate = False
            group_id = -1

            if i > 0:
                prev_phash = hashes[i - 1][3]
                similarity = self._compute_similarity(phash, prev_phash)

                if similarity >= self.config.duplicate_threshold:
                    is_duplicate = True
                    # 继承前一帧的分组
                    if prev_phash in duplicate_groups:
                        group_id = duplicate_groups[prev_phash]
                    else:
                        group_id = self._group_counter

            if not is_duplicate:
                # 新的非重复帧，创建新分组
                self._group_counter += 1
                group_id = self._group_counter
                duplicate_groups[phash] = group_id

                filtered.append(FilteredFrame(
                    frame=frame,
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    phash=phash,
                    is_duplicate=False,
                    is_cycle=False,
                    group_id=group_id
                ))
            else:
                # 重复帧，记录但不添加到输出
                duplicate_groups[phash] = group_id

            i += 1

        # 第三遍：检测循环帧序列
        cycle_frames = self._detect_cycles(filtered)

        # 标记循环帧
        for idx in cycle_frames:
            if idx < len(filtered):
                filtered[idx].is_cycle = True

        # 过滤掉循环帧（可选，这里保留但标记）
        # final_filtered = [f for f in filtered if not f.is_cycle]

        duplicate_count = len(hashes) - len(filtered)
        cycle_count = len(cycle_frames)
        logger.debug(
            f"帧过滤完成: 原始 {len(hashes)} 帧, "
            f"去除重复 {duplicate_count} 帧, "
            f"检测循环 {cycle_count} 帧, "
            f"保留 {len(filtered)} 帧"
        )

        return filtered

    def _compute_phash(self, frame: np.ndarray) -> str:
        """计算帧的 pHash"""
        # 转换为 PIL 图像
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # 计算 pHash
        phash = imagehash.phash(pil_image)
        return str(phash)

    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        """计算两个哈希的相似度

        Args:
            hash1: 第一个哈希
            hash2: 第二个哈希

        Returns:
            相似度 0-1
        """
        # 计算汉明距离
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        distance = h1 - h2

        # 转换为相似度（64 位哈希）
        similarity = 1.0 - distance / 64.0
        return similarity

    def _detect_cycles(self, frames: list[FilteredFrame]) -> list[int]:
        """检测循环帧序列

        使用滑动窗口检测周期性重复的帧模式

        Args:
            frames: 过滤后的帧列表

        Returns:
            循环帧的索引列表
        """
        if len(frames) < self.config.cycle_window_size:
            return []

        cycle_indices = set()

        # 尝试不同的周期长度
        for period in range(self.config.cycle_min_period, self.config.cycle_max_period + 1):
            # 滑动窗口检测
            for start in range(len(frames) - period * 2):
                # 比较两个连续的周期
                pattern1 = [frames[start + i].phash for i in range(period)]
                pattern2 = [frames[start + period + i].phash for i in range(period)]

                # 计算模式相似度
                match_count = 0
                for h1, h2 in zip(pattern1, pattern2):
                    if self._compute_similarity(h1, h2) >= self.config.cycle_similarity_threshold:
                        match_count += 1

                similarity = match_count / period

                # 如果两个周期高度相似，标记为循环
                if similarity >= self.config.cycle_similarity_threshold:
                    # 标记第二个周期的帧为循环帧
                    for i in range(period):
                        cycle_indices.add(start + period + i)

        return list(cycle_indices)

    def get_unique_frames(
        self,
        frames: list[tuple[np.ndarray, int, float]]
    ) -> list[tuple[np.ndarray, int, float]]:
        """获取唯一帧（去除重复和循环）

        Args:
            frames: [(frame, frame_idx, time_sec), ...] 帧列表

        Returns:
            去重后的帧列表
        """
        filtered = self.filter_frames(frames)
        return [
            (f.frame, f.frame_idx, f.time_sec)
            for f in filtered
            if not f.is_cycle
        ]

    def compute_frame_hash_batch(
        self,
        frames: list[np.ndarray]
    ) -> list[str]:
        """批量计算帧哈希

        Args:
            frames: 帧图像列表 (BGR)

        Returns:
            哈希字符串列表
        """
        return [self._compute_phash(frame) for frame in frames]


class DuplicateDetector:
    """简单的重复帧检测器（轻量版）

    仅检测连续重复帧，不检测循环序列。
    适用于实时流处理场景。
    """

    def __init__(self, threshold: float = 0.98):
        """初始化

        Args:
            threshold: 重复判定阈值
        """
        self.threshold = threshold
        self._prev_hash = None

    def is_duplicate(self, frame: np.ndarray) -> bool:
        """判断当前帧是否与上一帧重复

        Args:
            frame: BGR 格式帧

        Returns:
            True 表示重复
        """
        # 计算当前帧哈希
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        current_hash = imagehash.phash(pil_image)

        if self._prev_hash is None:
            self._prev_hash = current_hash
            return False

        # 计算距离
        distance = current_hash - self._prev_hash
        similarity = 1.0 - distance / 64.0

        # 更新状态
        self._prev_hash = current_hash

        return similarity >= self.threshold

    def reset(self):
        """重置状态"""
        self._prev_hash = None


# 便捷函数
def filter_duplicate_frames(
    frames: list[tuple[np.ndarray, int, float]],
    threshold: float = 0.98
) -> list[tuple[np.ndarray, int, float]]:
    """过滤重复帧的便捷函数

    Args:
        frames: [(frame, frame_idx, time_sec), ...] 帧列表
        threshold: 重复判定阈值

    Returns:
        去重后的帧列表
    """
    config = FilterConfig(duplicate_threshold=threshold)
    filter_obj = CycleFrameFilter(config)
    return filter_obj.get_unique_frames(frames)


def detect_duplicate_sequence(
    hashes: list[str],
    threshold: float = 0.98
) -> list[tuple[int, int]]:
    """检测重复序列的便捷函数

    Args:
        hashes: 哈希字符串列表
        threshold: 相似度阈值

    Returns:
        [(start_idx, end_idx), ...] 重复序列的起止索引
    """
    if not hashes:
        return []

    sequences = []
    start = 0

    for i in range(1, len(hashes)):
        h1 = imagehash.hex_to_hash(hashes[i - 1])
        h2 = imagehash.hex_to_hash(hashes[i])
        similarity = 1.0 - (h1 - h2) / 64.0

        if similarity < threshold:
            if i - start > 1:
                sequences.append((start, i - 1))
            start = i

    # 处理最后一个序列
    if len(hashes) - start > 1:
        sequences.append((start, len(hashes) - 1))

    return sequences
