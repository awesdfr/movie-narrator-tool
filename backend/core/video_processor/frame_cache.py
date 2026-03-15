"""帧缓存管理器

提供多级缓存机制加速帧读取：
1. L1 内存缓存 - LRU 淘汰策略
2. L2 磁盘缓存 - 持久化存储
3. 预加载机制 - 提前加载即将访问的帧
4. 批量读取优化 - 减少磁盘 seek 次数

核心目标：减少重复 I/O，加速帧匹配过程。
"""
import os
import cv2
import numpy as np
import hashlib
import pickle
import threading
import time
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from loguru import logger


@dataclass
class CacheConfig:
    """缓存配置"""
    # L1 内存缓存
    l1_max_frames: int = 1000         # 最大缓存帧数
    l1_max_memory_mb: int = 512       # 最大内存占用

    # L2 磁盘缓存
    l2_enabled: bool = True           # 启用磁盘缓存
    l2_cache_dir: str = ".frame_cache"  # 缓存目录
    l2_max_size_gb: float = 5.0       # 最大磁盘占用

    # 预加载
    prefetch_enabled: bool = True     # 启用预加载
    prefetch_window: int = 20         # 预加载窗口大小

    # 过期策略
    ttl_seconds: float = 3600.0       # 缓存过期时间


@dataclass
class CacheStats:
    """缓存统计"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    prefetch_hits: int = 0
    evictions: int = 0
    total_memory_mb: float = 0.0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0


class LRUCache:
    """LRU 内存缓存

    使用 OrderedDict 实现 O(1) 的插入、删除、查找
    """

    def __init__(self, max_items: int = 1000, max_memory_mb: int = 512):
        self._cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()
        self._max_items = max_items
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存项"""
        with self._lock:
            if key not in self._cache:
                return None

            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            return self._cache[key][0]

    def put(self, key: str, value: np.ndarray):
        """添加缓存项"""
        with self._lock:
            if key in self._cache:
                # 更新现有项
                old_frame, _ = self._cache[key]
                self._current_memory -= old_frame.nbytes
                self._cache[key] = (value, time.time())
                self._current_memory += value.nbytes
                self._cache.move_to_end(key)
                return

            # 检查是否需要淘汰
            value_size = value.nbytes
            while (
                len(self._cache) >= self._max_items or
                self._current_memory + value_size > self._max_memory_bytes
            ):
                if not self._cache:
                    break
                # 淘汰最旧的项（队首）
                _, (old_frame, _) = self._cache.popitem(last=False)
                self._current_memory -= old_frame.nbytes

            # 添加新项
            self._cache[key] = (value, time.time())
            self._current_memory += value_size

    def remove(self, key: str):
        """移除缓存项"""
        with self._lock:
            if key in self._cache:
                frame, _ = self._cache.pop(key)
                self._current_memory -= frame.nbytes

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def contains(self, key: str) -> bool:
        """检查是否存在"""
        return key in self._cache

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def memory_mb(self) -> float:
        return self._current_memory / (1024 * 1024)


class DiskCache:
    """磁盘缓存

    将帧数据序列化存储到磁盘，实现持久化缓存
    """

    def __init__(self, cache_dir: str, max_size_gb: float = 5.0):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._current_size = 0
        self._index: Dict[str, Tuple[str, int, float]] = {}  # key -> (path, size, time)
        self._lock = threading.Lock()
        self._load_index()

    def _get_index_path(self) -> Path:
        return self._cache_dir / "cache_index.pkl"

    def _load_index(self):
        """加载索引"""
        index_path = self._get_index_path()
        if index_path.exists():
            try:
                with open(index_path, 'rb') as f:
                    self._index = pickle.load(f)
                # 计算当前大小
                self._current_size = sum(
                    size for _, size, _ in self._index.values()
                )
            except Exception as e:
                logger.warning(f"加载磁盘缓存索引失败: {e}")
                self._index = {}

    def _save_index(self):
        """保存索引"""
        try:
            with open(self._get_index_path(), 'wb') as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logger.warning(f"保存磁盘缓存索引失败: {e}")

    def _key_to_filename(self, key: str) -> str:
        """将 key 转换为文件名"""
        hash_val = hashlib.md5(key.encode()).hexdigest()
        return f"{hash_val}.npy"

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存项"""
        with self._lock:
            if key not in self._index:
                return None

            path, size, _ = self._index[key]
            file_path = self._cache_dir / path

            if not file_path.exists():
                del self._index[key]
                self._current_size -= size
                return None

            try:
                frame = np.load(file_path)
                # 更新访问时间
                self._index[key] = (path, size, time.time())
                return frame
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {e}")
                return None

    def put(self, key: str, frame: np.ndarray):
        """添加缓存项"""
        with self._lock:
            filename = self._key_to_filename(key)
            file_path = self._cache_dir / filename
            frame_size = frame.nbytes

            # 检查是否需要淘汰
            while self._current_size + frame_size > self._max_size_bytes:
                if not self._index:
                    break
                # 淘汰最旧的项
                oldest_key = min(
                    self._index.keys(),
                    key=lambda k: self._index[k][2]
                )
                self._evict(oldest_key)

            # 保存帧
            try:
                np.save(file_path, frame)
                self._index[key] = (filename, frame_size, time.time())
                self._current_size += frame_size
            except Exception as e:
                logger.warning(f"写入磁盘缓存失败: {e}")

    def _evict(self, key: str):
        """淘汰缓存项"""
        if key not in self._index:
            return

        path, size, _ = self._index.pop(key)
        file_path = self._cache_dir / path
        self._current_size -= size

        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    def contains(self, key: str) -> bool:
        return key in self._index

    def clear(self):
        """清空缓存"""
        with self._lock:
            for key in list(self._index.keys()):
                self._evict(key)
            self._save_index()

    @property
    def size_gb(self) -> float:
        return self._current_size / (1024 * 1024 * 1024)


class FrameCache:
    """帧缓存管理器

    多级缓存：L1 内存 + L2 磁盘
    """

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._l1_cache = LRUCache(
            max_items=self.config.l1_max_frames,
            max_memory_mb=self.config.l1_max_memory_mb
        )
        self._l2_cache = DiskCache(
            cache_dir=self.config.l2_cache_dir,
            max_size_gb=self.config.l2_max_size_gb
        ) if self.config.l2_enabled else None
        self._stats = CacheStats()
        self._lock = threading.Lock()

    def _make_key(self, video_path: str, frame_idx: int) -> str:
        """生成缓存键"""
        # 使用视频路径哈希 + 帧索引
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:16]
        return f"{video_hash}:{frame_idx}"

    def get(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """获取帧

        Args:
            video_path: 视频路径
            frame_idx: 帧索引

        Returns:
            帧数据或 None
        """
        key = self._make_key(video_path, frame_idx)

        # 尝试 L1 缓存
        frame = self._l1_cache.get(key)
        if frame is not None:
            self._stats.l1_hits += 1
            return frame

        self._stats.l1_misses += 1

        # 尝试 L2 缓存
        if self._l2_cache:
            frame = self._l2_cache.get(key)
            if frame is not None:
                self._stats.l2_hits += 1
                # 提升到 L1
                self._l1_cache.put(key, frame)
                return frame
            self._stats.l2_misses += 1

        return None

    def put(self, video_path: str, frame_idx: int, frame: np.ndarray):
        """存储帧"""
        key = self._make_key(video_path, frame_idx)

        # 存入 L1
        self._l1_cache.put(key, frame)

        # 异步存入 L2（避免阻塞）
        if self._l2_cache and not self._l2_cache.contains(key):
            # 简单实现，直接存储
            self._l2_cache.put(key, frame)

    def get_batch(
        self,
        video_path: str,
        frame_indices: List[int]
    ) -> Dict[int, Optional[np.ndarray]]:
        """批量获取帧"""
        result = {}
        for idx in frame_indices:
            result[idx] = self.get(video_path, idx)
        return result

    def put_batch(
        self,
        video_path: str,
        frames: Dict[int, np.ndarray]
    ):
        """批量存储帧"""
        for idx, frame in frames.items():
            self.put(video_path, idx, frame)

    def prefetch(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ):
        """预加载帧范围"""
        if not self.config.prefetch_enabled:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        try:
            for frame_idx in range(start_frame, min(end_frame, start_frame + self.config.prefetch_window)):
                key = self._make_key(video_path, frame_idx)
                if self._l1_cache.contains(key):
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    self._l1_cache.put(key, frame)
                    self._stats.prefetch_hits += 1
        finally:
            cap.release()

    def clear(self):
        """清空所有缓存"""
        self._l1_cache.clear()
        if self._l2_cache:
            self._l2_cache.clear()

    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        self._stats.total_memory_mb = self._l1_cache.memory_mb
        return self._stats

    def evict_video(self, video_path: str):
        """清除特定视频的缓存"""
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:16]
        # L1 缓存需要遍历清除
        keys_to_remove = [
            k for k in self._l1_cache._cache.keys()
            if k.startswith(video_hash)
        ]
        for key in keys_to_remove:
            self._l1_cache.remove(key)


class VideoFrameReader:
    """视频帧读取器

    集成缓存的视频帧读取，优化批量读取性能
    """

    def __init__(self, video_path: str, cache: FrameCache = None):
        self._video_path = video_path
        self._cache = cache or FrameCache()
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 0.0
        self._total_frames = 0
        self._width = 0
        self._height = 0
        self._lock = threading.Lock()

    def open(self):
        """打开视频"""
        if self._cap is not None:
            return

        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            raise IOError(f"无法打开视频: {self._video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self):
        """关闭视频"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """读取单帧"""
        # 尝试缓存
        frame = self._cache.get(self._video_path, frame_idx)
        if frame is not None:
            return frame

        # 从视频读取
        with self._lock:
            if self._cap is None:
                self.open()

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()

            if ret:
                self._cache.put(self._video_path, frame_idx, frame)
                return frame

        return None

    def read_frame_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        """按时间读取帧"""
        frame_idx = int(time_sec * self._fps)
        return self.read_frame(frame_idx)

    def read_frames_batch(
        self,
        frame_indices: List[int],
        use_sequential: bool = True
    ) -> Dict[int, np.ndarray]:
        """批量读取帧

        Args:
            frame_indices: 帧索引列表
            use_sequential: 使用顺序读取优化（减少 seek）

        Returns:
            {frame_idx: frame, ...}
        """
        result = {}

        # 先检查缓存
        cache_results = self._cache.get_batch(self._video_path, frame_indices)
        missing_indices = [
            idx for idx, frame in cache_results.items()
            if frame is None
        ]
        result.update({
            idx: frame for idx, frame in cache_results.items()
            if frame is not None
        })

        if not missing_indices:
            return result

        # 排序以优化顺序读取
        if use_sequential:
            missing_indices.sort()

        with self._lock:
            if self._cap is None:
                self.open()

            for frame_idx in missing_indices:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self._cap.read()
                if ret:
                    result[frame_idx] = frame
                    self._cache.put(self._video_path, frame_idx, frame)

        return result

    def read_range(
        self,
        start_frame: int,
        end_frame: int,
        step: int = 1
    ) -> List[Tuple[int, np.ndarray]]:
        """读取帧范围"""
        indices = list(range(start_frame, end_frame, step))
        frames = self.read_frames_batch(indices)
        return [(idx, frames[idx]) for idx in indices if idx in frames]

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0


# 全局缓存实例
_global_cache: Optional[FrameCache] = None


def get_frame_cache() -> FrameCache:
    """获取全局帧缓存"""
    global _global_cache
    if _global_cache is None:
        _global_cache = FrameCache()
    return _global_cache


def create_video_reader(video_path: str) -> VideoFrameReader:
    """创建视频帧读取器"""
    return VideoFrameReader(video_path, get_frame_cache())
