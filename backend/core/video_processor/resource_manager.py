"""资源管理器

管理视频处理任务的资源分配：
1. 显存池管理
2. 任务队列调度
3. 超时控制
4. 资源申请/释放
5. 长视频分片处理
"""
import asyncio
import threading
import time
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
from contextlib import contextmanager
from loguru import logger


class TaskPriority(Enum):
    """任务优先级"""
    HIGH = 0      # 高优先级（短视频、实时处理）
    NORMAL = 1    # 普通优先级
    LOW = 2       # 低优先级（长视频、批处理）


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    priority: TaskPriority
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    timeout_sec: float = 300.0  # 默认5分钟超时
    estimated_memory_mb: int = 512


@dataclass
class ResourceConfig:
    """资源配置"""
    max_gpu_memory_mb: int = 4096       # 最大 GPU 显存
    max_cpu_memory_mb: int = 8192       # 最大 CPU 内存
    max_concurrent_tasks: int = 4       # 最大并发任务数
    task_timeout_sec: float = 600.0     # 任务超时时间（秒）
    chunk_size_frames: int = 1000       # 长视频分片大小（帧）
    chunk_overlap_frames: int = 50      # 分片重叠帧数
    memory_reserve_ratio: float = 0.1   # 预留显存比例


class MemoryPool:
    """显存池

    管理 GPU 显存的分配和释放，避免频繁的显存申请
    """

    def __init__(self, max_memory_mb: int = 4096):
        """初始化

        Args:
            max_memory_mb: 最大显存（MB）
        """
        self.max_memory_mb = max_memory_mb
        self._allocated_mb = 0
        self._lock = threading.Lock()
        self._allocations: Dict[str, int] = {}  # task_id -> allocated_mb

    def allocate(self, task_id: str, size_mb: int) -> bool:
        """申请显存

        Args:
            task_id: 任务 ID
            size_mb: 申请大小（MB）

        Returns:
            是否成功
        """
        with self._lock:
            if self._allocated_mb + size_mb > self.max_memory_mb:
                logger.warning(
                    f"显存不足: 请求 {size_mb}MB, "
                    f"已用 {self._allocated_mb}/{self.max_memory_mb}MB"
                )
                return False

            self._allocated_mb += size_mb
            self._allocations[task_id] = self._allocations.get(task_id, 0) + size_mb

            logger.debug(
                f"显存分配: {task_id} +{size_mb}MB, "
                f"当前 {self._allocated_mb}/{self.max_memory_mb}MB"
            )
            return True

    def release(self, task_id: str, size_mb: int = None):
        """释放显存

        Args:
            task_id: 任务 ID
            size_mb: 释放大小，None 表示释放该任务的全部
        """
        with self._lock:
            if task_id not in self._allocations:
                return

            if size_mb is None:
                size_mb = self._allocations[task_id]

            release_size = min(size_mb, self._allocations[task_id])
            self._allocations[task_id] -= release_size
            self._allocated_mb -= release_size

            if self._allocations[task_id] <= 0:
                del self._allocations[task_id]

            logger.debug(
                f"显存释放: {task_id} -{release_size}MB, "
                f"当前 {self._allocated_mb}/{self.max_memory_mb}MB"
            )

    def get_available(self) -> int:
        """获取可用显存（MB）"""
        with self._lock:
            return self.max_memory_mb - self._allocated_mb

    def get_usage(self) -> Dict[str, Any]:
        """获取使用情况"""
        with self._lock:
            return {
                'max_mb': self.max_memory_mb,
                'allocated_mb': self._allocated_mb,
                'available_mb': self.max_memory_mb - self._allocated_mb,
                'usage_ratio': self._allocated_mb / self.max_memory_mb,
                'allocations': dict(self._allocations)
            }


class TaskQueue:
    """任务队列

    优先级队列，支持任务调度和状态管理
    """

    def __init__(self, max_size: int = 100):
        """初始化

        Args:
            max_size: 最大队列长度
        """
        self.max_size = max_size
        self._queue = PriorityQueue()
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def submit(self, task: TaskInfo) -> bool:
        """提交任务

        Args:
            task: 任务信息

        Returns:
            是否成功
        """
        with self._lock:
            if len(self._tasks) >= self.max_size:
                logger.warning(f"任务队列已满: {len(self._tasks)}/{self.max_size}")
                return False

            self._tasks[task.task_id] = task
            # 优先级队列项: (priority, created_at, task_id)
            self._queue.put((
                task.priority.value,
                task.created_at,
                task.task_id
            ))

            logger.debug(f"任务提交: {task.task_id}, 优先级={task.priority.value}")
            return True

    def get_next(self) -> Optional[TaskInfo]:
        """获取下一个待处理任务

        Returns:
            任务信息或 None
        """
        try:
            while not self._queue.empty():
                _, _, task_id = self._queue.get_nowait()
                with self._lock:
                    if task_id in self._tasks:
                        task = self._tasks[task_id]
                        if task.status == TaskStatus.PENDING:
                            return task
            return None
        except Exception:
            return None

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float = None,
        result: Any = None,
        error: str = None
    ):
        """更新任务状态"""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.status = status

            if progress is not None:
                task.progress = progress
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

            if status == TaskStatus.RUNNING:
                task.started_at = time.time()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT):
                task.completed_at = time.time()

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        with self._lock:
            return self._tasks.get(task_id)

    def remove_task(self, task_id: str):
        """移除任务"""
        with self._lock:
            self._tasks.pop(task_id, None)

    def get_stats(self) -> Dict[str, int]:
        """获取队列统计"""
        with self._lock:
            stats = {s.value: 0 for s in TaskStatus}
            for task in self._tasks.values():
                stats[task.status.value] += 1
            stats['total'] = len(self._tasks)
            return stats


class ResourceManager:
    """资源管理器

    统一管理任务调度、资源分配、超时控制
    """

    def __init__(self, config: ResourceConfig = None):
        """初始化

        Args:
            config: 资源配置
        """
        self.config = config or ResourceConfig()
        self._memory_pool = MemoryPool(self.config.max_gpu_memory_mb)
        self._task_queue = TaskQueue()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
        self._lock = threading.Lock()

        # 启动超时检查
        self._timeout_checker = None

    async def start(self):
        """启动资源管理器"""
        self._shutdown = False
        self._timeout_checker = asyncio.create_task(self._check_timeouts())
        logger.info("资源管理器已启动")

    async def stop(self):
        """停止资源管理器"""
        self._shutdown = True
        if self._timeout_checker:
            self._timeout_checker.cancel()

        # 取消所有运行中的任务
        for task_id, task in list(self._running_tasks.items()):
            task.cancel()
            self._task_queue.update_status(task_id, TaskStatus.CANCELLED)

        logger.info("资源管理器已停止")

    async def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_sec: float = None,
        estimated_memory_mb: int = 512,
        **kwargs
    ) -> bool:
        """提交任务

        Args:
            task_id: 任务 ID
            func: 任务函数（异步）
            priority: 优先级
            timeout_sec: 超时时间
            estimated_memory_mb: 预估内存使用

        Returns:
            是否成功提交
        """
        task_info = TaskInfo(
            task_id=task_id,
            priority=priority,
            created_at=time.time(),
            timeout_sec=timeout_sec or self.config.task_timeout_sec,
            estimated_memory_mb=estimated_memory_mb
        )

        if not self._task_queue.submit(task_info):
            return False

        # 尝试立即执行
        await self._try_execute_next()
        return True

    async def _try_execute_next(self):
        """尝试执行下一个任务"""
        with self._lock:
            if len(self._running_tasks) >= self.config.max_concurrent_tasks:
                return

        task_info = self._task_queue.get_next()
        if task_info is None:
            return

        # 检查资源
        if not self._memory_pool.allocate(
            task_info.task_id,
            task_info.estimated_memory_mb
        ):
            # 资源不足，放回队列
            task_info.status = TaskStatus.PENDING
            return

        # 开始执行
        self._task_queue.update_status(task_info.task_id, TaskStatus.RUNNING)

        # 这里需要实际的任务函数，简化处理
        logger.info(f"任务开始执行: {task_info.task_id}")

    async def _check_timeouts(self):
        """检查超时任务"""
        while not self._shutdown:
            await asyncio.sleep(5)  # 每5秒检查一次

            current_time = time.time()

            for task_id, task in list(self._running_tasks.items()):
                task_info = self._task_queue.get_task(task_id)
                if task_info and task_info.started_at:
                    elapsed = current_time - task_info.started_at
                    if elapsed > task_info.timeout_sec:
                        logger.warning(f"任务超时: {task_id}, 已运行 {elapsed:.1f}s")
                        task.cancel()
                        self._task_queue.update_status(
                            task_id,
                            TaskStatus.TIMEOUT,
                            error=f"任务超时（{task_info.timeout_sec}s）"
                        )
                        self._memory_pool.release(task_id)

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        return self._task_queue.get_task(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        return {
            'memory': self._memory_pool.get_usage(),
            'tasks': self._task_queue.get_stats(),
            'running_count': len(self._running_tasks)
        }

    @contextmanager
    def allocate_memory(self, task_id: str, size_mb: int):
        """显存分配上下文管理器

        使用示例:
            with resource_manager.allocate_memory("task1", 512) as allocated:
                if allocated:
                    # 执行任务
                    pass
        """
        success = self._memory_pool.allocate(task_id, size_mb)
        try:
            yield success
        finally:
            if success:
                self._memory_pool.release(task_id, size_mb)


class ChunkProcessor:
    """分片处理器

    用于处理长视频的分片和合并
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 50
    ):
        """初始化

        Args:
            chunk_size: 分片大小（帧）
            overlap: 重叠帧数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_chunks(self, total_frames: int) -> list:
        """获取分片信息

        Args:
            total_frames: 总帧数

        Returns:
            [(start_frame, end_frame), ...] 列表
        """
        chunks = []
        start = 0

        while start < total_frames:
            end = min(start + self.chunk_size, total_frames)
            chunks.append((start, end))

            if end >= total_frames:
                break

            # 下一个分片的起始位置（考虑重叠）
            start = end - self.overlap

        return chunks

    def merge_results(
        self,
        chunk_results: list,
        chunks: list
    ) -> list:
        """合并分片结果

        Args:
            chunk_results: 各分片的结果列表
            chunks: 分片信息

        Returns:
            合并后的结果
        """
        if not chunk_results:
            return []

        if len(chunk_results) == 1:
            return chunk_results[0]

        merged = []

        for i, (result, (start, end)) in enumerate(zip(chunk_results, chunks)):
            if i == 0:
                # 第一个分片：保留全部，除了最后的重叠区域
                cutoff = end - self.overlap // 2
                merged.extend([r for r in result if r.get('frame', 0) < cutoff])
            elif i == len(chunk_results) - 1:
                # 最后一个分片：从重叠区域中点开始
                cutoff = start + self.overlap // 2
                merged.extend([r for r in result if r.get('frame', 0) >= cutoff])
            else:
                # 中间分片：两端都裁剪
                left_cutoff = start + self.overlap // 2
                right_cutoff = end - self.overlap // 2
                merged.extend([
                    r for r in result
                    if left_cutoff <= r.get('frame', 0) < right_cutoff
                ])

        return merged


# 全局资源管理器
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """获取全局资源管理器"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager
