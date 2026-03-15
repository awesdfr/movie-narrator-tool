"""并行帧匹配调度器

实现高效的并行匹配：
1. 多片段并行匹配（线程池/进程池）
2. 异步视频读取（预加载）
3. 流水线处理（读取→预处理→匹配重叠执行）
4. 批量结果聚合

在保持匹配精度的前提下提升处理速度。
"""
import asyncio
import concurrent.futures
import threading
import queue
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from loguru import logger


@dataclass
class MatchTask:
    """匹配任务"""
    segment_id: str
    narration_path: str
    start_time: float
    end_time: float
    priority: int = 0  # 越小优先级越高


@dataclass
class MatchResult:
    """匹配结果"""
    segment_id: str
    success: bool
    movie_start: Optional[float] = None
    movie_end: Optional[float] = None
    confidence: float = 0.0
    confidence_level: str = 'low'
    elapsed_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ParallelConfig:
    """并行配置"""
    max_workers: int = 4              # 最大并行工作线程
    prefetch_segments: int = 2        # 预加载片段数
    batch_size: int = 8               # 批处理大小
    use_process_pool: bool = False    # 使用进程池（CPU密集型任务）
    timeout_per_segment: float = 60.0 # 单片段超时（秒）


class FramePrefetcher:
    """帧预加载器

    异步预加载视频帧，减少 I/O 等待时间
    """

    def __init__(self, max_cache_frames: int = 500):
        """初始化

        Args:
            max_cache_frames: 最大缓存帧数
        """
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._max_frames = max_cache_frames
        self._lock = threading.Lock()
        self._prefetch_queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """启动预加载器"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """停止预加载器"""
        self._running = False
        if self._worker_thread:
            self._prefetch_queue.put(None)  # 发送停止信号
            self._worker_thread.join(timeout=2.0)

    def request_prefetch(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_fps: float = 4.0
    ):
        """请求预加载

        Args:
            video_path: 视频路径
            start_time: 起始时间
            end_time: 结束时间
            sample_fps: 采样帧率
        """
        self._prefetch_queue.put({
            'video_path': video_path,
            'start_time': start_time,
            'end_time': end_time,
            'sample_fps': sample_fps
        })

    def get_frame(self, video_path: str, time_sec: float) -> Optional[np.ndarray]:
        """获取缓存的帧

        Args:
            video_path: 视频路径
            time_sec: 时间戳

        Returns:
            帧数据或 None
        """
        key = f"{video_path}:{time_sec:.3f}"
        with self._lock:
            return self._cache.get(key)

    def get_frames(
        self,
        video_path: str,
        times: List[float]
    ) -> List[Optional[np.ndarray]]:
        """批量获取帧"""
        return [self.get_frame(video_path, t) for t in times]

    def _prefetch_worker(self):
        """预加载工作线程"""
        while self._running:
            try:
                task = self._prefetch_queue.get(timeout=1.0)
                if task is None:
                    break

                self._do_prefetch(
                    task['video_path'],
                    task['start_time'],
                    task['end_time'],
                    task['sample_fps']
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"预加载失败: {e}")

    def _do_prefetch(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_fps: float
    ):
        """执行预加载"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = 1.0 / sample_fps
        t = start_time

        while t < end_time:
            target_frame = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if ret:
                key = f"{video_path}:{t:.3f}"
                self._add_to_cache(key, frame)

            t += sample_interval

        cap.release()

    def _add_to_cache(self, key: str, frame: np.ndarray):
        """添加到缓存"""
        with self._lock:
            if key in self._cache:
                return

            # LRU 淘汰
            while len(self._cache) >= self._max_frames:
                if self._cache_order:
                    old_key = self._cache_order.pop(0)
                    self._cache.pop(old_key, None)
                else:
                    break

            self._cache[key] = frame
            self._cache_order.append(key)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()


class ParallelMatcher:
    """并行帧匹配器

    支持多片段并行匹配，最大化利用 CPU/GPU 资源
    """

    def __init__(
        self,
        frame_matcher,  # FrameMatcher 实例
        config: ParallelConfig = None
    ):
        """初始化

        Args:
            frame_matcher: FrameMatcher 实例
            config: 并行配置
        """
        self._matcher = frame_matcher
        self.config = config or ParallelConfig()
        self._prefetcher = FramePrefetcher()
        self._executor: Optional[concurrent.futures.Executor] = None
        self._results: Dict[str, MatchResult] = {}
        self._lock = threading.Lock()

    async def match_segments_parallel(
        self,
        tasks: List[MatchTask],
        progress_callback: Callable[[int, int, str], None] = None
    ) -> List[MatchResult]:
        """并行匹配多个片段

        Args:
            tasks: 匹配任务列表
            progress_callback: 进度回调 (completed, total, segment_id)

        Returns:
            匹配结果列表（与 tasks 顺序一致）
        """
        if not tasks:
            return []

        total = len(tasks)
        completed = [0]
        results: Dict[str, MatchResult] = {}

        # 启动预加载器
        self._prefetcher.start()

        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        # 预加载前几个片段
        for task in sorted_tasks[:self.config.prefetch_segments]:
            self._prefetcher.request_prefetch(
                task.narration_path,
                task.start_time,
                task.end_time
            )

        # 创建执行器
        if self.config.use_process_pool:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )

        try:
            # 创建异步任务
            async def match_single(task: MatchTask) -> MatchResult:
                import time
                start_time_ms = time.time() * 1000

                try:
                    # 预加载下一个片段
                    task_idx = sorted_tasks.index(task)
                    next_idx = task_idx + self.config.prefetch_segments
                    if next_idx < len(sorted_tasks):
                        next_task = sorted_tasks[next_idx]
                        self._prefetcher.request_prefetch(
                            next_task.narration_path,
                            next_task.start_time,
                            next_task.end_time
                        )

                    # 执行匹配（先尝试严格匹配，失败后回退到宽松匹配）
                    result = await asyncio.wait_for(
                        self._matcher.match_segment(
                            task.narration_path,
                            task.start_time,
                            task.end_time,
                            time_hint=None,
                            relaxed=False,
                            strict_window=False,
                        ),
                        timeout=self.config.timeout_per_segment
                    )

                    # 如果严格匹配失败，回退到宽松匹配
                    if result is None:
                        result = await asyncio.wait_for(
                            self._matcher.match_segment(
                                task.narration_path,
                                task.start_time,
                                task.end_time,
                                time_hint=None,
                                relaxed=True,
                                strict_window=False,
                            ),
                            timeout=self.config.timeout_per_segment
                        )

                    elapsed = time.time() * 1000 - start_time_ms

                    if result:
                        return MatchResult(
                            segment_id=task.segment_id,
                            success=True,
                            movie_start=result['start'],
                            movie_end=result['end'],
                            confidence=result['confidence'],
                            confidence_level=result.get('confidence_level', 'medium'),
                            elapsed_ms=elapsed
                        )
                    else:
                        return MatchResult(
                            segment_id=task.segment_id,
                            success=False,
                            elapsed_ms=elapsed
                        )

                except asyncio.TimeoutError:
                    return MatchResult(
                        segment_id=task.segment_id,
                        success=False,
                        error="匹配超时"
                    )
                except Exception as e:
                    return MatchResult(
                        segment_id=task.segment_id,
                        success=False,
                        error=str(e)
                    )
                finally:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0], total, task.segment_id)

            # 并行执行（使用信号量控制并发数）
            semaphore = asyncio.Semaphore(self.config.max_workers)

            async def match_with_semaphore(task: MatchTask) -> MatchResult:
                async with semaphore:
                    return await match_single(task)

            # 启动所有任务
            match_tasks = [match_with_semaphore(task) for task in sorted_tasks]
            match_results = await asyncio.gather(*match_tasks)

            # 按原始顺序整理结果
            result_map = {r.segment_id: r for r in match_results}
            ordered_results = [result_map[task.segment_id] for task in tasks]

            return ordered_results

        finally:
            executor.shutdown(wait=False)
            self._prefetcher.stop()

    async def match_segments_batch(
        self,
        tasks: List[MatchTask],
        batch_size: int = None
    ) -> List[MatchResult]:
        """分批并行匹配

        将任务分成多个批次，每批并行处理

        Args:
            tasks: 匹配任务列表
            batch_size: 批大小

        Returns:
            匹配结果列表
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await self.match_segments_parallel(batch)
            results.extend(batch_results)

        return results


class PipelineProcessor:
    """流水线处理器

    实现读取→预处理→匹配的流水线，最大化吞吐量
    """

    def __init__(
        self,
        frame_matcher,
        num_stages: int = 3
    ):
        """初始化

        Args:
            frame_matcher: FrameMatcher 实例
            num_stages: 流水线阶段数
        """
        self._matcher = frame_matcher
        self._stages = []
        self._running = False

        # 阶段队列
        self._read_queue = queue.Queue(maxsize=10)
        self._preprocess_queue = queue.Queue(maxsize=10)
        self._match_queue = queue.Queue(maxsize=10)
        self._result_queue = queue.Queue()

    def start(self):
        """启动流水线"""
        self._running = True

        # 读取阶段
        self._stages.append(threading.Thread(
            target=self._read_stage, daemon=True
        ))

        # 预处理阶段
        self._stages.append(threading.Thread(
            target=self._preprocess_stage, daemon=True
        ))

        for t in self._stages:
            t.start()

    def stop(self):
        """停止流水线"""
        self._running = False
        # 发送停止信号
        self._read_queue.put(None)
        self._preprocess_queue.put(None)
        self._match_queue.put(None)

        for t in self._stages:
            t.join(timeout=2.0)

    def submit(self, task: MatchTask):
        """提交任务"""
        self._read_queue.put(task)

    def get_result(self, timeout: float = None) -> Optional[MatchResult]:
        """获取结果"""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _read_stage(self):
        """读取阶段：从视频读取帧"""
        while self._running:
            try:
                task = self._read_queue.get(timeout=1.0)
                if task is None:
                    break

                # 读取帧
                frames = self._read_frames(
                    task.narration_path,
                    task.start_time,
                    task.end_time
                )

                self._preprocess_queue.put({
                    'task': task,
                    'frames': frames
                })

            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"读取阶段错误: {e}")

    def _preprocess_stage(self):
        """预处理阶段：帧归一化、去重"""
        while self._running:
            try:
                item = self._preprocess_queue.get(timeout=1.0)
                if item is None:
                    break

                task = item['task']
                frames = item['frames']

                # 预处理
                processed = self._preprocess_frames(frames)

                self._match_queue.put({
                    'task': task,
                    'frames': processed
                })

            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"预处理阶段错误: {e}")

    def _read_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_fps: float = 4.0
    ) -> List[tuple]:
        """读取帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return frames

        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = 1.0 / sample_fps
        t = start_time

        while t < end_time:
            target_frame = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if ret:
                frames.append((frame, target_frame, t))

            t += sample_interval

        cap.release()
        return frames

    def _preprocess_frames(self, frames: List[tuple]) -> List[tuple]:
        """预处理帧"""
        # 这里可以调用 DistortionNormalizer
        return frames


# 便捷函数
async def parallel_match_segments(
    frame_matcher,
    segments: List[Dict],
    narration_path: str,
    max_workers: int = 4
) -> List[MatchResult]:
    """并行匹配片段的便捷函数

    Args:
        frame_matcher: FrameMatcher 实例
        segments: 片段列表 [{'id': str, 'start': float, 'end': float}, ...]
        narration_path: 解说视频路径
        max_workers: 最大并行数

    Returns:
        匹配结果列表
    """
    tasks = [
        MatchTask(
            segment_id=seg['id'],
            narration_path=narration_path,
            start_time=seg['start'],
            end_time=seg['end'],
            priority=i
        )
        for i, seg in enumerate(segments)
    ]

    config = ParallelConfig(max_workers=max_workers)
    parallel_matcher = ParallelMatcher(frame_matcher, config)

    return await parallel_matcher.match_segments_parallel(tasks)
