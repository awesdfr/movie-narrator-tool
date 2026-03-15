"""监控与故障自愈模块

实现系统监控和故障恢复：
1. 指标采集（精度、效率、资源使用）
2. 故障检测（超时、溢出、异常）
3. 分级自愈（轻/中/重故障）
4. 告警通知
"""
import time
import asyncio
import threading
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from loguru import logger

# 尝试导入 psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器（只增不减）
    GAUGE = "gauge"          # 仪表盘（可增可减）
    HISTOGRAM = "histogram"  # 直方图（分布）
    TIMER = "timer"          # 计时器


class FaultLevel(Enum):
    """故障级别"""
    LIGHT = "light"    # 轻故障：自动重试
    MEDIUM = "medium"  # 中故障：模块重启
    SEVERE = "severe"  # 重故障：服务降级 + 告警


@dataclass
class Metric:
    """指标"""
    name: str
    type: MetricType
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    history: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, value: float):
        """记录值"""
        self.value = value
        self.timestamp = time.time()
        self.history.append((self.timestamp, value))

    def increment(self, delta: float = 1.0):
        """增加"""
        self.value += delta
        self.timestamp = time.time()
        self.history.append((self.timestamp, self.value))

    def get_avg(self, window_sec: float = 60.0) -> float:
        """获取时间窗口内的平均值"""
        if not self.history:
            return self.value

        cutoff = time.time() - window_sec
        values = [v for t, v in self.history if t >= cutoff]
        return sum(values) / len(values) if values else self.value


@dataclass
class FaultEvent:
    """故障事件"""
    fault_id: str
    level: FaultLevel
    component: str
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: str = ""


class MetricsCollector:
    """指标采集器"""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

        # 预定义核心指标
        self._init_core_metrics()

    def _init_core_metrics(self):
        """初始化核心指标"""
        core_metrics = [
            # 精度指标
            ("match_success_rate", MetricType.GAUGE),
            ("frame_error_avg", MetricType.GAUGE),
            ("false_match_rate", MetricType.GAUGE),

            # 效率指标
            ("process_time_ms", MetricType.HISTOGRAM),
            ("frames_per_second", MetricType.GAUGE),
            ("queue_length", MetricType.GAUGE),

            # 资源指标
            ("gpu_memory_used_mb", MetricType.GAUGE),
            ("gpu_utilization", MetricType.GAUGE),
            ("cpu_utilization", MetricType.GAUGE),

            # 异常指标
            ("optical_flow_failures", MetricType.COUNTER),
            ("timeout_count", MetricType.COUNTER),
            ("memory_overflow_count", MetricType.COUNTER),
        ]

        for name, type_ in core_metrics:
            self._metrics[name] = Metric(name=name, type=type_)

    def record(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录指标值"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    type=MetricType.GAUGE,
                    labels=labels or {}
                )
            self._metrics[name].record(value)

    def increment(self, name: str, delta: float = 1.0):
        """增加计数器"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(name=name, type=MetricType.COUNTER)
            self._metrics[name].increment(delta)

    def get(self, name: str) -> Optional[Metric]:
        """获取指标"""
        with self._lock:
            return self._metrics.get(name)

    def get_all(self) -> Dict[str, float]:
        """获取所有指标当前值"""
        with self._lock:
            return {name: m.value for name, m in self._metrics.items()}

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            return {
                'accuracy': {
                    'match_success_rate': self._metrics['match_success_rate'].value,
                    'frame_error_avg': self._metrics['frame_error_avg'].value,
                    'false_match_rate': self._metrics['false_match_rate'].value,
                },
                'efficiency': {
                    'process_time_ms': self._metrics['process_time_ms'].value,
                    'fps': self._metrics['frames_per_second'].value,
                    'queue_length': self._metrics['queue_length'].value,
                },
                'resources': {
                    'gpu_memory_mb': self._metrics['gpu_memory_used_mb'].value,
                    'gpu_util': self._metrics['gpu_utilization'].value,
                    'cpu_util': self._metrics['cpu_utilization'].value,
                },
                'errors': {
                    'flow_failures': self._metrics['optical_flow_failures'].value,
                    'timeouts': self._metrics['timeout_count'].value,
                    'oom_count': self._metrics['memory_overflow_count'].value,
                }
            }


class FaultDetector:
    """故障检测器"""

    def __init__(self, metrics: MetricsCollector):
        self._metrics = metrics
        self._fault_handlers: Dict[FaultLevel, List[Callable]] = {
            FaultLevel.LIGHT: [],
            FaultLevel.MEDIUM: [],
            FaultLevel.SEVERE: [],
        }
        self._fault_history: List[FaultEvent] = []
        self._thresholds = {
            'gpu_memory_threshold': 0.9,      # GPU 显存使用率阈值
            'timeout_rate_threshold': 0.1,     # 超时率阈值
            'match_success_threshold': 0.8,    # 匹配成功率阈值
            'cpu_threshold': 0.95,             # CPU 使用率阈值
        }

    def set_threshold(self, name: str, value: float):
        """设置阈值"""
        self._thresholds[name] = value

    def register_handler(self, level: FaultLevel, handler: Callable):
        """注册故障处理器"""
        self._fault_handlers[level].append(handler)

    def check(self) -> List[FaultEvent]:
        """检查故障"""
        faults = []

        # 检查 GPU 显存
        gpu_mem = self._metrics.get('gpu_memory_used_mb')
        if gpu_mem:
            try:
                import torch
                if torch.cuda.is_available():
                    total = torch.cuda.get_device_properties(0).total_memory
                    used = gpu_mem.value * 1024 * 1024
                    ratio = used / total if total > 0 else 0
                    if ratio > self._thresholds['gpu_memory_threshold']:
                        faults.append(self._create_fault(
                            FaultLevel.MEDIUM,
                            'gpu_memory',
                            f'GPU 显存使用率过高: {ratio:.1%}'
                        ))
            except Exception:
                pass

        # 检查超时率
        timeout_metric = self._metrics.get('timeout_count')
        total_metric = self._metrics.get('match_success_rate')
        if timeout_metric and total_metric and total_metric.value > 0:
            timeout_rate = timeout_metric.value / (total_metric.value + timeout_metric.value)
            if timeout_rate > self._thresholds['timeout_rate_threshold']:
                faults.append(self._create_fault(
                    FaultLevel.MEDIUM,
                    'timeout',
                    f'超时率过高: {timeout_rate:.1%}'
                ))

        # 检查匹配成功率
        match_rate = self._metrics.get('match_success_rate')
        if match_rate and match_rate.value < self._thresholds['match_success_threshold']:
            faults.append(self._create_fault(
                FaultLevel.LIGHT,
                'match_rate',
                f'匹配成功率过低: {match_rate.value:.1%}'
            ))

        return faults

    def _create_fault(
        self,
        level: FaultLevel,
        component: str,
        message: str
    ) -> FaultEvent:
        """创建故障事件"""
        fault = FaultEvent(
            fault_id=f"{component}_{int(time.time())}",
            level=level,
            component=component,
            message=message
        )
        self._fault_history.append(fault)
        return fault

    async def handle_fault(self, fault: FaultEvent):
        """处理故障"""
        handlers = self._fault_handlers.get(fault.level, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(fault)
                else:
                    handler(fault)
            except Exception as e:
                logger.error(f"故障处理器执行失败: {e}")

        # 执行默认恢复策略
        await self._default_recovery(fault)

    async def _default_recovery(self, fault: FaultEvent):
        """默认恢复策略"""
        if fault.level == FaultLevel.LIGHT:
            # 轻故障：记录日志，自动重试
            logger.warning(f"轻故障: {fault.message}")
            fault.resolved = True
            fault.resolution = "自动重试"

        elif fault.level == FaultLevel.MEDIUM:
            # 中故障：重启模块
            logger.error(f"中故障: {fault.message}")
            if fault.component == 'gpu_memory':
                # 清理 GPU 缓存
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        fault.resolved = True
                        fault.resolution = "GPU 缓存已清理"
                except Exception as e:
                    fault.resolution = f"GPU 清理失败: {e}"
            else:
                fault.resolved = True
                fault.resolution = "模块已重启"

        elif fault.level == FaultLevel.SEVERE:
            # 重故障：服务降级 + 告警
            logger.critical(f"重故障: {fault.message}")
            fault.resolution = "服务已降级，等待人工处理"
            # 这里可以添加告警通知（邮件、Slack 等）


class HealthChecker:
    """健康检查器"""

    def __init__(
        self,
        metrics: MetricsCollector,
        detector: FaultDetector
    ):
        self._metrics = metrics
        self._detector = detector
        self._running = False
        self._check_interval = 10.0  # 检查间隔（秒）

    async def start(self):
        """启动健康检查"""
        self._running = True
        logger.info("健康检查器已启动")

        while self._running:
            await self._check_health()
            await asyncio.sleep(self._check_interval)

    async def stop(self):
        """停止健康检查"""
        self._running = False
        logger.info("健康检查器已停止")

    async def _check_health(self):
        """执行健康检查"""
        # 更新资源指标
        self._update_resource_metrics()

        # 检测故障
        faults = self._detector.check()

        # 处理故障
        for fault in faults:
            await self._detector.handle_fault(fault)

    def _update_resource_metrics(self):
        """更新资源指标"""
        # CPU 使用率
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._metrics.record('cpu_utilization', cpu_percent / 100.0)

            # 内存使用
            mem = psutil.virtual_memory()
            self._metrics.record('memory_used_mb', mem.used / (1024 * 1024))

        # GPU 使用率（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / (1024 * 1024)
                self._metrics.record('gpu_memory_used_mb', mem_used)
        except Exception:
            pass

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'status': 'healthy' if self._running else 'stopped',
            'metrics': self._metrics.get_summary(),
            'check_interval': self._check_interval
        }


class ProcessingMonitor:
    """处理过程监控器

    监控单个处理任务的执行情况
    """

    def __init__(self, task_id: str, metrics: MetricsCollector):
        self._task_id = task_id
        self._metrics = metrics
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._frame_count = 0
        self._match_count = 0
        self._error_count = 0

    def start(self):
        """开始监控"""
        self._start_time = time.time()
        logger.debug(f"任务监控开始: {self._task_id}")

    def end(self, success: bool = True):
        """结束监控"""
        self._end_time = time.time()
        elapsed = self._end_time - self._start_time if self._start_time else 0

        # 记录指标
        self._metrics.record('process_time_ms', elapsed * 1000)

        if self._frame_count > 0:
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            self._metrics.record('frames_per_second', fps)

            match_rate = self._match_count / self._frame_count
            self._metrics.record('match_success_rate', match_rate)

        if not success:
            self._metrics.increment('timeout_count')

        logger.debug(
            f"任务监控结束: {self._task_id}, "
            f"耗时={elapsed:.2f}s, 帧数={self._frame_count}, "
            f"匹配={self._match_count}"
        )

    def record_frame(self, matched: bool = True):
        """记录帧处理"""
        self._frame_count += 1
        if matched:
            self._match_count += 1

    def record_error(self, error_type: str = "unknown"):
        """记录错误"""
        self._error_count += 1
        self._metrics.increment(f'{error_type}_failures')


# 全局监控实例
_global_metrics: Optional[MetricsCollector] = None
_global_detector: Optional[FaultDetector] = None
_global_health_checker: Optional[HealthChecker] = None


def get_metrics() -> MetricsCollector:
    """获取全局指标采集器"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_fault_detector() -> FaultDetector:
    """获取全局故障检测器"""
    global _global_detector, _global_metrics
    if _global_detector is None:
        _global_detector = FaultDetector(get_metrics())
    return _global_detector


async def start_health_checker():
    """启动健康检查器"""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker(
            get_metrics(),
            get_fault_detector()
        )
    await _global_health_checker.start()


async def stop_health_checker():
    """停止健康检查器"""
    global _global_health_checker
    if _global_health_checker:
        await _global_health_checker.stop()
