"""动态视频采样模块

实现二级关键帧检测：
1. 粗筛：帧间灰度直方图差异
2. 精准验证：Farneback 轻量光流

根据运动剧烈程度动态调整采样率：
- 运动剧烈区域：8-12 fps
- 普通区域：4 fps
- 静态画面：仅保留首帧
"""
import cv2
import numpy as np
from typing import Generator, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SampledFrame:
    """采样帧信息"""
    frame: np.ndarray       # 帧图像 (BGR)
    frame_idx: int          # 原始帧号
    time_sec: float         # 时间戳（秒）
    motion_level: str       # 运动级别: 'static', 'normal', 'dynamic'
    motion_score: float     # 运动分数 0-1


@dataclass
class SamplingConfig:
    """采样配置"""
    static_fps: float = 1.0       # 静态画面采样率
    normal_fps: float = 4.0       # 普通画面采样率
    dynamic_fps: float = 8.0      # 运动剧烈画面采样率
    max_fps: float = 15.0         # 最大采样率上限
    histogram_threshold: float = 0.1   # 直方图差异阈值（粗筛）
    motion_threshold_low: float = 2.0   # 光流运动阈值（低）
    motion_threshold_high: float = 8.0  # 光流运动阈值（高）


class VideoSampler:
    """动态视频采样器

    核心思想：
    1. 使用帧间灰度直方图差异快速过滤静态帧
    2. 使用 Farneback 轻量光流精确判断运动程度
    3. 根据运动程度动态调整采样率
    """

    def __init__(self, config: SamplingConfig = None):
        """初始化

        Args:
            config: 采样配置，为 None 时使用默认配置
        """
        self.config = config or SamplingConfig()
        self._prev_gray = None
        self._prev_histogram = None

    def sample_video(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = None
    ) -> Generator[SampledFrame, None, None]:
        """动态采样视频

        Args:
            video_path: 视频文件路径
            start_time: 起始时间（秒）
            end_time: 结束时间（秒），None 表示到视频结尾

        Yields:
            SampledFrame 对象
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_time is None or end_time > duration:
            end_time = duration

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        logger.debug(
            f"动态采样: {video_path}, "
            f"范围 [{start_time:.1f}s-{end_time:.1f}s], "
            f"帧 {start_frame}-{end_frame}"
        )

        # 重置状态
        self._prev_gray = None
        self._prev_histogram = None

        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        last_sample_frame = -1000  # 上次采样的帧号
        sampled_count = 0
        motion_history = []  # 最近的运动分数历史

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_idx / fps

            # 计算运动级别
            motion_level, motion_score = self._compute_motion_level(frame)
            motion_history.append(motion_score)

            # 保持最近 10 帧的运动历史
            if len(motion_history) > 10:
                motion_history.pop(0)

            # 根据运动级别确定采样间隔
            if motion_level == 'static':
                target_fps = self.config.static_fps
            elif motion_level == 'dynamic':
                target_fps = self.config.dynamic_fps
            else:
                target_fps = self.config.normal_fps

            # 应用最大采样率上限
            target_fps = min(target_fps, self.config.max_fps)

            # 计算采样间隔（帧数）
            sample_interval = max(1, int(fps / target_fps))

            # 判断是否应该采样
            should_sample = False

            # 静态画面：只采样首帧（或间隔很大时采样）
            if motion_level == 'static':
                if frame_idx - last_sample_frame >= fps * 2:  # 静态场景每2秒采样一帧
                    should_sample = True
            # 场景切换检测：直方图差异突变时强制采样
            elif motion_score > 0.5 and len(motion_history) >= 2:
                avg_recent = np.mean(motion_history[:-1])
                if motion_score > avg_recent * 2:
                    should_sample = True
            # 正常/动态：按采样间隔采样
            elif frame_idx - last_sample_frame >= sample_interval:
                should_sample = True

            if should_sample:
                yield SampledFrame(
                    frame=frame,
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    motion_level=motion_level,
                    motion_score=motion_score
                )
                last_sample_frame = frame_idx
                sampled_count += 1

            frame_idx += 1

        cap.release()

        actual_duration = end_time - start_time
        effective_fps = sampled_count / actual_duration if actual_duration > 0 else 0
        logger.debug(
            f"动态采样完成: 采样 {sampled_count} 帧, "
            f"有效采样率 {effective_fps:.1f} fps"
        )

    def _compute_motion_level(self, frame: np.ndarray) -> tuple[str, float]:
        """计算运动级别

        使用二级检测：
        1. 粗筛：灰度直方图差异（CPU 轻量）
        2. 精准：Farneback 光流（仅在需要时计算）

        Args:
            frame: BGR 格式帧

        Returns:
            (motion_level, motion_score)
            motion_level: 'static', 'normal', 'dynamic'
            motion_score: 0-1 运动分数
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算直方图
        histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
        histogram = histogram.flatten() / histogram.sum()  # 归一化

        if self._prev_gray is None or self._prev_histogram is None:
            self._prev_gray = gray
            self._prev_histogram = histogram
            return 'normal', 0.5  # 首帧默认 normal

        # 第一级：直方图差异（巴氏距离）
        hist_diff = cv2.compareHist(
            self._prev_histogram, histogram,
            cv2.HISTCMP_BHATTACHARYYA
        )

        # 静态判定：直方图几乎无变化
        if hist_diff < self.config.histogram_threshold:
            self._prev_gray = gray
            self._prev_histogram = histogram
            return 'static', float(hist_diff)

        # 第二级：光流检测（仅对非静态帧计算）
        motion_score = self._compute_optical_flow_score(gray)

        # 更新状态
        self._prev_gray = gray
        self._prev_histogram = histogram

        # 判定运动级别
        if motion_score < self.config.motion_threshold_low:
            return 'normal', motion_score / self.config.motion_threshold_high
        elif motion_score > self.config.motion_threshold_high:
            return 'dynamic', min(1.0, motion_score / 15.0)
        else:
            return 'normal', motion_score / self.config.motion_threshold_high

    def _compute_optical_flow_score(self, gray: np.ndarray) -> float:
        """计算 Farneback 光流运动分数

        Args:
            gray: 灰度图像

        Returns:
            运动分数（光流幅值的平均值）
        """
        if self._prev_gray is None:
            return 0.0

        # Farneback 光流（轻量参数）
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5,      # 金字塔缩放比例
            levels=3,           # 金字塔层数
            winsize=15,         # 窗口大小
            iterations=3,       # 迭代次数
            poly_n=5,           # 多项式展开大小
            poly_sigma=1.1,     # 多项式高斯标准差
            flags=0
        )

        # 计算光流幅值
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # 返回平均运动幅值
        return float(np.mean(magnitude))

    def get_frame_at_time(
        self,
        video_path: str,
        time_sec: float
    ) -> Optional[np.ndarray]:
        """获取指定时间的帧

        Args:
            video_path: 视频文件路径
            time_sec: 时间戳（秒）

        Returns:
            帧图像 (BGR) 或 None
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frame = int(time_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    def sample_at_fixed_fps(
        self,
        video_path: str,
        target_fps: float,
        start_time: float = 0.0,
        end_time: float = None
    ) -> Generator[SampledFrame, None, None]:
        """固定帧率采样（简单模式）

        Args:
            video_path: 视频文件路径
            target_fps: 目标采样率
            start_time: 起始时间
            end_time: 结束时间

        Yields:
            SampledFrame 对象
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_time is None or end_time > duration:
            end_time = duration

        # 计算采样间隔
        sample_interval = max(1, int(fps / target_fps))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % sample_interval == 0:
                yield SampledFrame(
                    frame=frame,
                    frame_idx=frame_idx,
                    time_sec=frame_idx / fps,
                    motion_level='normal',
                    motion_score=0.5
                )

            frame_idx += 1

        cap.release()


class KeyframeDetector:
    """关键帧检测器

    用于检测视频中的关键帧（场景切换、运动峰值等）
    """

    def __init__(
        self,
        histogram_threshold: float = 0.3,
        motion_threshold: float = 10.0
    ):
        """初始化

        Args:
            histogram_threshold: 直方图差异阈值（场景切换检测）
            motion_threshold: 运动阈值（运动峰值检测）
        """
        self.histogram_threshold = histogram_threshold
        self.motion_threshold = motion_threshold

    def detect_keyframes(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = None
    ) -> list[tuple[int, float, str]]:
        """检测关键帧

        Args:
            video_path: 视频文件路径
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            [(frame_idx, time_sec, reason), ...] 列表
            reason: 'scene_change', 'motion_peak', 'start', 'end'
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_time is None or end_time > duration:
            end_time = duration

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        keyframes = []
        prev_gray = None
        prev_histogram = None

        # 添加起始帧
        keyframes.append((start_frame, start_time, 'start'))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        # 每 5 帧检测一次，减少计算量
        check_interval = max(1, int(fps / 5))

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % check_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
                histogram = histogram.flatten() / histogram.sum()

                if prev_gray is not None and prev_histogram is not None:
                    # 直方图差异（场景切换检测）
                    hist_diff = cv2.compareHist(
                        prev_histogram, histogram,
                        cv2.HISTCMP_BHATTACHARYYA
                    )

                    if hist_diff > self.histogram_threshold:
                        keyframes.append((
                            frame_idx,
                            frame_idx / fps,
                            'scene_change'
                        ))

                prev_gray = gray
                prev_histogram = histogram

            frame_idx += 1

        cap.release()

        # 添加结束帧
        keyframes.append((end_frame - 1, end_time, 'end'))

        # 去重并排序
        keyframes = list(set(keyframes))
        keyframes.sort(key=lambda x: x[0])

        logger.debug(f"关键帧检测完成: 发现 {len(keyframes)} 个关键帧")
        return keyframes


# 便捷函数
def sample_video_dynamic(
    video_path: str,
    start_time: float = 0.0,
    end_time: float = None
) -> list[SampledFrame]:
    """动态采样视频的便捷函数"""
    sampler = VideoSampler()
    return list(sampler.sample_video(video_path, start_time, end_time))


def sample_video_fixed(
    video_path: str,
    target_fps: float,
    start_time: float = 0.0,
    end_time: float = None
) -> list[SampledFrame]:
    """固定帧率采样的便捷函数"""
    sampler = VideoSampler()
    return list(sampler.sample_at_fixed_fps(video_path, target_fps, start_time, end_time))
