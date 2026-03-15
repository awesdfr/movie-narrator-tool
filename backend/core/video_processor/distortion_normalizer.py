"""失真归一化预处理模块

对视频帧进行归一化预处理，减少失真影响：
1. 中值滤波去小水印/噪声
2. YCbCr 颜色归一化（抗调色）
3. 统一基准分辨率（可选）
4. 对比度/亮度归一化

全部使用 CPU 轻量计算，不依赖 GPU。
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class NormalizerConfig:
    """归一化配置"""
    # 分辨率归一化
    target_resolution: Tuple[int, int] = (640, 360)  # 目标分辨率 (width, height)
    enable_resize: bool = True                        # 是否启用分辨率归一化

    # 中值滤波（去水印/噪声）
    median_kernel_size: int = 3                       # 中值滤波核大小 (3, 5, 7)
    enable_median_filter: bool = True                 # 是否启用中值滤波

    # 颜色归一化
    enable_color_normalize: bool = True               # 是否启用颜色归一化
    target_brightness: float = 128.0                  # 目标亮度
    target_contrast: float = 1.0                      # 目标对比度

    # 直方图均衡化
    enable_histogram_eq: bool = False                 # 是否启用直方图均衡化（对亮度通道）

    # 边缘保持滤波（去噪同时保留边缘）
    enable_bilateral_filter: bool = False             # 是否启用双边滤波
    bilateral_d: int = 9                              # 双边滤波直径
    bilateral_sigma_color: float = 75                 # 颜色空间sigma
    bilateral_sigma_space: float = 75                 # 坐标空间sigma


class DistortionNormalizer:
    """失真归一化预处理器

    核心功能：
    1. 分辨率统一：将不同分辨率的视频统一到基准分辨率
    2. 去噪去水印：中值滤波去除小的水印和噪声
    3. 颜色归一化：YCbCr 空间进行亮度/对比度归一化
    4. 可选增强：直方图均衡化、双边滤波等
    """

    def __init__(self, config: NormalizerConfig = None):
        """初始化

        Args:
            config: 归一化配置
        """
        self.config = config or NormalizerConfig()

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """对帧进行归一化预处理

        Args:
            frame: 输入帧 (BGR)

        Returns:
            归一化后的帧 (BGR)
        """
        result = frame.copy()

        # 1. 分辨率归一化
        if self.config.enable_resize:
            result = self._resize_frame(result)

        # 2. 中值滤波去噪
        if self.config.enable_median_filter:
            result = self._apply_median_filter(result)

        # 3. 颜色归一化
        if self.config.enable_color_normalize:
            result = self._normalize_color(result)

        # 4. 可选：直方图均衡化
        if self.config.enable_histogram_eq:
            result = self._apply_histogram_eq(result)

        # 5. 可选：双边滤波
        if self.config.enable_bilateral_filter:
            result = self._apply_bilateral_filter(result)

        return result

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """分辨率归一化

        保持宽高比缩放到目标分辨率范围内
        """
        h, w = frame.shape[:2]
        target_w, target_h = self.config.target_resolution

        # 计算缩放比例（保持宽高比）
        scale = min(target_w / w, target_h / h)

        if scale < 1.0:  # 只缩小，不放大
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def _apply_median_filter(self, frame: np.ndarray) -> np.ndarray:
        """应用中值滤波去噪

        中值滤波对椒盐噪声和小水印特别有效
        """
        kernel_size = self.config.median_kernel_size
        # 确保核大小是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.medianBlur(frame, kernel_size)

    def _normalize_color(self, frame: np.ndarray) -> np.ndarray:
        """颜色归一化

        在 YCbCr 空间对亮度通道进行归一化，
        保持色度通道不变，实现抗调色效果
        """
        # BGR -> YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        y_channel = ycrcb[:, :, 0].astype(np.float32)

        # 计算当前亮度统计
        current_mean = np.mean(y_channel)
        current_std = np.std(y_channel)

        if current_std < 1e-6:
            current_std = 1.0

        # 线性变换到目标亮度和对比度
        target_mean = self.config.target_brightness
        target_std = current_std * self.config.target_contrast

        normalized_y = (y_channel - current_mean) / current_std * target_std + target_mean

        # 裁剪到有效范围
        normalized_y = np.clip(normalized_y, 0, 255).astype(np.uint8)

        ycrcb[:, :, 0] = normalized_y

        # YCrCb -> BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def _apply_histogram_eq(self, frame: np.ndarray) -> np.ndarray:
        """对亮度通道应用直方图均衡化

        增强对比度，使不同曝光条件下的帧更具可比性
        """
        # BGR -> YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # 对 Y 通道应用 CLAHE（自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])

        # YCrCb -> BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def _apply_bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        """应用双边滤波

        去噪同时保留边缘，适合处理模糊视频
        """
        return cv2.bilateralFilter(
            frame,
            self.config.bilateral_d,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space
        )

    def normalize_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """批量归一化

        Args:
            frames: 帧列表 (BGR)

        Returns:
            归一化后的帧列表
        """
        return [self.normalize(frame) for frame in frames]

    def extract_luminance(self, frame: np.ndarray) -> np.ndarray:
        """提取归一化后的亮度通道

        Args:
            frame: 输入帧 (BGR)

        Returns:
            亮度通道 (灰度图)
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)

        # 归一化
        current_mean = np.mean(y_channel)
        current_std = np.std(y_channel)

        if current_std < 1e-6:
            return y_channel.astype(np.uint8)

        normalized = (y_channel - current_mean) / current_std * 64 + 128
        return np.clip(normalized, 0, 255).astype(np.uint8)


class WatermarkRemover:
    """水印检测与去除

    基于帧间差异检测静态水印区域，
    使用周围区域进行修复。
    """

    def __init__(
        self,
        detection_threshold: float = 0.95,
        min_watermark_area: int = 100
    ):
        """初始化

        Args:
            detection_threshold: 静态区域检测阈值
            min_watermark_area: 最小水印面积（像素）
        """
        self.detection_threshold = detection_threshold
        self.min_watermark_area = min_watermark_area
        self._watermark_mask = None

    def detect_watermark(
        self,
        frames: list[np.ndarray],
        sample_count: int = 10
    ) -> Optional[np.ndarray]:
        """检测静态水印区域

        通过分析多帧，找出始终不变的区域

        Args:
            frames: 帧列表 (BGR)
            sample_count: 用于检测的采样帧数

        Returns:
            水印掩码 (二值图) 或 None
        """
        if len(frames) < 2:
            return None

        # 采样帧
        step = max(1, len(frames) // sample_count)
        sampled = frames[::step][:sample_count]

        if len(sampled) < 2:
            return None

        # 计算帧间差异
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sampled]

        # 累积静态区域掩码
        static_mask = np.ones(gray_frames[0].shape, dtype=np.float32)

        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[0], gray_frames[i])
            # 差异小的区域视为静态
            is_static = (diff < 10).astype(np.float32)
            static_mask *= is_static

        # 阈值化
        watermark_mask = (static_mask > self.detection_threshold).astype(np.uint8) * 255

        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        watermark_mask = cv2.morphologyEx(watermark_mask, cv2.MORPH_OPEN, kernel)
        watermark_mask = cv2.morphologyEx(watermark_mask, cv2.MORPH_CLOSE, kernel)

        # 检查是否有有效水印区域
        if np.sum(watermark_mask > 0) < self.min_watermark_area:
            return None

        self._watermark_mask = watermark_mask
        return watermark_mask

    def remove_watermark(
        self,
        frame: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """去除水印

        使用图像修复技术填充水印区域

        Args:
            frame: 输入帧 (BGR)
            mask: 水印掩码，None 则使用之前检测的

        Returns:
            去除水印后的帧
        """
        if mask is None:
            mask = self._watermark_mask

        if mask is None:
            return frame

        # 使用 OpenCV 的图像修复
        # INPAINT_TELEA: 基于快速行进方法
        # INPAINT_NS: 基于 Navier-Stokes 方程
        result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        return result


# 便捷函数
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """归一化帧的便捷函数"""
    normalizer = DistortionNormalizer()
    return normalizer.normalize(frame)


def normalize_frames_batch(frames: list[np.ndarray]) -> list[np.ndarray]:
    """批量归一化帧的便捷函数"""
    normalizer = DistortionNormalizer()
    return normalizer.normalize_batch(frames)


def extract_luminance(frame: np.ndarray) -> np.ndarray:
    """提取亮度通道的便捷函数"""
    normalizer = DistortionNormalizer()
    return normalizer.extract_luminance(frame)
