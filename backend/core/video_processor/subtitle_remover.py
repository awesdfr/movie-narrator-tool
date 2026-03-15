"""字幕检测与遮罩模块

使用 PaddleOCR 检测帧上的字幕文字区域，通过中值模糊快速遮罩。
PaddleOCR 为可选依赖，未安装时回退到底部 12% 裁剪。

设计要点：
- 固定字幕区域预扫描（采样少量帧投票合并，字幕位置通常固定）
- 中值模糊遮罩（~2ms/帧，远快于修复式去除）
- PaddleOCR 延迟初始化 + 单例（模型加载约 3-5 秒，只加载一次）
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from loguru import logger

# PaddleOCR 可选依赖
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False

# 模块级 PaddleOCR 单例
_ocr_instance: "PaddleOCR | None" = None


def _get_ocr() -> "PaddleOCR":
    """获取 PaddleOCR 单例（延迟初始化）"""
    global _ocr_instance
    if _ocr_instance is None:
        if not PADDLE_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR 未安装")
        logger.info("初始化 PaddleOCR（首次加载约 3-5 秒）...")
        _ocr_instance = PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            show_log=False,
            use_gpu=True,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
        )
        logger.info("PaddleOCR 初始化完成")
    return _ocr_instance


@dataclass
class SubtitleRemoverConfig:
    """字幕去除配置"""
    # 检测模式: "fast"=固定区域+中值模糊, "per_frame"=逐帧检测
    mode: str = "fast"
    # 固定区域检测的采样帧数
    fixed_region_sample_count: int = 10
    # OCR 置信度阈值（低于此值的检测结果忽略）
    ocr_confidence_threshold: float = 0.5
    # 字幕区域投票阈值（超过此比例的帧检测到的区域才认为是固定字幕）
    vote_ratio_threshold: float = 0.3
    # 中值模糊核大小（必须为奇数）
    median_blur_ksize: int = 21
    # 掩码膨胀像素（扩大字幕区域覆盖范围）
    mask_dilate_pixels: int = 5
    # 字幕区域限制：仅检测帧高度的底部百分比以内的区域
    bottom_ratio: float = 0.35
    # 回退模式：底部裁剪比例（PaddleOCR 不可用时使用）
    fallback_crop_ratio: float = 0.12


class SubtitleRemover:
    """字幕检测与遮罩

    工作流程：
    1. detect_fixed_regions() - 采样少量帧，投票合并得到固定字幕掩码
    2. process_frame() - 对每帧应用固定掩码（中值模糊填充字幕区域）

    PaddleOCR 不可用时自动回退到底部裁剪模式。
    """

    def __init__(self, config: SubtitleRemoverConfig = None):
        self.config = config or SubtitleRemoverConfig()
        self._fixed_mask: np.ndarray | None = None
        self._frame_shape: tuple | None = None
        self._ocr_available = PADDLE_OCR_AVAILABLE
        self._initialized = False

        if not self._ocr_available:
            logger.info("PaddleOCR 未安装，字幕去除将使用底部裁剪回退模式")

    @property
    def is_ocr_available(self) -> bool:
        return self._ocr_available

    def detect_fixed_regions(self, video_path: str, sample_count: int = None) -> np.ndarray | None:
        """采样帧检测固定字幕区域

        采样视频前 N 帧，用 PaddleOCR 检测文字区域，
        对出现频率超过投票阈值的区域合并为固定字幕掩码。

        Args:
            video_path: 视频路径
            sample_count: 采样帧数（默认用配置值）

        Returns:
            固定字幕区域的二值掩码（255=字幕区域），或 None 表示未检测到
        """
        if not self._ocr_available:
            logger.debug("PaddleOCR 不可用，跳过固定区域检测")
            return None

        sample_count = sample_count or self.config.fixed_region_sample_count
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"无法打开视频进行字幕检测: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
            cap.release()
            return None

        # 从视频均匀采样帧（跳过开头 5 秒避免片头）
        start_frame = min(int(fps * 5), total_frames // 4)
        end_frame = min(total_frames, int(fps * 120))  # 最多看前 2 分钟
        if end_frame <= start_frame:
            end_frame = total_frames

        sample_indices = np.linspace(start_frame, end_frame - 1, sample_count, dtype=int)
        vote_map = None
        valid_count = 0

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            self._frame_shape = frame.shape[:2]
            mask = self.detect_subtitle_regions(frame)
            if mask is not None and mask.any():
                if vote_map is None:
                    vote_map = np.zeros(frame.shape[:2], dtype=np.float32)
                vote_map += (mask > 0).astype(np.float32)
                valid_count += 1

        cap.release()

        if vote_map is None or valid_count == 0:
            logger.info("未检测到固定字幕区域")
            return None

        # 投票：超过阈值比例的帧都检测到的区域才保留
        threshold = valid_count * self.config.vote_ratio_threshold
        fixed_mask = (vote_map >= threshold).astype(np.uint8) * 255

        if fixed_mask.any():
            # 膨胀掩码，扩大覆盖范围
            if self.config.mask_dilate_pixels > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (self.config.mask_dilate_pixels * 2 + 1,
                     self.config.mask_dilate_pixels * 2 + 1)
                )
                fixed_mask = cv2.dilate(fixed_mask, kernel, iterations=1)

            self._fixed_mask = fixed_mask
            self._initialized = True

            subtitle_pixels = np.count_nonzero(fixed_mask)
            total_pixels = fixed_mask.shape[0] * fixed_mask.shape[1]
            coverage = subtitle_pixels / total_pixels * 100
            logger.info(
                f"固定字幕区域检测完成: 覆盖 {coverage:.1f}% 画面, "
                f"采样 {valid_count}/{sample_count} 帧有效"
            )
            return fixed_mask

        logger.info("投票后无固定字幕区域")
        return None

    def detect_subtitle_regions(self, frame: np.ndarray) -> np.ndarray | None:
        """单帧检测字幕文字区域

        Args:
            frame: BGR 格式的帧

        Returns:
            二值掩码（255=字幕区域），或 None
        """
        if not self._ocr_available:
            return None

        h, w = frame.shape[:2]
        bottom_limit = int(h * (1 - self.config.bottom_ratio))

        # 只截取底部区域送入 OCR（减少计算量）
        bottom_region = frame[bottom_limit:, :]

        try:
            ocr = _get_ocr()
            # PaddleOCR 接受 BGR numpy 数组
            results = ocr.ocr(bottom_region, cls=False)
        except Exception as e:
            logger.debug(f"OCR 检测异常: {e}")
            return None

        if not results or not results[0]:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        for line in results[0]:
            box, (text, confidence) = line[0], (line[1][0], line[1][1])
            if confidence < self.config.ocr_confidence_threshold:
                continue

            # box 是 4 个角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # 坐标转换回原始帧坐标系（加上 bottom_limit 偏移）
            pts = np.array(box, dtype=np.int32)
            pts[:, 1] += bottom_limit

            # 填充多边形区域
            cv2.fillPoly(mask, [pts], 255)

        return mask if mask.any() else None

    def remove_subtitle_fast(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """快速遮罩：用中值模糊填充字幕区域

        Args:
            frame: BGR 格式的帧
            mask: 二值掩码（255=字幕区域）

        Returns:
            处理后的帧
        """
        if mask is None or not mask.any():
            return frame

        # 中值模糊整帧
        blurred = cv2.medianBlur(frame, self.config.median_blur_ksize)

        # 仅在字幕区域用模糊结果替换
        result = frame.copy()
        mask_bool = mask > 0
        result[mask_bool] = blurred[mask_bool]
        return result

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """统一入口：处理单帧的字幕

        使用缓存的固定掩码（如果已检测到），否则直接返回原帧。

        Args:
            frame: BGR 格式的帧

        Returns:
            处理后的帧
        """
        if self._fixed_mask is not None:
            # 确保掩码尺寸与帧匹配
            h, w = frame.shape[:2]
            mask_h, mask_w = self._fixed_mask.shape[:2]
            if h != mask_h or w != mask_w:
                mask = cv2.resize(self._fixed_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = self._fixed_mask
            return self.remove_subtitle_fast(frame, mask)
        return frame

    def fallback_crop(self, frame: np.ndarray) -> np.ndarray:
        """回退模式：底部裁剪

        PaddleOCR 不可用时使用此方法。

        Args:
            frame: BGR 格式的帧

        Returns:
            裁剪后的帧
        """
        h = frame.shape[0]
        crop_bottom = int(h * (1 - self.config.fallback_crop_ratio))
        return frame[:crop_bottom, :]
