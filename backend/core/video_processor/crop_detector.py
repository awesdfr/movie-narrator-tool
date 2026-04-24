"""视频裁剪检测模块

检测解说视频是否对原电影进行了裁剪（中心、四边、四角等），
计算裁剪比例和方向，用于规范化匹配。
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger


@dataclass
class CropInfo:
    """裁剪信息"""
    is_cropped: bool           # 是否被裁剪
    crop_direction: str        # 裁剪方向: center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right
    crop_ratio: float          # 裁剪比例 (0-1)
    left: int                  # 裁剪边界 (像素)
    top: int
    right: int
    bottom: int
    confidence: float          # 检测置信度 (0-1)


class CropDetector:
    """视频裁剪检测器
    
    基于黑边检测和内容分析，识别解说视频是否存在以下裁剪：
    1. 中心裁剪：保留中心部分，四周黑边
    2. 边缘裁剪：移除某个方向，产生偏心黑边
    3. 角落裁剪：移除角落内容
    """
    
    def __init__(self, black_threshold: int = 30, min_crop_ratio: float = 0.05):
        """初始化
        
        Args:
            black_threshold: 黑边阈值（0-255），低于此值认为是黑边
            min_crop_ratio: 最小裁剪比例，低于此值不认为有裁剪
        """
        self.black_threshold = black_threshold
        self.min_crop_ratio = min_crop_ratio
    
    def detect_crop(self, frame: np.ndarray) -> CropInfo:
        """检测单帧的裁剪情况
        
        Args:
            frame: BGR 格式图像
            
        Returns:
            CropInfo 对象
        """
        height, width = frame.shape[:2]
        
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测黑边
        black_mask = gray < self.black_threshold
        
        # 从四个方向查找黑边边界
        top_crop = self._find_crop_top(black_mask)
        bottom_crop = self._find_crop_bottom(black_mask)
        left_crop = self._find_crop_left(black_mask)
        right_crop = self._find_crop_right(black_mask)
        
        # 分析裁剪模式
        crop_info = self._analyze_crop_pattern(
            width, height,
            top_crop, bottom_crop,
            left_crop, right_crop
        )
        
        return crop_info
    
    def detect_crop_from_video(
        self,
        video_path: str,
        sample_frames: int = 10
    ) -> CropInfo:
        """从视频检测裁剪（采样多帧）
        
        Args:
            video_path: 视频路径
            sample_frames: 采样帧数
            
        Returns:
            汇总的裁剪信息
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"无法打开视频: {video_path}")
            return CropInfo(False, "unknown", 0, 0, 0, 0, 0, 0)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        crop_results = []
        step = max(1, total_frames // sample_frames)
        
        for frame_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            crop_info = self.detect_crop(frame)
            if crop_info.is_cropped:
                crop_results.append(crop_info)
        
        cap.release()
        
        # 汇总结果
        if not crop_results:
            return CropInfo(False, "none", 0, 0, 0, 0, 0, 1.0)
        
        # 统计最常见的裁剪方向
        direction_counts = {}
        avg_ratio = 0
        for info in crop_results:
            direction_counts[info.crop_direction] = direction_counts.get(info.crop_direction, 0) + 1
            avg_ratio += info.crop_ratio
        
        avg_ratio /= len(crop_results)
        most_common_direction = max(direction_counts, key=direction_counts.get)
        confidence = direction_counts[most_common_direction] / len(crop_results)
        
        # 使用最常见的方向的边界值
        representative_info = next(
            (info for info in crop_results if info.crop_direction == most_common_direction),
            crop_results[0]
        )
        
        return CropInfo(
            is_cropped=True,
            crop_direction=most_common_direction,
            crop_ratio=avg_ratio,
            left=representative_info.left,
            top=representative_info.top,
            right=representative_info.right,
            bottom=representative_info.bottom,
            confidence=confidence
        )
    
    def apply_crop(self, frame: np.ndarray, crop_info: CropInfo) -> np.ndarray:
        """对帧应用裁剪
        
        Args:
            frame: 原始帧
            crop_info: 裁剪信息
            
        Returns:
            裁剪后的帧
        """
        if not crop_info.is_cropped:
            return frame
        
        return frame[crop_info.top:crop_info.bottom, crop_info.left:crop_info.right]
    
    def reverse_crop(
        self,
        cropped_frame: np.ndarray,
        crop_info: CropInfo,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """逆向操作：将裁剪帧还原为原始尺寸
        
        Args:
            cropped_frame: 裁剪后的帧
            crop_info: 裁剪信息
            original_shape: (height, width) 原始尺寸
            
        Returns:
            还原为原始尺寸的帧（周围补黑边）
        """
        if not crop_info.is_cropped:
            return cropped_frame
        
        height, width = original_shape
        result = np.zeros((height, width, 3), dtype=np.uint8)
        result[
            crop_info.top:crop_info.bottom,
            crop_info.left:crop_info.right
        ] = cropped_frame
        
        return result
    
    # 私有方法
    
    def _find_crop_top(self, black_mask: np.ndarray) -> int:
        """查找顶部黑边边界"""
        for i in range(black_mask.shape[0]):
            if black_mask[i].sum() < black_mask.shape[1] * 0.9:  # 不全是黑边
                return i
        return 0
    
    def _find_crop_bottom(self, black_mask: np.ndarray) -> int:
        """查找底部黑边边界"""
        for i in range(black_mask.shape[0] - 1, -1, -1):
            if black_mask[i].sum() < black_mask.shape[1] * 0.9:
                return i + 1
        return black_mask.shape[0]
    
    def _find_crop_left(self, black_mask: np.ndarray) -> int:
        """查找左边黑边边界"""
        for j in range(black_mask.shape[1]):
            if black_mask[:, j].sum() < black_mask.shape[0] * 0.9:
                return j
        return 0
    
    def _find_crop_right(self, black_mask: np.ndarray) -> int:
        """查找右边黑边边界"""
        for j in range(black_mask.shape[1] - 1, -1, -1):
            if black_mask[:, j].sum() < black_mask.shape[0] * 0.9:
                return j + 1
        return black_mask.shape[1]
    
    def _analyze_crop_pattern(
        self,
        width: int,
        height: int,
        top: int,
        bottom: int,
        left: int,
        right: int
    ) -> CropInfo:
        """分析裁剪模式
        
        Returns:
            CropInfo 对象
        """
        crop_height = bottom - top
        crop_width = right - left
        
        # 检查是否有意义的裁剪
        crop_ratio_h = 1.0 - (crop_height / height) if height > 0 else 0
        crop_ratio_w = 1.0 - (crop_width / width) if width > 0 else 0
        
        if crop_ratio_h < self.min_crop_ratio and crop_ratio_w < self.min_crop_ratio:
            return CropInfo(False, "none", 0, left, top, right, bottom, 1.0)
        
        # 判断裁剪类型
        top_black = top / height if height > 0 else 0
        bottom_black = (height - bottom) / height if height > 0 else 0
        left_black = left / width if width > 0 else 0
        right_black = (width - right) / width if width > 0 else 0
        
        # 中心裁剪（四周都有黑边）
        if (top_black > 0.05 and bottom_black > 0.05 and
            left_black > 0.05 and right_black > 0.05):
            avg_ratio = (crop_ratio_h + crop_ratio_w) / 2
            return CropInfo(True, "center", avg_ratio, left, top, right, bottom, 0.9)
        
        # 顶部裁剪
        if top_black > 0.05 and bottom_black < 0.02 and left_black < 0.02 and right_black < 0.02:
            return CropInfo(True, "top", crop_ratio_h, left, top, right, bottom, 0.85)
        
        # 底部裁剪
        if bottom_black > 0.05 and top_black < 0.02 and left_black < 0.02 and right_black < 0.02:
            return CropInfo(True, "bottom", crop_ratio_h, left, top, right, bottom, 0.85)
        
        # 左边裁剪
        if left_black > 0.05 and right_black < 0.02 and top_black < 0.02 and bottom_black < 0.02:
            return CropInfo(True, "left", crop_ratio_w, left, top, right, bottom, 0.85)
        
        # 右边裁剪
        if right_black > 0.05 and left_black < 0.02 and top_black < 0.02 and bottom_black < 0.02:
            return CropInfo(True, "right", crop_ratio_w, left, top, right, bottom, 0.85)
        
        # 角落裁剪
        if top_black > 0.05 and left_black > 0.05:
            return CropInfo(True, "top_left", max(crop_ratio_h, crop_ratio_w), left, top, right, bottom, 0.75)
        if top_black > 0.05 and right_black > 0.05:
            return CropInfo(True, "top_right", max(crop_ratio_h, crop_ratio_w), left, top, right, bottom, 0.75)
        if bottom_black > 0.05 and left_black > 0.05:
            return CropInfo(True, "bottom_left", max(crop_ratio_h, crop_ratio_w), left, top, right, bottom, 0.75)
        if bottom_black > 0.05 and right_black > 0.05:
            return CropInfo(True, "bottom_right", max(crop_ratio_h, crop_ratio_w), left, top, right, bottom, 0.75)
        
        # 复杂裁剪
        avg_ratio = (crop_ratio_h + crop_ratio_w) / 2
        return CropInfo(True, "complex", avg_ratio, left, top, right, bottom, 0.7)
