"""颜色自适应匹配模块

对抗视频变色干扰（调色、色温变化等），
通过多颜色空间和自适应阈值实现鲁棒匹配。
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from loguru import logger


@dataclass
class ColorProfile:
    """颜色特征"""
    hist_bgr: np.ndarray          # BGR 颜色直方图 (shape: 48)
    hist_hsv: np.ndarray          # HSV 颜色直方图 (shape: 48)
    hist_lab: np.ndarray          # Lab 颜色直方图 (shape: 48)
    brightness: float             # 平均亮度 (0-255)
    saturation: float             # 平均饱和度 (0-1)
    color_shift: float            # 颜色偏离度 (0-1)


class ColorAdaptiveMatcher:
    """颜色自适应匹配器
    
    特点：
    1. 多颜色空间分析：BGR、HSV、Lab
    2. 颜色偏离度检测：识别视频是否经过变色处理
    3. 自适应相似度计算：基于颜色偏离度动态调整阈值
    4. 颜色不变特征：使用在颜色变化下保持稳定的特征
    """
    
    def __init__(self):
        """初始化"""
        self.hist_bins = 8  # 每个通道 8 个 bin
    
    def extract_color_profile(self, frame: np.ndarray) -> ColorProfile:
        """提取帧的颜色特征
        
        Args:
            frame: BGR 格式图像
            
        Returns:
            ColorProfile 对象
        """
        # BGR 直方图
        hist_bgr = self._compute_histogram(frame, space='BGR')
        
        # HSV 直方图
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_hsv = self._compute_histogram(hsv, space='HSV')
        
        # Lab 直方图
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        hist_lab = self._compute_histogram(lab, space='Lab')
        
        # 亮度
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # 饱和度
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # 颜色偏离度（基于直方图熵）
        color_shift = self._compute_color_shift(hist_bgr, hist_hsv)
        
        return ColorProfile(
            hist_bgr=hist_bgr,
            hist_hsv=hist_hsv,
            hist_lab=hist_lab,
            brightness=float(brightness),
            saturation=float(saturation),
            color_shift=color_shift
        )
    
    def compute_color_similarity(
        self,
        profile1: ColorProfile,
        profile2: ColorProfile,
        use_weighted: bool = True
    ) -> Tuple[float, dict]:
        """计算两个颜色特征的相似度
        
        Args:
            profile1: 第一个颜色特征
            profile2: 第二个颜色特征
            use_weighted: 是否使用加权融合（考虑颜色偏离）
            
        Returns:
            (相似度 0-1, 详细信息字典)
        """
        # 分别计算各颜色空间的相似度
        sim_bgr = self._histogram_similarity(profile1.hist_bgr, profile2.hist_bgr)
        sim_hsv = self._histogram_similarity(profile1.hist_hsv, profile2.hist_hsv)
        sim_lab = self._histogram_similarity(profile1.hist_lab, profile2.hist_lab)
        
        # 亮度相似度
        brightness_diff = abs(profile1.brightness - profile2.brightness) / 255.0
        sim_brightness = 1.0 - min(1.0, brightness_diff / 0.2)
        
        # 饱和度相似度
        sat_diff = abs(profile1.saturation - profile2.saturation)
        sim_saturation = 1.0 - min(1.0, sat_diff / 0.3)
        
        if use_weighted:
            # 基于颜色偏离度调整权重
            max_shift = max(profile1.color_shift, profile2.color_shift)
            if max_shift > 0.3:  # 高度颜色偏离，降低色彩权重，提高亮度权重
                weights = {
                    'bgr': 0.15,
                    'hsv': 0.10,
                    'lab': 0.25,
                    'brightness': 0.35,
                    'saturation': 0.15
                }
            elif max_shift > 0.15:  # 中度偏离
                weights = {
                    'bgr': 0.20,
                    'hsv': 0.15,
                    'lab': 0.25,
                    'brightness': 0.25,
                    'saturation': 0.15
                }
            else:  # 低偏离，均衡权重
                weights = {
                    'bgr': 0.25,
                    'hsv': 0.20,
                    'lab': 0.20,
                    'brightness': 0.20,
                    'saturation': 0.15
                }
        else:
            # 等权重
            weights = {
                'bgr': 0.20,
                'hsv': 0.20,
                'lab': 0.20,
                'brightness': 0.20,
                'saturation': 0.20
            }
        
        # 加权融合
        combined_sim = (
            weights['bgr'] * sim_bgr +
            weights['hsv'] * sim_hsv +
            weights['lab'] * sim_lab +
            weights['brightness'] * sim_brightness +
            weights['saturation'] * sim_saturation
        )
        
        return float(combined_sim), {
            'sim_bgr': float(sim_bgr),
            'sim_hsv': float(sim_hsv),
            'sim_lab': float(sim_lab),
            'sim_brightness': float(sim_brightness),
            'sim_saturation': float(sim_saturation),
            'max_color_shift': float(max_shift),
            'weights': weights
        }
    
    def get_adaptive_threshold(
        self,
        color_shift: float,
        base_threshold: float = 0.7
    ) -> float:
        """基于颜色偏离度获取自适应相似度阈值
        
        Args:
            color_shift: 颜色偏离度 (0-1)
            base_threshold: 基础阈值
            
        Returns:
            调整后的阈值
        """
        # 颜色偏离度越高，阈值越低（容许更大的差异）
        if color_shift < 0.1:
            # 几乎无变色，使用严格阈值
            return base_threshold
        elif color_shift < 0.3:
            # 轻微变色，略降阈值
            return base_threshold - 0.05 * (color_shift / 0.3)
        else:
            # 显著变色，大幅降阈值
            return max(base_threshold - 0.15, base_threshold * 0.7)
    
    # 私有方法
    
    def _compute_histogram(self, frame: np.ndarray, space: str = 'BGR') -> np.ndarray:
        """计算直方图（L2 归一化）
        
        Args:
            frame: 图像
            space: 颜色空间 'BGR', 'HSV', 'Lab'
            
        Returns:
            扁平化的 L2 归一化直方图
        """
        if space == 'BGR':
            ranges = [0, 256, 0, 256, 0, 256]
        elif space == 'HSV':
            # H: 0-180, S: 0-255, V: 0-255
            ranges = [0, 180, 0, 256, 0, 256]
        elif space == 'Lab':
            # L: 0-255, a: 0-255, b: 0-255
            ranges = [0, 256, 0, 256, 0, 256]
        else:
            ranges = [0, 256, 0, 256, 0, 256]
        
        hist = cv2.calcHist(
            [frame],
            [0, 1, 2],
            None,
            [self.hist_bins, self.hist_bins, self.hist_bins],
            ranges
        )
        
        # 展平并 L2 归一化
        hist = hist.flatten().astype(np.float32)
        hist /= (np.linalg.norm(hist) + 1e-10)
        
        return hist
    
    def _histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """计算两个直方图的相似度（余弦距离）"""
        norm1 = np.linalg.norm(hist1) + 1e-10
        norm2 = np.linalg.norm(hist2) + 1e-10
        
        similarity = np.dot(hist1, hist2) / (norm1 * norm2)
        return float(np.clip(similarity, 0, 1))
    
    def _compute_color_shift(self, hist_bgr: np.ndarray, hist_hsv: np.ndarray) -> float:
        """计算颜色偏离度
        
        基于 BGR 和 HSV 直方图的熵差异来估计变色程度。
        """
        # 计算熵
        def entropy(hist):
            hist = np.clip(hist, 1e-10, 1.0)
            return -np.sum(hist * np.log2(hist + 1e-10))
        
        entropy_bgr = entropy(hist_bgr)
        entropy_hsv = entropy(hist_hsv)
        
        # 颜色偏离度：HSV 熵相对于 BGR 熵的偏离程度
        # 如果图像经过变色，HSV 分布会变得更均匀或更尖锐
        shift = abs(entropy_hsv - entropy_bgr) / max(entropy_bgr, entropy_hsv, 1.0)
        
        return float(np.clip(shift, 0, 1))
    
    @staticmethod
    def apply_color_correction(
        frame: np.ndarray,
        reference_profile: ColorProfile,
        target_profile: ColorProfile
    ) -> np.ndarray:
        """颜色校正：将帧调整到与参考帧类似的颜色
        
        Args:
            frame: 待校正帧
            reference_profile: 参考颜色特征
            target_profile: 目标颜色特征
            
        Returns:
            校正后的帧
        """
        # 基于亮度调整
        current_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        brightness_ratio = reference_profile.brightness / (current_brightness + 1e-10)
        frame_adjusted = np.clip(frame.astype(np.float32) * brightness_ratio, 0, 255).astype(np.uint8)
        
        return frame_adjusted
