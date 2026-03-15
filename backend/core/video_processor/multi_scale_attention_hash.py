"""多尺度注意力哈希模块

实现 24x24、32x32、48x48 三尺度 pHash 计算，
仅对 YCbCr 亮度通道 Y 计算（抗调色），
使用轻量化无训练注意力加权融合。
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class MultiScaleHash:
    """多尺度哈希结果"""
    hash_24: str          # 24x24 尺度哈希
    hash_32: str          # 32x32 尺度哈希（主尺度）
    hash_48: str          # 48x48 尺度哈希
    variance_24: float    # 24x24 尺度方差（区分度）
    variance_32: float    # 32x32 尺度方差
    variance_48: float    # 48x48 尺度方差
    fused_hash: str       # 加权融合后的哈希（用于快速匹配）
    quality_score: float  # 图像质量分数（用于动态阈值）


class MultiScaleAttentionHasher:
    """多尺度注意力哈希计算器

    特点：
    1. 三尺度计算：24x24（粗粒度，抗裁剪）、32x32（主尺度）、48x48（细粒度）
    2. 仅使用 YCbCr 亮度通道 Y，规避调色影响
    3. 轻量化注意力加权：按各尺度哈希的方差（区分度）分配权重
    4. 动态阈值：基于图像质量（PSNR/模糊度）自适应调整
    """

    # 尺度配置
    SCALES = [24, 32, 48]

    # 默认权重（当无法计算方差时使用）
    DEFAULT_WEIGHTS = {24: 0.25, 32: 0.50, 48: 0.25}

    # 动态阈值配置
    THRESHOLD_MIN = 6   # 高清晰视频的严格阈值
    THRESHOLD_MAX = 12  # 低质量视频的宽松阈值
    THRESHOLD_DEFAULT = 8

    def __init__(
        self,
        use_ycbcr: bool = True,
        adaptive_weights: bool = True
    ):
        """初始化

        Args:
            use_ycbcr: 是否使用 YCbCr 亮度通道（推荐开启，抗调色）
            adaptive_weights: 是否使用自适应权重（基于方差）
        """
        self.use_ycbcr = use_ycbcr
        self.adaptive_weights = adaptive_weights

    def compute_hash(self, image: np.ndarray) -> MultiScaleHash:
        """计算多尺度哈希

        Args:
            image: BGR 格式图像

        Returns:
            MultiScaleHash 对象
        """
        # 转换为灰度（如果使用 YCbCr，提取 Y 通道）
        if self.use_ycbcr:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            gray = ycbcr[:, :, 0]  # Y 通道（亮度）
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算图像质量分数（用于动态阈值）
        quality_score = self._compute_quality_score(gray)

        # 计算各尺度哈希
        hash_24, var_24 = self._compute_single_scale_hash(gray, 24)
        hash_32, var_32 = self._compute_single_scale_hash(gray, 32)
        hash_48, var_48 = self._compute_single_scale_hash(gray, 48)

        # 融合哈希（使用注意力加权）
        fused_hash = self._fuse_hashes(
            [(hash_24, var_24), (hash_32, var_32), (hash_48, var_48)]
        )

        return MultiScaleHash(
            hash_24=hash_24,
            hash_32=hash_32,
            hash_48=hash_48,
            variance_24=var_24,
            variance_32=var_32,
            variance_48=var_48,
            fused_hash=fused_hash,
            quality_score=quality_score
        )

    def compute_hash_from_pil(self, pil_image: Image.Image) -> MultiScaleHash:
        """从 PIL 图像计算多尺度哈希"""
        # PIL -> numpy BGR
        rgb = np.array(pil_image)
        if len(rgb.shape) == 2:
            # 灰度图
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.compute_hash(bgr)

    def _compute_single_scale_hash(
        self,
        gray: np.ndarray,
        scale: int
    ) -> tuple[str, float]:
        """计算单尺度 pHash

        Args:
            gray: 灰度图像
            scale: 缩放尺寸

        Returns:
            (哈希字符串, 方差)
        """
        # 缩放到目标尺寸
        resized = cv2.resize(gray, (scale, scale), interpolation=cv2.INTER_AREA)

        # DCT 变换
        dct = cv2.dct(np.float32(resized))

        # 取左上角 8x8 低频分量（与标准 pHash 兼容）
        dct_low = dct[:8, :8]

        # 计算中值
        median = np.median(dct_low)

        # 生成哈希（大于中值为 1，否则为 0）
        hash_bits = (dct_low > median).flatten()

        # 转换为十六进制字符串
        hash_bytes = np.packbits(hash_bits)
        hash_str = hash_bytes.tobytes().hex()

        # 计算方差（区分度指标）
        variance = float(np.var(dct_low))

        return hash_str, variance

    def _fuse_hashes(
        self,
        hash_var_pairs: list[tuple[str, float]]
    ) -> str:
        """融合多尺度哈希

        使用注意力机制：按方差（区分度）分配权重
        方差越大，说明该尺度的特征越丰富，权重越高

        Args:
            hash_var_pairs: [(hash_str, variance), ...] 列表

        Returns:
            融合后的哈希字符串
        """
        if not self.adaptive_weights:
            # 使用默认权重，直接返回主尺度（32x32）哈希
            return hash_var_pairs[1][0]

        # 计算注意力权重（softmax 归一化方差）
        variances = np.array([v for _, v in hash_var_pairs])

        # 避免除零
        if variances.sum() < 1e-6:
            # 方差都很小，使用默认权重
            return hash_var_pairs[1][0]

        # 使用 softmax 归一化（温度参数 = 1.0）
        exp_vars = np.exp(variances / (variances.mean() + 1e-6))
        weights = exp_vars / exp_vars.sum()

        # 将哈希转换为位数组
        hash_bits_list = []
        for hash_str, _ in hash_var_pairs:
            hash_bytes = bytes.fromhex(hash_str)
            bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
            hash_bits_list.append(bits.astype(np.float32))

        # 加权平均
        weighted_bits = np.zeros(64, dtype=np.float32)
        for bits, weight in zip(hash_bits_list, weights):
            weighted_bits += bits * weight

        # 四舍五入为 0/1
        fused_bits = (weighted_bits > 0.5).astype(np.uint8)

        # 转换为十六进制字符串
        fused_bytes = np.packbits(fused_bits)
        return fused_bytes.tobytes().hex()

    def _compute_quality_score(self, gray: np.ndarray) -> float:
        """计算图像质量分数

        基于：
        1. 拉普拉斯方差（清晰度/模糊度）
        2. 灰度直方图分布（对比度）

        Returns:
            质量分数 0-1，越高越清晰
        """
        # 拉普拉斯方差（清晰度指标）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # 归一化到 0-1（经验值：100-1000 对应清晰度范围）
        sharpness = min(1.0, laplacian_var / 500.0)

        # 灰度直方图标准差（对比度指标）
        hist_std = np.std(gray)
        contrast = min(1.0, hist_std / 80.0)  # 归一化

        # 综合质量分数
        quality = sharpness * 0.7 + contrast * 0.3

        return float(quality)

    def get_adaptive_threshold(
        self,
        quality_score: float,
        resolution: tuple[int, int] = None
    ) -> int:
        """获取自适应阈值

        基于图像质量和分辨率动态调整阈值：
        - 高清晰度 + 高分辨率 → 严格阈值（6）
        - 低清晰度 / 低分辨率 → 宽松阈值（12）

        Args:
            quality_score: 图像质量分数 0-1
            resolution: (width, height) 分辨率，可选

        Returns:
            阈值 6-12
        """
        # 基础阈值（基于质量分数）
        if quality_score >= 0.8:
            base_threshold = self.THRESHOLD_MIN
        elif quality_score >= 0.5:
            # 线性插值
            ratio = (quality_score - 0.5) / 0.3
            base_threshold = self.THRESHOLD_MAX - int(
                ratio * (self.THRESHOLD_MAX - self.THRESHOLD_MIN)
            )
        else:
            base_threshold = self.THRESHOLD_MAX

        # 分辨率调整
        if resolution is not None:
            width, height = resolution
            pixels = width * height

            # 低分辨率（<720P）放宽阈值
            if pixels < 1280 * 720:
                base_threshold = min(self.THRESHOLD_MAX, base_threshold + 2)
            # 高分辨率（>=1080P）收紧阈值
            elif pixels >= 1920 * 1080:
                base_threshold = max(self.THRESHOLD_MIN, base_threshold - 1)

        return int(base_threshold)

    def compute_distance(
        self,
        hash1: MultiScaleHash,
        hash2: MultiScaleHash,
        use_fused: bool = True
    ) -> tuple[int, float]:
        """计算两个多尺度哈希之间的距离

        Args:
            hash1: 第一个哈希
            hash2: 第二个哈希
            use_fused: 是否使用融合哈希（更快），否则使用加权多尺度距离（更准）

        Returns:
            (汉明距离, 相似度 0-1)
        """
        if use_fused:
            distance = self._hamming_distance(hash1.fused_hash, hash2.fused_hash)
        else:
            # 加权多尺度距离
            dist_24 = self._hamming_distance(hash1.hash_24, hash2.hash_24)
            dist_32 = self._hamming_distance(hash1.hash_32, hash2.hash_32)
            dist_48 = self._hamming_distance(hash1.hash_48, hash2.hash_48)

            # 使用注意力权重
            var_sum = hash1.variance_24 + hash1.variance_32 + hash1.variance_48
            if var_sum < 1e-6:
                weights = list(self.DEFAULT_WEIGHTS.values())
            else:
                weights = [
                    hash1.variance_24 / var_sum,
                    hash1.variance_32 / var_sum,
                    hash1.variance_48 / var_sum
                ]

            distance = int(
                dist_24 * weights[0] +
                dist_32 * weights[1] +
                dist_48 * weights[2]
            )

        # 相似度（64位哈希）
        similarity = 1.0 - distance / 64.0

        return distance, similarity

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算两个哈希的汉明距离"""
        bytes1 = bytes.fromhex(hash1)
        bytes2 = bytes.fromhex(hash2)

        distance = 0
        for b1, b2 in zip(bytes1, bytes2):
            distance += bin(b1 ^ b2).count('1')

        return distance

    def batch_compute_hash(
        self,
        images: list[np.ndarray]
    ) -> list[MultiScaleHash]:
        """批量计算多尺度哈希

        Args:
            images: BGR 格式图像列表

        Returns:
            MultiScaleHash 列表
        """
        return [self.compute_hash(img) for img in images]


# 便捷函数
def compute_multi_scale_hash(image: np.ndarray) -> MultiScaleHash:
    """计算多尺度哈希的便捷函数"""
    hasher = MultiScaleAttentionHasher()
    return hasher.compute_hash(image)


def get_adaptive_threshold(quality_score: float, resolution: tuple[int, int] = None) -> int:
    """获取自适应阈值的便捷函数"""
    hasher = MultiScaleAttentionHasher()
    return hasher.get_adaptive_threshold(quality_score, resolution)
