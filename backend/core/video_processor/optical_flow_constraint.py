"""光流引导的时序一致性约束模块

使用 Farneback 轻量光流验证匹配结果的时序一致性：
1. 光流运动平滑性约束：相邻匹配帧的运动应连贯
2. 哈希相似度时序约束：相邻帧哈希距离应平稳变化
3. 帧号单调性约束：电影时间应单调递增（允许小范围回退）

光流失效时自动切换到哈希时序约束。
"""
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TemporalConstraintResult:
    """时序约束验证结果"""
    is_valid: bool                # 是否通过验证
    confidence_adjustment: float  # 置信度调整值 (-0.2 ~ +0.1)
    motion_consistency: float     # 运动一致性得分 (0-1)
    hash_consistency: float       # 哈希一致性得分 (0-1)
    constraint_type: str          # 使用的约束类型: 'optical_flow', 'hash', 'fallback'
    details: dict                 # 详细信息


@dataclass
class OpticalFlowConfig:
    """光流约束配置"""
    # 光流计算参数
    pyr_scale: float = 0.5        # 金字塔缩放比例
    levels: int = 3               # 金字塔层数
    winsize: int = 15             # 窗口大小
    iterations: int = 3           # 迭代次数
    poly_n: int = 5               # 多项式展开大小
    poly_sigma: float = 1.1       # 多项式高斯标准差

    # 约束阈值
    motion_variance_threshold: float = 50.0   # 光流方差阈值（超过则失效）
    motion_magnitude_min: float = 0.5         # 最小运动幅值（低于则视为静态）
    motion_consistency_threshold: float = 0.6  # 运动一致性阈值
    hash_consistency_threshold: float = 0.7    # 哈希一致性阈值

    # 时序约束
    max_backtrack_frames: int = 30   # 最大允许回退帧数
    smoothness_window: int = 5       # 平滑性检测窗口大小


class OpticalFlowConstraint:
    """光流时序一致性约束器

    核心功能：
    1. 计算相邻匹配帧之间的光流
    2. 验证光流运动的平滑性和一致性
    3. 当光流失效时，使用哈希时序约束作为兜底
    4. 输出置信度调整建议
    """

    def __init__(self, config: OpticalFlowConfig = None):
        """初始化

        Args:
            config: 光流约束配置
        """
        self.config = config or OpticalFlowConfig()
        self._flow_cache = {}  # 缓存计算过的光流

    def verify_match_sequence(
        self,
        query_frames: list[np.ndarray],
        query_times: list[float],
        matched_frames: list[np.ndarray],
        matched_times: list[float],
        hash_distances: list[int] = None
    ) -> TemporalConstraintResult:
        """验证匹配序列的时序一致性

        Args:
            query_frames: 查询帧序列 (BGR)
            query_times: 查询帧时间戳
            matched_frames: 匹配到的电影帧序列 (BGR)
            matched_times: 匹配到的电影时间戳
            hash_distances: 对应的哈希距离列表（可选）

        Returns:
            TemporalConstraintResult 验证结果
        """
        if len(query_frames) < 2 or len(matched_frames) < 2:
            return TemporalConstraintResult(
                is_valid=True,
                confidence_adjustment=0.0,
                motion_consistency=1.0,
                hash_consistency=1.0,
                constraint_type='insufficient_data',
                details={'reason': '帧数不足，跳过验证'}
            )

        # 尝试使用光流约束
        flow_result = self._verify_with_optical_flow(
            query_frames, matched_frames
        )

        if flow_result is not None:
            # 光流约束有效
            motion_consistency = flow_result['consistency']
            is_valid = motion_consistency >= self.config.motion_consistency_threshold

            # 计算置信度调整
            if motion_consistency >= 0.9:
                adjustment = 0.1  # 高一致性，提升置信度
            elif motion_consistency >= 0.7:
                adjustment = 0.0  # 中等一致性，不调整
            elif motion_consistency >= 0.5:
                adjustment = -0.1  # 低一致性，降低置信度
            else:
                adjustment = -0.2  # 很低一致性，显著降低

            return TemporalConstraintResult(
                is_valid=is_valid,
                confidence_adjustment=adjustment,
                motion_consistency=motion_consistency,
                hash_consistency=1.0,
                constraint_type='optical_flow',
                details=flow_result
            )

        # 光流失效，使用哈希时序约束
        logger.debug("光流约束失效，切换到哈希时序约束")

        hash_result = self._verify_with_hash_consistency(
            matched_times, hash_distances
        )

        is_valid = hash_result['consistency'] >= self.config.hash_consistency_threshold

        if hash_result['consistency'] >= 0.85:
            adjustment = 0.05
        elif hash_result['consistency'] >= 0.7:
            adjustment = 0.0
        else:
            adjustment = -0.15

        return TemporalConstraintResult(
            is_valid=is_valid,
            confidence_adjustment=adjustment,
            motion_consistency=0.0,
            hash_consistency=hash_result['consistency'],
            constraint_type='hash',
            details=hash_result
        )

    def _verify_with_optical_flow(
        self,
        query_frames: list[np.ndarray],
        matched_frames: list[np.ndarray]
    ) -> Optional[dict]:
        """使用光流验证运动一致性

        Args:
            query_frames: 查询帧序列
            matched_frames: 匹配帧序列

        Returns:
            验证结果字典，或 None（光流失效）
        """
        query_flows = []
        matched_flows = []

        # 计算查询序列的光流
        for i in range(len(query_frames) - 1):
            flow = self._compute_optical_flow(
                query_frames[i], query_frames[i + 1]
            )
            if flow is None:
                return None
            query_flows.append(flow)

        # 计算匹配序列的光流
        for i in range(len(matched_frames) - 1):
            flow = self._compute_optical_flow(
                matched_frames[i], matched_frames[i + 1]
            )
            if flow is None:
                return None
            matched_flows.append(flow)

        # 检查光流是否有效（非静态，方差不过大）
        query_magnitudes = [self._flow_magnitude(f) for f in query_flows]
        matched_magnitudes = [self._flow_magnitude(f) for f in matched_flows]

        avg_query_mag = np.mean(query_magnitudes)
        avg_matched_mag = np.mean(matched_magnitudes)

        # 光流失效条件：幅值过小（静态）或方差过大（噪声）
        if avg_query_mag < self.config.motion_magnitude_min:
            logger.debug(f"查询序列光流幅值过小: {avg_query_mag:.2f}")
            return None

        query_variance = np.var(query_magnitudes)
        if query_variance > self.config.motion_variance_threshold:
            logger.debug(f"查询序列光流方差过大: {query_variance:.2f}")
            return None

        # 计算运动一致性：比较两个序列的光流模式
        consistency = self._compute_flow_consistency(query_flows, matched_flows)

        return {
            'consistency': consistency,
            'query_avg_magnitude': avg_query_mag,
            'matched_avg_magnitude': avg_matched_mag,
            'query_variance': query_variance
        }

    def _compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Optional[np.ndarray]:
        """计算两帧之间的 Farneback 光流

        Args:
            frame1: 第一帧 (BGR)
            frame2: 第二帧 (BGR)

        Returns:
            光流矩阵 [H, W, 2] 或 None
        """
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=self.config.pyr_scale,
                levels=self.config.levels,
                winsize=self.config.winsize,
                iterations=self.config.iterations,
                poly_n=self.config.poly_n,
                poly_sigma=self.config.poly_sigma,
                flags=0
            )

            return flow

        except Exception as e:
            logger.warning(f"光流计算失败: {e}")
            return None

    def _flow_magnitude(self, flow: np.ndarray) -> float:
        """计算光流的平均幅值"""
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(magnitude))

    def _flow_direction(self, flow: np.ndarray) -> float:
        """计算光流的主方向（弧度）"""
        # 使用加权平均计算主方向
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])

        # 加权平均（幅值大的权重高）
        weights = magnitude / (magnitude.sum() + 1e-6)
        weighted_dir = np.sum(direction * weights)

        return float(weighted_dir)

    def _compute_flow_consistency(
        self,
        flows1: list[np.ndarray],
        flows2: list[np.ndarray]
    ) -> float:
        """计算两个光流序列的一致性

        一致性基于：
        1. 幅值比例的稳定性
        2. 方向差异的平滑性

        Returns:
            一致性得分 0-1
        """
        if len(flows1) != len(flows2):
            min_len = min(len(flows1), len(flows2))
            flows1 = flows1[:min_len]
            flows2 = flows2[:min_len]

        if len(flows1) == 0:
            return 0.0

        # 幅值比例一致性
        mag_ratios = []
        for f1, f2 in zip(flows1, flows2):
            m1 = self._flow_magnitude(f1)
            m2 = self._flow_magnitude(f2)
            if m1 > 0.1 and m2 > 0.1:
                ratio = min(m1, m2) / max(m1, m2)
                mag_ratios.append(ratio)

        if mag_ratios:
            mag_consistency = np.mean(mag_ratios)
        else:
            mag_consistency = 0.5

        # 方向一致性（相对变化应相似）
        dir_diffs = []
        for i in range(len(flows1) - 1):
            d1_change = self._flow_direction(flows1[i+1]) - self._flow_direction(flows1[i])
            d2_change = self._flow_direction(flows2[i+1]) - self._flow_direction(flows2[i])
            diff = abs(d1_change - d2_change)
            # 归一化到 0-1
            dir_similarity = 1.0 - min(diff / np.pi, 1.0)
            dir_diffs.append(dir_similarity)

        if dir_diffs:
            dir_consistency = np.mean(dir_diffs)
        else:
            dir_consistency = 0.5

        # 综合一致性
        consistency = mag_consistency * 0.6 + dir_consistency * 0.4

        return float(consistency)

    def _verify_with_hash_consistency(
        self,
        matched_times: list[float],
        hash_distances: list[int] = None
    ) -> dict:
        """使用哈希时序约束验证

        Args:
            matched_times: 匹配的电影时间序列
            hash_distances: 哈希距离序列

        Returns:
            验证结果字典
        """
        # 时间单调性检查
        monotonic_violations = 0
        for i in range(1, len(matched_times)):
            if matched_times[i] < matched_times[i-1]:
                # 允许小范围回退
                backtrack = matched_times[i-1] - matched_times[i]
                if backtrack > 5.0:  # 超过5秒视为违规
                    monotonic_violations += 1

        monotonic_score = 1.0 - monotonic_violations / max(len(matched_times) - 1, 1)

        # 时间间隔平滑性检查
        if len(matched_times) >= 3:
            intervals = [
                matched_times[i] - matched_times[i-1]
                for i in range(1, len(matched_times))
            ]
            interval_variance = np.var(intervals)
            # 归一化（假设正常间隔变化在 10 以内）
            smoothness_score = 1.0 / (1.0 + interval_variance / 10.0)
        else:
            smoothness_score = 1.0

        # 哈希距离稳定性检查
        if hash_distances and len(hash_distances) >= 2:
            distance_variance = np.var(hash_distances)
            # 归一化（假设正常距离变化在 25 以内）
            hash_stability = 1.0 / (1.0 + distance_variance / 25.0)
        else:
            hash_stability = 0.8  # 默认值

        # 综合一致性
        consistency = (
            monotonic_score * 0.4 +
            smoothness_score * 0.3 +
            hash_stability * 0.3
        )

        return {
            'consistency': consistency,
            'monotonic_score': monotonic_score,
            'smoothness_score': smoothness_score,
            'hash_stability': hash_stability,
            'monotonic_violations': monotonic_violations
        }

    def verify_single_match(
        self,
        prev_query_frame: np.ndarray,
        curr_query_frame: np.ndarray,
        prev_matched_frame: np.ndarray,
        curr_matched_frame: np.ndarray,
        prev_matched_time: float,
        curr_matched_time: float
    ) -> tuple[bool, float]:
        """验证单个匹配的时序一致性

        Args:
            prev_query_frame: 上一个查询帧
            curr_query_frame: 当前查询帧
            prev_matched_frame: 上一个匹配帧
            curr_matched_frame: 当前匹配帧
            prev_matched_time: 上一个匹配时间
            curr_matched_time: 当前匹配时间

        Returns:
            (是否有效, 置信度调整)
        """
        # 时间顺序检查
        time_diff = curr_matched_time - prev_matched_time
        if time_diff < -5.0:  # 回退超过5秒
            return False, -0.2

        # 计算光流
        query_flow = self._compute_optical_flow(prev_query_frame, curr_query_frame)
        matched_flow = self._compute_optical_flow(prev_matched_frame, curr_matched_frame)

        if query_flow is None or matched_flow is None:
            # 光流计算失败，使用宽松验证
            if time_diff >= 0:
                return True, 0.0
            else:
                return True, -0.05

        # 运动幅值比较
        query_mag = self._flow_magnitude(query_flow)
        matched_mag = self._flow_magnitude(matched_flow)

        if query_mag < 0.5:  # 静态场景
            return True, 0.0

        # 幅值比例检查
        if matched_mag > 0.1:
            mag_ratio = min(query_mag, matched_mag) / max(query_mag, matched_mag)
        else:
            mag_ratio = 0.5

        if mag_ratio >= 0.7:
            return True, 0.05
        elif mag_ratio >= 0.4:
            return True, 0.0
        else:
            return True, -0.1


# 便捷函数
def verify_match_temporal_consistency(
    query_frames: list[np.ndarray],
    matched_frames: list[np.ndarray],
    matched_times: list[float]
) -> TemporalConstraintResult:
    """验证匹配时序一致性的便捷函数"""
    constraint = OpticalFlowConstraint()
    return constraint.verify_match_sequence(
        query_frames, [], matched_frames, matched_times
    )
