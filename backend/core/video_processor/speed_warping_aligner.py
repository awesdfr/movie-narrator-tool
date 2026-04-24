"""视频变速容限处理模块

实现 DTW (Dynamic Time Warping) 和时间映射，
支持 0.8x - 1.2x 变速的画面匹配。
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class SpeedWarpResult:
    """变速对齐结果"""
    speed_factor: float           # 检测到的速度因子 (0.8-1.2)
    confidence: float             # 检测置信度 (0-1)
    time_mapping: dict            # 时间映射函数 {query_time -> movie_time}
    aligned_pairs: List[Tuple]    # 对齐的 (query_idx, movie_idx, distance) 对
    dtw_cost: float              # DTW 总代价
    is_speed_changed: bool        # 是否检测到变速


class SpeedWarpingAligner:
    """变速容限对齐器
    
    特点：
    1. DTW 预搜索：检测是否存在变速
    2. 时间映射学习：学习非线性时间映射函数
    3. 多假设验证：尝试多个变速假设
    4. 鲁棒性：支持 0.8x-1.2x 的变速范围
    """
    
    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        dtw_window_size: Optional[int] = None,
        use_sakoe_chiba: bool = True
    ):
        """初始化
        
        Args:
            speed_range: 支持的变速范围 (min_speed, max_speed)
            dtw_window_size: DTW 累积距离窗口大小（None=无限）
            use_sakoe_chiba: 是否使用 Sakoe-Chiba 波段约束加速 DTW
        """
        self.speed_range = speed_range
        self.dtw_window_size = dtw_window_size
        self.use_sakoe_chiba = use_sakoe_chiba
    
    def detect_speed_change(
        self,
        query_times: List[float],
        movie_times: List[float],
        query_distances: np.ndarray,
        movie_distances: Optional[np.ndarray] = None
    ) -> SpeedWarpResult:
        """检测并学习时间映射
        
        Args:
            query_times: 查询帧的时间戳列表 (解说视频)
            movie_times: 电影帧的时间戳列表
            query_distances: 查询帧的距离矩阵 (shape: [N_query, N_movie])
            movie_distances: 电影帧的距离矩阵（可选，用于精化）
            
        Returns:
            SpeedWarpResult 对象
        """
        if len(query_times) < 2 or len(movie_times) < 2:
            logger.warning("时间序列太短，无法进行变速检测")
            return SpeedWarpResult(
                speed_factor=1.0,
                confidence=0.0,
                time_mapping={},
                aligned_pairs=[],
                dtw_cost=0,
                is_speed_changed=False
            )
        
        # 计算参考速度（线性映射）
        query_duration = query_times[-1] - query_times[0]
        movie_duration = movie_times[-1] - movie_times[0]
        linear_speed = movie_duration / (query_duration + 1e-10)
        
        # DTW 最优路径
        best_cost, best_path = self._compute_dtw(query_distances)
        
        # 从 DTW 路径学习时间映射
        time_mapping = self._extract_time_mapping(
            query_times, movie_times, best_path
        )
        
        # 分析时间映射以检测变速
        speed_factor, confidence = self._analyze_speed_from_mapping(
            query_times, time_mapping, linear_speed
        )
        
        # 提取对齐的帧对
        aligned_pairs = self._extract_aligned_pairs(
            best_path, query_distances
        )
        
        is_speed_changed = abs(speed_factor - 1.0) > 0.05  # 5% 容限
        
        return SpeedWarpResult(
            speed_factor=speed_factor,
            confidence=confidence,
            time_mapping=time_mapping,
            aligned_pairs=aligned_pairs,
            dtw_cost=best_cost,
            is_speed_changed=is_speed_changed
        )
    
    def apply_time_mapping(
        self,
        query_time: float,
        time_mapping: dict,
        query_times: List[float],
        movie_times: List[float]
    ) -> float:
        """应用时间映射函数
        
        Args:
            query_time: 查询时间
            time_mapping: 时间映射字典
            query_times: 查询时间戳列表（用于插值）
            movie_times: 电影时间戳列表
            
        Returns:
            映射后的电影时间
        """
        if not time_mapping:
            # 使用线性映射
            query_duration = query_times[-1] - query_times[0] if len(query_times) > 1 else 1.0
            movie_duration = movie_times[-1] - movie_times[0] if len(movie_times) > 1 else 1.0
            return (query_time - query_times[0]) * (movie_duration / query_duration) + movie_times[0]
        
        # 查找最近的两个映射点进行线性插值
        sorted_times = sorted(time_mapping.keys())
        
        if query_time <= sorted_times[0]:
            return time_mapping[sorted_times[0]]
        if query_time >= sorted_times[-1]:
            return time_mapping[sorted_times[-1]]
        
        # 二分查找
        idx = np.searchsorted(sorted_times, query_time)
        t1, t2 = sorted_times[idx - 1], sorted_times[idx]
        m1, m2 = time_mapping[t1], time_mapping[t2]
        
        # 线性插值
        ratio = (query_time - t1) / (t2 - t1)
        return m1 + ratio * (m2 - m1)
    
    # 私有方法
    
    def _compute_dtw(
        self,
        distance_matrix: np.ndarray,
        window_size: Optional[int] = None
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """计算 DTW 最优路径
        
        Args:
            distance_matrix: 距离矩阵 (shape: [N_query, N_movie])
            window_size: Sakoe-Chiba 窗口大小
            
        Returns:
            (最小代价, 最优路径)
        """
        if window_size is None:
            window_size = self.dtw_window_size
        
        n, m = distance_matrix.shape
        
        # 初始化累积代价矩阵
        dtw_matrix = np.full((n, m), np.inf)
        dtw_matrix[0, 0] = distance_matrix[0, 0]
        
        # 填充第一行和第一列
        for i in range(1, n):
            dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + distance_matrix[i, 0]
        for j in range(1, m):
            dtw_matrix[0, j] = dtw_matrix[0, j - 1] + distance_matrix[0, j]
        
        # 动态规划填充
        for i in range(1, n):
            # Sakoe-Chiba 波段约束
            if window_size is not None:
                j_min = max(1, i - window_size)
                j_max = min(m, i + window_size)
            else:
                j_min, j_max = 1, m
            
            for j in range(j_min, j_max):
                cost = min(
                    dtw_matrix[i - 1, j],      # 纵向
                    dtw_matrix[i, j - 1],      # 横向
                    dtw_matrix[i - 1, j - 1]   # 对角线
                )
                dtw_matrix[i, j] = cost + distance_matrix[i, j]
        
        # 回溯最优路径
        path = []
        i, j = n - 1, m - 1
        while i > 0 or j > 0:
            path.append((i, j))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # 选择代价最小的前驱
                candidates = [
                    (dtw_matrix[i - 1, j], i - 1, j),
                    (dtw_matrix[i, j - 1], i, j - 1),
                    (dtw_matrix[i - 1, j - 1], i - 1, j - 1)
                ]
                _, i, j = min(candidates)
        
        path.reverse()
        
        return float(dtw_matrix[n - 1, m - 1]), path
    
    def _extract_time_mapping(
        self,
        query_times: List[float],
        movie_times: List[float],
        dtw_path: List[Tuple[int, int]]
    ) -> dict:
        """从 DTW 路径提取时间映射
        
        Args:
            query_times: 查询时间戳列表
            movie_times: 电影时间戳列表
            dtw_path: DTW 最优路径
            
        Returns:
            时间映射字典 {query_time -> movie_time}
        """
        time_mapping = {}
        
        for query_idx, movie_idx in dtw_path:
            if query_idx < len(query_times) and movie_idx < len(movie_times):
                q_time = query_times[query_idx]
                m_time = movie_times[movie_idx]
                
                # 为了避免重复，保留该查询时间首次映射
                if q_time not in time_mapping:
                    time_mapping[q_time] = m_time
        
        return time_mapping
    
    def _analyze_speed_from_mapping(
        self,
        query_times: List[float],
        time_mapping: dict,
        linear_speed: float
    ) -> Tuple[float, float]:
        """从时间映射分析变速因子
        
        Args:
            query_times: 查询时间戳
            time_mapping: 时间映射
            linear_speed: 线性映射的速度
            
        Returns:
            (速度因子, 置信度)
        """
        if not time_mapping or len(time_mapping) < 2:
            return 1.0, 0.0
        
        # 计算局部速度
        sorted_times = sorted(time_mapping.keys())
        local_speeds = []
        
        for i in range(len(sorted_times) - 1):
            t1, t2 = sorted_times[i], sorted_times[i + 1]
            m1, m2 = time_mapping[t1], time_mapping[t2]
            
            query_delta = t2 - t1
            movie_delta = m2 - m1
            
            if query_delta > 1e-6:
                local_speed = movie_delta / query_delta
                local_speeds.append(local_speed)
        
        if not local_speeds:
            return 1.0, 0.0
        
        # 使用中位数作为总体速度
        speeds = np.array(local_speeds)
        median_speed = np.median(speeds)
        
        # 置信度：速度的一致性
        speed_std = np.std(speeds)
        confidence = np.exp(-speed_std)  # 速度越一致，置信度越高
        
        # 限制在支持范围内
        speed_factor = np.clip(median_speed, self.speed_range[0], self.speed_range[1])
        
        return float(speed_factor), float(confidence)
    
    def _extract_aligned_pairs(
        self,
        dtw_path: List[Tuple[int, int]],
        query_distances: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """从 DTW 路径提取对齐的帧对
        
        Args:
            dtw_path: DTW 最优路径
            query_distances: 距离矩阵
            
        Returns:
            [(query_idx, movie_idx, distance), ...] 列表
        """
        aligned_pairs = []
        
        for query_idx, movie_idx in dtw_path:
            if query_idx < query_distances.shape[0] and movie_idx < query_distances.shape[1]:
                distance = float(query_distances[query_idx, movie_idx])
                aligned_pairs.append((query_idx, movie_idx, distance))
        
        return aligned_pairs
