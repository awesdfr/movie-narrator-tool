"""匹配双向验证模块

实现前向-后向验证机制，消除误匹配。
"""
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool                 # 是否通过验证
    forward_confidence: float      # 前向置信度
    backward_confidence: float     # 后向置信度
    combined_confidence: float     # 综合置信度
    time_consistency: float        # 时间一致性 (0-1)
    distance_consistency: float    # 距离一致性 (0-1)
    validation_score: float        # 最终验证得分 (0-1)
    details: Dict                  # 详细信息


class MatchValidator:
    """匹配双向验证器
    
    特点：
    1. 前向验证：解说帧 → 匹配的电影帧 → 重新匹配解说
    2. 后向验证：电影帧 → 重新匹配解说 → 再回到电影
    3. 一致性检查：时间和距离的往返一致性
    4. 置信度融合：综合评估匹配可靠性
    """
    
    def __init__(
        self,
        time_threshold: float = 1.0,      # 时间往返误差阈值（秒）
        distance_threshold: float = 0.15, # 距离往返误差阈值（相对）
        min_confidence: float = 0.5       # 最小置信度阈值
    ):
        """初始化
        
        Args:
            time_threshold: 允许的时间误差
            distance_threshold: 允许的距离相对误差
            min_confidence: 最小置信度
        """
        self.time_threshold = time_threshold
        self.distance_threshold = distance_threshold
        self.min_confidence = min_confidence
    
    def validate_match(
        self,
        query_idx: int,
        query_times: List[float],
        query_features: np.ndarray,
        movie_idx: int,
        movie_times: List[float],
        movie_features: np.ndarray,
        forward_confidence: float,
        forward_distance: float
    ) -> ValidationResult:
        """验证单个匹配
        
        Args:
            query_idx: 查询帧索引
            query_times: 查询时间戳列表
            query_features: 查询特征矩阵 (shape: [N_query, D])
            movie_idx: 电影帧索引
            movie_times: 电影时间戳列表
            movie_features: 电影特征矩阵 (shape: [N_movie, D])
            forward_confidence: 前向匹配的置信度
            forward_distance: 前向匹配的距离
            
        Returns:
            ValidationResult 对象
        """
        # 前向信息
        forward_time = query_times[query_idx] if query_idx < len(query_times) else 0
        movie_time = movie_times[movie_idx] if movie_idx < len(movie_times) else 0
        
        # 后向验证：从电影特征匹配回查询特征
        backward_confidence, backward_distance, best_backward_idx = (
            self._backward_search(
                movie_features[movie_idx],
                query_features,
                query_idx
            )
        )
        
        # 时间一致性
        backward_time = query_times[best_backward_idx] if best_backward_idx < len(query_times) else forward_time
        time_error = abs(forward_time - backward_time)
        time_consistency = max(0.0, 1.0 - time_error / (self.time_threshold + 1e-6))
        
        # 距离一致性
        distance_error = abs(forward_distance - backward_distance)
        distance_consistency = max(
            0.0,
            1.0 - (distance_error / (max(forward_distance, backward_distance, 1e-6) * self.distance_threshold))
        )
        
        # 综合置信度
        combined_confidence = (
            forward_confidence * 0.5 +
            backward_confidence * 0.5 +
            time_consistency * 0.3 +
            distance_consistency * 0.2
        ) / 2.0  # 归一化
        
        # 综合验证得分
        validation_score = self._compute_validation_score(
            forward_confidence,
            backward_confidence,
            time_consistency,
            distance_consistency
        )
        
        # 判决
        is_valid = (
            validation_score >= self.min_confidence and
            time_consistency >= 0.5 and
            distance_consistency >= 0.5
        )
        
        return ValidationResult(
            is_valid=is_valid,
            forward_confidence=forward_confidence,
            backward_confidence=backward_confidence,
            combined_confidence=combined_confidence,
            time_consistency=time_consistency,
            distance_consistency=distance_consistency,
            validation_score=validation_score,
            details={
                'forward_time': forward_time,
                'backward_time': backward_time,
                'time_error': time_error,
                'forward_distance': forward_distance,
                'backward_distance': backward_distance,
                'distance_error': distance_error,
                'best_backward_idx': best_backward_idx
            }
        )
    
    def validate_batch(
        self,
        matches: List[Dict],
        query_times: List[float],
        query_features: np.ndarray,
        movie_times: List[float],
        movie_features: np.ndarray
    ) -> List[ValidationResult]:
        """批量验证多个匹配
        
        Args:
            matches: 匹配列表，每个包含 {
                'query_idx': int,
                'movie_idx': int,
                'confidence': float,
                'distance': float
            }
            query_times: 查询时间戳列表
            query_features: 查询特征矩阵
            movie_times: 电影时间戳列表
            movie_features: 电影特征矩阵
            
        Returns:
            ValidationResult 列表
        """
        results = []
        
        for match in matches:
            result = self.validate_match(
                match['query_idx'],
                query_times,
                query_features,
                match['movie_idx'],
                movie_times,
                movie_features,
                match.get('confidence', 0.5),
                match.get('distance', 0.0)
            )
            results.append(result)
        
        return results
    
    def filter_valid_matches(
        self,
        validation_results: List[ValidationResult],
        confidence_threshold: Optional[float] = None
    ) -> List[ValidationResult]:
        """过滤有效的匹配
        
        Args:
            validation_results: 验证结果列表
            confidence_threshold: 置信度阈值（使用 self.min_confidence 如果为 None）
            
        Returns:
            有效匹配的验证结果列表
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.min_confidence
        
        return [
            result for result in validation_results
            if result.is_valid and result.validation_score >= threshold
        ]
    
    def detect_and_remove_duplicates(
        self,
        validation_results: List[ValidationResult],
        movie_times: List[float],
        time_window: float = 2.0
    ) -> List[int]:
        """检测并移除重复的电影匹配
        
        在同一时间窗口内只保留置信度最高的匹配
        
        Args:
            validation_results: 验证结果列表
            movie_times: 电影时间戳列表
            time_window: 时间窗口（秒）
            
        Returns:
            要保留的结果索引列表
        """
        if not validation_results:
            return []
        
        # 按置信度排序
        indexed_results = [
            (i, result) for i, result in enumerate(validation_results)
        ]
        indexed_results.sort(key=lambda x: x[1].validation_score, reverse=True)
        
        kept_indices = set()
        
        for idx, result in indexed_results:
            # 检查是否与已保留的结果在时间窗口内重叠
            movie_idx = result.details.get('best_backward_idx', 0)
            movie_time = movie_times[movie_idx] if movie_idx < len(movie_times) else 0
            
            is_duplicate = False
            for kept_idx in kept_indices:
                kept_result = validation_results[kept_idx]
                kept_movie_idx = kept_result.details.get('best_backward_idx', 0)
                kept_movie_time = movie_times[kept_movie_idx] if kept_movie_idx < len(movie_times) else 0
                
                if abs(movie_time - kept_movie_time) < time_window:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_indices.add(idx)
        
        return sorted(list(kept_indices))
    
    # 私有方法
    
    def _backward_search(
        self,
        target_feature: np.ndarray,
        feature_matrix: np.ndarray,
        exclude_idx: Optional[int] = None
    ) -> Tuple[float, float, int]:
        """后向搜索：找到与目标特征最相似的特征
        
        Args:
            target_feature: 目标特征向量
            feature_matrix: 特征矩阵
            exclude_idx: 排除的索引（避免自匹配）
            
        Returns:
            (置信度, 距离, 最佳索引)
        """
        # 计算所有距离
        distances = np.zeros(feature_matrix.shape[0])
        for i, feature in enumerate(feature_matrix):
            if exclude_idx is not None and i == exclude_idx:
                distances[i] = np.inf
            else:
                distances[i] = self._compute_feature_distance(target_feature, feature)
        
        best_idx = np.argmin(distances)
        best_distance = float(distances[best_idx])
        
        # 转换为置信度 (距离小 -> 置信度高)
        confidence = np.exp(-best_distance)
        
        return confidence, best_distance, int(best_idx)
    
    @staticmethod
    def _compute_feature_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算两个特征向量的距离（欧氏距离）
        
        Args:
            feat1: 特征向量 1
            feat2: 特征向量 2
            
        Returns:
            距离值
        """
        return float(np.linalg.norm(feat1 - feat2))
    
    def _compute_validation_score(
        self,
        forward_conf: float,
        backward_conf: float,
        time_cons: float,
        distance_cons: float
    ) -> float:
        """计算综合验证得分
        
        Args:
            forward_conf: 前向置信度
            backward_conf: 后向置信度
            time_cons: 时间一致性
            distance_cons: 距离一致性
            
        Returns:
            综合得分 (0-1)
        """
        # 权重设置
        weights = {
            'forward': 0.25,
            'backward': 0.25,
            'time': 0.25,
            'distance': 0.25
        }
        
        score = (
            weights['forward'] * forward_conf +
            weights['backward'] * backward_conf +
            weights['time'] * time_cons +
            weights['distance'] * distance_cons
        )
        
        return float(score)
    
    @staticmethod
    def confidence_interval(
        validation_results: List[ValidationResult]
    ) -> Tuple[float, float]:
        """计算验证得分的置信区间
        
        Args:
            validation_results: 验证结果列表
            
        Returns:
            (下界, 上界)
        """
        if not validation_results:
            return 0.0, 0.0
        
        scores = np.array([r.validation_score for r in validation_results])
        mean = np.mean(scores)
        std = np.std(scores)
        
        # 使用 68-95-99.7 规则（±1σ）
        return float(mean - std), float(mean + std)
