"""Smith-Waterman 序列比对器

用于将解说视频帧序列与原电影帧序列做局部比对，
容忍帧丢失、重复、插入（解说视频的剪辑操作）。
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AlignmentResult:
    """比对结果"""
    movie_start: float       # 电影起始时间
    movie_end: float         # 电影结束时间
    score: float             # 比对得分
    confidence: float        # 置信度 (0-1)
    aligned_pairs: list      # 匹配的帧对 [(query_idx, movie_time, distance), ...]
    alignment_path: list     # 比对路径 (用于调试)


class SmithWatermanAligner:
    """Smith-Waterman 序列比对器

    将解说视频帧序列与原电影帧索引进行局部序列比对，
    找到最佳匹配位置，同时容忍剪辑操作带来的帧变化。
    """

    def __init__(
        self,
        match_score: float = 5.0,        # 完全匹配得分 (距离<=5)
        similar_score: float = 2.0,      # 高度相似得分 (距离5-10)
        mismatch_penalty: float = -3.0,  # 不匹配惩罚 (距离>10)
        gap_open: float = -10.0,         # 开启空位罚分
        gap_extend: float = -0.5,        # 延续空位罚分
        strict_threshold: int = 5,       # 严格匹配阈值
        loose_threshold: int = 10        # 宽松匹配阈值
    ):
        """初始化比对器

        Args:
            match_score: 完全匹配(距离<=strict_threshold)得分
            similar_score: 高度相似(距离strict_threshold-loose_threshold)得分
            mismatch_penalty: 不匹配(距离>loose_threshold)惩罚
            gap_open: 开启空位的罚分
            gap_extend: 延续空位的罚分
            strict_threshold: 严格匹配的距离阈值
            loose_threshold: 宽松匹配的距离阈值
        """
        self.match_score = match_score
        self.similar_score = similar_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.strict_threshold = strict_threshold
        self.loose_threshold = loose_threshold

    def _compute_score(self, distance: int) -> float:
        """根据汉明距离计算得分"""
        if distance <= self.strict_threshold:
            return self.match_score
        elif distance <= self.loose_threshold:
            return self.similar_score
        else:
            return self.mismatch_penalty

    def align(
        self,
        query_hashes: list[str],
        query_times: list[float],
        movie_hashes: np.ndarray,
        movie_times: list[float],
        search_window: Optional[tuple[float, float]] = None
    ) -> Optional[AlignmentResult]:
        """执行序列比对

        Args:
            query_hashes: 查询帧的pHash列表 (解说视频)
            query_times: 查询帧的时间戳列表
            movie_hashes: 电影帧的pHash数组 (uint8, shape: [N, 8])
            movie_times: 电影帧的时间戳列表
            search_window: 搜索窗口 (start_time, end_time)，None表示全局搜索

        Returns:
            AlignmentResult 或 None (如果没找到有效比对)
        """
        if not query_hashes or len(movie_times) == 0:
            return None

        # 将查询哈希转换为numpy数组
        query_arrays = []
        for h in query_hashes:
            try:
                hash_bytes = bytes.fromhex(h)
                query_arrays.append(np.frombuffer(hash_bytes, dtype=np.uint8).copy())
            except (ValueError, TypeError):
                continue

        if not query_arrays:
            return None

        query_array = np.array(query_arrays, dtype=np.uint8)  # [M, 8]

        # 应用搜索窗口过滤
        if search_window is not None:
            window_start, window_end = search_window
            window_mask = np.array([
                window_start <= t <= window_end for t in movie_times
            ])
            if not window_mask.any():
                return None

            movie_indices = np.where(window_mask)[0]
            filtered_movie_hashes = movie_hashes[window_mask]
            filtered_movie_times = [movie_times[i] for i in movie_indices]
        else:
            filtered_movie_hashes = movie_hashes
            filtered_movie_times = movie_times
            movie_indices = np.arange(len(movie_times))

        m = len(query_array)      # 查询序列长度
        n = len(filtered_movie_hashes)  # 电影序列长度

        if n == 0:
            return None

        # 预计算所有帧对之间的汉明距离
        # 使用向量化计算加速
        distances = self._compute_hamming_matrix(query_array, filtered_movie_hashes)

        # Smith-Waterman 动态规划
        # H[i][j] 表示 query[0:i] 和 movie[0:j] 的最佳局部比对得分
        H = np.zeros((m + 1, n + 1), dtype=np.float32)

        # 回溯指针: 0=终止, 1=对角线(匹配), 2=上方(query空位), 3=左方(movie空位)
        traceback = np.zeros((m + 1, n + 1), dtype=np.int8)

        # 空位状态追踪 (用于仿射空位罚分)
        in_query_gap = np.zeros(n + 1, dtype=bool)
        in_movie_gap = np.zeros(m + 1, dtype=bool)

        max_score = 0.0
        max_pos = (0, 0)

        for i in range(1, m + 1):
            prev_in_movie_gap = False
            for j in range(1, n + 1):
                # 对角线: 匹配/不匹配
                score = self._compute_score(distances[i-1, j-1])
                diag = H[i-1, j-1] + score

                # 上方: query 空位 (电影跳过帧)
                if in_query_gap[j]:
                    up = H[i-1, j] + self.gap_extend
                else:
                    up = H[i-1, j] + self.gap_open

                # 左方: movie 空位 (解说跳过帧)
                if prev_in_movie_gap:
                    left = H[i, j-1] + self.gap_extend
                else:
                    left = H[i, j-1] + self.gap_open

                # 取最大值 (局部比对允许从 0 重新开始)
                best = max(0, diag, up, left)
                H[i, j] = best

                # 更新空位状态
                if best == up:
                    in_query_gap[j] = True
                    traceback[i, j] = 2
                elif best == left:
                    prev_in_movie_gap = True
                    traceback[i, j] = 3
                elif best == diag:
                    in_query_gap[j] = False
                    prev_in_movie_gap = False
                    traceback[i, j] = 1
                else:  # best == 0
                    in_query_gap[j] = False
                    prev_in_movie_gap = False
                    traceback[i, j] = 0

                if best > max_score:
                    max_score = best
                    max_pos = (i, j)

        if max_score <= 0:
            return None

        # 回溯找到比对路径
        aligned_pairs = []
        alignment_path = []
        i, j = max_pos

        while i > 0 and j > 0 and H[i, j] > 0:
            direction = traceback[i, j]
            if direction == 1:  # 对角线 (匹配)
                query_idx = i - 1
                movie_idx = j - 1
                dist = distances[query_idx, movie_idx]
                movie_time = filtered_movie_times[movie_idx]
                aligned_pairs.append((query_idx, movie_time, int(dist)))
                alignment_path.append(('M', query_idx, movie_idx))
                i -= 1
                j -= 1
            elif direction == 2:  # 上方 (query 空位)
                alignment_path.append(('Q', i-1, j-1))
                i -= 1
            elif direction == 3:  # 左方 (movie 空位)
                alignment_path.append(('D', i-1, j-1))
                j -= 1
            else:
                break

        aligned_pairs.reverse()
        alignment_path.reverse()

        if not aligned_pairs:
            return None

        # 计算结果
        movie_times_matched = [t for _, t, _ in aligned_pairs]
        movie_start = min(movie_times_matched)
        movie_end = max(movie_times_matched)

        # 计算置信度
        # 基于: 匹配数量占比、平均距离、得分
        match_ratio = len(aligned_pairs) / len(query_hashes)
        avg_distance = np.mean([d for _, _, d in aligned_pairs])

        # 距离分数: 距离越小越好
        distance_score = max(0, 1.0 - avg_distance / 20.0)

        # 归一化得分
        max_possible_score = len(query_hashes) * self.match_score
        normalized_score = max_score / max_possible_score if max_possible_score > 0 else 0

        # 综合置信度
        confidence = (
            match_ratio * 0.3 +           # 匹配数量
            distance_score * 0.4 +        # 距离质量
            normalized_score * 0.3        # 比对得分
        )
        confidence = min(1.0, max(0.0, confidence))

        logger.debug(
            f"序列比对完成: score={max_score:.1f}, "
            f"matches={len(aligned_pairs)}/{len(query_hashes)}, "
            f"avg_dist={avg_distance:.1f}, confidence={confidence:.3f}, "
            f"range=[{movie_start:.1f}s-{movie_end:.1f}s]"
        )

        return AlignmentResult(
            movie_start=movie_start,
            movie_end=movie_end,
            score=max_score,
            confidence=confidence,
            aligned_pairs=aligned_pairs,
            alignment_path=alignment_path
        )

    def _compute_hamming_matrix(
        self,
        query_array: np.ndarray,
        movie_array: np.ndarray
    ) -> np.ndarray:
        """计算查询序列和电影序列之间的汉明距离矩阵

        Args:
            query_array: [M, 8] uint8 数组
            movie_array: [N, 8] uint8 数组

        Returns:
            [M, N] 距离矩阵
        """
        m = len(query_array)
        n = len(movie_array)

        # 使用广播计算 XOR
        # query_array[:, np.newaxis, :] -> [M, 1, 8]
        # movie_array[np.newaxis, :, :] -> [1, N, 8]
        # XOR result -> [M, N, 8]
        xor_result = np.bitwise_xor(
            query_array[:, np.newaxis, :],
            movie_array[np.newaxis, :, :]
        )

        # 查表法统计每字节的 popcount
        popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

        # 对每个字节查表并求和
        distances = np.zeros((m, n), dtype=np.int32)
        for byte_idx in range(xor_result.shape[2]):
            distances += popcount_table[xor_result[:, :, byte_idx]]

        return distances

    def find_best_alignment(
        self,
        query_hashes: list[str],
        query_times: list[float],
        movie_hashes: np.ndarray,
        movie_times: list[float],
        candidate_windows: list[tuple[float, float]] = None,
        min_confidence: float = 0.5
    ) -> Optional[AlignmentResult]:
        """在多个候选窗口中找到最佳比对

        Args:
            query_hashes: 查询帧的pHash列表
            query_times: 查询帧的时间戳
            movie_hashes: 电影帧的pHash数组
            movie_times: 电影帧的时间戳
            candidate_windows: 候选搜索窗口列表，None表示全局搜索
            min_confidence: 最低置信度阈值

        Returns:
            最佳 AlignmentResult 或 None
        """
        if candidate_windows is None:
            # 全局搜索
            result = self.align(
                query_hashes, query_times,
                movie_hashes, movie_times,
                search_window=None
            )
            if result and result.confidence >= min_confidence:
                return result
            return None

        best_result = None

        for window in candidate_windows:
            result = self.align(
                query_hashes, query_times,
                movie_hashes, movie_times,
                search_window=window
            )

            if result is None:
                continue

            if result.confidence < min_confidence:
                continue

            if best_result is None or result.score > best_result.score:
                best_result = result

        return best_result
