"""高性能哈希索引模块

使用多种索引结构加速哈希检索：
1. FAISS IVF 索引 - 大规模向量检索
2. BK-Tree - 汉明距离检索
3. LSH (局部敏感哈希) - 近似最近邻

在保持精度的前提下大幅提升检索速度。
"""
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger
import threading
from collections import defaultdict

# 尝试导入 FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，将使用纯 Python 实现")


@dataclass
class IndexConfig:
    """索引配置"""
    # IVF 配置
    use_ivf: bool = True              # 使用 IVF 索引
    nlist: int = 100                  # IVF 聚类中心数
    nprobe: int = 10                  # 搜索时检查的聚类数

    # 索引类型
    index_type: str = "flat"          # flat, ivf, hnsw
    use_gpu: bool = False             # GPU 加速

    # BK-Tree 配置
    use_bktree: bool = True           # 使用 BK-Tree
    bktree_threshold: int = 10        # BK-Tree 搜索阈值

    # 缓存配置
    cache_size: int = 10000           # 缓存大小


class BKTreeNode:
    """BK-Tree 节点

    BK-Tree 适合汉明距离等离散距离度量的近似搜索
    """

    def __init__(self, hash_str: str, frame_idx: int, time_sec: float):
        self.hash_str = hash_str
        self.frame_idx = frame_idx
        self.time_sec = time_sec
        self.children: Dict[int, 'BKTreeNode'] = {}


class BKTree:
    """BK-Tree 索引

    用于快速汉明距离检索，时间复杂度 O(n^0.6) 平均情况
    """

    def __init__(self):
        self.root: Optional[BKTreeNode] = None
        self.size = 0
        self._lock = threading.Lock()

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """计算汉明距离"""
        if len(hash1) != len(hash2):
            return 64  # 最大距离

        # 转换为整数计算
        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            xor = int1 ^ int2
            return bin(xor).count('1')
        except ValueError:
            return 64

    def insert(self, hash_str: str, frame_idx: int, time_sec: float):
        """插入节点"""
        with self._lock:
            if self.root is None:
                self.root = BKTreeNode(hash_str, frame_idx, time_sec)
                self.size = 1
                return

            self._insert_node(self.root, hash_str, frame_idx, time_sec)
            self.size += 1

    def _insert_node(
        self,
        node: BKTreeNode,
        hash_str: str,
        frame_idx: int,
        time_sec: float
    ):
        """递归插入"""
        d = self.hamming_distance(node.hash_str, hash_str)

        if d in node.children:
            self._insert_node(node.children[d], hash_str, frame_idx, time_sec)
        else:
            node.children[d] = BKTreeNode(hash_str, frame_idx, time_sec)

    def search(
        self,
        hash_str: str,
        threshold: int
    ) -> List[Tuple[str, int, float, int]]:
        """搜索汉明距离在阈值内的节点

        Args:
            hash_str: 查询哈希
            threshold: 距离阈值

        Returns:
            [(hash, frame_idx, time_sec, distance), ...]
        """
        if self.root is None:
            return []

        results = []
        self._search_node(self.root, hash_str, threshold, results)
        return sorted(results, key=lambda x: x[3])  # 按距离排序

    def _search_node(
        self,
        node: BKTreeNode,
        hash_str: str,
        threshold: int,
        results: List
    ):
        """递归搜索"""
        d = self.hamming_distance(node.hash_str, hash_str)

        if d <= threshold:
            results.append((node.hash_str, node.frame_idx, node.time_sec, d))

        # 只搜索距离在 [d-threshold, d+threshold] 范围内的子节点
        for child_d in range(max(0, d - threshold), d + threshold + 1):
            if child_d in node.children:
                self._search_node(node.children[child_d], hash_str, threshold, results)

    def clear(self):
        """清空树"""
        with self._lock:
            self.root = None
            self.size = 0


class HashVectorizer:
    """哈希向量化器

    将十六进制哈希字符串转换为二进制向量，用于 FAISS 索引
    """

    def __init__(self, hash_bits: int = 64):
        """初始化

        Args:
            hash_bits: 哈希位数（默认 64 位 = 16 字符十六进制）
        """
        self.hash_bits = hash_bits

    def hash_to_vector(self, hash_str: str) -> np.ndarray:
        """将哈希转换为二进制向量"""
        try:
            hash_int = int(hash_str, 16)
            bits = [(hash_int >> i) & 1 for i in range(self.hash_bits)]
            return np.array(bits, dtype=np.float32)
        except ValueError:
            return np.zeros(self.hash_bits, dtype=np.float32)

    def vector_to_hash(self, vector: np.ndarray) -> str:
        """将向量转回哈希（用于调试）"""
        hash_int = sum(int(v > 0.5) << i for i, v in enumerate(vector))
        return format(hash_int, f'0{self.hash_bits // 4}x')

    def batch_hash_to_vectors(self, hashes: List[str]) -> np.ndarray:
        """批量转换"""
        vectors = [self.hash_to_vector(h) for h in hashes]
        return np.array(vectors, dtype=np.float32)


class FAISSIndex:
    """FAISS 索引封装

    支持多种索引类型：
    - Flat: 精确搜索，适合小规模数据
    - IVF: 倒排索引，适合大规模数据
    - HNSW: 图索引，适合高维数据
    """

    def __init__(self, config: IndexConfig = None):
        self.config = config or IndexConfig()
        self.vectorizer = HashVectorizer()
        self.index: Optional[Any] = None
        self.frame_indices: List[int] = []      # frame_idx 映射
        self.frame_times: List[float] = []      # time_sec 映射
        self.hash_strings: List[str] = []       # 原始哈希字符串
        self._trained = False
        self._lock = threading.Lock()

    def build(self, hashes: List[str], frame_indices: List[int], times: List[float]):
        """构建索引

        Args:
            hashes: 哈希字符串列表
            frame_indices: 帧索引列表
            times: 时间戳列表
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，跳过索引构建")
            return

        if len(hashes) == 0:
            return

        with self._lock:
            # 向量化
            vectors = self.vectorizer.batch_hash_to_vectors(hashes)
            dimension = vectors.shape[1]
            n_vectors = vectors.shape[0]

            # 选择索引类型
            if self.config.index_type == "flat" or n_vectors < 1000:
                # 小规模数据使用 Flat 索引
                self.index = faiss.IndexFlatL2(dimension)
            elif self.config.index_type == "ivf":
                # IVF 索引
                nlist = min(self.config.nlist, n_vectors // 10)
                nlist = max(nlist, 1)
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.index.train(vectors)
                self.index.nprobe = self.config.nprobe
            elif self.config.index_type == "hnsw":
                # HNSW 图索引
                self.index = faiss.IndexHNSWFlat(dimension, 32)

            # 添加向量
            self.index.add(vectors)

            # 保存映射
            self.frame_indices = list(frame_indices)
            self.frame_times = list(times)
            self.hash_strings = list(hashes)
            self._trained = True

            logger.debug(
                f"FAISS 索引构建完成: {n_vectors} 向量, "
                f"类型={self.config.index_type}"
            )

    def search(
        self,
        query_hashes: List[str],
        k: int = 10
    ) -> List[List[Tuple[int, float, float]]]:
        """搜索最近邻

        Args:
            query_hashes: 查询哈希列表
            k: 返回的最近邻数量

        Returns:
            [[(frame_idx, time_sec, distance), ...], ...]
        """
        if not FAISS_AVAILABLE or self.index is None:
            return [[] for _ in query_hashes]

        with self._lock:
            # 向量化查询
            query_vectors = self.vectorizer.batch_hash_to_vectors(query_hashes)

            # 搜索
            distances, indices = self.index.search(query_vectors, k)

            # 转换结果
            results = []
            for i in range(len(query_hashes)):
                matches = []
                for j in range(k):
                    idx = indices[i, j]
                    if idx >= 0 and idx < len(self.frame_indices):
                        matches.append((
                            self.frame_indices[idx],
                            self.frame_times[idx],
                            float(distances[i, j])
                        ))
                results.append(matches)

            return results

    def search_radius(
        self,
        query_hash: str,
        radius: float
    ) -> List[Tuple[int, float, float]]:
        """范围搜索

        Args:
            query_hash: 查询哈希
            radius: 搜索半径（L2 距离）

        Returns:
            [(frame_idx, time_sec, distance), ...]
        """
        if not FAISS_AVAILABLE or self.index is None:
            return []

        with self._lock:
            query_vector = self.vectorizer.hash_to_vector(query_hash).reshape(1, -1)

            # FAISS range_search
            lims, distances, indices = self.index.range_search(query_vector, radius)

            results = []
            for i in range(lims[1]):
                idx = indices[i]
                if idx >= 0 and idx < len(self.frame_indices):
                    results.append((
                        self.frame_indices[idx],
                        self.frame_times[idx],
                        float(distances[i])
                    ))

            return sorted(results, key=lambda x: x[2])

    def clear(self):
        """清空索引"""
        with self._lock:
            self.index = None
            self.frame_indices = []
            self.frame_times = []
            self.hash_strings = []
            self._trained = False


class HybridHashIndex:
    """混合哈希索引

    结合 FAISS 和 BK-Tree 的优势：
    - FAISS 用于快速候选检索
    - BK-Tree 用于精确汉明距离验证
    """

    def __init__(self, config: IndexConfig = None):
        self.config = config or IndexConfig()
        self.faiss_index = FAISSIndex(config)
        self.bktree = BKTree() if self.config.use_bktree else None
        self._hash_to_info: Dict[str, Tuple[int, float]] = {}  # hash -> (frame_idx, time)
        self._lock = threading.Lock()

    def build(
        self,
        hashes: List[str],
        frame_indices: List[int],
        times: List[float]
    ):
        """构建混合索引"""
        with self._lock:
            # 构建 FAISS 索引
            if FAISS_AVAILABLE:
                self.faiss_index.build(hashes, frame_indices, times)

            # 构建 BK-Tree
            if self.bktree:
                self.bktree.clear()
                for h, idx, t in zip(hashes, frame_indices, times):
                    self.bktree.insert(h, idx, t)
                    self._hash_to_info[h] = (idx, t)

            logger.info(f"混合索引构建完成: {len(hashes)} 条目")

    def search(
        self,
        query_hash: str,
        threshold: int = 8,
        max_results: int = 50
    ) -> List[Tuple[int, float, int]]:
        """搜索相似哈希

        Args:
            query_hash: 查询哈希
            threshold: 汉明距离阈值
            max_results: 最大返回数量

        Returns:
            [(frame_idx, time_sec, hamming_distance), ...]
        """
        # 优先使用 BK-Tree（精确汉明距离）
        if self.bktree and self.bktree.size > 0:
            results = self.bktree.search(query_hash, threshold)
            return [(r[1], r[2], r[3]) for r in results[:max_results]]

        # 回退到 FAISS
        if FAISS_AVAILABLE and self.faiss_index.index is not None:
            # FAISS 返回 L2 距离，需要转换为汉明距离估计
            faiss_results = self.faiss_index.search([query_hash], k=max_results * 2)
            if faiss_results and faiss_results[0]:
                results = []
                for frame_idx, time_sec, l2_dist in faiss_results[0]:
                    # L2 距离到汉明距离的近似转换
                    # 二进制向量的 L2 距离平方 = 汉明距离
                    hamming_est = int(l2_dist)
                    if hamming_est <= threshold:
                        results.append((frame_idx, time_sec, hamming_est))
                return results[:max_results]

        return []

    def batch_search(
        self,
        query_hashes: List[str],
        threshold: int = 8,
        max_results_per_query: int = 20
    ) -> List[List[Tuple[int, float, int]]]:
        """批量搜索

        Args:
            query_hashes: 查询哈希列表
            threshold: 汉明距离阈值
            max_results_per_query: 每个查询的最大返回数量

        Returns:
            [[（frame_idx, time_sec, distance), ...], ...]
        """
        return [
            self.search(h, threshold, max_results_per_query)
            for h in query_hashes
        ]

    def clear(self):
        """清空索引"""
        with self._lock:
            self.faiss_index.clear()
            if self.bktree:
                self.bktree.clear()
            self._hash_to_info.clear()

    @property
    def size(self) -> int:
        """索引大小"""
        if self.bktree:
            return self.bktree.size
        return len(self.faiss_index.frame_indices)


class LSHIndex:
    """局部敏感哈希索引

    适合近似最近邻搜索，内存效率高
    """

    def __init__(self, num_tables: int = 10, hash_size: int = 8):
        """初始化

        Args:
            num_tables: 哈希表数量（越多越精确，但内存越大）
            hash_size: 每个表的哈希位数
        """
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.tables: List[Dict[int, List[Tuple[str, int, float]]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]
        self.hash_bits = 64  # 输入哈希位数
        self._masks: List[List[int]] = []
        self._init_masks()

    def _init_masks(self):
        """初始化随机投影掩码"""
        np.random.seed(42)  # 可重现
        for _ in range(self.num_tables):
            # 每个表选择 hash_size 个随机位
            mask_bits = np.random.choice(
                self.hash_bits,
                size=self.hash_size,
                replace=False
            ).tolist()
            self._masks.append(mask_bits)

    def _compute_lsh(self, hash_str: str, table_idx: int) -> int:
        """计算 LSH 值"""
        try:
            hash_int = int(hash_str, 16)
        except ValueError:
            return 0

        mask = self._masks[table_idx]
        lsh_value = 0
        for i, bit_pos in enumerate(mask):
            if (hash_int >> bit_pos) & 1:
                lsh_value |= (1 << i)
        return lsh_value

    def insert(self, hash_str: str, frame_idx: int, time_sec: float):
        """插入"""
        for i, table in enumerate(self.tables):
            lsh = self._compute_lsh(hash_str, i)
            table[lsh].append((hash_str, frame_idx, time_sec))

    def search(
        self,
        query_hash: str,
        threshold: int = 8
    ) -> List[Tuple[int, float, int]]:
        """搜索"""
        candidates = set()

        # 从所有表收集候选
        for i, table in enumerate(self.tables):
            lsh = self._compute_lsh(query_hash, i)
            for item in table.get(lsh, []):
                candidates.add(item)

        # 精确验证汉明距离
        results = []
        for hash_str, frame_idx, time_sec in candidates:
            dist = BKTree.hamming_distance(query_hash, hash_str)
            if dist <= threshold:
                results.append((frame_idx, time_sec, dist))

        return sorted(results, key=lambda x: x[2])

    def clear(self):
        """清空"""
        for table in self.tables:
            table.clear()


# 便捷函数
def create_hash_index(config: IndexConfig = None) -> HybridHashIndex:
    """创建混合哈希索引"""
    return HybridHashIndex(config)


def create_bktree() -> BKTree:
    """创建 BK-Tree 索引"""
    return BKTree()


def create_lsh_index(num_tables: int = 10) -> LSHIndex:
    """创建 LSH 索引"""
    return LSHIndex(num_tables=num_tables)
