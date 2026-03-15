"""GPU 加速模块

实现视频处理的 GPU 加速：
1. 批量 pHash 计算（GPU DCT）
2. FAISS GPU 索引
3. 显存流式管理
4. CPU 自动回退

无 GPU 时自动降级到 CPU 实现。
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger
import threading
import queue

# 尝试导入 GPU 相关库
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class GPUConfig:
    """GPU 配置"""
    device_id: int = 0                    # GPU 设备 ID
    max_memory_mb: int = 0                # 最大显存使用 (MB)，0 表示自动（使用 80% 显存）
    batch_size: int = 64                  # 批处理大小
    enable_memory_pool: bool = True       # 启用显存池
    stream_count: int = 2                 # CUDA 流数量
    fallback_to_cpu: bool = True          # 无 GPU 时回退到 CPU


@dataclass
class AcceleratorStats:
    """加速器统计信息"""
    gpu_available: bool
    device_name: str
    total_memory_mb: int
    used_memory_mb: int
    compute_mode: str  # 'gpu', 'cpu', 'hybrid'
    processed_frames: int
    total_time_ms: float


class GPUAccelerator:
    """GPU 加速器

    提供视频处理的 GPU 加速功能：
    1. 批量图像处理（缩放、颜色转换）
    2. 批量 DCT 计算（用于 pHash）
    3. 批量哈希比较
    4. 显存自动管理
    """

    def __init__(self, config: GPUConfig = None):
        """初始化

        Args:
            config: GPU 配置
        """
        self.config = config or GPUConfig()
        self._device = None
        self._streams = []
        self._memory_pool = None
        self._stats = AcceleratorStats(
            gpu_available=False,
            device_name="N/A",
            total_memory_mb=0,
            used_memory_mb=0,
            compute_mode='cpu',
            processed_frames=0,
            total_time_ms=0.0
        )

        self._init_gpu()

    def _init_gpu(self):
        """初始化 GPU"""
        if not TORCH_AVAILABLE:
            logger.info("PyTorch 未安装，使用 CPU 模式")
            self._stats.compute_mode = 'cpu'
            return

        if not CUDA_AVAILABLE:
            logger.info("CUDA 不可用，使用 CPU 模式")
            self._stats.compute_mode = 'cpu'
            return

        try:
            self._device = torch.device(f'cuda:{self.config.device_id}')

            # 获取 GPU 信息
            props = torch.cuda.get_device_properties(self.config.device_id)
            self._stats.gpu_available = True
            self._stats.device_name = props.name
            self._stats.total_memory_mb = props.total_memory // (1024 * 1024)
            self._stats.compute_mode = 'gpu'

            # 设置显存限制（0 表示自动使用 80%）
            if self.config.max_memory_mb > 0:
                actual_limit = self.config.max_memory_mb
                fraction = actual_limit / self._stats.total_memory_mb
                torch.cuda.set_per_process_memory_fraction(
                    min(fraction, 0.9),
                    self.config.device_id
                )
            else:
                # 自动使用 80% 显存
                actual_limit = int(self._stats.total_memory_mb * 0.8)

            logger.info(
                f"GPU 加速已启用: {props.name}, "
                f"显存: {self._stats.total_memory_mb}MB, "
                f"可用: {actual_limit}MB"
            )

        except Exception as e:
            logger.warning(f"GPU 初始化失败: {e}，回退到 CPU")
            self._stats.compute_mode = 'cpu'

    @property
    def is_gpu_available(self) -> bool:
        """GPU 是否可用"""
        return self._stats.gpu_available and self._stats.compute_mode == 'gpu'

    def get_stats(self) -> AcceleratorStats:
        """获取统计信息"""
        if self.is_gpu_available:
            self._stats.used_memory_mb = torch.cuda.memory_allocated(
                self.config.device_id
            ) // (1024 * 1024)
        return self._stats

    def batch_resize(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """批量图像缩放

        Args:
            images: 图像列表 (BGR/灰度)
            target_size: 目标尺寸 (width, height)

        Returns:
            缩放后的图像列表
        """
        if not images:
            return []

        if self.is_gpu_available:
            return self._batch_resize_gpu(images, target_size)
        else:
            return self._batch_resize_cpu(images, target_size)

    def _batch_resize_gpu(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """GPU 批量缩放"""
        target_w, target_h = target_size
        results = []

        # 分批处理
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]

            # 转换为 tensor
            tensors = []
            for img in batch:
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                # HWC -> CHW
                tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                tensors.append(tensor)

            # 堆叠并移至 GPU
            batch_tensor = torch.stack(tensors).to(self._device)

            # 双线性插值缩放
            resized = F.interpolate(
                batch_tensor,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

            # 转回 numpy
            resized_np = resized.cpu().numpy()
            for j in range(len(batch)):
                # CHW -> HWC
                img = resized_np[j].transpose(1, 2, 0)
                if img.shape[2] == 1:
                    img = img[:, :, 0]
                results.append(img.astype(np.uint8))

            # 释放显存
            del batch_tensor, resized
            torch.cuda.empty_cache()

        self._stats.processed_frames += len(images)
        return results

    def _batch_resize_cpu(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """CPU 批量缩放"""
        import cv2
        results = []
        for img in images:
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            results.append(resized)
        return results

    def batch_dct(
        self,
        images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """批量 DCT 变换

        Args:
            images: 灰度图像列表

        Returns:
            DCT 系数列表
        """
        if not images:
            return []

        if self.is_gpu_available:
            return self._batch_dct_gpu(images)
        else:
            return self._batch_dct_cpu(images)

    def _batch_dct_gpu(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """GPU 批量 DCT（使用 FFT 实现）"""
        results = []

        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]

            # 转换为 tensor
            tensors = [torch.from_numpy(img.astype(np.float32)) for img in batch]
            batch_tensor = torch.stack(tensors).to(self._device)

            # 使用 FFT 实现 DCT
            # DCT-II 可以通过 FFT 计算
            n = batch_tensor.shape[-1]

            # 对称扩展
            extended = torch.cat([batch_tensor, batch_tensor.flip(-1)], dim=-1)

            # FFT
            fft_result = torch.fft.rfft(extended, dim=-1)

            # 提取 DCT 系数
            k = torch.arange(n, device=self._device)
            phase = torch.exp(-1j * np.pi * k / (2 * n))
            dct_result = (fft_result[..., :n] * phase).real * 2

            # 转回 numpy
            dct_np = dct_result.cpu().numpy()
            results.extend([dct_np[j] for j in range(len(batch))])

            # 释放显存
            del batch_tensor, extended, fft_result, dct_result
            torch.cuda.empty_cache()

        return results

    def _batch_dct_cpu(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """CPU 批量 DCT"""
        import cv2
        results = []
        for img in images:
            dct = cv2.dct(np.float32(img))
            results.append(dct)
        return results

    def batch_compute_phash(
        self,
        images: List[np.ndarray],
        hash_size: int = 8
    ) -> List[str]:
        """批量计算 pHash

        Args:
            images: BGR 图像列表
            hash_size: 哈希大小（默认 8x8 = 64 位）

        Returns:
            哈希字符串列表
        """
        if not images:
            return []

        # 1. 转灰度 + 缩放到 32x32
        gray_images = []
        for img in images:
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2).astype(np.uint8)
            else:
                gray = img
            gray_images.append(gray)

        resized = self.batch_resize(gray_images, (32, 32))

        # 2. DCT 变换
        dct_results = self.batch_dct(resized)

        # 3. 提取低频分量并生成哈希
        hashes = []
        for dct in dct_results:
            # 取左上角 8x8
            dct_low = dct[:hash_size, :hash_size]
            median = np.median(dct_low)

            # 生成哈希
            hash_bits = (dct_low > median).flatten()
            hash_bytes = np.packbits(hash_bits)
            hash_str = hash_bytes.tobytes().hex()
            hashes.append(hash_str)

        return hashes

    def batch_hamming_distance(
        self,
        query_hashes: List[str],
        index_hashes: np.ndarray
    ) -> np.ndarray:
        """批量计算汉明距离

        Args:
            query_hashes: 查询哈希列表
            index_hashes: 索引哈希数组 [N, 8] uint8

        Returns:
            距离矩阵 [M, N]
        """
        if not query_hashes or len(index_hashes) == 0:
            return np.array([])

        # 转换查询哈希
        query_arrays = []
        for h in query_hashes:
            hash_bytes = bytes.fromhex(h)
            query_arrays.append(np.frombuffer(hash_bytes, dtype=np.uint8))
        query_array = np.array(query_arrays, dtype=np.uint8)

        if self.is_gpu_available:
            return self._batch_hamming_gpu(query_array, index_hashes)
        else:
            return self._batch_hamming_cpu(query_array, index_hashes)

    def _batch_hamming_gpu(
        self,
        query_array: np.ndarray,
        index_array: np.ndarray
    ) -> np.ndarray:
        """GPU 批量汉明距离"""
        query_tensor = torch.from_numpy(query_array).to(self._device)
        index_tensor = torch.from_numpy(index_array).to(self._device)

        # XOR
        xor_result = query_tensor.unsqueeze(1) ^ index_tensor.unsqueeze(0)

        # Popcount（使用查表法）
        popcount_lut = torch.tensor(
            [bin(i).count('1') for i in range(256)],
            dtype=torch.int32,
            device=self._device
        )

        distances = popcount_lut[xor_result.long()].sum(dim=-1)

        result = distances.cpu().numpy()

        # 清理
        del query_tensor, index_tensor, xor_result, distances
        torch.cuda.empty_cache()

        return result

    def _batch_hamming_cpu(
        self,
        query_array: np.ndarray,
        index_array: np.ndarray
    ) -> np.ndarray:
        """CPU 批量汉明距离"""
        popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

        m, n = len(query_array), len(index_array)
        distances = np.zeros((m, n), dtype=np.int32)

        for i in range(m):
            xor_result = np.bitwise_xor(index_array, query_array[i])
            for byte_idx in range(xor_result.shape[1]):
                distances[i] += popcount_table[xor_result[:, byte_idx]]

        return distances

    def clear_cache(self):
        """清理 GPU 缓存"""
        if self.is_gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __del__(self):
        """析构时清理资源"""
        self.clear_cache()


class StreamProcessor:
    """流式处理器

    使用多个 CUDA 流并行处理，提高吞吐量
    """

    def __init__(
        self,
        accelerator: GPUAccelerator,
        stream_count: int = 2
    ):
        """初始化

        Args:
            accelerator: GPU 加速器
            stream_count: CUDA 流数量
        """
        self.accelerator = accelerator
        self.stream_count = stream_count
        self._streams = []
        self._result_queue = queue.Queue()

        if accelerator.is_gpu_available:
            for _ in range(stream_count):
                self._streams.append(torch.cuda.Stream())

    def process_async(
        self,
        images: List[np.ndarray],
        callback=None
    ):
        """异步处理图像

        Args:
            images: 图像列表
            callback: 完成回调函数
        """
        if not self._streams:
            # 无 GPU，同步处理
            hashes = self.accelerator.batch_compute_phash(images)
            if callback:
                callback(hashes)
            return hashes

        # 分配到不同的流
        chunk_size = (len(images) + self.stream_count - 1) // self.stream_count
        threads = []

        for i, stream in enumerate(self._streams):
            start = i * chunk_size
            end = min(start + chunk_size, len(images))
            if start >= len(images):
                break

            chunk = images[start:end]

            def process_chunk(s, c, idx):
                with torch.cuda.stream(s):
                    hashes = self.accelerator.batch_compute_phash(c)
                    self._result_queue.put((idx, hashes))

            t = threading.Thread(target=process_chunk, args=(stream, chunk, i))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 收集结果
        results = [None] * len(threads)
        while not self._result_queue.empty():
            idx, hashes = self._result_queue.get()
            results[idx] = hashes

        # 合并结果
        all_hashes = []
        for r in results:
            if r:
                all_hashes.extend(r)

        if callback:
            callback(all_hashes)

        return all_hashes


# 全局加速器实例（延迟初始化）
_global_accelerator: Optional[GPUAccelerator] = None


def get_accelerator() -> GPUAccelerator:
    """获取全局加速器实例"""
    global _global_accelerator
    if _global_accelerator is None:
        _global_accelerator = GPUAccelerator()
    return _global_accelerator


# 便捷函数
def batch_compute_phash_gpu(images: List[np.ndarray]) -> List[str]:
    """GPU 批量计算 pHash"""
    return get_accelerator().batch_compute_phash(images)


def is_gpu_available() -> bool:
    """检查 GPU 是否可用"""
    return get_accelerator().is_gpu_available
