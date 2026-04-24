"""音频匹配模块

使用 MFCC 特征 + FAISS 索引实现音频相似度匹配
"""

import asyncio
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

from config import settings
from .audio_extractor import AudioExtractor

# 尝试导入 librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa 未安装，音频匹配功能不可用。请运行: pip install librosa")

# 尝试导入 FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，将使用暴力搜索")


class AudioMatcher:
    """音频匹配器

    使用 MFCC 特征进行音频相似度匹配：
    1. 构建索引：滑动窗口提取 MFCC 特征，构建 FAISS 索引
    2. 匹配：提取查询片段的 MFCC 特征，搜索最相似的位置
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        hop_length: int = 512
    ):
        """初始化

        Args:
            sample_rate: 采样率
            n_mfcc: MFCC 系数数量
            hop_length: 帧移
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa 未安装，无法使用音频匹配功能")

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

        # 索引数据
        self._faiss_index = None
        self._feature_times: list[float] = []
        self._feature_dim = n_mfcc * 2  # mean + std

        # 音频提取器
        self._audio_extractor = AudioExtractor()

    async def build_index(
        self,
        video_path: str,
        window_sec: float = 2.0,
        step_sec: float = 0.5,
        cache_path: Optional[Path] = None
    ):
        """构建音频指纹索引

        Args:
            video_path: 视频文件路径
            window_sec: 窗口大小（秒）
            step_sec: 滑动步长（秒）
            cache_path: 缓存文件路径
        """
        video_path = str(video_path)

        # 检查缓存
        if cache_path and cache_path.exists():
            logger.info(f"从缓存加载音频索引: {cache_path}")
            await self._load_cache(cache_path)
            return

        logger.info(f"开始构建音频索引: {video_path}")

        # 提取音频
        audio_path = await self._audio_extractor.extract_full(
            video_path,
            output_dir=settings.temp_dir,
            sample_rate=self.sample_rate,
            mono=True
        )

        def _build():
            # 加载音频
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            duration = len(y) / sr
            logger.info(f"音频时长: {duration:.1f} 秒")

            # 滑动窗口提取特征
            window_samples = int(window_sec * sr)
            step_samples = int(step_sec * sr)

            features = []
            times = []
            
            # 确保有足够的样本用于处理
            if len(y) < window_samples:
                logger.warning(f"音频太短 ({len(y)} < {window_samples} samples), 使用整个音频作为单个特征")
                window = y[:len(y)]
                mfcc = librosa.feature.mfcc(
                    y=window,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length
                )
                feature = np.concatenate([
                    mfcc.mean(axis=1),
                    mfcc.std(axis=1)
                ])
                features.append(feature)
                times.append(0)
            else:
                total_windows = (len(y) - window_samples) // step_samples + 1
                processed = 0

                for start in range(0, len(y) - window_samples, step_samples):
                    window = y[start:start + window_samples]

                    # 计算 MFCC
                    mfcc = librosa.feature.mfcc(
                        y=window,
                        sr=sr,
                        n_mfcc=self.n_mfcc,
                        hop_length=self.hop_length
                    )

                    # 聚合为单一向量：均值 + 标准差
                    feature = np.concatenate([
                        mfcc.mean(axis=1),
                        mfcc.std(axis=1)
                    ])
                    features.append(feature)
                    times.append(start / sr)

                processed += 1
                if processed % 500 == 0:
                    logger.debug(f"音频索引构建进度: {100 * processed / total_windows:.1f}%")

            return features, times

        features, self._feature_times = await asyncio.to_thread(_build)

        # 构建 FAISS 索引
        if features:
            features_array = np.array(features).astype('float32')

            # L2 归一化，使内积等于余弦相似度
            if FAISS_AVAILABLE:
                faiss.normalize_L2(features_array)
                # 使用内积索引（归一化后等价于余弦相似度）
                self._faiss_index = faiss.IndexFlatIP(features_array.shape[1])
                self._faiss_index.add(features_array)
                logger.info(f"FAISS 索引构建完成: {len(features)} 个特征向量")
            else:
                # 使用 numpy 数组
                features_array_norm = features_array / (np.linalg.norm(features_array, axis=1, keepdims=True) + 1e-10)
                self._faiss_index = features_array_norm
                logger.info(f"Numpy 索引构建完成: {len(features)} 个特征向量")

        # 保存缓存
        if cache_path:
            await self._save_cache(cache_path)

    async def match_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        top_k: int = 10
    ) -> Optional[dict]:
        """匹配音频片段

        Args:
            video_path: 视频文件路径
            start_time: 起始时间（秒）
            end_time: 结束时间（秒）
            top_k: 返回前 k 个候选

        Returns:
            匹配结果 {start, end, confidence} 或 None
        """
        if self._faiss_index is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        # 提取查询片段的音频
        audio_path = await self._audio_extractor.extract_segment(
            video_path,
            start_time,
            end_time,
            output_dir=settings.temp_dir,
            sample_rate=self.sample_rate
        )

        def _match():
            # 加载音频
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

            if len(y) < self.hop_length * 2:
                logger.warning(f"音频片段太短: {len(y)} 采样点")
                return None

            # 计算 MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length
            )

            # 聚合为单一向量
            query_feature = np.concatenate([
                mfcc.mean(axis=1),
                mfcc.std(axis=1)
            ]).reshape(1, -1).astype('float32')

            if FAISS_AVAILABLE:
                faiss.normalize_L2(query_feature)
                distances, indices = self._faiss_index.search(query_feature, k=top_k)

                if len(indices[0]) == 0 or indices[0][0] < 0:
                    return None

                # 内积值越大越相似（范围 -1 到 1）
                best_idx = indices[0][0]
                best_confidence = float(distances[0][0])

                # 转换为 0-1 范围
                best_confidence = (best_confidence + 1) / 2

            else:
                # 暴力搜索：余弦相似度
                query_norm_val = np.linalg.norm(query_feature)
                query_norm = query_feature / (query_norm_val + 1e-10)
                index_norms = np.linalg.norm(self._faiss_index, axis=1, keepdims=True)
                index_norm = self._faiss_index / (index_norms + 1e-10)
                similarities = np.dot(index_norm, query_norm.T).flatten()

                best_idx = np.argmax(similarities)
                best_confidence = float(similarities[best_idx])

                # 转换为 0-1 范围
                best_confidence = (best_confidence + 1) / 2

            if best_idx < 0 or best_idx >= len(self._feature_times):
                return None

            best_start = self._feature_times[best_idx]
            segment_duration = end_time - start_time

            return {
                "start": best_start,
                "end": best_start + segment_duration,
                "confidence": best_confidence
            }

        return await asyncio.to_thread(_match)

    async def match_full_audio(
        self,
        query_audio_path: str,
        window_sec: float = 2.0,
        step_sec: float = 0.5,
        threshold: float = 0.6
    ) -> list[tuple[float, float, float]]:
        """匹配完整音频文件

        对查询音频按窗口滑动匹配，返回所有匹配位置

        Args:
            query_audio_path: 查询音频路径
            window_sec: 窗口大小（秒）
            step_sec: 步长（秒）
            threshold: 置信度阈值

        Returns:
            [(query_time, movie_time, confidence), ...]
        """
        if self._faiss_index is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        def _match_full():
            y, sr = librosa.load(str(query_audio_path), sr=self.sample_rate, mono=True)

            window_samples = int(window_sec * sr)
            step_samples = int(step_sec * sr)

            results = []

            for start in range(0, len(y) - window_samples, step_samples):
                window = y[start:start + window_samples]
                query_time = start / sr

                # 计算 MFCC
                mfcc = librosa.feature.mfcc(
                    y=window,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length
                )

                query_feature = np.concatenate([
                    mfcc.mean(axis=1),
                    mfcc.std(axis=1)
                ]).reshape(1, -1).astype('float32')

                if FAISS_AVAILABLE:
                    faiss.normalize_L2(query_feature)
                    distances, indices = self._faiss_index.search(query_feature, k=1)

                    if len(indices[0]) > 0 and indices[0][0] >= 0:
                        idx = indices[0][0]
                        confidence = (float(distances[0][0]) + 1) / 2

                        if confidence >= threshold and idx < len(self._feature_times):
                            movie_time = self._feature_times[idx]
                            results.append((query_time, movie_time, confidence))
                else:
                    query_norm_val = np.linalg.norm(query_feature)
                    query_norm = query_feature / (query_norm_val + 1e-10)
                    index_norms = np.linalg.norm(self._faiss_index, axis=1, keepdims=True)
                    index_norm = self._faiss_index / (index_norms + 1e-10)
                    similarities = np.dot(index_norm, query_norm.T).flatten()

                    best_idx = np.argmax(similarities)
                    confidence = (float(similarities[best_idx]) + 1) / 2

                    if confidence >= threshold:
                        movie_time = self._feature_times[best_idx]
                        results.append((query_time, movie_time, confidence))

            return results

        return await asyncio.to_thread(_match_full)

    async def _save_cache(self, cache_path: Path):
        """保存索引缓存"""
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "feature_times": self._feature_times,
            "sample_rate": self.sample_rate,
            "n_mfcc": self.n_mfcc,
        }

        # 分开保存 FAISS 索引
        if FAISS_AVAILABLE and isinstance(self._faiss_index, faiss.Index):
            faiss_path = cache_path.with_suffix(".audio.faiss")
            faiss.write_index(self._faiss_index, str(faiss_path))
        elif isinstance(self._faiss_index, np.ndarray):
            data["features"] = self._faiss_index

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"音频索引缓存已保存: {cache_path}")

    async def _load_cache(self, cache_path: Path):
        """加载索引缓存"""
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        self._feature_times = data["feature_times"]
        self.sample_rate = data.get("sample_rate", self.sample_rate)
        self.n_mfcc = data.get("n_mfcc", self.n_mfcc)

        # 加载 FAISS 索引
        faiss_path = cache_path.with_suffix(".audio.faiss")
        if FAISS_AVAILABLE and faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
        elif "features" in data:
            self._faiss_index = data["features"]

        logger.info(f"音频索引缓存已加载: {len(self._feature_times)} 个时间点")

    def is_ready(self) -> bool:
        """检查索引是否已构建"""
        return self._faiss_index is not None and len(self._feature_times) > 0
