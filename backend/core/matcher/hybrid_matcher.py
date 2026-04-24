"""混合匹配器

融合画面匹配和音频匹配，实现双重匹配
"""

import asyncio
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings
from core.video_processor.frame_matcher import FrameMatcher
from core.video_processor.scene_detector import SceneDetector
from core.audio_processor.audio_matcher import AudioMatcher
from .match_result import MatchResult, MatchConfig


class HybridMatcher:
    """混合匹配器

    使用画面 + 音频双重匹配策略：
    1. 对解说视频进行场景检测，切分为片段
    2. 对每个片段同时进行画面和音频匹配
    3. 加权融合匹配结果
    """

    def __init__(self, config: Optional[MatchConfig] = None):
        """初始化

        Args:
            config: 匹配配置
        """
        self.config = config or MatchConfig()

        self._frame_matcher = FrameMatcher()
        self._audio_matcher = AudioMatcher()
        self._scene_detector = SceneDetector()

        self._movie_path: Optional[str] = None
        self._indexes_built = False

    async def build_indexes(
        self,
        movie_path: str,
        cache_dir: Optional[Path] = None
    ):
        """构建原电影的帧索引和音频索引

        Args:
            movie_path: 原电影路径
            cache_dir: 缓存目录
        """
        self._movie_path = str(movie_path)

        if cache_dir is None:
            cache_dir = settings.temp_dir / "match_cache"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 生成缓存文件名
        movie_name = Path(movie_path).stem
        frame_cache = cache_dir / f"{movie_name}_frame.pkl"
        audio_cache = cache_dir / f"{movie_name}_audio.pkl"

        logger.info(f"开始构建索引: {movie_path}")

        # 并行构建帧索引和音频索引
        await asyncio.gather(
            self._frame_matcher.build_index(
                movie_path,
                sample_interval=self.config.frame_sample_interval,
                cache_path=frame_cache
            ),
            self._audio_matcher.build_index(
                movie_path,
                window_sec=self.config.audio_window_sec,
                step_sec=self.config.audio_step_sec,
                cache_path=audio_cache
            )
        )

        self._indexes_built = True
        logger.info("索引构建完成")

    async def match_video(
        self,
        narration_path: str,
        use_scene_detection: bool = True
    ) -> list[MatchResult]:
        """匹配解说视频到原电影

        Args:
            narration_path: 解说视频路径
            use_scene_detection: 是否使用场景检测自动切分

        Returns:
            匹配结果列表
        """
        if not self._indexes_built:
            raise RuntimeError("索引未构建，请先调用 build_indexes()")

        narration_path = str(narration_path)

        # 场景检测，获取片段列表
        if use_scene_detection:
            logger.info("正在进行场景检测...")
            # 临时设置场景检测阈值
            self._scene_detector.threshold = self.config.scene_threshold
            scene_list = await self._scene_detector.detect_scenes(
                narration_path,
                min_scene_duration=self.config.min_segment_duration
            )
            # 转换为 (start, end) 元组列表
            segments = [(s["start"], s["end"]) for s in scene_list]
            logger.info(f"检测到 {len(segments)} 个场景片段")
        else:
            # 不使用场景检测，获取视频总时长作为单个片段
            import cv2
            cap = cv2.VideoCapture(narration_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            segments = [(0, duration)]

        # 对每个片段进行匹配
        results = []
        for idx, (start, end) in enumerate(segments):
            logger.debug(f"匹配片段 {idx + 1}/{len(segments)}: {start:.1f}s - {end:.1f}s")

            result = await self._match_segment(
                narration_path,
                start,
                end,
                segment_index=idx
            )

            if result:
                results.append(result)
            else:
                logger.warning(f"片段 {idx + 1} 匹配失败")

        logger.info(f"匹配完成: {len(results)}/{len(segments)} 个片段成功匹配")
        return results

    async def _match_segment(
        self,
        narration_path: str,
        start_time: float,
        end_time: float,
        segment_index: int = 0
    ) -> Optional[MatchResult]:
        """匹配单个片段

        Args:
            narration_path: 解说视频路径
            start_time: 起始时间
            end_time: 结束时间
            segment_index: 片段索引

        Returns:
            匹配结果或 None
        """
        # 并行进行帧匹配和音频匹配
        frame_task = self._frame_matcher.match_segment(
            narration_path, start_time, end_time
        )
        audio_task = self._audio_matcher.match_segment(
            narration_path, start_time, end_time
        )

        frame_result, audio_result = await asyncio.gather(
            frame_task, audio_task, return_exceptions=True
        )

        # 处理异常
        if isinstance(frame_result, Exception):
            logger.warning(f"帧匹配异常: {frame_result}")
            frame_result = None
        if isinstance(audio_result, Exception):
            logger.warning(f"音频匹配异常: {audio_result}")
            audio_result = None

        # 融合结果
        return self._fuse_results(
            frame_result,
            audio_result,
            start_time,
            end_time,
            segment_index
        )

    def _fuse_results(
        self,
        frame_result: Optional[dict],
        audio_result: Optional[dict],
        narration_start: float,
        narration_end: float,
        segment_index: int
    ) -> Optional[MatchResult]:
        """融合帧匹配和音频匹配结果

        融合策略：
        1. 如果两者都有结果且时间相近，加权融合
        2. 如果两者时间差距大，优先采用帧匹配结果（解说可能有配乐干扰）
        3. 如果只有一个结果，直接使用
        4. 如果都没有结果，返回 None
        """
        frame_conf = frame_result.get("confidence", 0) if frame_result else 0
        audio_conf = audio_result.get("confidence", 0) if audio_result else 0

        # 两者都没有结果
        if not frame_result and not audio_result:
            return None

        # 只有帧匹配结果
        if frame_result and not audio_result:
            if frame_conf >= self.config.frame_threshold:
                return MatchResult(
                    narration_start=narration_start,
                    narration_end=narration_end,
                    movie_start=frame_result["start"],
                    movie_end=frame_result["end"],
                    frame_confidence=frame_conf,
                    audio_confidence=0,
                    combined_confidence=frame_conf,
                    match_source="frame",
                    segment_index=segment_index
                )
            return None

        # 只有音频匹配结果
        if audio_result and not frame_result:
            if audio_conf >= self.config.audio_threshold:
                return MatchResult(
                    narration_start=narration_start,
                    narration_end=narration_end,
                    movie_start=audio_result["start"],
                    movie_end=audio_result["end"],
                    frame_confidence=0,
                    audio_confidence=audio_conf,
                    combined_confidence=audio_conf,
                    match_source="audio",
                    segment_index=segment_index
                )
            return None

        # 两者都有结果，检查时间差
        time_diff = abs(frame_result["start"] - audio_result["start"])

        # 时间差小于 5 秒，认为是同一位置，加权融合
        if time_diff < 5.0:
            # 动态权重：如果音频置信度太低，降低其权重
            frame_w = self.config.frame_weight
            audio_w = self.config.audio_weight
            
            if audio_conf < 0.3 and frame_conf > 0.6:
                # 音频质量差但画面匹配好 -> 忽略音频
                frame_w = 0.9
                audio_w = 0.1
                logger.debug(f"音频置信度低({audio_conf:.2f})，启用动态权重: frame={frame_w}, audio={audio_w}")
            
            combined_conf = (
                frame_w * frame_conf +
                audio_w * audio_conf
            )

            # 加权平均时间
            movie_start = (
                frame_w * frame_result["start"] +
                audio_w * audio_result["start"]
            )
            movie_end = movie_start + (narration_end - narration_start)

            if combined_conf >= self.config.min_confidence:
                return MatchResult(
                    narration_start=narration_start,
                    narration_end=narration_end,
                    movie_start=movie_start,
                    movie_end=movie_end,
                    frame_confidence=frame_conf,
                    audio_confidence=audio_conf,
                    combined_confidence=combined_conf,
                    match_source="hybrid",
                    segment_index=segment_index
                )

        # 时间差大，优先使用帧匹配结果（更可靠）
        if frame_conf >= self.config.frame_threshold:
            return MatchResult(
                narration_start=narration_start,
                narration_end=narration_end,
                movie_start=frame_result["start"],
                movie_end=frame_result["end"],
                frame_confidence=frame_conf,
                audio_confidence=audio_conf,
                combined_confidence=frame_conf,
                match_source="frame",
                segment_index=segment_index,
                notes=f"时间差较大({time_diff:.1f}s)，使用帧匹配结果"
            )

        # 帧匹配置信度不够，尝试使用音频匹配
        if audio_conf >= self.config.audio_threshold:
            return MatchResult(
                narration_start=narration_start,
                narration_end=narration_end,
                movie_start=audio_result["start"],
                movie_end=audio_result["end"],
                frame_confidence=frame_conf,
                audio_confidence=audio_conf,
                combined_confidence=audio_conf,
                match_source="audio",
                segment_index=segment_index,
                notes="帧匹配置信度不足，使用音频匹配结果"
            )

        return None

    async def match_segment_manual(
        self,
        narration_path: str,
        start_time: float,
        end_time: float
    ) -> Optional[MatchResult]:
        """手动匹配指定时间范围的片段

        Args:
            narration_path: 解说视频路径
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            匹配结果
        """
        if not self._indexes_built:
            raise RuntimeError("索引未构建，请先调用 build_indexes()")

        return await self._match_segment(
            str(narration_path),
            start_time,
            end_time,
            segment_index=0
        )

    def is_ready(self) -> bool:
        """检查是否已准备就绪"""
        return self._indexes_built
