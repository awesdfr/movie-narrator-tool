"""处理API"""
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from models.project import Project, ProjectStatus, ProcessingProgress
from models.segment import Segment, SegmentUpdate, SegmentBatchUpdate, TTSStatus
from api.routes.project import load_project, save_project
from api.websocket import manager

router = APIRouter()

# 处理任务状态
_processing_tasks: dict[str, asyncio.Task] = {}


def _enforce_monotonicity(segments: list, max_backtrack: float = 600.0):
    """使用 LIS（最长递增子序列）过滤时序矛盾的匹配

    核心思想：
    1. 提取所有已匹配片段的 (解说时间, 电影时间) 对
    2. 计算电影时间序列的最长递增子序列
    3. 不在 LIS 中的匹配被视为异常，清除其匹配结果

    Args:
        segments: 片段列表
        max_backtrack: 允许的最大回退时间（秒），超过此值视为异常
    """
    import bisect

    matched = [(i, seg) for i, seg in enumerate(segments) if seg.movie_start is not None]

    if len(matched) < 2:
        logger.info(f"单调性检查: 匹配数量不足 ({len(matched)}), 跳过")
        return

    # 提取电影时间序列
    movie_times = [seg.movie_start for _, seg in matched]

    # 计算 LIS（使用二分查找优化，O(n log n)）
    def compute_lis_indices(arr):
        """计算最长递增子序列的索引列表"""
        n = len(arr)
        if n == 0:
            return []

        # dp[i] 存储长度为 i+1 的递增子序列的最小末尾元素的索引
        dp = []
        # dp_vals[i] 同步存储 dp[i] 对应的值，避免每次重建列表（O(n²) → O(n log n)）
        dp_vals = []
        # parent[i] 存储元素 i 在 LIS 中的前驱索引
        parent = [-1] * n

        for i in range(n):
            # 找到 arr[i] 应该插入的位置
            pos = bisect.bisect_left(dp_vals, arr[i])

            if pos == len(dp):
                dp.append(i)
                dp_vals.append(arr[i])
            else:
                dp[pos] = i
                dp_vals[pos] = arr[i]

            # 记录前驱
            if pos > 0:
                parent[i] = dp[pos - 1]

        # 回溯构建 LIS 索引列表
        lis_indices = []
        idx = dp[-1] if dp else -1
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]

        lis_indices.reverse()
        return lis_indices

    # 计算 LIS
    lis_indices = set(compute_lis_indices(movie_times))

    # 清除不在 LIS 中的匹配
    removed = 0
    for local_idx, (global_idx, seg) in enumerate(matched):
        if local_idx not in lis_indices:
            # 检查是否允许小范围回退
            if local_idx > 0:
                prev_movie_time = matched[local_idx - 1][1].movie_start
                if prev_movie_time is not None:
                    backtrack = prev_movie_time - seg.movie_start
                    # 允许小范围回退（可能是闪回镜头）
                    if 0 < backtrack <= max_backtrack:
                        continue

            # 清除异常匹配
            seg.movie_start = None
            seg.movie_end = None
            seg.match_confidence = 0.0
            removed += 1

    if removed > 0:
        logger.info(
            f"单调性检查 (LIS): 清除 {removed}/{len(matched)} 个时序异常匹配, "
            f"保留 {len(matched) - removed} 个"
        )
    else:
        logger.info(f"单调性检查 (LIS): 全部 {len(matched)} 个匹配时序正常")


async def process_project_task(project_id: str):
    """项目处理任务（后台执行）"""
    from core.video_processor.frame_extractor import FrameExtractor
    from core.video_processor.scene_detector import SceneDetector
    from core.video_processor.frame_matcher import FrameMatcher
    from core.video_processor.non_movie_detector import NonMovieDetector
    from core.audio_processor.speech_recognizer import SpeechRecognizer
    from core.audio_processor.voiceprint import VoiceprintRecognizer

    project = load_project(project_id)
    if not project:
        return

    # 在开头加载用户设置（后续多处需要使用）
    from api.routes.settings import load_settings
    app_settings = load_settings()

    try:
        # 阶段1: 视频分析
        await update_progress(project_id, "analyzing", 0, "正在分析视频...")

        frame_extractor = FrameExtractor()
        scene_detector = SceneDetector()

        # 提取解说视频信息
        narration_info = await frame_extractor.get_video_info(project.narration_path)
        project.narration_duration = narration_info["duration"]
        project.narration_fps = narration_info["fps"]

        # 提取原电影信息
        movie_info = await frame_extractor.get_video_info(project.movie_path)
        project.movie_duration = movie_info["duration"]
        project.movie_fps = movie_info["fps"]
        project.movie_resolution = (movie_info["width"], movie_info["height"])

        await update_progress(project_id, "analyzing", 20, "视频信息提取完成")

        # 阶段2: 场景检测
        await update_progress(project_id, "analyzing", 30, "正在检测场景切换...")
        scenes = await scene_detector.detect_scenes(project.narration_path)
        await update_progress(project_id, "analyzing", 50, f"检测到 {len(scenes)} 个场景")

        # 阶段3: 语音识别（如果提供了SRT字幕则跳过Whisper）
        project.status = ProjectStatus.RECOGNIZING

        if project.subtitle_path:
            await update_progress(project_id, "recognizing", 0, "正在解析SRT字幕...")
            from core.audio_processor.subtitle_parser import SubtitleParser
            subtitle_parser = SubtitleParser()
            transcription = subtitle_parser.parse_srt(project.subtitle_path)
            await update_progress(project_id, "recognizing", 50,
                                  f"SRT字幕解析完成，共 {len(transcription)} 条")
        else:
            await update_progress(project_id, "recognizing", 0, "正在进行语音识别...")
            speech_recognizer = SpeechRecognizer()
            transcription = await speech_recognizer.transcribe(project.narration_path)
            await update_progress(project_id, "recognizing", 50, "语音识别完成")

        # 阶段4: 声纹识别（区分解说/电影对白）
        voiceprint_recognizer = VoiceprintRecognizer()
        if project.reference_audio_path:
            await voiceprint_recognizer.load_reference(project.reference_audio_path)

        narration_segments = await voiceprint_recognizer.identify_narrator(
            project.narration_path, transcription
        )
        await update_progress(project_id, "recognizing", 100, "声纹识别完成")

        # 阶段5: 创建初始Segment对象（从声纹识别结果）
        segments = []
        for i, seg_info in enumerate(narration_segments):
            segment = Segment(
                id=f"seg_{i:04d}",
                index=i,
                narration_start=seg_info["start"],
                narration_end=seg_info["end"],
                movie_start=None,
                movie_end=None,
                segment_type=seg_info["type"],
                match_confidence=0.0,
                original_text=seg_info.get("text", ""),
            )
            segments.append(segment)

        project.segments = segments
        save_project(project)

        # 阶段6: 帧匹配（匹配完成后暂停，等待用户确认再润色）
        project.status = ProjectStatus.MATCHING
        await update_progress(project_id, "matching", 0, "正在执行帧匹配...")

        # --- 帧匹配任务 ---
        async def run_frame_matching():
            match_settings = app_settings.match
            frame_matcher = FrameMatcher(
                phash_threshold=match_settings.phash_threshold,
                match_threshold=match_settings.frame_match_threshold,
                use_deep_learning=match_settings.use_deep_learning,
                index_sample_fps=match_settings.index_sample_fps,
                fast_mode=match_settings.fast_mode
            )
            non_movie_detector = NonMovieDetector()

            if frame_matcher.fast_mode:
                logger.warning(
                    "帧匹配使用快速模式（仅pHash），仅适合短片或画面高度相似的场景。"
                    "如匹配效果不佳，请在设置中关闭 fast_mode 以启用深度学习特征匹配。"
                )

            from config import settings as app_config
            cache_dir = app_config.temp_dir / "match_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            movie_name = Path(project.movie_path).stem
            frame_cache_path = cache_dir / f"{movie_name}_frame.pkl"

            sample_interval = match_settings.sample_interval if match_settings else None
            await frame_matcher.build_index(
                project.movie_path,
                sample_interval=sample_interval,
                cache_path=frame_cache_path
            )

            total_segments = len(segments)
            match_completed = [0]
            match_task_status = {
                i: {"id": f"seg_{i:04d}", "name": f"片段 {i+1}", "status": "pending", "progress": 0}
                for i in range(total_segments)
            }

            # 初始速率估算：用电影总时长/解说总时长作为起点
            # 这比"无锚点盲匹配"靠谱得多，避免首批片段匹配到电影远端
            initial_pace = None
            if project.movie_duration and project.narration_duration and project.narration_duration > 0:
                initial_pace = project.movie_duration / project.narration_duration
                logger.info(
                    f"初始速率估算: {initial_pace:.2f}x "
                    f"(电影{project.movie_duration:.0f}s / 解说{project.narration_duration:.0f}s)"
                )
            else:
                logger.warning(
                    f"无法计算初始速率: movie_duration={project.movie_duration}, "
                    f"narration_duration={project.narration_duration}"
                )

            # 使用一次性快速匹配模式（类青柠剪吧速度）
            logger.info("使用一次性快速匹配模式")

            # 准备片段任务
            segment_tasks = [
                {
                    'id': segment.id,
                    'start': segment.narration_start,
                    'end': segment.narration_end
                }
                for segment in segments
            ]

            # 进度回调
            def on_progress(stage, progress_pct, message):
                # 映射到 20-70% 区间
                mapped_progress = 20 + int(50 * progress_pct / 100)
                logger.info(f"快速匹配 [{stage}]: {message} ({mapped_progress}%)")

            # 一次性快速匹配（读取一遍视频，批量搜索）
            allow_non_sequential = getattr(match_settings, 'allow_non_sequential', True)
            batch_results = await frame_matcher.match_all_segments_fast(
                narration_path=project.narration_path,
                segments=segment_tasks,
                sample_fps=4.0,  # 每秒4帧，平衡速度和精度
                progress_callback=on_progress,
                movie_duration=project.movie_duration,
                narration_duration=project.narration_duration,
                allow_non_sequential=allow_non_sequential
            )

            # 更新进度
            await update_progress(
                project_id, "matching", 70,
                f"快速匹配完成",
                total_tasks=total_segments,
                completed_tasks=total_segments
            )

            # 应用匹配结果
            result_map = {r['id']: r for r in batch_results}
            for segment in segments:
                result = result_map.get(segment.id)
                if result and result.get('success'):
                    segment.movie_start = result['start']
                    segment.movie_end = result['end']
                    segment.match_confidence = result['confidence']
                    logger.debug(
                        f"片段 {segment.id} 匹配成功: "
                        f"电影[{result['start']:.1f}s-{result['end']:.1f}s], "
                        f"置信度={result['confidence']:.2f}"
                    )

            matched_in_batch = sum(1 for r in batch_results if r.get('success'))
            logger.info(f"批量匹配完成: {matched_in_batch}/{total_segments} 个片段成功")

            # 后处理：用 LIS（最长递增子序列）剔除打断时间顺序的错误匹配
            # 注意：跳跃剪辑的解说视频应禁用此功能
            use_lis = getattr(match_settings, 'use_lis_filter', False)
            if use_lis:
                _enforce_monotonicity(segments)
            else:
                logger.info("LIS 过滤已禁用（支持跳跃剪辑）")

            # ===== 第二轮：用 LIS 存活的匹配做插值，重新匹配被清除和未匹配的片段 =====
            surviving = [(seg.narration_start, seg.movie_start) for seg in segments if seg.movie_start is not None]
            unmatched_indices = [
                i for i, seg in enumerate(segments)
                if seg.movie_start is None and seg.segment_type != "non_movie"
            ]

            if len(surviving) >= 2 and len(unmatched_indices) > 0:
                logger.info(
                    f"二次匹配: {len(surviving)} 个存活匹配, "
                    f"对 {len(unmatched_indices)} 个未匹配片段进行重新匹配..."
                )
                surviving.sort(key=lambda x: x[0])
                rematch_success = 0
                total_unmatched = len(unmatched_indices)

                for rematch_i, idx in enumerate(unmatched_indices):
                    segment = segments[idx]
                    narr_t = segment.narration_start

                    # 在存活匹配中找前后锚点做插值
                    before = [(n, m) for n, m in surviving if n <= narr_t]
                    after = [(n, m) for n, m in surviving if n > narr_t]

                    # 判断是否是早期片段（只有后置锚点，没有前置锚点）
                    is_early_segment = not before and after

                    if before and after:
                        n1, m1 = before[-1]
                        n2, m2 = after[0]
                        ratio = (narr_t - n1) / (n2 - n1) if n2 != n1 else 0
                        interpolated_hint = m1 + ratio * (m2 - m1)
                    elif before:
                        n_last, m_last = before[-1]
                        pace = initial_pace or 1.0
                        if len(before) >= 2:
                            n_prev, m_prev = before[-2]
                            if n_last - n_prev > 0.5:
                                pace = (m_last - m_prev) / (n_last - n_prev)
                        interpolated_hint = m_last + (narr_t - n_last) * pace
                    elif after:
                        n_first, m_first = after[0]
                        pace = initial_pace or 1.0
                        if len(after) >= 2:
                            n_next, m_next = after[1]
                            if n_next - n_first > 0.5:
                                pace = (m_next - m_first) / (n_next - n_first)
                        interpolated_hint = max(0, m_first - (n_first - narr_t) * pace)
                    else:
                        continue

                    # 早期片段使用宽松模式（没有可靠的前置锚点）
                    # 非早期片段使用严格窗口模式
                    use_strict = not is_early_segment

                    match_result = await frame_matcher.match_segment(
                        project.narration_path,
                        segment.narration_start,
                        segment.narration_end,
                        time_hint=interpolated_hint,
                        strict_window=use_strict
                    )
                    if match_result is None:
                        match_result = await frame_matcher.match_segment(
                            project.narration_path,
                            segment.narration_start,
                            segment.narration_end,
                            time_hint=interpolated_hint,
                            relaxed=True,
                            strict_window=use_strict
                        )

                    if match_result:
                        segment.movie_start = match_result["start"]
                        segment.movie_end = match_result["end"]
                        segment.match_confidence = match_result["confidence"]
                        rematch_success += 1
                        logger.debug(
                            f"二次匹配 片段 {idx+1} 成功: "
                            f"电影[{match_result['start']:.1f}s-{match_result['end']:.1f}s] "
                            f"(hint={interpolated_hint:.1f}s)"
                        )

                    # 更新进度
                    rematch_progress = 70 + int(25 * (rematch_i + 1) / total_unmatched)
                    await update_progress(
                        project_id, "matching", rematch_progress,
                        f"二次匹配 {rematch_i + 1}/{total_unmatched} (已恢复{rematch_success}个)"
                    )

                logger.info(f"二次匹配完成: 恢复 {rematch_success}/{len(unmatched_indices)} 个片段")

                # 二次匹配后再做一次 LIS 清理（仅当启用时）
                if rematch_success > 0 and use_lis:
                    _enforce_monotonicity(segments)

        # 执行帧匹配
        await run_frame_matching()

        # 统计匹配结果
        matched_count = sum(1 for s in segments if s.movie_start is not None)
        total_count = len(segments)
        avg_conf = 0.0
        if matched_count > 0:
            avg_conf = sum(s.match_confidence for s in segments if s.movie_start is not None) / matched_count

        await update_progress(project_id, "matching", 100, "帧匹配完成")

        # 帧匹配完成，进入"待确认润色"状态
        # 用户可以先预览匹配效果，确认无误后再开始AI润色
        project.status = ProjectStatus.READY_FOR_POLISH
        save_project(project)
        await update_progress(
            project_id, "ready_for_polish", 100,
            f"帧匹配完成: {matched_count}/{total_count} 个片段匹配成功，平均置信度 {avg_conf:.0%}"
        )

        logger.info(
            f"项目帧匹配完成: {project_id}, "
            f"匹配 {matched_count}/{total_count}, 平均置信度 {avg_conf:.2f}"
        )

    except Exception as e:
        import traceback
        logger.error(f"项目处理失败: {project_id}, 错误: {e}\n{traceback.format_exc()}")
        project.status = ProjectStatus.ERROR
        project.progress.message = str(e)
        save_project(project)
        await update_progress(project_id, "error", 0, f"处理失败: {e}")


async def update_progress(
    project_id: str,
    stage: str,
    progress: float,
    message: str,
    parallel_tasks: list = None,
    total_tasks: int = None,
    completed_tasks: int = None
):
    """更新并广播处理进度"""
    project = load_project(project_id)
    if project:
        project.progress = ProcessingProgress(
            stage=stage,
            progress=progress,
            message=message
        )
        save_project(project)

    # 通过WebSocket广播进度
    data = {
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message
    }

    # 添加并行任务详情
    if parallel_tasks is not None:
        data["parallel_tasks"] = parallel_tasks
    if total_tasks is not None:
        data["total_tasks"] = total_tasks
    if completed_tasks is not None:
        data["completed_tasks"] = completed_tasks

    await manager.broadcast_to_project(project_id, data)


@router.post("/{project_id}/start")
async def start_processing(project_id: str, background_tasks: BackgroundTasks):
    """开始处理项目"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    if project_id in _processing_tasks:
        task = _processing_tasks[project_id]
        if not task.done():
            raise HTTPException(status_code=400, detail="项目正在处理中")

    # 重跑时清理旧数据
    if project.status in (ProjectStatus.READY_FOR_POLISH, ProjectStatus.READY_FOR_TTS, ProjectStatus.COMPLETED, ProjectStatus.ERROR):
        if project.segments:
            project.segments = []
        project.progress = ProcessingProgress()

        # 删除旧的TTS文件
        from config import settings as app_config
        import shutil
        tts_dir = app_config.temp_dir / project_id / "tts"
        if tts_dir.exists():
            shutil.rmtree(tts_dir, ignore_errors=True)

    # 创建后台任务
    task = asyncio.create_task(process_project_task(project_id))
    _processing_tasks[project_id] = task

    project.status = ProjectStatus.ANALYZING
    save_project(project)

    return {"message": "处理已开始", "project_id": project_id}


@router.post("/{project_id}/stop")
async def stop_processing(project_id: str):
    """停止处理"""
    if project_id in _processing_tasks:
        task = _processing_tasks[project_id]
        if not task.done():
            task.cancel()
            del _processing_tasks[project_id]

    project = load_project(project_id)
    if project:
        project.status = ProjectStatus.ERROR
        project.progress.message = "用户取消处理"
        save_project(project)

    return {"message": "处理已停止"}


@router.get("/{project_id}/progress")
async def get_progress(project_id: str):
    """获取处理进度"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    return {
        "status": project.status,
        "progress": project.progress
    }


@router.get("/{project_id}/segments")
async def get_segments(project_id: str):
    """获取片段列表"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    return project.segments


@router.put("/{project_id}/segments/{segment_id}")
async def update_segment(project_id: str, segment_id: str, update: SegmentUpdate):
    """更新单个片段"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    segment = next((s for s in project.segments if s.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="片段不存在")

    # 更新字段
    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(segment, key, value)

    save_project(project)
    return segment


@router.post("/{project_id}/segments/batch")
async def batch_update_segments(project_id: str, batch: SegmentBatchUpdate):
    """批量更新片段"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    update_data = batch.model_dump(exclude={"segment_ids"}, exclude_unset=True)
    updated_count = 0

    for segment in project.segments:
        if segment.id in batch.segment_ids:
            for key, value in update_data.items():
                if value is not None:
                    setattr(segment, key, value)
            updated_count += 1

    save_project(project)
    return {"message": f"已更新 {updated_count} 个片段"}


@router.post("/{project_id}/segments/{segment_id}/regenerate-tts")
async def regenerate_segment_tts(project_id: str, segment_id: str):
    """重新生成片段TTS"""
    from core.tts_service.tts_client import TTSClient
    from api.routes.settings import load_settings

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    segment = next((s for s in project.segments if s.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="片段不存在")

    text = segment.polished_text if segment.use_polished_text else segment.original_text
    if not text:
        raise HTTPException(status_code=400, detail="片段没有文案")

    # 从用户设置中读取TTS配置
    app_settings = load_settings()
    tts_client = TTSClient(
        api_base=app_settings.tts.api_base,
        api_endpoint=app_settings.tts.api_endpoint if hasattr(app_settings.tts, 'api_endpoint') else '/tts',
        reference_audio=app_settings.tts.reference_audio,
        infer_mode=app_settings.tts.infer_mode if hasattr(app_settings.tts, 'infer_mode') else '批次推理'
    )

    try:
        audio_path = await tts_client.generate(text, f"{project_id}_{segment_id}")
        segment.tts_audio_path = str(audio_path)
        segment.tts_duration = await tts_client.get_duration(str(audio_path))
        segment.tts_status = TTSStatus.GENERATED
        segment.tts_error = None
    except Exception as e:
        segment.tts_status = TTSStatus.FAILED
        segment.tts_error = str(e)
        save_project(project)
        raise HTTPException(status_code=500, detail=f"TTS生成失败: {e}")

    save_project(project)
    return {
        "audio_path": audio_path,
        "duration": segment.tts_duration,
        "tts_status": segment.tts_status
    }


@router.post("/{project_id}/segments/{segment_id}/repolish")
async def repolish_segment(project_id: str, segment_id: str):
    """重新润色片段文案"""
    from core.ai_service.text_polisher import TextPolisher
    from api.routes.settings import load_settings
    from core.ai_service.api_manager import APIManager

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    segment = next((s for s in project.segments if s.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="片段不存在")

    if not segment.original_text:
        raise HTTPException(status_code=400, detail="片段没有原始文案")

    # 从用户设置中读取AI配置
    app_settings = load_settings()
    api_manager = APIManager(
        api_base=app_settings.ai.api_base,
        api_key=app_settings.ai.api_key,
        model=app_settings.ai.model
    )
    text_polisher = TextPolisher(
        api_manager=api_manager,
        template=app_settings.ai.polish_template,
        temperature=app_settings.ai.temperature,
        max_tokens=app_settings.ai.max_tokens
    )

    try:
        segment.polished_text = await text_polisher.polish(segment.original_text)
    except RuntimeError as e:
        logger.error(f"润色失败: {segment_id}, 错误: {e}")
        raise HTTPException(status_code=500, detail=f"润色失败: {e}")
    except Exception as e:
        logger.error(f"润色失败: {segment_id}, 错误: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"润色失败: {type(e).__name__}: {e}")

    save_project(project)
    return {"polished_text": segment.polished_text}


@router.post("/{project_id}/start-polish")
async def start_polishing(project_id: str):
    """开始AI文案润色（用户确认帧匹配后手动触发）"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    if project_id in _processing_tasks:
        task = _processing_tasks[project_id]
        if not task.done():
            raise HTTPException(status_code=400, detail="项目正在处理中")

    if not project.segments:
        raise HTTPException(status_code=400, detail="项目没有片段数据")

    # 创建后台润色任务
    task = asyncio.create_task(_polish_project_task(project_id))
    _processing_tasks[project_id] = task

    project.status = ProjectStatus.POLISHING
    save_project(project)

    return {"message": "文案润色已开始", "project_id": project_id}


async def _polish_project_task(project_id: str):
    """文案润色后台任务"""
    from core.ai_service.text_polisher import TextPolisher
    from core.ai_service.api_manager import APIManager
    from api.routes.settings import load_settings

    project = load_project(project_id)
    if not project:
        return

    app_settings = load_settings()

    try:
        await update_progress(project_id, "polishing", 0, "正在进行AI文案润色...")

        api_manager = APIManager(
            api_base=app_settings.ai.api_base,
            api_key=app_settings.ai.api_key,
            model=app_settings.ai.model
        )
        text_polisher = TextPolisher(
            api_manager=api_manager,
            template=app_settings.ai.polish_template,
            temperature=app_settings.ai.temperature,
            max_tokens=app_settings.ai.max_tokens
        )

        segments_to_polish = [(i, s) for i, s in enumerate(project.segments) if s.original_text]
        total_polish = len(segments_to_polish)

        if total_polish == 0:
            project.status = ProjectStatus.READY_FOR_TTS
            save_project(project)
            await update_progress(project_id, "ready_for_tts", 100, "无需润色，请编辑文案后生成TTS")
            return

        polish_completed = [0]
        polish_concurrency = app_settings.concurrency.polish_concurrency if hasattr(app_settings, 'concurrency') else 5
        polish_semaphore = asyncio.Semaphore(polish_concurrency)

        async def polish_single(idx: int, segment: Segment):
            async with polish_semaphore:
                try:
                    target_duration = segment.narration_end - segment.narration_start
                    segment.polished_text = await text_polisher.rewrite_for_tts(
                        segment.original_text,
                        target_duration=target_duration
                    )
                except Exception as e:
                    logger.warning(f"润色失败: {segment.id}, 错误: {e}")
                    segment.polished_text = segment.original_text

                polish_completed[0] += 1
                progress = int(100 * polish_completed[0] / total_polish)
                await update_progress(
                    project_id, "polishing", progress,
                    f"文案润色 {polish_completed[0]}/{total_polish}"
                )

        await asyncio.gather(*[
            polish_single(idx, segment)
            for idx, segment in segments_to_polish
        ])

        await update_progress(project_id, "polishing", 100, "文案润色完成")

        # 润色完成，进入"待编辑/生成TTS"状态
        project.status = ProjectStatus.READY_FOR_TTS
        save_project(project)
        await update_progress(project_id, "ready_for_tts", 100, "润色完成，请编辑文案后生成TTS")

        logger.info(f"项目文案润色完成: {project_id}")

    except Exception as e:
        import traceback
        logger.error(f"文案润色失败: {project_id}, 错误: {e}\n{traceback.format_exc()}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(e)
            save_project(project)
        await update_progress(project_id, "error", 0, f"文案润色失败: {e}")


@router.post("/{project_id}/generate-tts")
async def batch_generate_tts(project_id: str):
    """批量生成所有片段的TTS音频（用户手动触发）"""
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    if project_id in _processing_tasks:
        task = _processing_tasks[project_id]
        if not task.done():
            raise HTTPException(status_code=400, detail="项目正在处理中")

    # 创建后台TTS生成任务
    task = asyncio.create_task(_batch_generate_tts_task(project_id))
    _processing_tasks[project_id] = task

    project.status = ProjectStatus.GENERATING_TTS
    save_project(project)

    return {"message": "TTS生成已开始", "project_id": project_id}


async def _batch_generate_tts_task(project_id: str):
    """批量TTS生成后台任务"""
    from core.tts_service.tts_client import TTSClient
    from api.routes.settings import load_settings

    project = load_project(project_id)
    if not project:
        return

    app_settings = load_settings()

    try:
        await update_progress(project_id, "generating_tts", 0, "正在生成TTS音频...")

        tts_client = TTSClient(
            api_base=app_settings.tts.api_base,
            api_endpoint=app_settings.tts.api_endpoint if hasattr(app_settings.tts, 'api_endpoint') else '/tts',
            reference_audio=app_settings.tts.reference_audio,
            infer_mode=app_settings.tts.infer_mode if hasattr(app_settings.tts, 'infer_mode') else '批次推理'
        )
        from config import settings as app_config
        tts_output_dir = app_config.temp_dir / project_id / "tts"

        segments_with_text = [
            (i, s) for i, s in enumerate(project.segments)
            if s.original_text or s.polished_text
        ]
        total_tts = len(segments_with_text)
        tts_completed = [0]
        tts_task_status = {
            i: {"id": seg.id, "name": f"片段 {i+1}", "status": "pending", "progress": 0}
            for i, (_, seg) in enumerate(segments_with_text)
        }

        tts_concurrency = app_settings.concurrency.tts_concurrency if hasattr(app_settings, 'concurrency') else 3
        tts_semaphore = asyncio.Semaphore(tts_concurrency)

        async def generate_single_tts(task_idx: int, idx: int, segment):
            text = segment.polished_text if (segment.use_polished_text and segment.polished_text) else segment.original_text
            audio_path = None

            if text:
                # 信号量只包裹 TTS API 调用，尽早释放并发槽位
                async with tts_semaphore:
                    tts_task_status[task_idx]["status"] = "running"
                    tts_task_status[task_idx]["progress"] = 50

                    try:
                        ref_audio = project.tts_reference_audio_path if project.tts_reference_audio_path else project.reference_audio_path

                        # 删除旧的TTS缓存文件，强制重新生成
                        old_path = tts_output_dir / f"{project_id}_{segment.id}.wav"
                        if old_path.exists():
                            old_path.unlink()

                        audio_path = await tts_client.generate(
                            text,
                            output_name=f"{project_id}_{segment.id}",
                            output_dir=tts_output_dir,
                            reference_audio=ref_audio
                        )
                    except Exception as e:
                        logger.warning(f"TTS生成失败: {segment.id}, 错误: {e}")
                        segment.tts_duration = await tts_client.estimate_duration(text)
                        segment.tts_status = TTSStatus.FAILED
                        segment.tts_error = str(e)
                        tts_task_status[task_idx]["status"] = "error"

                # 以下操作不占用 TTS 并发槽位
                if audio_path:
                    segment.tts_audio_path = str(audio_path)
                    segment.tts_duration = await tts_client.get_duration(str(audio_path))
                    segment.tts_status = TTSStatus.GENERATED
                    segment.tts_error = None
                    logger.debug(f"TTS生成成功: {segment.id}, 时长: {segment.tts_duration:.2f}s")
                    tts_task_status[task_idx]["status"] = "completed"

            tts_task_status[task_idx]["progress"] = 100
            tts_completed[0] += 1

            progress = int(100 * tts_completed[0] / total_tts) if total_tts > 0 else 100
            visible_tasks = [tts_task_status[j] for j in range(min(20, total_tts))]
            await update_progress(
                project_id, "generating_tts", progress,
                f"生成TTS {tts_completed[0]}/{total_tts}",
                parallel_tasks=visible_tasks,
                total_tasks=total_tts,
                completed_tasks=tts_completed[0]
            )

        if segments_with_text:
            await asyncio.gather(*[
                generate_single_tts(task_idx, idx, segment)
                for task_idx, (idx, segment) in enumerate(segments_with_text)
            ])

        await update_progress(project_id, "generating_tts", 100, "TTS生成完成")

        # 重新加载项目以获取最新数据（TTS过程中段落数据可能被编辑）
        project = load_project(project_id)
        # 更新段落TTS状态（从内存中的segments_with_text同步）
        seg_map = {seg.id: seg for _, seg in segments_with_text}
        for seg in project.segments:
            if seg.id in seg_map:
                updated = seg_map[seg.id]
                seg.tts_audio_path = updated.tts_audio_path
                seg.tts_duration = updated.tts_duration
                seg.tts_status = updated.tts_status
                seg.tts_error = updated.tts_error

        project.status = ProjectStatus.COMPLETED
        save_project(project)
        await update_progress(project_id, "completed", 100, "TTS生成完成！")

        logger.info(f"批量TTS生成完成: {project_id}")

    except Exception as e:
        logger.error(f"批量TTS生成失败: {project_id}, 错误: {e}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(e)
            save_project(project)
        await update_progress(project_id, "error", 0, f"TTS生成失败: {e}")
