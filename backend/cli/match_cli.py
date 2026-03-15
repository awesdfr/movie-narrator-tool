"""视频匹配命令行工具

使用方法:
    python -m cli.match_cli match --movie "原电影.mp4" --narration "解说视频.mp4" --output "项目名"
    python -m cli.match_cli analyze --movie "原电影.mp4" --narration "解说视频.mp4" --output results.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config import settings
from core.matcher import HybridMatcher, MatchResult, MatchConfig


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )


async def cmd_match(args):
    """执行匹配并导出剪映草稿"""
    movie_path = Path(args.movie)
    narration_path = Path(args.narration)

    if not movie_path.exists():
        logger.error(f"原电影文件不存在: {movie_path}")
        return 1

    if not narration_path.exists():
        logger.error(f"解说视频文件不存在: {narration_path}")
        return 1

    # 创建匹配配置
    config = MatchConfig(
        frame_weight=args.frame_weight,
        audio_weight=args.audio_weight,
        min_confidence=args.min_confidence
    )

    # 创建匹配器
    matcher = HybridMatcher(config)

    logger.info("=" * 50)
    logger.info("视频匹配工具")
    logger.info("=" * 50)
    logger.info(f"原电影: {movie_path}")
    logger.info(f"解说视频: {narration_path}")
    logger.info(f"权重配置: 画面 {config.frame_weight:.0%} / 音频 {config.audio_weight:.0%}")
    logger.info("=" * 50)

    # 构建索引
    logger.info("\n[1/3] 构建索引...")
    cache_dir = settings.temp_dir / "match_cache"
    await matcher.build_indexes(str(movie_path), cache_dir)

    # 执行匹配
    logger.info("\n[2/3] 执行匹配...")
    results = await matcher.match_video(
        str(narration_path),
        use_scene_detection=not args.no_scene_detection
    )

    if not results:
        logger.warning("未找到任何匹配结果")
        return 1

    # 打印匹配结果
    logger.info(f"\n匹配完成，共 {len(results)} 个片段:")
    for i, r in enumerate(results):
        logger.info(
            f"  [{i+1}] 解说 {r.narration_start:.1f}s-{r.narration_end:.1f}s "
            f"-> 电影 {r.movie_start:.1f}s-{r.movie_end:.1f}s "
            f"(置信度: {r.combined_confidence:.2f}, 来源: {r.match_source})"
        )

    # 导出剪映草稿
    logger.info("\n[3/3] 导出剪映草稿...")
    from core.exporter.jianying_exporter import JianyingExporter

    exporter = JianyingExporter()

    # 使用新的导出方法
    project_name = args.output or f"匹配项目_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    draft_dir = await exporter.export_from_matches(
        results=results,
        movie_path=str(movie_path),
        narration_path=str(narration_path),
        project_name=project_name
    )

    logger.info("=" * 50)
    logger.info(f"草稿已生成: {draft_dir}")
    logger.info("请在剪映中打开该草稿")
    logger.info("=" * 50)

    return 0


async def cmd_analyze(args):
    """仅执行匹配分析，输出 JSON 结果"""
    movie_path = Path(args.movie)
    narration_path = Path(args.narration)

    if not movie_path.exists():
        logger.error(f"原电影文件不存在: {movie_path}")
        return 1

    if not narration_path.exists():
        logger.error(f"解说视频文件不存在: {narration_path}")
        return 1

    # 创建匹配器
    config = MatchConfig(
        frame_weight=args.frame_weight,
        audio_weight=args.audio_weight,
        min_confidence=args.min_confidence
    )
    matcher = HybridMatcher(config)

    logger.info("构建索引...")
    await matcher.build_indexes(str(movie_path))

    logger.info("执行匹配...")
    results = await matcher.match_video(
        str(narration_path),
        use_scene_detection=not args.no_scene_detection
    )

    # 转换为 JSON
    output_data = {
        "movie_path": str(movie_path),
        "narration_path": str(narration_path),
        "config": {
            "frame_weight": config.frame_weight,
            "audio_weight": config.audio_weight,
            "min_confidence": config.min_confidence
        },
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_segments": len(results),
            "average_confidence": sum(r.combined_confidence for r in results) / len(results) if results else 0,
            "match_sources": {
                "hybrid": sum(1 for r in results if r.match_source == "hybrid"),
                "frame": sum(1 for r in results if r.match_source == "frame"),
                "audio": sum(1 for r in results if r.match_source == "audio")
            }
        }
    }

    # 输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"分析结果已保存: {output_path}")
    return 0


async def cmd_build_index(args):
    """仅构建索引（用于预处理）"""
    movie_path = Path(args.movie)

    if not movie_path.exists():
        logger.error(f"视频文件不存在: {movie_path}")
        return 1

    matcher = HybridMatcher()

    logger.info(f"构建索引: {movie_path}")
    cache_dir = Path(args.cache_dir) if args.cache_dir else settings.temp_dir / "match_cache"

    await matcher.build_indexes(str(movie_path), cache_dir)

    logger.info(f"索引已保存到: {cache_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="视频匹配工具 - 匹配解说视频与原电影画面",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # match 命令
    match_parser = subparsers.add_parser("match", help="匹配并导出剪映草稿")
    match_parser.add_argument("--movie", "-m", required=True, help="原电影文件路径")
    match_parser.add_argument("--narration", "-n", required=True, help="解说视频文件路径")
    match_parser.add_argument("--output", "-o", help="输出项目名称")
    match_parser.add_argument("--frame-weight", type=float, default=0.6, help="画面匹配权重 (默认: 0.6)")
    match_parser.add_argument("--audio-weight", type=float, default=0.4, help="音频匹配权重 (默认: 0.4)")
    match_parser.add_argument("--min-confidence", type=float, default=0.5, help="最低置信度阈值 (默认: 0.5)")
    match_parser.add_argument("--no-scene-detection", action="store_true", help="禁用场景检测（整体匹配）")

    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="仅分析匹配，输出 JSON 结果")
    analyze_parser.add_argument("--movie", "-m", required=True, help="原电影文件路径")
    analyze_parser.add_argument("--narration", "-n", required=True, help="解说视频文件路径")
    analyze_parser.add_argument("--output", "-o", default="match_results.json", help="输出 JSON 文件路径")
    analyze_parser.add_argument("--frame-weight", type=float, default=0.6, help="画面匹配权重")
    analyze_parser.add_argument("--audio-weight", type=float, default=0.4, help="音频匹配权重")
    analyze_parser.add_argument("--min-confidence", type=float, default=0.5, help="最低置信度阈值")
    analyze_parser.add_argument("--no-scene-detection", action="store_true", help="禁用场景检测")

    # build-index 命令
    index_parser = subparsers.add_parser("build-index", help="预构建索引")
    index_parser.add_argument("--movie", "-m", required=True, help="视频文件路径")
    index_parser.add_argument("--cache-dir", help="缓存目录")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    # 执行命令
    if args.command == "match":
        return asyncio.run(cmd_match(args))
    elif args.command == "analyze":
        return asyncio.run(cmd_analyze(args))
    elif args.command == "build-index":
        return asyncio.run(cmd_build_index(args))

    return 0


if __name__ == "__main__":
    sys.exit(main())
