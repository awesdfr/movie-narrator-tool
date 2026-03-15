"""HTML match review report exporter."""
from __future__ import annotations

import asyncio
import base64
import html
import re
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger

from models.project import Project
from models.segment import AlignmentStatus, SegmentType


class MatchReportExporter:
    """Export a self-contained HTML review report for current segment matches."""

    def __init__(self, thumbnail_size: tuple[int, int] = (240, 135), jpeg_quality: int = 76):
        self.thumbnail_size = thumbnail_size
        self.jpeg_quality = jpeg_quality

    async def export(self, project: Project, output_dir: Path | None = None) -> Path:
        from core.video_processor.frame_extractor import FrameExtractor

        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        extractor = FrameExtractor()
        segments = list(project.segments)
        matched_segments = [segment for segment in segments if self._has_movie_match(segment)]
        report_rows: list[str] = []

        for segment in segments:
            report_rows.append(await self._render_segment(project, segment, extractor, output_dir))

        report_path = output_dir / f"{self._slugify(project.name)}_match_report.html"
        report_path.write_text(
            self._render_document(project, segments, matched_segments, report_rows),
            encoding="utf-8",
        )
        logger.info(f"Match report exported: {report_path}")
        return report_path

    async def _render_segment(self, project: Project, segment, extractor, output_dir: Path) -> str:
        narration_preview = await self._build_preview_image(
            extractor,
            project.narration_path,
            segment.narration_start + max(0.0, (segment.narration_end - segment.narration_start) * 0.35),
            output_dir / "report_frames",
        )
        movie_preview = None
        if self._has_movie_match(segment):
            movie_preview = await self._build_preview_image(
                extractor,
                project.movie_path,
                segment.movie_start + max(0.0, (segment.movie_end - segment.movie_start) * 0.5),
                output_dir / "report_frames",
            )

        tags = self._segment_tags(segment)
        badge_html = "".join(f'<span class="badge">{html.escape(tag)}</span>' for tag in tags)
        candidate_rows = "".join(self._render_candidate(candidate) for candidate in segment.match_candidates[:3])
        if not candidate_rows:
            candidate_rows = '<div class="candidate-empty">当前没有候选记录</div>'

        return f"""
<details class="segment-card" data-tags="{html.escape(' '.join(tags))}">
  <summary>
    <div class="segment-summary-line">
      <span class="segment-index">#{segment.index + 1:03d}</span>
      <span class="segment-status">{html.escape(self._alignment_text(segment.alignment_status))}</span>
      <span class="segment-score">{self._percent(segment.match_confidence)}</span>
      <span class="segment-time">{self._format_time(segment.narration_start)} - {self._format_time(segment.narration_end)}</span>
    </div>
    <div class="segment-badges">{badge_html}</div>
  </summary>
  <div class="segment-body">
    <div class="preview-grid">
      <div class="preview-item">
        <h4>解说片段</h4>
        {self._render_image_or_empty(narration_preview, "解说预览")}
        <div class="time-note">{self._format_time(segment.narration_start)} - {self._format_time(segment.narration_end)}</div>
      </div>
      <div class="preview-item">
        <h4>匹配电影片段</h4>
        {self._render_image_or_empty(movie_preview, "尚未匹配到电影片段")}
        <div class="time-note">{self._format_time(segment.movie_start)} - {self._format_time(segment.movie_end)}</div>
      </div>
    </div>

    <div class="metric-grid">
      <div>总置信度 <strong>{self._percent(segment.match_confidence)}</strong></div>
      <div>visual <strong>{self._percent(segment.visual_confidence)}</strong></div>
      <div>audio <strong>{self._percent(segment.audio_confidence)}</strong></div>
      <div>temporal <strong>{self._percent(segment.temporal_confidence)}</strong></div>
      <div>stability <strong>{self._percent(segment.stability_score)}</strong></div>
      <div>speech <strong>{self._percent(segment.speech_likelihood)}</strong></div>
      <div>duration gap <strong>{(segment.duration_gap or 0.0):.1f}s</strong></div>
      <div>boundary error <strong>{self._format_optional_seconds(segment.estimated_boundary_error)}</strong></div>
    </div>

    <div class="reason-block">
      <h4>匹配说明</h4>
      <p>{html.escape(segment.match_reason or "暂无匹配说明")}</p>
    </div>

    <div class="text-grid">
      <div class="text-item">
        <h4>原文</h4>
        <p>{html.escape(segment.original_text or "(无)")}</p>
      </div>
      <div class="text-item">
        <h4>润色</h4>
        <p>{html.escape(segment.polished_text or "(未润色)")}</p>
      </div>
    </div>

    <div class="candidate-block">
      <h4>候选摘要</h4>
      {candidate_rows}
    </div>
  </div>
</details>
"""

    async def _build_preview_image(self, extractor, video_path: str | None, time_sec: float | None, output_dir: Path) -> str | None:
        if not video_path or time_sec is None:
            return None
        try:
            frame_path = await extractor.extract_frame(video_path, float(time_sec), output_dir=output_dir)
        except Exception as exc:
            logger.warning(f"Failed to extract report frame for {video_path}: {exc}")
            return None

        def _encode() -> str | None:
            image = cv2.imread(str(frame_path))
            if image is None:
                return None
            resized = cv2.resize(image, self.thumbnail_size)
            ok, buffer = cv2.imencode(
                ".jpg",
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                return None
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"

        return await asyncio.to_thread(_encode)

    def _render_document(self, project: Project, segments: list, matched_segments: list, report_rows: list[str]) -> str:
        avg_confidence = (
            sum((segment.match_confidence or 0.0) for segment in matched_segments) / len(matched_segments)
            if matched_segments else 0.0
        )
        review_count = sum(1 for segment in segments if segment.review_required)
        skipped_count = sum(1 for segment in segments if segment.skip_matching)
        unmatched_count = sum(
            1 for segment in segments
            if segment.use_segment and not self._has_movie_match(segment) and segment.segment_type != SegmentType.NON_MOVIE
        )
        disabled_count = sum(1 for segment in segments if not segment.use_segment)
        benchmark_text = "--" if project.benchmark_accuracy is None else self._percent(project.benchmark_accuracy)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>匹配审阅报告 - {html.escape(project.name)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d7dde5;
      --text: #1f2937;
      --muted: #667085;
      --accent: #0f766e;
      --warn: #b45309;
      --blue: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #eef4f8 0%, var(--bg) 100%);
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--text);
    }}
    .page {{ max-width: 1280px; margin: 0 auto; }}
    .hero, .toolbar, .segment-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }}
    .hero {{ padding: 24px; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }}
    .summary-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      background: #fbfcfd;
    }}
    .summary-value {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .summary-label {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 16px;
      margin-bottom: 16px;
      position: sticky;
      top: 12px;
      z-index: 5;
    }}
    .toolbar button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 14px;
    }}
    .toolbar button.active {{
      border-color: var(--accent);
      color: var(--accent);
      background: #ecfdf5;
    }}
    .segment-list {{ display: grid; gap: 14px; }}
    .segment-card {{ overflow: hidden; }}
    .segment-card summary {{
      list-style: none;
      padding: 16px 18px;
      cursor: pointer;
    }}
    .segment-card summary::-webkit-details-marker {{ display: none; }}
    .segment-summary-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      font-weight: 600;
    }}
    .segment-index {{ color: var(--blue); min-width: 58px; }}
    .segment-status {{ color: var(--accent); }}
    .segment-score {{ color: var(--warn); }}
    .segment-time {{ color: var(--muted); font-weight: 500; }}
    .segment-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      border: 1px solid #c7d2fe;
    }}
    .segment-body {{
      border-top: 1px solid var(--line);
      padding: 18px;
      display: grid;
      gap: 18px;
    }}
    .preview-grid, .text-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .preview-item, .text-item, .reason-block, .candidate-block {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .preview-item h4, .text-item h4, .reason-block h4, .candidate-block h4 {{ margin: 0 0 12px; }}
    .preview-item img {{
      width: 100%;
      display: block;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #111827;
    }}
    .preview-empty {{
      min-height: 135px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f8fafc;
      color: var(--muted);
      border: 1px dashed var(--line);
      text-align: center;
      padding: 12px;
    }}
    .time-note {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      font-size: 14px;
    }}
    .metric-grid > div {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
    }}
    p {{ margin: 0; white-space: pre-wrap; line-height: 1.6; }}
    .candidate-row {{
      display: grid;
      gap: 4px;
      border-top: 1px solid var(--line);
      padding: 12px 0;
    }}
    .candidate-row:first-of-type {{
      border-top: 0;
      padding-top: 0;
    }}
    .candidate-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .candidate-empty {{ color: var(--muted); }}
    @media (max-width: 768px) {{
      body {{ padding: 14px; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>匹配审阅报告</h1>
      <div class="meta">
        <span>项目: {html.escape(project.name)}</span>
        <span>匹配版本: {html.escape(project.match_version or "v2")}</span>
        <span>Benchmark: {benchmark_text}</span>
        <span>生成时间: {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</span>
      </div>
      <div class="summary-grid">
        <div class="summary-card"><div class="summary-value">{len(segments)}</div><div class="summary-label">总片段</div></div>
        <div class="summary-card"><div class="summary-value">{len(matched_segments)}</div><div class="summary-label">已匹配</div></div>
        <div class="summary-card"><div class="summary-value">{review_count}</div><div class="summary-label">待复核</div></div>
        <div class="summary-card"><div class="summary-value">{skipped_count}</div><div class="summary-label">跳过匹配</div></div>
        <div class="summary-card"><div class="summary-value">{unmatched_count}</div><div class="summary-label">未匹配</div></div>
        <div class="summary-card"><div class="summary-value">{disabled_count}</div><div class="summary-label">已禁用</div></div>
        <div class="summary-card"><div class="summary-value">{self._percent(avg_confidence)}</div><div class="summary-label">平均置信度</div></div>
      </div>
    </section>

    <section class="toolbar">
      <button type="button" class="active" data-filter="all">全部</button>
      <button type="button" data-filter="review">待复核</button>
      <button type="button" data-filter="matched">已匹配</button>
      <button type="button" data-filter="unmatched">未匹配</button>
      <button type="button" data-filter="skipped">已跳过</button>
      <button type="button" data-filter="disabled">已禁用</button>
      <button type="button" data-filter="non_movie">非电影</button>
    </section>

    <section class="segment-list">
      {''.join(report_rows)}
    </section>
  </div>

  <script>
    const buttons = document.querySelectorAll('.toolbar button');
    const cards = document.querySelectorAll('.segment-card');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        cards.forEach((card) => {{
          const tags = (card.dataset.tags || '').split(' ');
          card.style.display = filter === 'all' || tags.includes(filter) ? '' : 'none';
        }});
      }});
    }});
  </script>
</body>
</html>
"""

    def _render_candidate(self, candidate) -> str:
        return f"""
<div class="candidate-row">
  <div><strong>#{candidate.rank}</strong> {self._format_time(candidate.start)} - {self._format_time(candidate.end)} · {self._percent(candidate.confidence)}</div>
  <div class="candidate-meta">
    <span>visual {self._percent(candidate.visual_confidence)}</span>
    <span>audio {self._percent(candidate.audio_confidence)}</span>
    <span>temporal {self._percent(candidate.temporal_confidence)}</span>
    <span>stability {self._percent(candidate.stability_score)}</span>
    <span>gap {(candidate.duration_gap or 0.0):.1f}s</span>
  </div>
  <div>{html.escape(candidate.reason or "无说明")}</div>
</div>
"""

    def _render_image_or_empty(self, image_data: str | None, label: str) -> str:
        if not image_data:
            return f'<div class="preview-empty">{html.escape(label)}</div>'
        return f'<img src="{image_data}" alt="{html.escape(label)}" />'

    def _segment_tags(self, segment) -> list[str]:
        tags = []
        if segment.review_required:
            tags.append("review")
        if segment.skip_matching:
            tags.append("skipped")
        if not segment.use_segment:
            tags.append("disabled")
        if segment.segment_type == SegmentType.NON_MOVIE:
            tags.append("non_movie")
        if self._has_movie_match(segment):
            tags.append("matched")
        else:
            tags.append("unmatched")
        return tags

    def _alignment_text(self, status: AlignmentStatus | str | None) -> str:
        mapping = {
            "auto_accepted": "自动通过",
            "needs_review": "待复核",
            "unmatched": "未匹配",
            "skipped": "已跳过",
            "non_movie": "非电影",
            "manual": "手动调整",
            "rematched": "重匹配",
            "pending": "待处理",
        }
        return mapping.get(str(status or "pending"), str(status or "pending"))

    def _has_movie_match(self, segment) -> bool:
        return segment.movie_start is not None and segment.movie_end is not None

    def _format_time(self, seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        total = max(0.0, float(seconds))
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        secs = int(total % 60)
        tenths = int(round((total - int(total)) * 10))
        if tenths == 10:
            secs += 1
            tenths = 0
        if secs == 60:
            minutes += 1
            secs = 0
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}.{tenths}"
        return f"{minutes:02d}:{secs:02d}.{tenths}"

    def _format_optional_seconds(self, value: float | None) -> str:
        if value is None:
            return "--"
        return f"{float(value):.1f}s"

    def _percent(self, value: float | None) -> str:
        return f"{(float(value or 0.0) * 100):.0f}%"

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "project")
        slug = slug.strip("._")
        return slug or "project"
