"""Run post-fix validation and generate a Claude-ready report."""
from __future__ import annotations

import argparse
import compileall
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

REPO_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from config import settings  # noqa: E402
from main import app  # noqa: E402
from models.project import Project  # noqa: E402


GOAL_LINES = [
    "视频匹配 benchmark >= 95%",
    "没有阻塞性 bug",
    "文案润色尽量自然，减少 AI 味",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-fix validation and generate a Claude handoff report.")
    parser.add_argument("--project-id", default="", help="Optional project id to inspect")
    parser.add_argument("--skip-frontend-build", action="store_true", help="Skip frontend build validation")
    return parser.parse_args()


def find_latest_project() -> Project | None:
    project_files = sorted(
        settings.projects_dir.glob("proj_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in project_files:
        try:
            return Project.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return None


def load_project(project_id: str) -> Project | None:
    path = settings.projects_dir / f"{project_id}.json"
    if not path.exists():
        return None
    return Project.model_validate(json.loads(path.read_text(encoding="utf-8")))


def compile_backend() -> dict[str, Any]:
    ok = compileall.compile_dir(str(BACKEND_DIR), quiet=1)
    return {"ok": ok, "label": "backend compileall"}


def run_api_smoke(project_id: str | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    project_payload = None
    progress_payload = None

    with TestClient(app) as client:
        for method, url in [
            ("GET", "/health"),
            ("GET", "/api"),
            ("GET", "/api/project/list"),
        ]:
            response = client.request(method, url)
            checks.append({"name": f"{method} {url}", "ok": response.status_code == 200, "status": response.status_code})

        if project_id:
            response = client.get(f"/api/project/{project_id}")
            checks.append({"name": f"GET /api/project/{project_id}", "ok": response.status_code == 200, "status": response.status_code})
            if response.status_code == 200:
                project_payload = response.json()

            response = client.get(f"/api/process/{project_id}/progress")
            checks.append({"name": f"GET /api/process/{project_id}/progress", "ok": response.status_code == 200, "status": response.status_code})
            if response.status_code == 200:
                progress_payload = response.json()

    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "project_payload": project_payload,
        "progress_payload": progress_payload,
    }


def run_frontend_build(skip: bool) -> dict[str, Any]:
    if skip:
        return {"ok": True, "skipped": True, "output": "frontend build skipped"}

    result = subprocess.run(
        ["cmd", "/c", "npm run build"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_DIR / "frontend"),
    )
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return {
        "ok": result.returncode == 0,
        "skipped": False,
        "output": tail_text(output, 60),
    }


def collect_recent_log_lines(limit: int = 25) -> list[str]:
    logs_dir = settings.temp_dir / "logs"
    candidates = sorted(logs_dir.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
    lines: list[str] = []
    for path in candidates[:2]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        interesting = [line for line in content if "ERROR" in line or "WARNING" in line]
        lines.extend(interesting[-limit:])
        if len(lines) >= limit:
            break
    return lines[-limit:]


def summarize_project(project: Project | None) -> dict[str, Any]:
    if not project:
        return {}

    segments = project.segments
    matched = [segment for segment in segments if segment.movie_start is not None and segment.movie_end is not None]
    review = [segment for segment in segments if segment.review_required]
    skipped = [segment for segment in segments if segment.skip_matching]
    avg_confidence = (
        sum(float(segment.match_confidence or 0.0) for segment in matched) / len(matched)
        if matched else 0.0
    )
    boundary_errors = [segment.estimated_boundary_error for segment in matched if segment.estimated_boundary_error is not None]
    return {
        "id": project.id,
        "name": project.name,
        "status": project.status,
        "segment_total": len(segments),
        "matched_total": len(matched),
        "review_total": len(review),
        "skipped_total": len(skipped),
        "avg_confidence": avg_confidence,
        "avg_boundary_error": (
            sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
        ),
        "benchmark_accuracy": project.benchmark_accuracy,
        "benchmark_manifest": project.benchmark_manifest,
        "progress_stage": project.progress.stage,
        "progress_message": project.progress.message,
        "progress_value": project.progress.progress,
    }


def render_report(
    project_summary: dict[str, Any],
    compile_result: dict[str, Any],
    api_result: dict[str, Any],
    frontend_result: dict[str, Any],
    log_lines: list[str],
) -> str:
    goal_block = "\n".join(f"- {line}" for line in GOAL_LINES)
    api_checks = "\n".join(
        f"- {'OK' if item['ok'] else 'FAIL'} {item['name']} ({item['status']})"
        for item in api_result["checks"]
    ) or "- 无 API 检查"

    if project_summary:
        benchmark_text = (
            f"{project_summary['benchmark_accuracy'] * 100:.1f}%"
            if project_summary.get("benchmark_accuracy") is not None
            else "未评测"
        )
        boundary_text = (
            f"{project_summary['avg_boundary_error']:.2f}s"
            if project_summary.get("avg_boundary_error") is not None
            else "--"
        )
        project_block = (
            f"- 项目: {project_summary['name']} ({project_summary['id']})\n"
            f"- 状态: {project_summary['status']}\n"
            f"- 片段: {project_summary['segment_total']}\n"
            f"- 已匹配: {project_summary['matched_total']}\n"
            f"- 待复核: {project_summary['review_total']}\n"
            f"- 跳过匹配: {project_summary['skipped_total']}\n"
            f"- 平均置信度: {project_summary['avg_confidence'] * 100:.1f}%\n"
            f"- 平均边界误差: {boundary_text}\n"
            f"- Benchmark: {benchmark_text}\n"
            f"- 当前进度: {project_summary.get('progress_stage') or '--'} / "
            f"{project_summary.get('progress_value') or 0:.1f}% / "
            f"{project_summary.get('progress_message') or '--'}"
        )
    else:
        project_block = "- 未找到可总结的项目"

    if frontend_result.get("skipped"):
        frontend_block = "- frontend build skipped"
    else:
        frontend_block = (
            f"- {'OK' if frontend_result['ok'] else 'FAIL'} frontend build\n\n"
            f"```text\n{frontend_result['output']}\n```"
        )

    logs_block = "\n".join(f"- {line}" for line in log_lines) if log_lines else "- 无最近错误/警告"

    return f"""# Claude Handoff Report

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 当前目标
{goal_block}

## 本轮验证摘要
- {'OK' if compile_result['ok'] else 'FAIL'} {compile_result['label']}
- {'OK' if api_result['ok'] else 'FAIL'} API smoke

## 项目状态
{project_block}

## API Smoke
{api_checks}

## Frontend Build
{frontend_block}

## 最近错误/警告
{logs_block}

## 发给 Claude 的请求
请基于上面的验证结果继续修复当前最高优先级问题。优先级顺序：
1. 阻塞运行或导致处理卡住的问题
2. 影响视频匹配度的错误和低置信热点
3. 导致文案润色出现明显 AI 味的提示词或后处理问题

修复后请再次运行同一份验证脚本，并基于新的报告继续迭代。
"""


def tail_text(text: str, line_limit: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-line_limit:])


def main() -> int:
    args = parse_args()

    project = load_project(args.project_id) if args.project_id else None
    if project is None:
        project = find_latest_project()

    compile_result = compile_backend()
    api_result = run_api_smoke(project.id if project else None)
    frontend_result = run_frontend_build(args.skip_frontend_build)
    project_summary = summarize_project(project)
    log_lines = collect_recent_log_lines()

    report = render_report(
        project_summary=project_summary,
        compile_result=compile_result,
        api_result=api_result,
        frontend_result=frontend_result,
        log_lines=log_lines,
    )

    output_dir = settings.temp_dir / "claude_feedback"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latest_report.md"
    prompt_path = output_dir / "latest_prompt.txt"
    report_path.write_text(report, encoding="utf-8")
    prompt_path.write_text(report, encoding="utf-8")

    print(f"Report written: {report_path}")
    print(f"Prompt written: {prompt_path}")
    if project_summary:
        print(
            "Project summary:",
            project_summary["id"],
            f"matched={project_summary['matched_total']}/{project_summary['segment_total']}",
            f"avg_confidence={project_summary['avg_confidence'] * 100:.1f}%",
        )
    print(f"Compile: {'OK' if compile_result['ok'] else 'FAIL'}")
    print(f"API smoke: {'OK' if api_result['ok'] else 'FAIL'}")
    print(f"Frontend build: {'OK' if frontend_result['ok'] else 'FAIL'}")
    return 0 if compile_result["ok"] and api_result["ok"] and frontend_result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
