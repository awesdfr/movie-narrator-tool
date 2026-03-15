"""项目管理器"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings
from models.project import Project


class ProjectManager:
    """项目管理器

    负责项目的保存、加载、备份等操作
    """

    def __init__(self, projects_dir: Optional[Path] = None):
        """初始化

        Args:
            projects_dir: 项目存储目录
        """
        self.projects_dir = Path(projects_dir or settings.projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def get_project_path(self, project_id: str) -> Path:
        """获取项目文件路径"""
        return self.projects_dir / f"{project_id}.json"

    def save(self, project: Project) -> Path:
        """保存项目

        Args:
            project: 项目数据

        Returns:
            保存的文件路径
        """
        project.updated_at = datetime.now()
        file_path = self.get_project_path(project.id)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        logger.debug(f"项目已保存: {file_path}")
        return file_path

    def load(self, project_id: str) -> Optional[Project]:
        """加载项目

        Args:
            project_id: 项目ID

        Returns:
            项目数据，不存在则返回None
        """
        file_path = self.get_project_path(project_id)

        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Project.from_dict(data)

    def delete(self, project_id: str) -> bool:
        """删除项目

        Args:
            project_id: 项目ID

        Returns:
            是否删除成功
        """
        file_path = self.get_project_path(project_id)

        if file_path.exists():
            file_path.unlink()
            logger.info(f"项目已删除: {project_id}")
            return True

        return False

    def list_projects(self) -> list[dict]:
        """列出所有项目

        Returns:
            项目摘要列表
        """
        projects = []

        for file_path in self.projects_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                projects.append({
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "segment_count": len(data.get("segments", [])),
                })
            except Exception as e:
                logger.warning(f"读取项目文件失败: {file_path}, 错误: {e}")

        # 按更新时间排序
        projects.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return projects

    def backup(self, project_id: str) -> Optional[Path]:
        """备份项目

        Args:
            project_id: 项目ID

        Returns:
            备份文件路径
        """
        source_path = self.get_project_path(project_id)
        if not source_path.exists():
            return None

        backup_dir = self.projects_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{project_id}_{timestamp}.json"

        shutil.copy2(source_path, backup_path)
        logger.info(f"项目已备份: {backup_path}")

        return backup_path

    def restore(self, backup_path: Path) -> Optional[Project]:
        """从备份恢复项目

        Args:
            backup_path: 备份文件路径

        Returns:
            恢复的项目
        """
        if not backup_path.exists():
            return None

        with open(backup_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = Project.from_dict(data)
        self.save(project)

        logger.info(f"项目已恢复: {project.id}")
        return project

    def export_project(self, project_id: str, output_path: Path) -> bool:
        """导出项目到指定位置

        Args:
            project_id: 项目ID
            output_path: 输出路径

        Returns:
            是否成功
        """
        source_path = self.get_project_path(project_id)
        if not source_path.exists():
            return False

        shutil.copy2(source_path, output_path)
        return True

    def import_project(self, import_path: Path) -> Optional[Project]:
        """从文件导入项目

        Args:
            import_path: 导入文件路径

        Returns:
            导入的项目
        """
        if not import_path.exists():
            return None

        with open(import_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = Project.from_dict(data)
        self.save(project)

        logger.info(f"项目已导入: {project.id}")
        return project

    def get_project_size(self, project_id: str) -> int:
        """获取项目文件大小（字节）"""
        file_path = self.get_project_path(project_id)
        if file_path.exists():
            return file_path.stat().st_size
        return 0

    def cleanup_backups(self, keep_count: int = 10):
        """清理旧备份

        Args:
            keep_count: 保留的备份数量
        """
        backup_dir = self.projects_dir / "backups"
        if not backup_dir.exists():
            return

        backups = sorted(backup_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

        for backup in backups[keep_count:]:
            backup.unlink()
            logger.debug(f"清理旧备份: {backup}")
