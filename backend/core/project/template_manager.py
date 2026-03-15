"""模板管理器"""
import json
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings


class TemplateManager:
    """模板管理器

    管理AI润色模板和其他可复用配置
    """

    # 内置润色模板
    BUILTIN_TEMPLATES = {
        "default": {
            "name": "默认风格",
            "description": "通用的电影解说润色风格",
            "template": """你是一位专业的电影解说文案润色师。请对以下解说词进行润色，要求：
1. 保持原意，但使语言更加生动、有感染力
2. 适当使用修辞手法，如比喻、拟人等
3. 控制字数，润色后的文字长度与原文相近（±20%）
4. 保持口语化，适合朗读

原解说词：
{text}

请直接输出润色后的文案，不要有任何解释："""
        },
        "dramatic": {
            "name": "戏剧风格",
            "description": "更加戏剧化、悬疑感强的风格",
            "template": """你是一位擅长制造悬念的电影解说文案润色师。请对以下解说词进行润色，要求：
1. 增强戏剧张力和悬念感
2. 使用更加紧凑有力的句式
3. 适当留白，制造悬念
4. 字数与原文相近

原解说词：
{text}

请直接输出润色后的文案："""
        },
        "humorous": {
            "name": "幽默风格",
            "description": "轻松幽默的解说风格",
            "template": """你是一位风趣幽默的电影解说文案润色师。请对以下解说词进行润色，要求：
1. 保持原意的同时增加幽默感
2. 可以适当加入调侃和吐槽
3. 语言轻松活泼
4. 字数与原文相近

原解说词：
{text}

请直接输出润色后的文案："""
        },
        "professional": {
            "name": "专业风格",
            "description": "专业影评人风格",
            "template": """你是一位专业的影评人。请对以下解说词进行润色，要求：
1. 使用专业但不晦涩的电影术语
2. 分析角度专业、观点独到
3. 语言精炼准确
4. 字数与原文相近

原解说词：
{text}

请直接输出润色后的文案："""
        },
        "minimal": {
            "name": "极简风格",
            "description": "精简直白的解说风格",
            "template": """请精简以下解说词，要求：
1. 保留核心信息
2. 去除冗余表达
3. 简洁有力
4. 字数减少20-30%

原解说词：
{text}

请直接输出精简后的文案："""
        }
    }

    def __init__(self, templates_dir: Optional[Path] = None):
        """初始化

        Args:
            templates_dir: 自定义模板存储目录
        """
        self.templates_dir = Path(templates_dir or settings.projects_dir / "templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def get_template(self, template_id: str) -> Optional[dict]:
        """获取模板

        Args:
            template_id: 模板ID

        Returns:
            模板数据
        """
        # 先检查内置模板
        if template_id in self.BUILTIN_TEMPLATES:
            return self.BUILTIN_TEMPLATES[template_id]

        # 检查自定义模板
        template_path = self.templates_dir / f"{template_id}.json"
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return None

    def list_templates(self) -> list[dict]:
        """列出所有模板

        Returns:
            模板列表
        """
        templates = []

        # 内置模板
        for tid, data in self.BUILTIN_TEMPLATES.items():
            templates.append({
                "id": tid,
                "name": data["name"],
                "description": data["description"],
                "builtin": True
            })

        # 自定义模板
        for template_path in self.templates_dir.glob("*.json"):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                templates.append({
                    "id": template_path.stem,
                    "name": data.get("name", template_path.stem),
                    "description": data.get("description", ""),
                    "builtin": False
                })
            except Exception as e:
                logger.warning(f"读取模板失败: {template_path}, 错误: {e}")

        return templates

    def save_template(self, template_id: str, name: str, description: str, template: str) -> bool:
        """保存自定义模板

        Args:
            template_id: 模板ID
            name: 模板名称
            description: 模板描述
            template: 模板内容

        Returns:
            是否成功
        """
        if template_id in self.BUILTIN_TEMPLATES:
            logger.warning(f"不能覆盖内置模板: {template_id}")
            return False

        if "{text}" not in template:
            logger.error("模板必须包含 {text} 占位符")
            return False

        template_path = self.templates_dir / f"{template_id}.json"
        data = {
            "name": name,
            "description": description,
            "template": template
        }

        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"模板已保存: {template_id}")
        return True

    def delete_template(self, template_id: str) -> bool:
        """删除自定义模板

        Args:
            template_id: 模板ID

        Returns:
            是否成功
        """
        if template_id in self.BUILTIN_TEMPLATES:
            logger.warning(f"不能删除内置模板: {template_id}")
            return False

        template_path = self.templates_dir / f"{template_id}.json"
        if template_path.exists():
            template_path.unlink()
            logger.info(f"模板已删除: {template_id}")
            return True

        return False

    def get_template_content(self, template_id: str) -> Optional[str]:
        """获取模板内容

        Args:
            template_id: 模板ID

        Returns:
            模板文本内容
        """
        template = self.get_template(template_id)
        if template:
            return template.get("template")
        return None

    def apply_template(self, template_id: str, text: str) -> str:
        """应用模板

        Args:
            template_id: 模板ID
            text: 要处理的文本

        Returns:
            填充后的提示词
        """
        template_content = self.get_template_content(template_id)
        if template_content:
            return template_content.format(text=text)
        return text
