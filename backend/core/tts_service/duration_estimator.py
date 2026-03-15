"""TTS时长预估模块"""
import re
from typing import Optional


class DurationEstimator:
    """TTS时长预估器

    基于文本特征预估语音合成后的时长
    """

    # 不同语言的平均语速（字/秒）
    SPEECH_RATES = {
        "zh": 4.0,  # 中文
        "en": 2.5,  # 英文（单词/秒）
        "ja": 5.0,  # 日语
        "ko": 4.5,  # 韩语
    }

    # 标点符号停顿时间（秒）
    PUNCTUATION_PAUSES = {
        "。": 0.4,
        "！": 0.4,
        "？": 0.4,
        ".": 0.4,
        "!": 0.4,
        "?": 0.4,
        "，": 0.2,
        ",": 0.2,
        "、": 0.15,
        "；": 0.3,
        ";": 0.3,
        "：": 0.25,
        ":": 0.25,
        "……": 0.5,
        "...": 0.5,
        "——": 0.3,
        "--": 0.3,
    }

    def __init__(self, language: str = "zh", speed_factor: float = 1.0):
        """初始化

        Args:
            language: 语言代码
            speed_factor: 语速倍率（1.0为正常）
        """
        self.language = language
        self.speed_factor = speed_factor
        self.base_rate = self.SPEECH_RATES.get(language, 4.0)

    def estimate(self, text: str) -> float:
        """预估TTS时长

        Args:
            text: 文本内容

        Returns:
            预估时长(秒)
        """
        if not text or not text.strip():
            return 0.0

        # 计算基础时长
        if self.language == "en":
            # 英文按单词计算
            words = len(text.split())
            base_duration = words / self.base_rate
        else:
            # 其他语言按字符计算
            # 移除标点和空格
            clean_text = re.sub(r'[^\w]', '', text)
            char_count = len(clean_text)
            base_duration = char_count / self.base_rate

        # 计算标点停顿
        pause_duration = 0.0
        for punct, pause in self.PUNCTUATION_PAUSES.items():
            pause_duration += text.count(punct) * pause

        # 应用语速倍率
        total_duration = (base_duration + pause_duration) / self.speed_factor

        return round(total_duration, 2)

    def estimate_with_details(self, text: str) -> dict:
        """详细预估TTS时长

        Args:
            text: 文本内容

        Returns:
            详细预估结果
        """
        if not text or not text.strip():
            return {
                "duration": 0.0,
                "char_count": 0,
                "word_count": 0,
                "punctuation_count": 0,
                "base_duration": 0.0,
                "pause_duration": 0.0
            }

        # 字符统计
        clean_text = re.sub(r'[^\w]', '', text)
        char_count = len(clean_text)
        word_count = len(text.split())

        # 标点统计
        punctuation_count = sum(text.count(p) for p in self.PUNCTUATION_PAUSES.keys())

        # 基础时长
        if self.language == "en":
            base_duration = word_count / self.base_rate
        else:
            base_duration = char_count / self.base_rate

        # 停顿时长
        pause_duration = sum(
            text.count(p) * t
            for p, t in self.PUNCTUATION_PAUSES.items()
        )

        # 总时长
        total_duration = (base_duration + pause_duration) / self.speed_factor

        return {
            "duration": round(total_duration, 2),
            "char_count": char_count,
            "word_count": word_count,
            "punctuation_count": punctuation_count,
            "base_duration": round(base_duration, 2),
            "pause_duration": round(pause_duration, 2)
        }

    def adjust_text_for_duration(
        self,
        text: str,
        target_duration: float,
        tolerance: float = 0.1
    ) -> Optional[str]:
        """调整文本以匹配目标时长

        返回建议：文本过长/过短需要调整的幅度

        Args:
            text: 原文本
            target_duration: 目标时长(秒)
            tolerance: 容差比例

        Returns:
            调整建议，None表示无需调整
        """
        if target_duration <= 0:
            return None

        current_duration = self.estimate(text)

        if current_duration == 0:
            return None

        ratio = current_duration / target_duration

        if 1 - tolerance <= ratio <= 1 + tolerance:
            return None  # 在容差范围内

        if ratio > 1 + tolerance:
            # 文本过长
            excess_percent = int((ratio - 1) * 100)
            return f"文本过长约{excess_percent}%，建议精简"
        else:
            # 文本过短
            deficit_percent = int((1 - ratio) * 100)
            return f"文本过短约{deficit_percent}%，建议扩展"

    def calculate_required_speed(self, text: str, target_duration: float) -> float:
        """计算达到目标时长需要的语速倍率

        Args:
            text: 文本内容
            target_duration: 目标时长(秒)

        Returns:
            需要的语速倍率
        """
        # 用正常语速估算
        normal_duration = DurationEstimator(
            self.language, speed_factor=1.0
        ).estimate(text)

        if target_duration <= 0:
            return 1.0

        return round(normal_duration / target_duration, 2)
