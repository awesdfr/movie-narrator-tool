"""字幕匹配模块 — 四信号融合匹配

通过英文台词直接匹配、专有名词匹配、时间线性插值、上下文窗口关键词聚合
四种信号的融合来定位解说字幕对应的电影时间点。

两阶段执行：
1. 锚点发现：用英文台词 + 专有名词找高置信度锚点 → LIS 过滤 → 注入插值器
2. 全量匹配：四信号融合，锚点插值兜底，确保全部条目有时间估计
"""
import re
import math
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from loguru import logger


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SubtitleEntry:
    """字幕条目"""
    index: int
    start_time: float       # 秒
    end_time: float         # 秒
    text: str               # 纯文本内容（完整）
    raw_text: str           # 原始文本（含格式标签）
    chinese_text: str = ""  # 中文部分
    english_text: str = ""  # 英文部分
    entities: list = field(default_factory=list)  # 提取到的实体词


# ---------------------------------------------------------------------------
# 字幕解析器
# ---------------------------------------------------------------------------

class SubtitleParser:
    """字幕解析器，支持 SRT 和 ASS 格式"""

    @staticmethod
    def parse_time_srt(time_str: str) -> float:
        """解析 SRT 时间格式 00:00:00,000"""
        time_str = time_str.strip().replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        return 0.0

    @staticmethod
    def parse_time_ass(time_str: str) -> float:
        """解析 ASS 时间格式 0:00:00.00"""
        time_str = time_str.strip()
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        return 0.0

    @staticmethod
    def clean_ass_text(text: str) -> str:
        """清理 ASS 格式标签，保留纯文本"""
        text = re.sub(r'\{[^}]*\}', '', text)
        text = text.replace('\\N', ' ').replace('\\n', ' ')
        return text.strip()

    @staticmethod
    def clean_srt_text(text: str) -> str:
        """清理 SRT 格式标签"""
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    @staticmethod
    def _split_ass_chinese_english(raw_text: str) -> tuple[str, str]:
        """从 ASS Dialogue 的 raw_text 中分离中英文

        典型格式: "杜弗伦先生  请你\\N{\\rEng}Mr. Dufresne, describe"
        - \\N{\\rEng} 之前是中文
        - \\N{\\rEng} 之后是英文
        """
        # 匹配 \N{\rEng} 或 \N{\r Eng} 等变体
        pattern = r'\\N\s*\{\\r\s*Eng\s*\}'
        parts = re.split(pattern, raw_text, maxsplit=1)
        if len(parts) == 2:
            chinese_raw = parts[0]
            english_raw = parts[1]
        else:
            # 没有英文样式标记，判断内容语言
            chinese_raw = raw_text
            english_raw = ""

        # 清理格式标签
        chinese = re.sub(r'\{[^}]*\}', '', chinese_raw).strip()
        english = re.sub(r'\{[^}]*\}', '', english_raw).strip()
        return chinese, english

    @staticmethod
    def _detect_english_in_srt(text: str) -> tuple[str, str]:
        """检测 SRT 文本中的英文内容

        解说字幕中可能夹杂英文原台词。
        判断标准：英文字符（含空格）占比 > 50% 视为英文行。
        """
        if not text:
            return text, ""
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if not c.isspace())
        if total_chars == 0:
            return text, ""

        ratio = ascii_chars / total_chars
        if ratio > 0.5:
            # 主要是英文
            return "", text
        return text, ""

    def parse_file(self, filepath: str) -> list[SubtitleEntry]:
        """解析字幕文件，自动检测格式"""
        path = Path(filepath)
        content = path.read_text(encoding='utf-8', errors='ignore')

        if '[Script Info]' in content or 'Dialogue:' in content:
            return self._parse_ass(content)
        else:
            return self._parse_srt(content)

    def _parse_srt(self, content: str) -> list[SubtitleEntry]:
        """解析 SRT 格式，检测英文行"""
        entries = []
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0])
                time_line = lines[1]
                text_lines = lines[2:]

                match = re.match(
                    r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
                    time_line
                )
                if not match:
                    continue

                start = self.parse_time_srt(match.group(1))
                end = self.parse_time_srt(match.group(2))
                raw_text = '\n'.join(text_lines)
                clean_text = self.clean_srt_text(raw_text)
                chinese, english = self._detect_english_in_srt(clean_text)

                entries.append(SubtitleEntry(
                    index=index,
                    start_time=start,
                    end_time=end,
                    text=clean_text,
                    raw_text=raw_text,
                    chinese_text=chinese,
                    english_text=english,
                ))
            except (ValueError, IndexError):
                continue

        return entries

    def _parse_ass(self, content: str) -> list[SubtitleEntry]:
        """解析 ASS 格式，按 \\N{\\rEng} 分离中英文"""
        entries = []
        index = 0

        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith('Dialogue:'):
                continue

            # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line[9:].split(',', 9)
            if len(parts) < 10:
                continue

            try:
                start = self.parse_time_ass(parts[1])
                end = self.parse_time_ass(parts[2])
                raw_text = parts[9]
                clean_text = self.clean_ass_text(raw_text)
                chinese, english = self._split_ass_chinese_english(raw_text)

                if not clean_text or len(clean_text) < 2:
                    continue

                index += 1
                entries.append(SubtitleEntry(
                    index=index,
                    start_time=start,
                    end_time=end,
                    text=clean_text,
                    raw_text=raw_text,
                    chinese_text=chinese,
                    english_text=english,
                ))
            except (ValueError, IndexError):
                continue

        return entries


# ---------------------------------------------------------------------------
# 信号 1：英文台词直接匹配
# ---------------------------------------------------------------------------

class EnglishLineMatcher:
    """英文台词匹配器

    从电影字幕提取英文部分，建立 n-gram 倒排索引。
    对解说中的英文行进行 n-gram 匹配 + 模糊匹配。
    """

    def __init__(self, movie_subs: list[SubtitleEntry]):
        # n-gram 倒排索引: ngram_str -> [(entry, ngram位置)]
        self._index_3gram: dict[str, list[SubtitleEntry]] = defaultdict(list)
        self._index_4gram: dict[str, list[SubtitleEntry]] = defaultdict(list)
        self._index_5gram: dict[str, list[SubtitleEntry]] = defaultdict(list)
        self._movie_english: list[tuple[SubtitleEntry, str]] = []  # (entry, normalized_english)
        self._build_index(movie_subs)

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化英文文本：小写，去标点，合并空格"""
        text = text.lower()
        text = re.sub(r"[^\w\s']", ' ', text)  # 保留撇号（it's, don't）
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _make_ngrams(text: str, n: int) -> list[str]:
        """生成词级 n-gram"""
        words = text.split()
        if len(words) < n:
            return [text] if words else []
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def _build_index(self, movie_subs: list[SubtitleEntry]):
        """为电影字幕英文部分建立 n-gram 倒排索引"""
        for entry in movie_subs:
            eng = entry.english_text
            if not eng or len(eng) < 3:
                continue
            norm = self._normalize(eng)
            if not norm:
                continue
            self._movie_english.append((entry, norm))

            for gram in self._make_ngrams(norm, 3):
                self._index_3gram[gram].append(entry)
            for gram in self._make_ngrams(norm, 4):
                self._index_4gram[gram].append(entry)
            for gram in self._make_ngrams(norm, 5):
                self._index_5gram[gram].append(entry)

        logger.info(f"英文索引: {len(self._movie_english)} 条电影英文字幕, "
                    f"3-gram {len(self._index_3gram)}, "
                    f"4-gram {len(self._index_4gram)}, "
                    f"5-gram {len(self._index_5gram)}")

    def match(self, narration_entry: SubtitleEntry) -> Optional[tuple[SubtitleEntry, float]]:
        """匹配一条解说英文到电影字幕

        Returns:
            (匹配到的电影字幕条目, 置信度) 或 None
        """
        eng = narration_entry.english_text
        if not eng or len(eng) < 3:
            return None

        norm = self._normalize(eng)
        if not norm or len(norm.split()) < 2:
            return None

        # 阶段 1: n-gram 候选收集
        candidate_scores: dict[int, float] = defaultdict(float)  # entry.index -> score
        candidate_map: dict[int, SubtitleEntry] = {}

        # 5-gram 权重最高
        for gram in self._make_ngrams(norm, 5):
            for entry in self._index_5gram.get(gram, []):
                candidate_scores[entry.index] += 3.0
                candidate_map[entry.index] = entry

        # 4-gram
        for gram in self._make_ngrams(norm, 4):
            for entry in self._index_4gram.get(gram, []):
                candidate_scores[entry.index] += 2.0
                candidate_map[entry.index] = entry

        # 3-gram
        for gram in self._make_ngrams(norm, 3):
            for entry in self._index_3gram.get(gram, []):
                candidate_scores[entry.index] += 1.0
                candidate_map[entry.index] = entry

        if not candidate_scores:
            # n-gram 无命中，尝试全量模糊匹配（仅当解说英文足够长时）
            if len(norm.split()) >= 4:
                return self._fuzzy_match_all(norm)
            return None

        # 取 top-5 候选，用 SequenceMatcher 精排
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        best_entry = None
        best_ratio = 0.0

        for idx, ngram_score in top_candidates:
            entry = candidate_map[idx]
            movie_norm = self._normalize(entry.english_text)
            ratio = SequenceMatcher(None, norm, movie_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_entry = entry

        if best_entry is None:
            return None

        # 置信度映射: ratio 0.4-1.0 → confidence 0.3-1.0
        if best_ratio >= 0.8:
            confidence = 0.9 + (best_ratio - 0.8) * 0.5  # 0.9-1.0
        elif best_ratio >= 0.6:
            confidence = 0.6 + (best_ratio - 0.6) * 1.5  # 0.6-0.9
        elif best_ratio >= 0.4:
            confidence = 0.3 + (best_ratio - 0.4) * 1.5  # 0.3-0.6
        else:
            return None  # ratio < 0.4 太低，不可靠

        confidence = min(1.0, confidence)
        return best_entry, confidence

    def _fuzzy_match_all(self, norm_query: str) -> Optional[tuple[SubtitleEntry, float]]:
        """对全部电影英文字幕做模糊匹配（慢，仅用于 n-gram 无命中时的兜底）"""
        best_entry = None
        best_ratio = 0.0

        for entry, movie_norm in self._movie_english:
            # 快速过滤：长度差异过大的跳过
            if abs(len(norm_query) - len(movie_norm)) > max(len(norm_query), len(movie_norm)) * 0.6:
                continue
            ratio = SequenceMatcher(None, norm_query, movie_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_entry = entry

        if best_entry and best_ratio >= 0.5:
            confidence = 0.3 + (best_ratio - 0.5) * 1.4
            confidence = min(1.0, confidence)
            return best_entry, confidence
        return None


# ---------------------------------------------------------------------------
# 信号 2：专有名词匹配
# ---------------------------------------------------------------------------

class EntityExtractor:
    """实体提取器

    从电影字幕自动学习高频实体词典（角色名、地名），
    并内置别名映射表，用于解说与电影字幕的实体共现匹配。
    """

    # 内置别名映射（同一实体的不同表述）
    # 每组中的任意一个出现都视为同一实体
    ALIAS_GROUPS = [
        {"andy", "安迪", "dufresne", "杜弗伦", "杜弗雷恩"},
        {"red", "瑞德", "redding", "雷丁"},
        {"shawshank", "肖申克", "鲨堡"},
        {"norton", "诺顿", "典狱长"},
        {"brooks", "布鲁克斯", "老布"},
        {"tommy", "汤米", "汤姆"},
        {"hadley", "海利", "哈德利"},
        {"bogs", "伯格斯", "博格斯", "三姐妹"},
        {"jake", "杰克"},
        {"zihuatanejo", "芝华塔内欧", "芝华塔尼欧"},
        {"mexico", "墨西哥"},
        {"rita", "丽塔", "hayworth", "海华丝"},
        {"raquel", "拉蔻儿", "welch", "韦尔奇"},
    ]

    # 实体权重
    WEIGHT_CHARACTER = 3.0    # 角色名
    WEIGHT_LOCATION = 2.0     # 地名
    WEIGHT_COMMON = 0.5       # 通用词

    def __init__(self, movie_subs: list[SubtitleEntry]):
        # 标准化别名 → 规范名
        self._alias_to_canon: dict[str, str] = {}
        for group in self.ALIAS_GROUPS:
            canon = sorted(group)[0]  # 取字母序最小的作为规范名
            for alias in group:
                self._alias_to_canon[alias.lower()] = canon

        # 从电影字幕学习高频实体
        self._entity_dict: dict[str, float] = {}  # entity -> weight
        self._movie_entity_index: dict[str, list[SubtitleEntry]] = defaultdict(list)
        self._build_entity_dict(movie_subs)

    def _build_entity_dict(self, movie_subs: list[SubtitleEntry]):
        """从电影字幕学习实体词典并建索引"""
        # 统计中英文词频
        cn_counter: Counter = Counter()
        en_counter: Counter = Counter()

        for entry in movie_subs:
            cn_words = re.findall(r'[\u4e00-\u9fff]{2,4}', entry.chinese_text)
            en_words = [w.lower() for w in re.findall(r"[a-zA-Z']{2,}", entry.english_text)]
            cn_counter.update(cn_words)
            en_counter.update(en_words)

        # 停用词（高频但无意义的词）
        cn_stopwords = {"这个", "那个", "什么", "就是", "可以", "我们", "他们", "你们",
                        "不是", "没有", "知道", "现在", "已经", "但是", "因为", "所以",
                        "如果", "这样", "那样", "的是", "是的", "怎么", "为什么", "这里",
                        "那里", "一个", "一些", "一样", "还是", "只是", "不过", "而且",
                        "先生", "请你", "不要", "应该", "真的", "觉得", "看到", "告诉",
                        "认为", "希望", "喜欢", "需要", "不能", "也许", "可能", "或者"}
        en_stopwords = {"the", "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "do", "does", "did", "will", "would",
                        "could", "should", "may", "might", "shall", "can", "need",
                        "and", "but", "or", "nor", "not", "no", "yes", "you", "your",
                        "he", "him", "his", "she", "her", "it", "its", "they", "them",
                        "their", "we", "our", "my", "me", "who", "what", "which",
                        "that", "this", "these", "those", "there", "here", "where",
                        "when", "how", "why", "all", "any", "some", "one", "two",
                        "with", "for", "from", "about", "into", "than", "then",
                        "just", "don", "didn", "doesn", "won", "wouldn", "couldn",
                        "very", "too", "also", "only", "now", "out", "well", "get",
                        "got", "going", "know", "think", "want", "like", "say", "said",
                        "tell", "told", "come", "came", "take", "took", "make", "made",
                        "let", "look", "see", "saw", "give", "gave", "back", "over"}

        # 实体词典：出现频次在 [2, 50] 区间的可能是角色名/地名
        # 太高频是通用词，太低频可能是噪声
        for word, count in cn_counter.items():
            if word in cn_stopwords or count < 2:
                continue
            canon = self._alias_to_canon.get(word.lower(), word.lower())
            # 在别名表中的给高权重
            if word.lower() in self._alias_to_canon:
                self._entity_dict[canon] = self.WEIGHT_CHARACTER
            elif 2 <= count <= 50:
                self._entity_dict.setdefault(canon, self.WEIGHT_COMMON)

        for word, count in en_counter.items():
            if word in en_stopwords or count < 2 or len(word) < 3:
                continue
            canon = self._alias_to_canon.get(word.lower(), word.lower())
            if word.lower() in self._alias_to_canon:
                self._entity_dict[canon] = self.WEIGHT_CHARACTER
            elif 2 <= count <= 50:
                self._entity_dict.setdefault(canon, self.WEIGHT_COMMON)

        # 为电影字幕建实体索引
        for entry in movie_subs:
            entities = self.extract_entities(entry.text + " " + entry.chinese_text + " " + entry.english_text)
            entry.entities = [e for e, _ in entities]
            for entity, _ in entities:
                self._movie_entity_index[entity].append(entry)

        logger.info(f"实体词典: {len(self._entity_dict)} 个实体, "
                    f"别名映射: {len(self._alias_to_canon)} 个")

    def extract_entities(self, text: str) -> list[tuple[str, float]]:
        """从文本中提取实体及其权重

        Returns:
            [(规范名, 权重), ...]
        """
        found = {}
        text_lower = text.lower()

        # 检查别名表中的所有词
        for alias, canon in self._alias_to_canon.items():
            if alias in text_lower:
                weight = self._entity_dict.get(canon, self.WEIGHT_COMMON)
                found[canon] = max(found.get(canon, 0), weight)

        # 检查学习到的实体词典
        cn_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        en_words = [w.lower() for w in re.findall(r"[a-zA-Z']{2,}", text)]
        for word in cn_words + en_words:
            canon = self._alias_to_canon.get(word.lower(), word.lower())
            if canon in self._entity_dict:
                found[canon] = max(found.get(canon, 0), self._entity_dict[canon])

        return list(found.items())

    def match(self, narration_entry: SubtitleEntry) -> list[tuple[SubtitleEntry, float]]:
        """基于实体共现匹配解说到电影字幕

        Returns:
            [(电影字幕条目, 置信度), ...] 按置信度降序，最多返回 3 个
        """
        narr_entities = self.extract_entities(narration_entry.text)
        if not narr_entities:
            return []

        narr_entity_set = {e for e, _ in narr_entities}
        narr_total_weight = sum(w for _, w in narr_entities)

        # 收集候选：与解说有共同实体的电影字幕
        candidate_scores: dict[int, tuple[SubtitleEntry, float]] = {}

        for entity, weight in narr_entities:
            for movie_entry in self._movie_entity_index.get(entity, []):
                idx = movie_entry.index
                if idx not in candidate_scores:
                    candidate_scores[idx] = (movie_entry, 0.0)
                entry, score = candidate_scores[idx]
                candidate_scores[idx] = (entry, score + weight)

        if not candidate_scores:
            return []

        # 计算置信度并排序
        results = []
        for idx, (entry, score) in candidate_scores.items():
            # 置信度 = 共现权重 / 解说实体总权重
            confidence = min(1.0, score / max(narr_total_weight, 1.0))
            # 惩罚只有通用词匹配的低质量结果
            if confidence < 0.15:
                continue
            results.append((entry, confidence))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:3]


# ---------------------------------------------------------------------------
# 信号 3：时间线性映射 + 锚点插值
# ---------------------------------------------------------------------------

class TimelineInterpolator:
    """时间线插值器

    解说视频按时间线顺序覆盖原片，基于时长比例做线性映射，
    有锚点后改为分段线性插值提升精度。
    """

    def __init__(self, narration_duration: float, movie_duration: float):
        self._narration_duration = narration_duration
        self._movie_duration = movie_duration
        self._ratio = movie_duration / max(narration_duration, 0.1)
        # 锚点列表: [(narration_time, movie_time)] 按 narration_time 排序
        self._anchors: list[tuple[float, float]] = []
        logger.info(f"时间线插值器: 解说 {narration_duration:.1f}s, "
                    f"电影 {movie_duration:.1f}s, 比例 {self._ratio:.2f}x")

    def add_anchors(self, anchors: list[tuple[float, float]]):
        """注入锚点（已经过 LIS 过滤），按 narration_time 排序"""
        self._anchors = sorted(anchors, key=lambda x: x[0])
        logger.info(f"注入 {len(self._anchors)} 个锚点到插值器")

    def interpolate(self, narration_time: float) -> tuple[float, float]:
        """根据解说时间估算电影时间

        Returns:
            (估算的电影时间, 置信度)
        """
        if not self._anchors:
            # 无锚点，纯线性映射
            movie_time = narration_time * self._ratio
            movie_time = max(0, min(movie_time, self._movie_duration))
            return movie_time, 0.3  # 基础置信度低

        # 分段线性插值
        # 找到 narration_time 所在的锚点区间
        if narration_time <= self._anchors[0][0]:
            # 在第一个锚点之前：用第一个锚点和起点插值
            n0, m0 = 0.0, 0.0
            n1, m1 = self._anchors[0]
        elif narration_time >= self._anchors[-1][0]:
            # 在最后一个锚点之后：用最后锚点和终点插值
            n0, m0 = self._anchors[-1]
            n1, m1 = self._narration_duration, self._movie_duration
        else:
            # 在两个锚点之间：二分查找
            left, right = 0, len(self._anchors) - 1
            while left < right - 1:
                mid = (left + right) // 2
                if self._anchors[mid][0] <= narration_time:
                    left = mid
                else:
                    right = mid
            n0, m0 = self._anchors[left]
            n1, m1 = self._anchors[right]

        # 线性插值
        if abs(n1 - n0) < 0.01:
            movie_time = m0
        else:
            t = (narration_time - n0) / (n1 - n0)
            movie_time = m0 + t * (m1 - m0)

        movie_time = max(0, min(movie_time, self._movie_duration))

        # 置信度：距最近锚点越近越高
        min_dist = min(abs(narration_time - a[0]) for a in self._anchors)
        # 距离 0 → 置信度 0.7，距离 60s → 0.3，距离 120s+ → 0.2
        confidence = 0.7 * math.exp(-min_dist / 60.0) + 0.2
        confidence = max(0.2, min(0.7, confidence))

        return movie_time, confidence


# ---------------------------------------------------------------------------
# 信号 4：上下文窗口关键词聚合
# ---------------------------------------------------------------------------

class ContextWindowMatcher:
    """上下文窗口关键词聚合匹配器

    单条解说太短，滑动窗口聚合 5-10 条后关键词更丰富。
    手写轻量 TF-IDF 加权，在电影字幕中搜索相同实体组合出现的时间区域。
    """

    def __init__(self, movie_subs: list[SubtitleEntry], entity_extractor: EntityExtractor,
                 window_size: int = 7):
        self._movie_subs = movie_subs
        self._entity_extractor = entity_extractor
        self._window_size = window_size

        # 建电影字幕分时段索引（每 30 秒一个区间）
        self._time_buckets: dict[int, list[SubtitleEntry]] = defaultdict(list)
        self._bucket_entities: dict[int, Counter] = defaultdict(Counter)
        self._build_time_index()

        # 计算 IDF
        self._idf: dict[str, float] = {}
        self._compute_idf()

    def _build_time_index(self):
        """将电影字幕按 30 秒桶分组"""
        for entry in self._movie_subs:
            bucket = int(entry.start_time // 30)
            self._time_buckets[bucket].append(entry)
            for entity in entry.entities:
                self._bucket_entities[bucket][entity] += 1

    def _compute_idf(self):
        """计算实体的 IDF 值"""
        total_buckets = max(len(self._time_buckets), 1)
        entity_doc_count: Counter = Counter()

        for bucket_entities in self._bucket_entities.values():
            for entity in bucket_entities:
                entity_doc_count[entity] += 1

        for entity, count in entity_doc_count.items():
            self._idf[entity] = math.log(total_buckets / (1 + count))

    def match_window(self, narration_entries: list[SubtitleEntry]) -> Optional[tuple[float, float]]:
        """对一个解说窗口做关键词聚合匹配

        Args:
            narration_entries: 窗口内的解说字幕列表

        Returns:
            (估算的电影时间, 置信度) 或 None
        """
        # 收集窗口内所有实体
        window_entities: Counter = Counter()
        for entry in narration_entries:
            entities = self._entity_extractor.extract_entities(entry.text)
            for entity, weight in entities:
                window_entities[entity] += weight

        if not window_entities:
            return None

        # TF-IDF 加权匹配每个时间桶
        best_bucket = -1
        best_score = 0.0

        for bucket_id, bucket_entities in self._bucket_entities.items():
            score = 0.0
            for entity, tf_weight in window_entities.items():
                if entity in bucket_entities:
                    idf = self._idf.get(entity, 1.0)
                    score += tf_weight * idf * bucket_entities[entity]
            if score > best_score:
                best_score = score
                best_bucket = bucket_id

        if best_bucket < 0 or best_score < 0.5:
            return None

        # 桶的中心时间
        movie_time = (best_bucket + 0.5) * 30.0

        # 置信度：基于匹配得分
        total_weight = sum(window_entities.values())
        confidence = min(0.7, best_score / max(total_weight * 3, 1.0))
        confidence = max(0.1, confidence)

        return movie_time, confidence


# ---------------------------------------------------------------------------
# 四信号融合匹配器
# ---------------------------------------------------------------------------

class SubtitleMatcher:
    """字幕匹配器 — 四信号融合

    通过比较解说字幕和电影字幕来定位对应的电影时间点。
    信号 1: 英文台词直接匹配 (权重 0.45)
    信号 2: 专有名词匹配 (权重 0.25)
    信号 3: 时间线性映射 + 锚点插值 (权重 0.15)
    信号 4: 上下文窗口关键词聚合 (权重 0.15)
    """

    WEIGHT_ENGLISH = 0.45
    WEIGHT_ENTITY = 0.25
    WEIGHT_TIMELINE = 0.15
    WEIGHT_CONTEXT = 0.15

    def __init__(self, movie_subtitle_path: str, narration_subtitle_path: str):
        self.parser = SubtitleParser()

        logger.info(f"加载电影字幕: {movie_subtitle_path}")
        self.movie_subs = self.parser.parse_file(movie_subtitle_path)
        logger.info(f"电影字幕条目数: {len(self.movie_subs)}")

        logger.info(f"加载解说字幕: {narration_subtitle_path}")
        self.narration_subs = self.parser.parse_file(narration_subtitle_path)
        logger.info(f"解说字幕条目数: {len(self.narration_subs)}")

        # 时长
        narration_duration = max((e.end_time for e in self.narration_subs), default=0)
        movie_duration = max((e.end_time for e in self.movie_subs), default=0)

        # 初始化四个信号组件
        self._english_matcher = EnglishLineMatcher(self.movie_subs)
        self._entity_extractor = EntityExtractor(self.movie_subs)
        self._timeline = TimelineInterpolator(narration_duration, movie_duration)
        self._context_matcher = ContextWindowMatcher(
            self.movie_subs, self._entity_extractor, window_size=7
        )

    def match_all_narration_segments(self) -> list[dict]:
        """匹配所有解说片段到电影时间（两阶段执行）

        Returns:
            匹配结果列表
        """
        # === 阶段 1：锚点发现 ===
        logger.info("=== 阶段 1：锚点发现 ===")
        anchors = self._discover_anchors()
        logger.info(f"发现 {len(anchors)} 个锚点")

        # LIS 单调性过滤
        filtered_anchors = self._lis_filter(anchors)
        logger.info(f"LIS 过滤后 {len(filtered_anchors)} 个锚点")

        # 注入插值器
        anchor_pairs = [(a['narration_time'], a['movie_time']) for a in filtered_anchors]
        self._timeline.add_anchors(anchor_pairs)

        # === 阶段 2：全量匹配 ===
        logger.info("=== 阶段 2：全量匹配 ===")
        results = self._full_match()

        # 统计
        self._log_statistics(results)

        return results

    def _discover_anchors(self) -> list[dict]:
        """用信号 1（英文）和信号 2（实体）发现高置信度锚点"""
        anchors = []

        for entry in self.narration_subs:
            best_time = None
            best_confidence = 0.0
            source = "none"

            # 信号 1：英文台词匹配
            eng_match = self._english_matcher.match(entry)
            if eng_match:
                movie_entry, confidence = eng_match
                if confidence >= 0.5:
                    best_time = movie_entry.start_time
                    best_confidence = confidence
                    source = "english"

            # 信号 2：实体匹配（仅当英文匹配不够好时补充）
            if best_confidence < 0.7:
                entity_matches = self._entity_extractor.match(entry)
                if entity_matches:
                    top_entry, top_conf = entity_matches[0]
                    # 实体匹配要求更高的阈值才能成为锚点
                    if top_conf >= 0.4 and top_conf > best_confidence:
                        best_time = top_entry.start_time
                        best_confidence = top_conf
                        source = "entity"

            if best_time is not None and best_confidence >= 0.4:
                anchors.append({
                    'narration_time': entry.start_time,
                    'movie_time': best_time,
                    'confidence': best_confidence,
                    'source': source,
                    'text': entry.text[:30],
                })

        return anchors

    @staticmethod
    def _lis_filter(anchors: list[dict]) -> list[dict]:
        """用最长递增子序列过滤时序矛盾的锚点

        保留 movie_time 单调递增的最大子序列。
        使用带权 LIS：优先保留高置信度的锚点。
        """
        if not anchors:
            return []

        # 按 narration_time 排序
        sorted_anchors = sorted(anchors, key=lambda a: a['narration_time'])
        n = len(sorted_anchors)

        # dp[i] = 以 i 结尾的 LIS 长度（带置信度加权）
        dp = [1.0] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if sorted_anchors[j]['movie_time'] < sorted_anchors[i]['movie_time']:
                    # 加权：长度 + 置信度 bonus
                    score = dp[j] + 1.0 + sorted_anchors[i]['confidence'] * 0.1
                    if score > dp[i]:
                        dp[i] = score
                        parent[i] = j

        # 找最优结尾
        best_end = max(range(n), key=lambda i: dp[i])

        # 回溯
        result_indices = []
        k = best_end
        while k != -1:
            result_indices.append(k)
            k = parent[k]
        result_indices.reverse()

        return [sorted_anchors[i] for i in result_indices]

    def _full_match(self) -> list[dict]:
        """阶段 2：对所有解说条目做四信号融合匹配"""
        results = []
        n = len(self.narration_subs)

        for i, entry in enumerate(self.narration_subs):
            signals = []  # [(movie_time, confidence, weight, source)]

            # 信号 1：英文台词
            eng_match = self._english_matcher.match(entry)
            if eng_match:
                movie_entry, confidence = eng_match
                signals.append((
                    movie_entry.start_time,
                    confidence,
                    self.WEIGHT_ENGLISH,
                    'english'
                ))

            # 信号 2：专有名词
            entity_matches = self._entity_extractor.match(entry)
            if entity_matches:
                top_entry, top_conf = entity_matches[0]
                signals.append((
                    top_entry.start_time,
                    top_conf,
                    self.WEIGHT_ENTITY,
                    'entity'
                ))

            # 信号 3：时间线插值（总是有值）
            timeline_time, timeline_conf = self._timeline.interpolate(entry.start_time)
            signals.append((
                timeline_time,
                timeline_conf,
                self.WEIGHT_TIMELINE,
                'timeline'
            ))

            # 信号 4：上下文窗口
            window_start = max(0, i - 3)
            window_end = min(n, i + 4)
            window_entries = self.narration_subs[window_start:window_end]
            ctx_match = self._context_matcher.match_window(window_entries)
            if ctx_match:
                ctx_time, ctx_conf = ctx_match
                signals.append((
                    ctx_time,
                    ctx_conf,
                    self.WEIGHT_CONTEXT,
                    'context'
                ))

            # 融合
            movie_start, movie_end, confidence, match_source = self._fuse_signals(
                signals, entry
            )

            results.append({
                'narration_start': entry.start_time,
                'narration_end': entry.end_time,
                'narration_text': entry.text,
                'movie_start': movie_start,
                'movie_end': movie_end,
                'confidence': confidence,
                'match_source': match_source,
            })

        return results

    @staticmethod
    def _fuse_signals(
        signals: list[tuple[float, float, float, str]],
        narration_entry: SubtitleEntry
    ) -> tuple[float, float, float, str]:
        """融合多个信号计算最终电影时间和置信度

        Args:
            signals: [(movie_time, confidence, weight, source), ...]
            narration_entry: 当前解说条目

        Returns:
            (movie_start, movie_end, confidence, match_source)
        """
        if not signals:
            return 0.0, 0.0, 0.0, 'none'

        # 加权平均时间
        total_weight = 0.0
        weighted_time = 0.0
        best_source = 'none'
        best_source_weight = 0.0

        for movie_time, conf, weight, source in signals:
            effective_weight = weight * conf
            weighted_time += movie_time * effective_weight
            total_weight += effective_weight
            if effective_weight > best_source_weight:
                best_source_weight = effective_weight
                best_source = source

        if total_weight > 0:
            movie_start = weighted_time / total_weight
        else:
            movie_start = signals[0][0]

        # 估算持续时间：解说条目时长 × 时间比例
        narration_duration = narration_entry.end_time - narration_entry.start_time
        # 用最高置信度信号附近的电影字幕持续时间，或默认比例
        movie_end = movie_start + max(narration_duration, 0.5)

        # 综合置信度
        # final_confidence = Σ(weight_i × score_i) / Σ(weight_i)
        weight_sum = sum(w for _, _, w, _ in signals)
        if weight_sum > 0:
            confidence = sum(w * c for _, c, w, _ in signals) / weight_sum
        else:
            confidence = 0.0

        confidence = min(1.0, max(0.0, confidence))

        return movie_start, movie_end, confidence, best_source

    @staticmethod
    def _log_statistics(results: list[dict]):
        """打印匹配统计信息"""
        total = len(results)
        if total == 0:
            logger.info("无匹配结果")
            return

        source_counts = Counter(r['match_source'] for r in results)
        source_avg_conf = defaultdict(list)
        for r in results:
            source_avg_conf[r['match_source']].append(r['confidence'])

        has_movie_time = sum(1 for r in results if r['movie_start'] is not None and r['movie_start'] > 0)
        avg_confidence = sum(r['confidence'] for r in results) / total

        logger.info(f"=== 匹配统计 ===")
        logger.info(f"总条目: {total}, 有时间估计: {has_movie_time} ({has_movie_time/total*100:.1f}%)")
        logger.info(f"平均置信度: {avg_confidence:.3f}")

        for source in ['english', 'entity', 'timeline', 'context', 'none']:
            count = source_counts.get(source, 0)
            if count > 0:
                avg = sum(source_avg_conf[source]) / count
                logger.info(f"  {source}: {count} 条 ({count/total*100:.1f}%), 平均置信度 {avg:.3f}")

        # 时间单调性检查
        times = [r['movie_start'] for r in results if r['movie_start'] is not None and r['movie_start'] > 0]
        if len(times) >= 2:
            monotonic_count = sum(1 for i in range(1, len(times)) if times[i] >= times[i-1])
            logger.info(f"时间单调性: {monotonic_count}/{len(times)-1} "
                        f"({monotonic_count/(len(times)-1)*100:.1f}%)")


# ---------------------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------------------

def test_subtitle_matching():
    """测试字幕匹配"""
    movie_sub = r"C:\Users\23730\movie-narrator-tool\videos\subtitles\The.Shawshank.Redemption.1994.2160p.BluRay.x265.10bit.SDR.DTS-HD.MA.5.1-SWTYBLZ.ass"
    narration_sub = r"C:\Users\23730\movie-narrator-tool\videos\subtitles\2月1日.srt"

    matcher = SubtitleMatcher(movie_sub, narration_sub)
    results = matcher.match_all_narration_segments()

    print(f"\n{'='*60}")
    print(f"匹配结果总览")
    print(f"{'='*60}")

    # 分类统计
    source_counts = Counter(r['match_source'] for r in results)
    source_conf = defaultdict(list)
    for r in results:
        source_conf[r['match_source']].append(r['confidence'])

    print(f"\n总条目: {len(results)}")
    for source in ['english', 'entity', 'timeline', 'context', 'none']:
        count = source_counts.get(source, 0)
        if count > 0:
            avg = sum(source_conf[source]) / count
            print(f"  {source}: {count} 条 ({count/len(results)*100:.1f}%), 平均置信度 {avg:.3f}")

    # 全量覆盖检查
    no_time = sum(1 for r in results if r['movie_start'] is None or r['movie_start'] == 0)
    print(f"\n全量覆盖: {len(results) - no_time}/{len(results)} "
          f"({(len(results) - no_time)/len(results)*100:.1f}%)")
    if no_time > 0:
        print(f"  缺失时间: {no_time} 条")

    # 时间单调性
    times = [r['movie_start'] for r in results if r['movie_start'] is not None and r['movie_start'] > 0]
    if len(times) >= 2:
        mono = sum(1 for i in range(1, len(times)) if times[i] >= times[i-1])
        print(f"\n时间单调性: {mono}/{len(times)-1} ({mono/(len(times)-1)*100:.1f}%)")

    # 英文匹配样本
    eng_results = [r for r in results if r['match_source'] == 'english']
    if eng_results:
        print(f"\n--- 英文匹配样本 (前10) ---")
        for r in eng_results[:10]:
            m_min = int(r['movie_start'] // 60)
            m_sec = r['movie_start'] % 60
            print(f"  [{r['narration_start']:.1f}s] {r['narration_text'][:40]}")
            print(f"    -> 电影 {m_min}:{m_sec:05.2f}, 置信度 {r['confidence']:.3f}")

    # 实体匹配样本
    ent_results = [r for r in results if r['match_source'] == 'entity']
    if ent_results:
        print(f"\n--- 实体匹配样本 (前10) ---")
        for r in ent_results[:10]:
            m_min = int(r['movie_start'] // 60)
            m_sec = r['movie_start'] % 60
            print(f"  [{r['narration_start']:.1f}s] {r['narration_text'][:40]}")
            print(f"    -> 电影 {m_min}:{m_sec:05.2f}, 置信度 {r['confidence']:.3f}")

    # 全部结果前 20 条
    print(f"\n--- 全部结果 (前20) ---")
    for r in results[:20]:
        if r['movie_start'] and r['movie_start'] > 0:
            m_min = int(r['movie_start'] // 60)
            m_sec = r['movie_start'] % 60
            print(f"  [{r['narration_start']:7.1f}s] {r['narration_text'][:35]:35s} "
                  f"-> {m_min:3d}:{m_sec:05.2f} [{r['match_source']:8s}] "
                  f"conf={r['confidence']:.3f}")
        else:
            print(f"  [{r['narration_start']:7.1f}s] {r['narration_text'][:35]:35s} "
                  f"-> 无匹配")


if __name__ == '__main__':
    test_subtitle_matching()
