"""文件工具函数"""
import hashlib
import re
from pathlib import Path


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """计算文件哈希值

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5/sha1/sha256)

    Returns:
        哈希值字符串
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def ensure_dir(path: Path) -> Path:
    """确保目录存在

    Args:
        path: 目录路径

    Returns:
        目录路径
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str, max_length: int = 200) -> str:
    """生成安全的文件名

    移除或替换不安全字符

    Args:
        name: 原始文件名
        max_length: 最大长度

    Returns:
        安全的文件名
    """
    # 移除/替换不安全字符
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 移除控制字符
    safe_name = re.sub(r'[\x00-\x1f\x7f]', '', safe_name)
    # 移除首尾空格和点
    safe_name = safe_name.strip(' .')
    # 截断
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]

    return safe_name or "unnamed"


def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）

    Args:
        file_path: 文件路径

    Returns:
        文件大小
    """
    path = Path(file_path)
    if path.exists():
        return path.stat().st_size
    return 0


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_extension(file_path: str) -> str:
    """获取文件扩展名（小写，不含点）

    Args:
        file_path: 文件路径

    Returns:
        扩展名
    """
    return Path(file_path).suffix.lower().lstrip('.')


def is_video_file(file_path: str) -> bool:
    """判断是否为视频文件

    Args:
        file_path: 文件路径

    Returns:
        是否为视频文件
    """
    video_extensions = {'mp4', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'webm', 'm4v', 'mpeg', 'mpg'}
    return get_extension(file_path) in video_extensions


def is_audio_file(file_path: str) -> bool:
    """判断是否为音频文件

    Args:
        file_path: 文件路径

    Returns:
        是否为音频文件
    """
    audio_extensions = {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma'}
    return get_extension(file_path) in audio_extensions
