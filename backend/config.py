"""Application configuration."""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings."""

    app_name: str = "电影解说重制工具"
    debug: bool = Field(default=True)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    temp_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "temp")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    projects_dir: Path = Field(default_factory=lambda: Path.home() / "MovieNarratorProjects")
    videos_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "videos")

    jianying_drafts_dir: Path = Field(default=Path(""))

    ai_api_base: str = Field(default="https://api.openai.com/v1")
    ai_api_key: Optional[str] = Field(default=None)
    ai_model: str = Field(default="gpt-4o")
    ai_max_tokens: int = Field(default=2000)
    ai_temperature: float = Field(default=0.7)

    tts_api_base: str = Field(default="http://127.0.0.1:5000")
    tts_reference_audio: Optional[str] = Field(default=None)

    whisper_model: str = Field(default="medium")
    whisper_device: str = Field(default="cuda")
    whisper_language: str = Field(default="zh")

    frame_match_threshold: float = Field(default=0.60)
    phash_threshold: int = Field(default=12)
    phash_strict_threshold: int = Field(default=5)
    phash_loose_threshold: int = Field(default=12)
    scene_threshold: float = Field(default=30.0)
    index_sample_fps: float = Field(default=5.0)
    use_multi_scale_hash: bool = Field(default=True)
    use_sequence_alignment: bool = Field(default=True)
    use_dynamic_sampling: bool = Field(default=True)

    use_geometric_verification: bool = Field(default=True)
    geometric_min_inliers: int = Field(default=20)
    geometric_check_top_k: int = Field(default=10)

    enable_subtitle_removal: bool = Field(default=True)
    subtitle_removal_mode: str = Field(default="fast")
    non_linear_narration: bool = Field(default=True)

    voiceprint_threshold: float = Field(default=0.75)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_dirs(self) -> None:
        """Ensure required directories exist."""

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        (self.videos_dir / "movies").mkdir(parents=True, exist_ok=True)
        (self.videos_dir / "narrations").mkdir(parents=True, exist_ok=True)
        (self.videos_dir / "reference_audio").mkdir(parents=True, exist_ok=True)
        (self.videos_dir / "subtitles").mkdir(parents=True, exist_ok=True)


settings = Settings()
