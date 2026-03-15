"""Application settings models."""
from typing import Optional

from pydantic import BaseModel, Field


class AISettings(BaseModel):
    """AI provider settings."""

    api_base: str = Field(default="https://api.openai.com/v1", description="AI API base URL")
    api_key: Optional[str] = Field(default=None, description="AI API key")
    model: str = Field(default="gpt-4o", description="Default model")
    max_tokens: int = Field(default=2000, description="Maximum completion tokens")
    temperature: float = Field(default=0.4, description="Sampling temperature")
    polish_style_preset: str = Field(default="movie_pro", description="Default polishing preset")
    enable_de_ai_pass: bool = Field(default=True, description="Run anti-AI cleanup before style rewrite")
    enable_self_review: bool = Field(default=True, description="Run automatic self review after polishing")

    polish_template: str = Field(
        default=(
            "你是一位专业的电影解说文案校对师。请对下面的解说文案做轻度润色。\n"
            "要求：保持原意，不扩写剧情，不改关键名词，让表达自然、口语化、适合配音。\n"
            "原文：\n{text}\n\n"
            "请只输出润色后的文案。"
        ),
        description="Fallback polishing template",
    )


class TTSSettings(BaseModel):
    """TTS provider settings."""

    api_base: str = Field(default="http://127.0.0.1:7860", description="TTS service base URL")
    api_endpoint: str = Field(default="/gradio_api/call/gen_single", description="TTS API endpoint")
    reference_audio: Optional[str] = Field(None, description="Default reference audio")
    speed: float = Field(default=1.0, description="TTS playback speed")
    infer_mode: str = Field(default="批次推理", description="TTS inference mode")


class SegmentationSettings(BaseModel):
    """Narration segmentation settings."""

    min_segment_duration: float = Field(default=1.2, ge=0.4, le=10.0, description="Minimum target duration")
    max_segment_duration: float = Field(default=8.0, ge=2.0, le=20.0, description="Maximum target duration")
    split_pause_seconds: float = Field(default=0.55, ge=0.15, le=3.0, description="Pause threshold for splitting")
    merge_gap_seconds: float = Field(default=0.35, ge=0.0, le=3.0, description="Gap threshold for merging")
    sentence_snap_tolerance: float = Field(default=0.4, ge=0.0, le=2.0, description="Scene snap tolerance")
    enable_scene_snap: bool = Field(default=True, description="Snap boundaries to detected scene cuts")
    prefer_word_timestamps: bool = Field(default=True, description="Prefer ASR word timestamps when available")


class MatchSettings(BaseModel):
    """Frame matching settings."""

    frame_match_threshold: float = Field(default=0.65, description="Minimum visual match threshold")
    phash_threshold: int = Field(default=8, description="pHash search threshold")
    phash_strict_threshold: int = Field(default=5, description="Strict pHash threshold")
    phash_loose_threshold: int = Field(default=10, description="Loose pHash threshold")
    scene_threshold: float = Field(default=30.0, description="Scene detection threshold")
    use_deep_learning: bool = Field(default=True, description="Use deep visual features")
    sample_interval: int = Field(default=1, ge=1, le=30, description="Index sample interval (seconds)")
    index_sample_fps: float = Field(default=8.0, ge=0.5, le=15.0, description="Index sample fps")
    fast_mode: bool = Field(default=False, description="Fast but less accurate mode")
    use_multi_scale_hash: bool = Field(default=True, description="Use multi-scale hashes")
    use_sequence_alignment: bool = Field(default=True, description="Use Smith-Waterman alignment")
    use_dynamic_sampling: bool = Field(default=True, description="Use adaptive sampling")
    use_prefilter: bool = Field(default=True, description="Use lightweight prefilter")
    high_confidence_threshold: float = Field(default=0.82, ge=0.5, le=1.0, description="Auto-accept threshold")
    medium_confidence_threshold: float = Field(default=0.65, ge=0.3, le=0.95, description="Review threshold")
    low_confidence_threshold: float = Field(default=0.50, ge=0.1, le=0.9, description="Low confidence cutoff")
    candidate_top_k: int = Field(default=8, ge=1, le=20, description="Number of candidates kept per segment")
    allow_non_sequential: bool = Field(default=True, description="Allow non sequential narration")
    use_lis_filter: bool = Field(default=False, description="Use LIS cleanup for time order")
    rerank_low_confidence: bool = Field(default=True, description="Rerank low-confidence segments")
    use_multimodal_rerank: bool = Field(default=False, description="Enable optional multimodal rerank")
    global_backtrack_penalty: float = Field(default=1.4, ge=0.0, le=5.0, description="Penalty for time backtracking")
    duplicate_scene_penalty: float = Field(default=0.7, ge=0.0, le=3.0, description="Penalty for repeated overlapping scenes")


class ConcurrencySettings(BaseModel):
    """Concurrency settings."""

    polish_concurrency: int = Field(default=5, ge=1, le=20, description="Polish concurrency")
    tts_concurrency: int = Field(default=5, ge=1, le=20, description="TTS concurrency")
    match_concurrency: int = Field(default=8, ge=1, le=10, description="Match concurrency")


class VoiceprintSettings(BaseModel):
    """Voiceprint settings."""

    threshold: float = Field(default=0.75, description="Voice similarity threshold")
    min_speech_duration: float = Field(default=0.5, description="Minimum speech duration")


class WhisperSettings(BaseModel):
    """Whisper settings."""

    model: str = Field(default="medium", description="Whisper model size")
    device: str = Field(default="cuda", description="Whisper device")
    language: str = Field(default="zh", description="Whisper language")
    word_timestamps: bool = Field(default=True, description="Request word timestamps")


class ExportSettings(BaseModel):
    """Export settings."""

    jianying_drafts_dir: str = Field(
        default="",
        description="Jianying drafts directory",
    )
    output_fps: int = Field(default=0, ge=0, le=120, description="Output fps")
    output_resolution: str = Field(default="original", description="Output resolution")
    audio_source: str = Field(default="original", description="Audio source")
    export_subtitles: bool = Field(default=True, description="Export subtitles")
    subtitle_format: str = Field(default="srt", description="Subtitle format")
    min_playback_speed: float = Field(default=0.5, ge=0.25, le=1.0, description="Minimum playback speed")
    max_playback_speed: float = Field(default=2.0, ge=1.0, le=4.0, description="Maximum playback speed")


class UISettings(BaseModel):
    """UI settings."""

    language: str = Field(default="zh-CN", description="UI language")
    theme: str = Field(default="light", description="Theme")


class AppSettings(BaseModel):
    """Complete application settings."""

    ai: AISettings = Field(default_factory=AISettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    match: MatchSettings = Field(default_factory=MatchSettings)
    voiceprint: VoiceprintSettings = Field(default_factory=VoiceprintSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    ui: UISettings = Field(default_factory=UISettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
