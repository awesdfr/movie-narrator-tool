"""Project data models."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .segment import Segment


class ProjectStatus(str, Enum):
    """Project processing status."""

    CREATED = "created"
    IMPORTING = "importing"
    ANALYZING = "analyzing"
    MATCHING = "matching"
    RECOGNIZING = "recognizing"
    READY_FOR_POLISH = "ready_for_polish"
    POLISHING = "polishing"
    READY_FOR_TTS = "ready_for_tts"
    GENERATING_TTS = "generating_tts"
    COMPLETED = "completed"
    ERROR = "error"


class ProcessingProgress(BaseModel):
    """Live processing progress."""

    stage: str = Field(default="", description="Current stage")
    progress: float = Field(default=0.0, description="Progress percentage")
    message: str = Field(default="", description="Status message")
    current_step: int = Field(default=0, description="Current step")
    total_steps: int = Field(default=0, description="Total steps")


class SubtitleMaskMode(str, Enum):
    """How subtitle masking should be applied."""

    HYBRID = "hybrid"
    AUTO_ONLY = "auto_only"
    MANUAL_ONLY = "manual_only"


class SubtitleRegion(BaseModel):
    """Normalized subtitle region for a video source."""

    id: str = Field(..., description="Stable region id")
    x: float = Field(..., ge=0.0, le=1.0, description="Left position ratio")
    y: float = Field(..., ge=0.0, le=1.0, description="Top position ratio")
    width: float = Field(..., gt=0.0, le=1.0, description="Width ratio")
    height: float = Field(..., gt=0.0, le=1.0, description="Height ratio")
    enabled: bool = Field(default=True, description="Whether the region is active")
    label: Optional[str] = Field(default=None, description="Optional display label")
    start_time: Optional[float] = Field(default=None, ge=0.0, description="Optional start time in seconds")
    end_time: Optional[float] = Field(default=None, ge=0.0, description="Optional end time in seconds")


class Project(BaseModel):
    """Project data."""

    id: str = Field(..., description="Unique project id")
    name: str = Field(..., description="Project name")
    created_at: datetime = Field(default_factory=datetime.now, description="Created at")
    updated_at: datetime = Field(default_factory=datetime.now, description="Updated at")

    status: ProjectStatus = Field(default=ProjectStatus.CREATED, description="Project status")
    progress: ProcessingProgress = Field(default_factory=ProcessingProgress, description="Processing progress")

    movie_path: Optional[str] = Field(None, description="Movie path")
    narration_path: Optional[str] = Field(None, description="Narration video path")
    reference_audio_path: Optional[str] = Field(None, description="Voiceprint reference audio path")
    tts_reference_audio_path: Optional[str] = Field(None, description="TTS reference audio path")
    subtitle_path: Optional[str] = Field(None, description="Optional subtitle path")

    movie_duration: Optional[float] = Field(None, description="Movie duration in seconds")
    movie_fps: Optional[float] = Field(None, description="Movie fps")
    movie_resolution: Optional[tuple[int, int]] = Field(None, description="Movie resolution")
    narration_duration: Optional[float] = Field(None, description="Narration duration in seconds")
    narration_fps: Optional[float] = Field(None, description="Narration fps")
    narration_resolution: Optional[tuple[int, int]] = Field(None, description="Narration resolution")

    segments: list[Segment] = Field(default_factory=list, description="Project segments")
    subtitle_mask_mode: SubtitleMaskMode = Field(
        default=SubtitleMaskMode.HYBRID,
        description="How subtitle masking is applied during frame matching",
    )
    narration_subtitle_regions: list[SubtitleRegion] = Field(
        default_factory=list,
        description="Manual subtitle regions for narration video",
    )
    movie_subtitle_regions: list[SubtitleRegion] = Field(
        default_factory=list,
        description="Manual subtitle regions for movie video",
    )

    output_name: Optional[str] = Field(None, description="Output name")
    benchmark_accuracy: Optional[float] = Field(None, description="Last benchmark accuracy")
    benchmark_manifest: Optional[str] = Field(None, description="Last benchmark manifest path")
    match_version: str = Field(default="v2_alignment_pipeline", description="Active match pipeline version")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "proj_abc123",
                "name": "电影解说项目1",
                "status": "completed",
                "movie_path": "D:/movies/movie.mp4",
                "narration_path": "D:/videos/narration.mp4",
            }
        }

    def to_dict(self) -> dict:
        """Serialize as json-friendly dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Create project from serialized data."""
        return cls.model_validate(data)


class ProjectCreate(BaseModel):
    """Project creation payload."""

    name: str = Field(..., description="Project name")
    movie_path: str = Field(..., description="Movie path")
    narration_path: str = Field(..., description="Narration path")
    reference_audio_path: Optional[str] = Field(None, description="Voiceprint reference audio path")
    tts_reference_audio_path: Optional[str] = Field(None, description="TTS reference audio path")
    subtitle_path: Optional[str] = Field(None, description="Optional SRT subtitle path")


class ProjectSummary(BaseModel):
    """Summary item for project lists."""

    id: str
    name: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    movie_path: Optional[str]
    narration_path: Optional[str]
    segment_count: int = 0
    thumbnail: Optional[str] = None


class SubtitleRegionsUpdate(BaseModel):
    """Manual subtitle region update payload."""

    subtitle_mask_mode: SubtitleMaskMode = Field(default=SubtitleMaskMode.HYBRID)
    narration_subtitle_regions: list[SubtitleRegion] = Field(default_factory=list)
    movie_subtitle_regions: list[SubtitleRegion] = Field(default_factory=list)


class VideoSourceInfo(BaseModel):
    """Frame picker metadata for a project video source."""

    source: str
    path: str
    duration: float
    fps: float
    width: int
    height: int
