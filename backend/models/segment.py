"""Segment-related models."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SegmentType(str, Enum):
    """Segment classification."""

    HAS_NARRATION = "has_narration"
    NO_NARRATION = "no_narration"
    NON_MOVIE = "non_movie"


class SegmentStatus(str, Enum):
    """Processing status for a segment."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class TTSStatus(str, Enum):
    """TTS generation status."""

    NOT_GENERATED = "not_generated"
    GENERATED = "generated"
    FAILED = "failed"


class AlignmentStatus(str, Enum):
    """Alignment review state."""

    PENDING = "pending"
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    UNMATCHED = "unmatched"
    SKIPPED = "skipped"
    NON_MOVIE = "non_movie"
    MANUAL = "manual"
    REMATCHED = "rematched"


class MatchCandidate(BaseModel):
    """A candidate movie segment for one narration segment."""

    id: str = Field(..., description="Candidate id")
    start: float = Field(..., description="Movie start time in seconds")
    end: float = Field(..., description="Movie end time in seconds")
    score: float = Field(default=0.0, description="Overall rerank score")
    confidence: float = Field(default=0.0, description="Raw confidence")
    visual_confidence: float = Field(default=0.0, description="Visual match confidence")
    audio_confidence: float = Field(default=0.0, description="Audio match confidence")
    temporal_confidence: float = Field(default=0.0, description="Temporal consistency confidence")
    stability_score: float = Field(default=0.0, description="Evidence stability score")
    candidate_quality: float = Field(default=0.0, description="Average matched movie-frame quality")
    query_quality: float = Field(default=0.0, description="Average narration-frame quality")
    low_info_ratio: float = Field(default=0.0, description="Ratio of low-information matched frames")
    duration_gap: float = Field(default=0.0, description="Absolute duration gap in seconds")
    match_count: int = Field(default=0, description="Supporting sample frames")
    reason: str = Field(default="", description="Human readable reason")
    source: str = Field(default="visual", description="Candidate source")
    rank: int = Field(default=0, description="Candidate rank")


class Segment(BaseModel):
    """A narration-aligned segment."""

    id: str = Field(..., description="Unique segment id")
    index: int = Field(..., description="Display order")

    narration_start: float = Field(..., description="Narration start time in seconds")
    narration_end: float = Field(..., description="Narration end time in seconds")
    movie_start: Optional[float] = Field(None, description="Matched movie start time in seconds")
    movie_end: Optional[float] = Field(None, description="Matched movie end time in seconds")

    segment_type: SegmentType = Field(default=SegmentType.HAS_NARRATION, description="Segment type")
    status: SegmentStatus = Field(default=SegmentStatus.PENDING, description="Segment processing status")

    match_confidence: float = Field(default=0.0, description="Selected match confidence")
    visual_confidence: float = Field(default=0.0, description="Selected visual confidence")
    audio_confidence: float = Field(default=0.0, description="Selected audio confidence")
    temporal_confidence: float = Field(default=0.0, description="Selected temporal confidence")
    stability_score: float = Field(default=0.0, description="Selected evidence stability")
    duration_gap: float = Field(default=0.0, description="Absolute duration gap")
    match_reason: str = Field(default="", description="Why the match was selected")
    speech_likelihood: float = Field(default=0.0, description="Estimated narration speech likelihood")
    audio_activity_label: str = Field(default="unknown", description="Audio activity label")
    voiceprint_similarity: Optional[float] = Field(default=None, description="Narrator voice similarity")
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.PENDING, description="Alignment review state")
    review_required: bool = Field(default=False, description="Whether the segment should be reviewed")
    is_manual_match: bool = Field(default=False, description="Whether the current match was manually selected")
    skip_matching: bool = Field(default=False, description="Whether this segment should be excluded from matching")
    selected_candidate_id: Optional[str] = Field(None, description="Currently selected candidate id")
    match_candidates: list[MatchCandidate] = Field(default_factory=list, description="Alternative movie candidates")
    estimated_boundary_error: Optional[float] = Field(
        default=None,
        description="Estimated boundary uncertainty in seconds",
    )

    original_text: str = Field(default="", description="Original narration text")
    polished_text: str = Field(default="", description="Polished narration text")

    tts_status: TTSStatus = Field(default=TTSStatus.NOT_GENERATED, description="TTS generation status")
    tts_duration: Optional[float] = Field(None, description="TTS duration in seconds")
    tts_audio_path: Optional[str] = Field(None, description="Generated TTS audio path")
    tts_error: Optional[str] = Field(None, description="TTS error message")

    thumbnail_path: Optional[str] = Field(None, description="Segment thumbnail path")

    use_segment: bool = Field(default=True, description="Whether to keep this segment")
    keep_bgm: bool = Field(default=False, description="Keep narration video BGM")
    keep_movie_audio: bool = Field(default=False, description="Keep movie original audio")
    mute_movie_audio: bool = Field(default=False, description="Mute movie original audio")
    use_polished_text: bool = Field(default=True, description="Use polished narration text")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "seg_001",
                "index": 1,
                "narration_start": 5.0,
                "narration_end": 15.5,
                "movie_start": 120.0,
                "movie_end": 130.5,
                "segment_type": "has_narration",
                "status": "completed",
                "alignment_status": "auto_accepted",
                "match_confidence": 0.95,
                "visual_confidence": 0.93,
                "audio_confidence": 0.41,
                "temporal_confidence": 0.88,
                "stability_score": 0.81,
                "duration_gap": 0.4,
                "match_reason": "Strong visual cluster with stable timeline",
                "speech_likelihood": 0.92,
                "audio_activity_label": "active",
                "review_required": False,
                "skip_matching": False,
                "original_text": "主角走进了那间阴暗的房间",
                "polished_text": "主角缓缓推开那扇昏暗房门",
                "tts_duration": 5.2,
                "use_segment": True,
                "match_candidates": [
                    {
                        "id": "seg_001_cand_1",
                        "start": 120.0,
                        "end": 130.5,
                        "score": 0.93,
                        "confidence": 0.95,
                        "visual_confidence": 0.93,
                        "audio_confidence": 0.41,
                        "temporal_confidence": 0.88,
                        "stability_score": 0.81,
                        "duration_gap": 0.4,
                        "match_count": 9,
                        "reason": "Strong visual cluster with stable timeline",
                        "source": "visual",
                        "rank": 1,
                    }
                ],
            }
        }


def compute_segment_duration(segment: "Segment", audio_source: str = "tts") -> float:
    """Compute playback duration for exporters and timeline views."""

    narration_duration = segment.narration_end - segment.narration_start
    if audio_source == "original":
        return narration_duration
    if segment.tts_duration:
        return segment.tts_duration
    if segment.movie_end is not None and segment.movie_start is not None:
        return segment.movie_end - segment.movie_start
    return narration_duration


class SegmentUpdate(BaseModel):
    """Patch payload for a single segment."""

    movie_start: Optional[float] = None
    movie_end: Optional[float] = None
    original_text: Optional[str] = None
    polished_text: Optional[str] = None
    use_segment: Optional[bool] = None
    keep_bgm: Optional[bool] = None
    keep_movie_audio: Optional[bool] = None
    mute_movie_audio: Optional[bool] = None
    use_polished_text: Optional[bool] = None
    is_manual_match: Optional[bool] = None
    skip_matching: Optional[bool] = None
    match_confidence: Optional[float] = None
    visual_confidence: Optional[float] = None
    audio_confidence: Optional[float] = None
    temporal_confidence: Optional[float] = None
    stability_score: Optional[float] = None
    duration_gap: Optional[float] = None
    match_reason: Optional[str] = None
    alignment_status: Optional[AlignmentStatus] = None
    review_required: Optional[bool] = None
    selected_candidate_id: Optional[str] = None


class SegmentBatchUpdate(BaseModel):
    """Batch patch payload."""

    segment_ids: list[str] = Field(..., description="Target segment ids")
    use_segment: Optional[bool] = None
    use_polished_text: Optional[bool] = None
    skip_matching: Optional[bool] = None
    keep_bgm: Optional[bool] = None
    keep_movie_audio: Optional[bool] = None
    mute_movie_audio: Optional[bool] = None
