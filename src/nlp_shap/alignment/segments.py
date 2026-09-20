"""Aligned audio segment records produced by SGPA."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One transcript token aligned to an audio time range."""

    token: str
    """Transcript token or word for this segment."""

    start_time: float
    """Segment start time in seconds."""

    end_time: float
    """Segment end time in seconds."""

    confidence: float
    """Alignment confidence in ``[0, 1]``."""

    audio: bytes = b""
    """Optional encoded audio bytes for this segment."""

    audio_format: str = "wav"
    """Container or codec label for ``audio``."""

    sample_rate: int | None = None
    """Sample rate used when sample indices are set."""

    start_sample: int | None = None
    """Inclusive start sample index in the source waveform."""

    end_sample: int | None = None
    """Exclusive end sample index in the source waveform."""

    boundary_refined: bool = True
    """Whether both edges used acoustic boundary refinement."""

    def __post_init__(self) -> None:
        if self.end_time <= self.start_time:
            msg = "end_time must be greater than start_time"
            raise ValueError(msg)

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def __add__(self, other: "AudioSegment") -> "AudioSegment":
        """Merge two segments that share the same token."""
        if self.token != other.token:
            msg = "Cannot combine AudioSegments with different tokens."
            raise ValueError(msg)
        return AudioSegment(
            token=self.token,
            start_time=min(self.start_time, other.start_time),
            end_time=max(self.end_time, other.end_time),
            confidence=(self.confidence + other.confidence) / 2.0,
            audio=self.audio + other.audio,
            audio_format=self.audio_format,
            sample_rate=self.sample_rate or other.sample_rate,
            start_sample=_min_optional(self.start_sample, other.start_sample),
            end_sample=_max_optional(self.end_sample, other.end_sample),
            boundary_refined=self.boundary_refined and other.boundary_refined,
        )


def _min_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _max_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)
