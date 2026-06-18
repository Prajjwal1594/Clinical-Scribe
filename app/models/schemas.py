"""
Pydantic models for the audio ingestion + diarization pipeline.
These schemas define the contract between the API and its clients,
and will be reused/extended in Week 2 for SOAP note structuring.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TranscriptSegment(BaseModel):
    """A single utterance/segment from the transcription engine,
    tagged with a speaker label."""

    speaker: str = Field(..., description="Speaker label, e.g. 'Doctor' or 'Patient'")
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class TranscriptionResponse(BaseModel):
    """Full response returned after processing an uploaded audio file."""

    filename: str
    duration_seconds: float
    language: str
    full_text: str
    segments: List[TranscriptSegment]
    speaker_count: int


class HealthCheckResponse(BaseModel):
    status: str
    whisper_model_loaded: bool
    model_size: str


class ErrorResponse(BaseModel):
    detail: str
