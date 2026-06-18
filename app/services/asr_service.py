"""
ASR (Automatic Speech Recognition) and speaker diarization service.

Week 1 scope:
  - Transcribe audio using faster-whisper (CTranslate2-backed Whisper).
  - Apply a SIMPLE speaker-labeling heuristic on top of Whisper's
    segment-level timestamps: we alternate "Doctor" / "Patient" labels
    whenever a sufficiently long pause is detected between segments.

This is intentionally lightweight. It is NOT acoustic diarization
(it doesn't analyze voice characteristics) — it's a pause-based
heuristic that's good enough for a two-party clinical conversation
in a Week 1 MVP. A note on the upgrade path to pyannote.audio for
true acoustic diarization is included in the README.
"""

from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_SIZE,
    SPEAKER_LABELS,
    SPEAKER_TURN_GAP_THRESHOLD,
)
from app.models.schemas import TranscriptSegment, TranscriptionResponse


class ASRService:
    """Wraps the faster-whisper model and adds speaker-turn labeling."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model = None  # Lazy-loaded on first use

    def load_model(self):
        """Loads the Whisper model into memory if not already loaded.
        Lazy loading avoids slow startup time on every server reload."""
        if self._model is None:
            print(f"[ASRService] Loading Whisper model: {self.model_size} ...")
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
            print("[ASRService] Whisper model loaded.")
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: Path) -> TranscriptionResponse:
        """
        Runs Whisper transcription on the given audio file and applies
        simple pause-based speaker labeling to the resulting segments.
        """
        model = self.load_model()

        segments_gen, info = model.transcribe(str(audio_path), beam_size=5)

        # Materialize the generator into a list of segment objects
        raw_segments = list(segments_gen)

        language = info.language or "unknown"
        full_text = " ".join(seg.text.strip() for seg in raw_segments).strip()

        labeled_segments = self._apply_speaker_labels(raw_segments)

        duration = raw_segments[-1].end if raw_segments else 0.0
        speaker_count = (
            len(set(seg.speaker for seg in labeled_segments))
            if labeled_segments
            else 0
        )

        return TranscriptionResponse(
            filename=audio_path.name,
            duration_seconds=round(duration, 2),
            language=language,
            full_text=full_text,
            segments=labeled_segments,
            speaker_count=speaker_count,
        )

    def _apply_speaker_labels(self, raw_segments: list) -> List[TranscriptSegment]:
        """
        Heuristic speaker labeling:
        - Start with Speaker 0 (Doctor).
        - Whenever the gap between the end of one segment and the start
          of the next exceeds SPEAKER_TURN_GAP_THRESHOLD seconds, flip
          to the other speaker (simulating a conversational turn change).

        This is a deliberately simple Week 1 placeholder. It assumes a
        two-party conversation (Doctor/Patient) and does not use voice
        embeddings. See README for the pyannote.audio upgrade path.
        """
        labeled: List[TranscriptSegment] = []
        current_speaker_idx = 0
        previous_end_time = None

        for seg in raw_segments:
            start = seg.start
            end = seg.end
            text = seg.text.strip()

            if previous_end_time is not None:
                gap = start - previous_end_time
                if gap >= SPEAKER_TURN_GAP_THRESHOLD:
                    current_speaker_idx = 1 - current_speaker_idx  # flip 0/1

            labeled.append(
                TranscriptSegment(
                    speaker=SPEAKER_LABELS[current_speaker_idx],
                    start_time=round(start, 2),
                    end_time=round(end, 2),
                    text=text,
                )
            )
            previous_end_time = end

        return labeled


# Singleton instance shared across the app so the model is loaded once
asr_service = ASRService()

