"""
Configuration settings for the Ambient Clinical Scribe backend.
Centralizes paths, model choices, and constants so Week 2+ features
(SOAP synthesis, RAG) can import the same settings.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Where uploaded audio files are stored temporarily before/after processing
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Where mock medical audio datasets live for local testing
MOCK_DATA_DIR = BASE_DIR / "mock_data"
MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Whisper model size. "base" is a good balance of speed/accuracy for dev.
# Options: tiny, base, small, medium, large
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Allowed audio file extensions for upload validation
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# Max upload size in bytes (50 MB) — guards against runaway uploads
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

# Simple speaker labeling heuristic settings (Week 1 scope: no real
# diarization model yet — pyannote.audio is a Week-2+ upgrade path).
# We alternate speaker labels based on pause-based segmentation from
# Whisper's own segment timestamps.
SPEAKER_LABELS = ["Doctor", "Patient"]

# Minimum silence gap (seconds) between Whisper segments that we treat
# as a likely speaker turn change. Tuned empirically against short
# conversational test clips — 0.4s catches quick back-and-forth replies
# better than longer thresholds, but it's still a heuristic: rapid
# interruptions or same-speaker pauses longer than this will mislabel.
SPEAKER_TURN_GAP_THRESHOLD = 0.4
