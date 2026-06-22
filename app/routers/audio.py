"""
Router for audio ingestion endpoints.

Exposes:
  POST /audio/transcribe        -> upload an audio file, get back a
                                    diarized transcript
  GET  /audio/mock-files        -> list available mock medical audio
                                    files bundled for local testing
  POST /audio/transcribe-mock/{filename} -> transcribe one of the
                                    bundled mock files by name
  GET  /audio/stats             -> summary statistics about audio
                                    files and storage
"""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import (
    UPLOAD_DIR,
    MOCK_DATA_DIR,
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
)
from app.models.schemas import TranscriptionResponse, AudioStatsResponse
from app.services.asr_service import asr_service

router = APIRouter(prefix="/audio", tags=["Audio Ingestion"])


def _validate_extension(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            ),
        )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file (e.g. a simulated doctor-patient
    conversation), saves it temporarily, runs it through Whisper for
    transcription, and applies speaker labeling.
    """
    _validate_extension(file.filename)

    # Stream to disk while enforcing a max size limit
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = UPLOAD_DIR / unique_name

    size = 0
    try:
        with open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    out_file.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds max upload size of "
                        f"{MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    try:
        result = asr_service.transcribe(dest_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        # Clean up the uploaded file after processing. In a real clinical
        # deployment, this is where audio retention/privacy policy applies
        # (see Business Objectives: strict adherence to data privacy).
        dest_path.unlink(missing_ok=True)

    return result


@router.get("/mock-files")
async def list_mock_files():
    """Lists mock medical audio files available in mock_data/ for testing
    the pipeline without needing to upload a real file."""
    files = [
        f.name
        for f in MOCK_DATA_DIR.iterdir()
        if f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
    ]
    return {"mock_files": files, "count": len(files)}


@router.get("/stats", response_model=AudioStatsResponse)
async def audio_stats():
    """Returns summary statistics about available audio files,
    upload directory status, and supported formats."""
    mock_files = [
        f for f in MOCK_DATA_DIR.iterdir()
        if f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
    ]
    return AudioStatsResponse(
        mock_file_count=len(mock_files),
        upload_dir_exists=UPLOAD_DIR.exists(),
        supported_formats=sorted(ALLOWED_AUDIO_EXTENSIONS),
    )


@router.post("/transcribe-mock/{filename}", response_model=TranscriptionResponse)
async def transcribe_mock_file(filename: str):
    """Transcribes a mock audio file already present in mock_data/ by
    filename, useful for quick local testing/demos."""
    # Security: reject path traversal attempts (e.g. "../../etc/passwd")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Directory traversal is not allowed.",
        )

    target = (MOCK_DATA_DIR / filename).resolve()

    # Double-check the resolved path is still inside mock_data/
    if not str(target).startswith(str(MOCK_DATA_DIR.resolve())):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Path escapes the mock data directory.",
        )

    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Mock file '{filename}' not found in mock_data/.",
        )
    try:
        result = asr_service.transcribe(target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    return result
