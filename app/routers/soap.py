"""
Router for SOAP note generation endpoints.

Exposes:
  POST /soap/generate                     -> generate SOAP note from
                                             transcript segments
  POST /soap/generate-from-mock/{filename} -> end-to-end: transcribe a
                                             mock file and generate SOAP
"""

import logging

from fastapi import APIRouter, HTTPException
from typing import List

from app.config import MOCK_DATA_DIR, ALLOWED_AUDIO_EXTENSIONS
from app.models.schemas import TranscriptSegment
from app.models.soap_schemas import SOAPNoteResponse
from app.services.asr_service import asr_service
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/soap", tags=["SOAP Note Generation"])


@router.post("/generate", response_model=SOAPNoteResponse)
async def generate_soap_note(segments: List[TranscriptSegment]):
    """
    Accepts a list of transcript segments (from a previous transcription
    call) and generates a structured SOAP note using the LLM.
    """
    if not segments:
        raise HTTPException(
            status_code=400,
            detail="At least one transcript segment is required.",
        )

    try:
        result = llm_service.generate_soap_note(segments)
    except ValueError as exc:
        # API key not configured
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        # LLM failed after retries
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@router.post(
    "/generate-from-mock/{filename}",
    response_model=SOAPNoteResponse,
)
async def generate_soap_from_mock(filename: str):
    """
    End-to-end pipeline: transcribes a mock audio file and then
    generates a SOAP note from the transcript — all in one call.
    Useful for demos and testing.
    """
    # Security: reject path traversal attempts
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Directory traversal is not allowed.",
        )

    target = (MOCK_DATA_DIR / filename).resolve()

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

    # Step 1: Transcribe
    logger.info("SOAP pipeline — transcribing: %s", filename)
    try:
        transcription = asr_service.transcribe(target)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        )

    # Step 2: Generate SOAP note
    logger.info("SOAP pipeline — generating note from %d segments",
                len(transcription.segments))
    try:
        result = llm_service.generate_soap_note(
            transcription.segments,
            filename=filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result
