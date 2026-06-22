"""
Ambient Clinical Scribe — Backend entrypoint.

Run locally with:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env before any config imports

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)

from app.routers import audio, soap
from app.services.asr_service import asr_service
from app.config import WHISPER_MODEL_SIZE, LLM_MODEL_NAME
from app.models.schemas import HealthCheckResponse

app = FastAPI(
    title="Ambient Clinical Scribe API",
    description=(
        "Audio ingestion, speaker-diarized transcription, and "
        "LLM-powered SOAP note generation for the Automated "
        "SOAP Note Generator project."
    ),
    version="0.2.0",
)

app.include_router(audio.router)
app.include_router(soap.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Ambient Clinical Scribe API is running.",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Root"])
async def health_check():
    """Basic health check, also reports whether the Whisper model has
    been loaded into memory yet (it lazy-loads on first transcription
    request to keep server startup fast)."""
    return HealthCheckResponse(
        status="ok",
        whisper_model_loaded=asr_service.is_loaded,
        model_size=WHISPER_MODEL_SIZE,
    )


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    logger.info("Ambient Clinical Scribe API starting up...")
    logger.info("Whisper model configured: %s", WHISPER_MODEL_SIZE)
    logger.info("LLM model configured: %s", LLM_MODEL_NAME)
    logger.info("Models will load lazily on first request.")
