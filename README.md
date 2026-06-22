# Ambient Clinical Scribe — Audio Ingestion, Diarization & SOAP Note Generation

A FastAPI backend that ingests audio of a doctor-patient conversation,
transcribes it using OpenAI Whisper, applies speaker labeling
("Doctor" / "Patient"), and generates structured SOAP clinical notes
using an LLM (Google Gemini) with few-shot prompting.

## What's included

### Week 1 — Audio Ingestion & Speaker Diarization

- `app/services/asr_service.py` — Whisper transcription + speaker
  labeling logic
- `app/routers/audio.py` — API endpoints for upload and mock-file
  transcription
- `app/models/schemas.py` — Pydantic request/response models
- `generate_mock_audio.py` — generates 3 mock doctor-patient
  conversations for local testing (no real dataset needed)

### Week 2 — Prompt Engineering for Clinical Structuring

- `app/services/llm_service.py` — LLM integration with few-shot
  prompting for SOAP note generation
- `app/routers/soap.py` — SOAP generation endpoints
- `app/models/soap_schemas.py` — Pydantic schemas for structured
  SOAP notes (Subjective, Objective, Assessment, Plan)

### Shared

- `app/main.py` — FastAPI app entrypoint
- `app/config.py` — central settings (paths, model size, thresholds,
  LLM configuration)
- `mock_data/` — where generated/sample audio files live
- `uploads/` — temporary storage for user-uploaded files (auto-cleaned
  after processing)

## Setup

```bash
python -m venv venv
source venv\Scripts\activate        

pip install -r requirements.txt
```

Whisper also requires `ffmpeg` installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

### Configure LLM (Week 2)

Copy the example environment file and add your API key:

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-key-here
```

Get a free Google Gemini API key at [aistudio.google.com](https://aistudio.google.com).

## Generate mock test files (no real dataset required)

```bash
python generate_mock_audio.py
```

This creates three mock consultation audio files in `mock_data/`:
- `mock_consultation_01.wav` — sore throat / fever visit
- `mock_consultation_02.wav` — knee pain / orthopedic visit
- `mock_consultation_03.wav` — headache / fever visit

## Run the server

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

## Endpoints

### Audio Ingestion (Week 1)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check, reports if Whisper model is loaded |
| GET | `/audio/mock-files` | List available mock audio files |
| GET | `/audio/stats` | System statistics (mock file count, formats) |
| POST | `/audio/transcribe-mock/{filename}` | Transcribe a bundled mock file |
| POST | `/audio/transcribe` | Upload and transcribe your own audio file |

### SOAP Note Generation (Week 2)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/soap/generate` | Generate SOAP note from transcript segments |
| POST | `/soap/generate-from-mock/{filename}` | End-to-end: transcribe mock file → generate SOAP |

### Example: transcribe the mock file

```bash
curl -X POST "http://127.0.0.1:8000/audio/transcribe-mock/mock_consultation_01.wav"
```

### Example: generate a SOAP note from a mock file

```bash
curl -X POST "http://127.0.0.1:8000/soap/generate-from-mock/mock_consultation_01.wav"
```

### Example SOAP note response

```json
{
  "filename": "mock_consultation_01.wav",
  "soap_note": {
    "subjective": {
      "chief_complaint": "Sore throat and mild fever since yesterday",
      "history_of_present_illness": "Patient presents with a one-day history of sore throat accompanied by mild fever. Denies cough but reports odynophagia.",
      "review_of_systems": "Denies cough. Reports painful swallowing."
    },
    "objective": {
      "vitals": null,
      "physical_exam": "Oropharyngeal examination reveals erythema and mild swelling of the posterior pharynx."
    },
    "assessment": {
      "diagnosis": "Acute pharyngitis, likely viral",
      "differential_diagnosis": ["Streptococcal pharyngitis", "Viral upper respiratory infection"]
    },
    "plan": {
      "medications": ["Paracetamol 500mg PO every 6 hours PRN for pain and fever"],
      "follow_up": "Follow-up visit scheduled; rapid strep test ordered",
      "patient_education": "Gargle with warm salt water for symptomatic relief"
    }
  },
  "model_used": "gemini-2.0-flash",
  "processing_time_seconds": 3.21
}
```

## Architecture

```
Audio File → Whisper ASR → Speaker Labeling → Transcript Segments
                                                      ↓
                              SOAP Note ← LLM (Gemini) ← Few-shot Prompt
```

**Technology Stack:**

| Component | Technology |
|---|---|
| Backend & API | Python, FastAPI |
| Speech-to-Text (ASR) | faster-whisper (CTranslate2) |
| Speaker Labeling | Pause-based heuristic (Week 1) |
| LLM Orchestration | LangChain + Google Gemini |
| Schema Validation | Pydantic v2 |

## Known limitations

1. **Speaker labeling is heuristic, not acoustic.** It alternates
   "Doctor"/"Patient" based on pause length between Whisper segments
   (`SPEAKER_TURN_GAP_THRESHOLD` in `config.py`, default `0.4`s, tuned
   against short test clips). This works reasonably well for clean,
   alternating two-party dialogue, but will still mislabel cases like
   interruptions, overlapping speech, or one speaker pausing mid-turn
   longer than the threshold.

2. **Upgrade path:** True acoustic diarization (clustering by voice
   characteristics, not just pauses) should be added with
   **pyannote.audio** in a later iteration. That library identifies
   "who spoke when" using a pretrained diarization pipeline, which is
   more robust than the pause heuristic above. Swapping it in would
   mean replacing `_apply_speaker_labels()` in `asr_service.py` with a
   call to a pyannote pipeline, then aligning its speaker timestamps
   with Whisper's segment timestamps.

3. **Privacy note:** Uploaded audio files are deleted immediately
   after transcription completes (see `audio.py`), in line with the
   project's data privacy requirement. No audio is persisted by
   default.

## Next steps (Week 3 preview)

Week 3 will introduce a lightweight RAG system with a vector database
of ICD-10 codes. The AI will cross-reference its generated "Assessment"
section against this database to automatically suggest billing codes
for physician review.
