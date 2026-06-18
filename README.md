# Ambient Clinical Scribe — Week 1: Audio Ingestion & Speaker Diarization

This is the Week 1 deliverable for the Healthcare AI Scribe project:
a FastAPI backend that ingests audio of a doctor-patient conversation,
transcribes it using OpenAI Whisper, and applies speaker labeling
("Doctor" / "Patient") to the resulting transcript segments.

## What's included

- `app/main.py` — FastAPI app entrypoint
- `app/config.py` — central settings (paths, model size, thresholds)
- `app/models/schemas.py` — Pydantic request/response models
- `app/services/asr_service.py` — Whisper transcription + speaker
  labeling logic
- `app/routers/audio.py` — API endpoints for upload and mock-file
  transcription
- `generate_mock_audio.py` — generates a mock doctor-patient
  conversation audio file for local testing (no real dataset needed)
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

## Generate a mock test file (no real dataset required)

```bash
python generate_mock_audio.py
```

This creates `mock_data/mock_consultation_01.wav`, a short synthesized
doctor-patient conversation you can use to test the pipeline end to end.

## Run the server

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check, reports if Whisper model is loaded |
| GET | `/audio/mock-files` | List available mock audio files |
| POST | `/audio/transcribe-mock/{filename}` | Transcribe a bundled mock file |
| POST | `/audio/transcribe` | Upload and transcribe your own audio file |

### Example: transcribe the mock file

```bash
curl -X POST "http://127.0.0.1:8000/audio/transcribe-mock/mock_consultation_01.wav"
```

### Example: upload your own audio

```bash
curl -X POST "http://127.0.0.1:8000/audio/transcribe" \
  -F "file=@/path/to/your/audio.wav"
```

### Example response shape

```json
{
  "filename": "mock_consultation_01.wav",
  "duration_seconds": 42.1,
  "language": "en",
  "full_text": "Good morning, what brings you in today? ...",
  "segments": [
    {
      "speaker": "Doctor",
      "start_time": 0.0,
      "end_time": 2.4,
      "text": "Good morning, what brings you in today?"
    },
    {
      "speaker": "Patient",
      "start_time": 3.1,
      "end_time": 6.8,
      "text": "Hi doctor, I've had a sore throat and a mild fever since yesterday."
    }
  ],
  "speaker_count": 2
}
```

## Week 1 scope and known limitations

This week's goal was the foundational ingestion pipeline, not
production-grade diarization. Two important notes:

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

## Next steps (Week 2 preview)

Week 2 will take the `TranscriptSegment` list produced here and feed
it into an LLM (Claude or GPT-4o) with few-shot prompting to synthesize
a structured SOAP note matching a strict JSON schema.
