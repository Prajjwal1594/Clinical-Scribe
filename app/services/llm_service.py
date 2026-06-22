"""
LLM service for generating structured SOAP notes from transcribed
doctor-patient conversations.

Uses Google Gemini via LangChain with few-shot prompting to convert
raw transcript segments into the formal SOAP (Subjective, Objective,
Assessment, Plan) clinical note format.

The prompt is carefully engineered to:
  - Extract only medically relevant information
  - Filter out casual small talk and pleasantries
  - Map conversational dialogue into formal clinical language
  - Return strict JSON matching our Pydantic schemas
"""

import json
import logging
import time
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import (
    GOOGLE_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_RETRIES,
)
from app.models.schemas import TranscriptSegment
from app.models.soap_schemas import SOAPNote, SOAPNoteResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt: instructs the LLM to behave as a medical scribe
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert medical scribe AI. Your job is to read a transcribed
doctor-patient conversation and produce a structured SOAP note in JSON format.

RULES:
1. Extract ONLY medically relevant information from the conversation.
2. Ignore greetings, small talk, and non-clinical dialogue.
3. Use formal clinical language in the output (e.g., "Patient reports
   odynophagia" instead of "patient says throat hurts").
4. If information for a field is not mentioned in the conversation,
   use null for optional fields or "Not mentioned" for required fields.
5. For differential_diagnosis, list plausible alternatives based on the
   symptoms discussed. If none are discussed, return an empty list.
6. For medications, include dosage and frequency if mentioned.
7. Return ONLY valid JSON — no markdown, no code fences, no explanation.

OUTPUT SCHEMA:
{
  "subjective": {
    "chief_complaint": "string (required)",
    "history_of_present_illness": "string (required)",
    "review_of_systems": "string or null"
  },
  "objective": {
    "vitals": "string or null",
    "physical_exam": "string or null"
  },
  "assessment": {
    "diagnosis": "string (required)",
    "differential_diagnosis": ["string", ...]
  },
  "plan": {
    "medications": ["string", ...],
    "follow_up": "string or null",
    "patient_education": "string or null"
  }
}
"""

# ---------------------------------------------------------------------------
# Few-shot example: shows the LLM the expected input → output mapping
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLE_INPUT = """\
[0.00s - 2.40s] Doctor: Good morning, what brings you in today?
[3.10s - 6.80s] Patient: Hi doctor, I've had a sore throat and a mild fever since yesterday.
[7.50s - 11.20s] Doctor: Okay, have you noticed any cough or difficulty swallowing?
[12.00s - 15.30s] Patient: No cough, but swallowing is a bit painful.
[16.00s - 20.50s] Doctor: Let me take a look. Your throat appears red with some swelling. I'd recommend a rapid strep test.
[21.30s - 24.00s] Patient: Sure, doctor.
[25.00s - 30.00s] Doctor: In the meantime, take paracetamol 500mg every 6 hours for the pain and fever, and gargle with warm salt water. We'll schedule a follow up visit.
"""

FEW_SHOT_EXAMPLE_OUTPUT = """\
{
  "subjective": {
    "chief_complaint": "Sore throat and mild fever since yesterday",
    "history_of_present_illness": "Patient presents with a one-day history of sore throat accompanied by mild fever. Denies cough but reports odynophagia (painful swallowing).",
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
}
"""


def _format_transcript(segments: List[TranscriptSegment]) -> str:
    """Formats transcript segments into a readable string for the LLM prompt."""
    lines = []
    for seg in segments:
        lines.append(
            f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s] "
            f"{seg.speaker}: {seg.text}"
        )
    return "\n".join(lines)


class LLMService:
    """Manages LLM interactions for SOAP note generation."""

    def __init__(self):
        self._llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-initialize the LLM client."""
        if self._llm is None:
            if not GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Get a free key at "
                    "https://aistudio.google.com and set it in your "
                    ".env file or as an environment variable."
                )
            logger.info("Initializing LLM: %s", LLM_MODEL_NAME)
            self._llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                temperature=LLM_TEMPERATURE,
            )
        return self._llm

    def generate_soap_note(
        self,
        segments: List[TranscriptSegment],
        filename: str = None,
    ) -> SOAPNoteResponse:
        """
        Generate a structured SOAP note from transcript segments.

        1. Formats the segments into a readable transcript
        2. Sends the transcript to Gemini with system + few-shot prompts
        3. Parses and validates the JSON response against Pydantic schemas
        4. Retries on malformed output up to LLM_MAX_RETRIES times
        """
        llm = self._get_llm()
        transcript_text = _format_transcript(segments)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            # Few-shot example
            HumanMessage(content=(
                "Here is an example.\n\n"
                f"TRANSCRIPT:\n{FEW_SHOT_EXAMPLE_INPUT}\n\n"
                "SOAP NOTE:"
            )),
            HumanMessage(content=FEW_SHOT_EXAMPLE_OUTPUT),
            # Actual request
            HumanMessage(content=(
                "Now generate a SOAP note for the following transcript.\n\n"
                f"TRANSCRIPT:\n{transcript_text}\n\n"
                "SOAP NOTE:"
            )),
        ]

        start_time = time.time()
        last_error = None

        for attempt in range(1, LLM_MAX_RETRIES + 2):  # +2 because range is exclusive and we want retries + 1
            try:
                logger.info(
                    "SOAP generation attempt %d/%d",
                    attempt,
                    LLM_MAX_RETRIES + 1,
                )
                response = llm.invoke(messages)
                raw_text = response.content.strip()

                # Strip markdown code fences if the LLM wraps the output
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[1]  # remove first line
                if raw_text.endswith("```"):
                    raw_text = raw_text.rsplit("```", 1)[0]  # remove last fence
                raw_text = raw_text.strip()

                parsed = json.loads(raw_text)
                soap_note = SOAPNote.model_validate(parsed)

                elapsed = round(time.time() - start_time, 2)
                logger.info("SOAP note generated successfully in %.2fs", elapsed)

                return SOAPNoteResponse(
                    filename=filename,
                    soap_note=soap_note,
                    model_used=LLM_MODEL_NAME,
                    processing_time_seconds=elapsed,
                )

            except (json.JSONDecodeError, Exception) as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d failed: %s — %s",
                    attempt,
                    type(exc).__name__,
                    str(exc)[:200],
                )

        raise RuntimeError(
            f"Failed to generate valid SOAP note after "
            f"{LLM_MAX_RETRIES + 1} attempts. Last error: {last_error}"
        )


# Singleton instance shared across the app
llm_service = LLMService()
