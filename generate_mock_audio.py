"""
Generates a mock doctor-patient conversation audio file for local
testing of the ASR + diarization pipeline, since real clinical audio
datasets are not available in this environment.

Uses pyttsx3 (offline text-to-speech) to synthesize a short scripted
conversation and saves it to mock_data/. Run this once before testing
the /audio/transcribe-mock/ endpoint.

Usage:
    python generate_mock_audio.py
"""

import pyttsx3
from pathlib import Path

MOCK_DATA_DIR = Path(__file__).resolve().parent / "mock_data"
MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# A short scripted doctor-patient conversation. Pauses between lines
# (achieved by saving as separate utterances with the engine) help
# simulate natural conversational gaps for the speaker-labeling
# heuristic to pick up on.
CONVERSATION = [
    "Good morning, what brings you in today?",
    "Hi doctor, I've had a sore throat and a mild fever since yesterday.",
    "Okay, have you noticed any cough or difficulty swallowing?",
    "A little cough, but no real trouble swallowing.",
    "Let's check your temperature and take a look at your throat.",
    "Sure, go ahead.",
    "Your throat looks a bit red and your temperature is 100.4 Fahrenheit.",
    "Is that something to worry about?",
    "It's likely a mild viral infection. I recommend rest, fluids, and "
    "over the counter acetaminophen for the fever.",
    "Should I come back if it doesn't improve?",
    "Yes, if symptoms persist beyond five days or worsen, please schedule "
    "a follow up visit.",
]

OUTPUT_PATH = MOCK_DATA_DIR / "mock_consultation_01.wav"


def generate():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)

    full_script = " ... ".join(CONVERSATION)
    engine.save_to_file(full_script, str(OUTPUT_PATH))
    engine.runAndWait()
    print(f"Mock audio saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
