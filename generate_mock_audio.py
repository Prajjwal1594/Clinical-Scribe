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
import subprocess
import sys
from pathlib import Path

MOCK_DATA_DIR = Path(__file__).resolve().parent / "mock_data"
MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# A short scripted doctor-patient conversation. Pauses between lines
# (achieved by saving as separate utterances with the engine) help
# simulate natural conversational gaps for the speaker-labeling
# heuristic to pick up on.
CONVERSATION_1 = [
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

# Second conversation: an orthopedic / knee-pain visit.
CONVERSATION_2 = [
    "Hello, how can I help you today?",
    "Hi doctor, I've been having pain in my right knee for the past two weeks.",
    "Can you describe the pain? Is it sharp, dull, or more of an ache?",
    "It's a dull ache most of the time, but it gets sharp when I climb stairs.",
    "Have you had any recent injuries or started any new physical activities?",
    "I started jogging again about a month ago after a long break.",
    "That could be contributing. Any swelling or stiffness in the morning?",
    "A little swelling in the evening, but no morning stiffness.",
    "Let me examine the knee. Does it hurt when I press here?",
    "Yes, right there on the inner side is quite tender.",
    "It looks like you may have mild patellofemoral syndrome. I'd recommend "
    "reducing your running for now and doing some quad strengthening exercises.",
    "Should I be worried about long term damage?",
    "Not at this stage. If the pain doesn't improve in three to four weeks, "
    "we can order an MRI to rule out any cartilage issues.",
    "Thank you, doctor. I'll follow your advice.",
]

CONVERSATION_3 = [
    "Good morning, what brings you in today?",
    "Hi Doctor, I am having headache since five days.",
    "I see. Is the pain mild, moderate or severe?",
    "Moderately severe",
    "Any associated symptoms like nausea vomiting or sensitivity to light?",
    "No",
    "Are you having any fever?",
    "Yes",
    "Any visual disturbances or neck stiffness?",
    "No",
    "Have you had any head injury recently?",
    "No",
    "Are you taking any medications currently?",
    "Yes, I am taking Metformin for diabetes.",
    "And you mentioned a fever. How high has it been?",
    "Around 101 Fahrenheit.",
    "Any chills or body aches?",
    "Yes, body aches.",
    "I think it's best to get a CBC and a chest X-ray to rule out infection.",
    "Sure, doctor."
]

CONVERSATIONS = [
    ("mock_consultation_01.wav", CONVERSATION_1),
    ("mock_consultation_02.wav", CONVERSATION_2),
    ("mock_consultation_03.wav", CONVERSATION_3),   
]


def _generate_single(index: int):
    """Generate a single audio file (called in a subprocess)."""
    filename, conversation = CONVERSATIONS[index]
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    output_path = MOCK_DATA_DIR / filename
    full_script = " ... ".join(conversation)
    engine.save_to_file(full_script, str(output_path))
    engine.runAndWait()
    print(f"Mock audio saved to: {output_path}")


def generate():
    """Generate all mock audio files, each in its own subprocess
    to work around the pyttsx3 SAPI5 hang-on-reuse bug on Windows."""
    for i in range(len(CONVERSATIONS)):
        subprocess.run(
            [sys.executable, __file__, "--single", str(i)],
            check=True,
        )


if __name__ == "__main__":
    if "--single" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--single") + 1])
        _generate_single(idx)
    else:
        generate()
