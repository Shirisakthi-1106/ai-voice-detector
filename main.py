from fastapi import FastAPI, Form
import base64
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment
import os

app = FastAPI(
    title="AI Generated Voice Detection API",
    description="Detects whether a voice sample is AI-generated or human-generated",
    version="1.0"
)

# -------------------------------
# API KEY
# -------------------------------
API_KEY = "AI_VOICE_DETECTOR_2026_SECRET"

# -------------------------------
# ENDPOINT
# -------------------------------
@app.post("/detect")
async def detect_voice(
    audio_base64: str = Form(...),
    api_key: str = Form(...)
):
    # -------- AUTH CHECK --------
    if api_key != API_KEY:
        return {
            "error": "Unauthorized: Invalid API Key"
        }

    try:
        # -------- DECODE BASE64 --------
        audio_bytes = base64.b64decode(audio_base64)

        # -------- SAVE TEMP MP3 --------
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
            mp3_file.write(audio_bytes)
            mp3_path = mp3_file.name

        # -------- MP3 → WAV --------
        audio = AudioSegment.from_file(mp3_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            audio.export(wav_file.name, format="wav")
            wav_path = wav_file.name

        # -------- LOAD AUDIO --------
        y, sr = librosa.load(wav_path, sr=None)

        # -------- FEATURE EXTRACTION --------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        variability = np.std(mfccs)

        # -------- SIMPLE DECISION LOGIC --------
        if variability < 20:
            return {
                "classification": "AI-generated",
                "confidence": 0.85,
                "explanation": "Low spectral variability commonly observed in synthetic speech."
            }
        else:
            return {
                "classification": "Human-generated",
                "confidence": 0.80,
                "explanation": "Natural spectral variability indicates human speech."
            }

    except Exception as e:
        return {
            "error": f"Audio processing failed: {str(e)}"
        }
