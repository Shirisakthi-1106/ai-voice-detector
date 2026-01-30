from fastapi import FastAPI
import base64
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment

app = FastAPI()

API_KEY = "AI_VOICE_DETECTOR_2026_SECRET"

@app.post("/detect")
async def detect_voice(audio_base64: str, api_key: str):
    # -------- AUTH CHECK --------
    if api_key != API_KEY:
        return {"error": "Unauthorized: Invalid API Key"}

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)

        # Save as temp mp3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
            mp3_file.write(audio_bytes)
            mp3_path = mp3_file.name

        # Convert to wav
        audio = AudioSegment.from_mp3(mp3_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            audio.export(wav_file.name, format="wav")
            wav_path = wav_file.name

        # Load audio
        y, sr = librosa.load(wav_path, sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        variability = np.std(mfccs)

        # Simple logic
        if variability < 20:
            return {
                "classification": "AI-generated",
                "confidence": 0.85,
                "explanation": "Low spectral variability indicates synthetic speech patterns."
            }
        else:
            return {
                "classification": "Human-generated",
                "confidence": 0.80,
                "explanation": "Natural spectral variability indicates human speech."
            }

    except Exception as e:
        return {"error": str(e)}
