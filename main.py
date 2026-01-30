from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment

app = FastAPI()

API_KEY = "AI_VOICE_DETECTOR_2026_SECRET"

# -----------------------------
# Request Body Model
# -----------------------------
class AudioRequest(BaseModel):
    audio_base64: str
    language: str | None = None
    audio_format: str | None = None

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/detect")
async def detect_voice(
    data: AudioRequest,
    x_api_key: str = Header(None)
):
    # -------- AUTH --------
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        audio_bytes = base64.b64decode(data.audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        audio = AudioSegment.from_file(audio_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            audio.export(wf.name, format="wav")
            wav_path = wf.name

        y, sr = librosa.load(wav_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        variability = np.std(mfccs)

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
