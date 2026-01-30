from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import base64
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment

app = FastAPI()

API_KEY = "AI_VOICE_DETECTOR_2026_SECRET"


# ==============================
# Request Model (matches tester)
# ==============================
class AudioRequest(BaseModel):
    audio_base64: str = Field(..., alias="audioBase64")
    language: str | None = None
    audio_format: str | None = Field(None, alias="audioFormat")

    class Config:
        populate_by_name = True


# ==============================
# Detection Endpoint
# ==============================
@app.post("/detect")
async def detect_voice(
    data: AudioRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Decode base64
        audio_bytes = base64.b64decode(data.audio_base64)

        # Save raw audio
        with tempfile.NamedTemporaryFile(delete=False) as raw:
            raw.write(audio_bytes)
            raw_path = raw.name

        # Convert to WAV
        audio = AudioSegment.from_file(raw_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
            audio.export(wav.name, format="wav")
            wav_path = wav.name

        # Load audio
        y, sr = librosa.load(wav_path, sr=None, mono=True)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Variability metric
        variability = float(np.std(mfccs))

        # Normalize variability to 0–1 range
        # (empirically chosen safe bounds)
        min_var, max_var = 10.0, 50.0
        score = (variability - min_var) / (max_var - min_var)
        score = float(np.clip(score, 0.0, 1.0))

        # Decision logic
        if score < 0.4:
            classification = "AI-generated"
            confidence = round(1.0 - score, 2)
            explanation = (
                "Low spectral variability detected. "
                "Synthetic speech often exhibits uniform acoustic patterns."
            )

        elif score > 0.6:
            classification = "Human-generated"
            confidence = round(score, 2)
            explanation = (
                "High spectral variability detected. "
                "Natural human speech shows irregular acoustic dynamics."
            )

        else:
            classification = "Uncertain"
            confidence = 0.5
            explanation = (
                "Acoustic features fall in an overlapping region "
                "between human and AI-generated speech."
            )

        return {
            "classification": classification,
            "confidence": confidence,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "error": f"Audio processing failed: {str(e)}"
        }
