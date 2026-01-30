from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import base64
import requests

app = FastAPI()

# ==============================
# Bolt configuration
# ==============================
BOLT_ENDPOINT = "https://slrcnkzdzlvgbzhqdli.supabase.co/functions/v1/voice-detection"
BOLT_API_KEY = "vd_test_37837267-2d32-4185-a6ea-ccdac85d6881"


# ==============================
# Hackathon request model
# ==============================
class HackathonRequest(BaseModel):
    language: str
    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        populate_by_name = True


# ==============================
# Health check (optional but useful)
# ==============================
@app.get("/")
def health():
    return {"status": "ok"}


# ==============================
# Detection endpoint
# ==============================
@app.post("/detect")
async def detect_voice(
    data: HackathonRequest,
    x_api_key: str = Header(None)
):
    # --- Basic header presence check (hackathon requirement) ---
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    # --- Validate Base64 early ---
    try:
        base64.b64decode(data.audio_base64, validate=True)
    except Exception:
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "explanation": "Invalid Base64 audio provided."
        }

    # --- Prepare Bolt request ---
    bolt_payload = {
        "audio": data.audio_base64,
        "language": data.language[:2] if data.language else "en"
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": BOLT_API_KEY
    }

    # --- Call Bolt safely ---
    try:
        response = requests.post(
            BOLT_ENDPOINT,
            json=bolt_payload,
            headers=headers,
            timeout=15
        )
    except Exception:
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "explanation": "Upstream service unreachable."
        }

    # --- Handle non-JSON or error responses ---
    try:
        bolt_response = response.json()
    except Exception:
        bolt_response = {
            "classification": "unknown",
            "confidence": 0.0,
            "explanation": "Upstream service returned invalid response."
        }

    if not response.ok:
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "explanation": "Upstream service error.",
            "details": bolt_response
        }

    # --- Success ---
    return bolt_response
