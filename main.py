from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import base64
import requests

app = FastAPI()

# Bolt details
BOLT_ENDPOINT = "https://slrcnkzdzlvgbzhqdli.supabase.co/functions/v1/voice-detection"
BOLT_API_KEY = "vd_test_37837267-2d32-4185-a6ea-ccdac85d6881"


# ==============================
# Hackathon Request Model
# ==============================
class HackathonRequest(BaseModel):
    language: str
    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        populate_by_name = True


# ==============================
# Adapter Endpoint
# ==============================
@app.post("/detect")
async def detect_voice(
    data: HackathonRequest,
    x_api_key: str = Header(None)
):
    # Optional: simple auth for hackathon
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    # -------- Base64 validation --------
    try:
        base64.b64decode(data.audio_base64, validate=True)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid Base64 audio. Please provide real encoded audio."
        )

    # -------- Forward to Bolt --------
    bolt_payload = {
        "audio": data.audio_base64,   # IMPORTANT MAPPING
        "language": data.language[:2] # en / ta / hi / ml / te
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": BOLT_API_KEY
    }

    response = requests.post(
        BOLT_ENDPOINT,
        json=bolt_payload,
        headers=headers,
        timeout=20
    )

    if not response.ok:
        return {
            "error": "Bolt API error",
            "details": response.text
        }

    return response.json()
