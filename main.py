from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal, Any
import os
import json
import base64
import requests
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

app = FastAPI(title="Smart Reply Generator API")

# Allow frontend origin in dev; in production, restrict this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    mode: Literal["text", "image", "camera"]
    tone: Literal["flirty", "friendly", "cool", "empathetic", "assertive"]
    text: Optional[str] = None
    note: Optional[str] = None
    imageDataUrl: Optional[str] = Field(default=None, description="Data URL for uploaded or captured image")


class GenerateResponse(BaseModel):
    suggestions: List[str]


@app.get("/")
def root() -> dict:
    return {"message": "Smart Reply Generator backend is running"}


@app.get("/api/hello")
def api_hello() -> dict:
    return {"hello": "world"}


@app.get("/test")
def test() -> dict:
    key_present = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "ok", "openai_key_configured": key_present}


def build_prompt(req: GenerateRequest) -> str:
    base = (
        "You're an assistant that crafts concise, natural-sounding chat reply suggestions. "
        "Provide exactly two different replies, short and ready to send. "
        "Match the requested tone: {tone}. "
    ).format(tone=req.tone)

    extras = []
    if req.text:
        extras.append(f"Original message: {req.text}")
    if req.note:
        extras.append(f"Personal note to incorporate: {req.note}")

    joined = "\n".join(extras)

    instructions = (
        "Respond ONLY as JSON array of two strings, for example: [\"reply1\", \"reply2\"]. "
        "No additional text."
    )

    return f"{base}\n{joined}\n{instructions}".strip()


def call_openai_chat(messages: List[dict], api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.7,
        "response_format": {"type": "text"},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {r.text}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid response from OpenAI")


def parse_suggestions(raw: str) -> List[str]:
    # Try JSON parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            out = [str(x).strip() for x in parsed if isinstance(x, (str, int, float))]
            return out[:2]
    except Exception:
        pass

    # Fallback: split by lines and take non-empty lines
    lines = [ln.strip("- •* \t") for ln in raw.splitlines() if ln.strip()]
    return lines[:2] if lines else []


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    # Validate inputs by mode
    if req.mode == "text":
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Text is required for text mode.")
    elif req.mode in ("image", "camera"):
        if not req.imageDataUrl:
            raise HTTPException(status_code=400, detail="imageDataUrl is required for image/camera mode.")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    system_msg = {
        "role": "system",
        "content": "You are a helpful AI that crafts short, ready-to-send chat replies.",
    }

    # Build user content depending on mode
    if req.mode == "text":
        user_text = build_prompt(req)
        messages = [system_msg, {"role": "user", "content": user_text}]
    else:
        # Vision-like format: include the instruction plus the image
        instruction = build_prompt(req)
        # OpenAI vision supports either URL or data URL. We'll pass the data URL directly.
        messages = [
            system_msg,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "input_image", "image_url": req.imageDataUrl},
                ],
            },
        ]

    raw = call_openai_chat(messages, api_key)
    suggestions = parse_suggestions(raw)
    if len(suggestions) < 2:
        # Make a second attempt: ask for short bullet list
        fallback_messages = messages[:-1] + [
            {
                "role": "user",
                "content": (
                    "Give exactly two short reply suggestions, each on its own line, without numbering."
                ),
            }
        ]
        raw2 = call_openai_chat(fallback_messages, api_key)
        suggestions = parse_suggestions(raw2)

    if len(suggestions) < 2:
        raise HTTPException(status_code=500, detail="Failed to produce two suggestions.")

    return {"suggestions": suggestions[:2]}
