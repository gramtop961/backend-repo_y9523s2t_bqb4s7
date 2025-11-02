import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeneratePayload(BaseModel):
    mode: str = Field(description="text | image | camera")
    tone: str
    text: Optional[str] = None
    note: Optional[str] = None
    imageDataUrl: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


@app.post("/generate")
def generate_replies(payload: GeneratePayload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server is not configured with an OpenAI API key.")

    if payload.mode not in {"text", "image", "camera"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    if not payload.tone:
        raise HTTPException(status_code=400, detail="Tone is required")

    if payload.mode == "text" and not (payload.text and payload.text.strip()):
        raise HTTPException(status_code=400, detail="Text content is required for text mode")

    if payload.mode in {"image", "camera"} and not payload.imageDataUrl:
        raise HTTPException(status_code=400, detail="Image is required for image/camera mode")

    system_prompt = (
        "You are an assistant that crafts concise, natural-sounding chat replies. "
        "Always return a compact JSON object with a `suggestions` array of exactly two short strings. "
        "Keep each under 25 words, casual, context-aware, and human."
    )

    user_summary = (
        f"Here is the input content.\n"
        f"Tone: {payload.tone}.\n"
        f"Personal note: {payload.note or 'N/A'}.\n"
        "If an image is provided, analyze it for chat context and extract any text you can read.\n"
        "Return JSON strictly as {\"suggestions\":[\"...\",\"...\"]}."
    )

    content = [{"type": "text", "text": user_summary}]

    if payload.mode == "text" and payload.text:
        content.append({"type": "text", "text": f"User text: {payload.text}"})

    if payload.mode in {"image", "camera"} and payload.imageDataUrl:
        content.append({
            "type": "image_url",
            "image_url": {"url": payload.imageDataUrl},
        })

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 200,
                "temperature": 0.8,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        suggestions = []
        # Try parse JSON block
        try:
            import json as pyjson
            import re
            m = re.search(r"\{[\s\S]*\}", message)
            if m:
                parsed = pyjson.loads(m.group(0))
                if isinstance(parsed, dict) and isinstance(parsed.get("suggestions"), list):
                    suggestions = [str(x) for x in parsed["suggestions"]][:2]
        except Exception:
            suggestions = []

        if len(suggestions) < 2:
            # fallback: split lines
            lines = [
                l.strip() for l in message.splitlines()
                if l.strip()
            ]
            # remove bullets and numbering
            import re
            lines = [re.sub(r"^\d+\.|^-\s*", "", l).strip() for l in lines]
            suggestions = lines[:2]

        if len(suggestions) < 2:
            raise HTTPException(status_code=500, detail="Could not parse two suggestions from AI response")

        return {"suggestions": suggestions}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
