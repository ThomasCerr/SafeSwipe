import os, io, base64
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import imagehash
import requests

# Hugging Face model (still configurable via env var)
MODEL_ID = os.getenv("SAFESWIPE_MODEL_ID", "umm-maybe/ai-art-detector")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="SafeSwipe — AI Profile Safety Check")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LIKELY_LABELS = {"ai", "fake", "generated", "synthetic", "art"}

def classify_image(img: Image.Image) -> Optional[float]:
    """Send image to Hugging Face Inference API and return AI probability"""
    if not HF_TOKEN:
        return None
    
    try:
        # Convert PIL image to bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Call Hugging Face Inference API
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_ID}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=img_bytes,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        best_ai = 0.0
        for r in results:
            label = str(r.get("label", "")).lower()
            score = float(r.get("score", 0.0))
            if any(k in label for k in LIKELY_LABELS):
                best_ai = max(best_ai, score)
        return best_ai
    except Exception as e:
        print("HF API error:", e)
        return None

def verdict_label(ai_prob: Optional[float], heuristic_risk: int) -> str:
    base = ai_prob if ai_prob is not None else 0.0
    adj = min(0.15, heuristic_risk / 100.0)  # small heuristic boost
    p = base + adj
    if p >= 0.85:
        return "Definitely Made with AI"
    if p >= 0.55 or heuristic_risk >= 20:
        return "Potentially Made with AI"
    return "Not Made with AI"

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_id": MODEL_ID,
        "model_error": None if HF_TOKEN else "Missing HF_TOKEN"
    })

@app.get("/tool", response_class=HTMLResponse)
async def tool(request: Request):
    return templates.TemplateResponse("tool.html", {
        "request": request,
        "model_id": MODEL_ID,
        "model_error": None if HF_TOKEN else "Missing HF_TOKEN"
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request,
                  files: List[UploadFile] = File(...),
                  bio: str = Form("")):
    signals = []
    ai_probs = []
    phashes = []

    for f in files[:5]:
        content = await f.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # Run AI classification via Hugging Face
        ai_prob = classify_image(img)
        if ai_prob is not None:
            ai_probs.append(ai_prob)
            if ai_prob >= 0.55:
                signals.append(f"AI indicator present in an image (confidence {ai_prob*100:.1f}%).")

        # Perceptual hash check
        phashes.append(str(imagehash.phash(img)))

    if len(phashes) > 1 and len(set(phashes)) < len(phashes):
        signals.append("Multiple uploaded photos are near-duplicates.")

    # Bio cliché checks
    cliches = ["love to travel", "adventure", "foodie", "spontaneous", "work hard play hard", "down to earth"]
    bio_lower = (bio or "").lower()
    c_hits = [c for c in cliches if c in bio_lower]
    if c_hits:
        signals.append("Bio uses common cliches: " + ", ".join(c_hits[:4]))

    heuristic_risk = 0
    if any("near-duplicates" in s.lower() for s in signals):
        heuristic_risk += 12
    if c_hits:
        heuristic_risk += min(12, 4 * len(c_hits))

    top_ai = max(ai_probs) if ai_probs else None
    verdict = verdict_label(top_ai, heuristic_risk)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "verdict": verdict,
        "signals": signals,
        "model_id": MODEL_ID,
        "model_error": None if HF_TOKEN else "Missing HF_TOKEN"
    })
