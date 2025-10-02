import os, io
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import imagehash
import requests

APP_NAME = "SwipeSafe"
HF_MODEL = "manelhaj/ai-vs-human-faces"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

app = FastAPI(title=f"{APP_NAME} — AI Profile Safety Check")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LIKELY_LABELS = {"ai", "fake", "generated", "synthetic", "art"}

def hf_classify(img: Image.Image):
    if not HF_TOKEN:
        return None
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = requests.post(HF_API_URL, headers=HEADERS, data=buf.getvalue(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print("HF raw response:", data)

        best = 0.0
        if isinstance(data, list):
            for r in data:
                label = str(r.get("label","")).lower()
                score = float(r.get("score",0.0))
                if "ai" in label or "generated" in label:
                    best = max(best, score)
        elif isinstance(data, dict):
            label = str(data.get("label","")).lower()
            score = float(data.get("score",0.0))
            if "ai" in label or "generated" in label:
                best = max(best, score)
        return best
    except Exception as e:
        print("HF classify error:", e)
        return None

def verdict_label(ai_prob: Optional[float], heuristic_risk: int) -> str:
    base = ai_prob if ai_prob is not None else 0.0
    p = min(1.0, base + min(0.15, heuristic_risk / 100.0))
    if p >= 0.75:
        return "Definitely Made with AI"
    if p >= 0.45 or heuristic_risk >= 20:
        return "Potentially Made with AI"
    return "Not Made with AI"

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": APP_NAME,
        "model_id": HF_MODEL,
        "detector_ready": bool(HF_TOKEN),
    })

@app.get("/tool", response_class=HTMLResponse)
async def tool(request: Request):
    return templates.TemplateResponse("tool.html", {
        "request": request,
        "app_name": APP_NAME,
        "model_id": HF_MODEL,
        "detector_ready": bool(HF_TOKEN),
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

        ai_prob = hf_classify(img)
        if ai_prob is not None:
            ai_probs.append(ai_prob)
            if ai_prob >= 0.55:
                signals.append(f"AI indicator in one image (confidence {ai_prob*100:.1f}%).")

        phashes.append(str(imagehash.phash(img)))

    if len(phashes) > 1 and len(set(phashes)) < len(phashes):
        signals.append("Multiple uploaded photos are near-duplicates.")

    cliches = ["love to travel", "adventure", "foodie", "spontaneous", "work hard play hard", "down to earth"]
    bio_lower = (bio or "").lower()
    c_hits = [c for c in cliches if c in bio_lower]
    if c_hits:
        signals.append("Bio uses common cliches: " + ", ".join(c_hits[:4]))

    risk = 0
    if any("near-duplicates" in s.lower() for s in signals): risk += 12
    if c_hits: risk += min(12, 4*len(c_hits))

    top_ai = max(ai_probs) if ai_probs else None
    verdict = verdict_label(top_ai, risk)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "app_name": APP_NAME,
        "verdict": verdict,
        "signals": signals,
        "model_id": HF_MODEL,
        "detector_ready": bool(HF_TOKEN),
    })
