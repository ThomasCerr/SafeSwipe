import os, io
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import imagehash
import requests
import cv2
import numpy as np

APP_NAME = "SwipeSafe"
HF_MODEL = "prithivMLmods/deepfake-detector-model-v1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

app = FastAPI(title=f"{APP_NAME} — AI Profile Safety Check")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LIKELY_LABELS = {'fake','ai','synthetic','generated','render','cgi','art'}

def hf_classify(img: Image.Image):
    if not HF_TOKEN:
        return None
    try:
        # Face crop via OpenCV Haar cascade; fall back to full image
        np_img = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
        if len(faces) > 0:
            # choose largest face
            x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
            crop = Image.fromarray(np_img[y:y+h, x:x+w]).resize((512,512))
        else:
            crop = img

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        resp = requests.post(HF_API_URL, headers=HEADERS, data=buf.getvalue(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print("HF raw response:", data)

        best = 0.0
        # list of dicts
        if isinstance(data, list):
            for r in data:
                label = str(r.get("label","")).lower()
                score = float(r.get("score",0.0))
                if any(k in label for k in ('fake','ai','synthetic','generated','render','cgi','art')):
                    best = max(best, score)
        # dict forms
        if isinstance(data, dict):
            label = str(data.get("label","")).lower()
            if label:
                score = float(data.get("score",0.0))
                if any(k in label for k in ('fake','ai','synthetic','generated','render','cgi','art')):
                    best = max(best, score)
            # nested lists/dicts
            for v in data.values():
                if isinstance(v, list):
                    for r in v:
                        if isinstance(r, dict):
                            L = str(r.get("label","")).lower()
                            S = float(r.get("score",0.0))
                            if any(k in L for k in ('fake','ai','synthetic','generated','render','cgi','art')):
                                best = max(best, S)

        return best
    except Exception as e:
        print("HF classify error:", e)
        return None

def verdict_label(ai_prob: Optional[float], heuristic_risk: int) -> str:
    base = ai_prob if ai_prob is not None else 0.0
    # combine simple heuristic
    p = min(1.0, base + min(0.15, heuristic_risk / 100.0))
    if p >= 0.75:
        return "Definitely Made with AI"
    if p >= 0.45 or heuristic_risk >= 20:
        return "Potentially Made with AI"
    return "Not Made with AI"

@app.get("/", response_class=HTMLResponse)
@app.head("/")
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
