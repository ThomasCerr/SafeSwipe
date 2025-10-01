import os, io
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import imagehash

MODEL_ID = os.getenv('SAFESWIPE_MODEL_ID', 'orion-ai/ai-image-detector')

detector = None
model_load_error = None
try:
    from transformers import pipeline
    detector = pipeline('image-classification', model=MODEL_ID)
except Exception as e:
    model_load_error = str(e)
    detector = None

app = FastAPI(title='SafeSwipe — AI Profile Safety Check')
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

LIKELY_LABELS = {'ai', 'fake', 'generated', 'synthetic'}

def classify_image(img: Image.Image) -> Optional[float]:
    global detector
    if detector is None:
        return None
    try:
        res = detector(img.convert('RGB'))
        best_ai = 0.0
        for r in res:
            label = str(r.get('label','')).lower()
            score = float(r.get('score', 0.0))
            if any(k in label for k in LIKELY_LABELS):
                if score > best_ai:
                    best_ai = score
        return best_ai
    except Exception:
        return None

def bucketize(ai_prob: Optional[float], heuristic_risk: int) -> str:
    score = 0
    if ai_prob is not None:
        score += int(ai_prob * 70)
    score += min(30, heuristic_risk)
    if score >= 70:
        return 'Very Likely Fake'
    elif score >= 40:
        return 'Likely Fake'
    else:
        return 'Not Likely Fake'

@app.get('/', response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'model_id': MODEL_ID,
        'model_error': model_load_error
    })

@app.post('/analyze', response_class=HTMLResponse)
async def analyze(request: Request,
                  files: List[UploadFile] = File(...),
                  bio: str = Form('')):
    signals = []
    ai_probs = []
    phashes = []

    for f in files[:5]:
        content = await f.read()
        img = Image.open(io.BytesIO(content)).convert('RGB')
        ai_prob = classify_image(img)
        if ai_prob is not None:
            ai_probs.append(ai_prob)
            if ai_prob >= 0.7:
                signals.append(f'AI detector flagged an image with {ai_prob*100:.1f}% confidence')
        ph = str(imagehash.phash(img))
        phashes.append(ph)

    if len(phashes) > 1 and len(set(phashes)) < len(phashes):
        signals.append('Multiple images look unusually similar (near-duplicates)')

    cliches = ['love to travel', 'adventure', 'foodie', 'spontaneous', 'work hard play hard']
    bio_lower = (bio or '').lower()
    c_hits = [c for c in cliches if c in bio_lower]
    if c_hits:
        signals.append(f"Bio contains common clichés: {', '.join(c_hits[:4])}")

    heuristic_risk = 0
    if any('near-duplicates' in s.lower() for s in signals): heuristic_risk += 15
    if c_hits: heuristic_risk += min(15, 5 * len(c_hits))
    top_ai = max(ai_probs) if ai_probs else None
    risk_label = bucketize(top_ai, heuristic_risk)

    return templates.TemplateResponse('results.html', {
        'request': request,
        'risk': risk_label,
        'signals': signals,
        'model_id': MODEL_ID,
        'model_error': model_load_error
    })
