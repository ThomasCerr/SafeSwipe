# SafeSwipe — AI Profile Safety Check (Deploy-Ready)

A slick, professional site to check dating screenshots for AI-generated or fake profiles.
- Clean landing page (safety-first messaging)
- Clear results: Not Likely Fake / Likely Fake / Very Likely Fake
- Real AI detector (Hugging Face transformers, pixel-based)

## Deploy on Render (no localhost)
1. Push these files to a GitHub repo (name it "safeswipe").
2. In Render, click "New +" → "Web Service" → connect your repo.
3. Render auto-detects Dockerfile; choose the **Free** plan (or Starter for reliability).
4. Add env var: `SAFESWIPE_MODEL_ID = orion-ai/ai-image-detector`
5. Deploy. First boot downloads the model.
6. Your public URL will look like: https://safeswipe.onrender.com
