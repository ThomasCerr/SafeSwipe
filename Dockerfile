FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1 libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
