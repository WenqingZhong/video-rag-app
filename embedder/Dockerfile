# Model server for CLIP embeddings. Separate image: torch is ~1 GB and only this service needs it.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
EXPOSE 8000
# One process on purpose: the model is loaded once. Scale with more replicas, not more workers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
