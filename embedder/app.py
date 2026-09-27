"""CLIP embedding service: images and text → vectors in the same space (L2-normalised: cosine = dot product).

One model instance serves the API (query text) and the worker (keyframes), instead of a copy in every process.
"""

import base64
import binascii
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import open_clip
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("embedder")

MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    state.update(
        model=model,
        preprocess=preprocess,
        tokenizer=open_clip.get_tokenizer(MODEL_NAME),
        name=f"{MODEL_NAME}/{PRETRAINED}",
        dim=model.text_projection.shape[1],
    )
    logger.info("Loaded %s (dim=%s) in %.1fs", state["name"], state["dim"], time.perf_counter() - t0)
    yield
    state.clear()


app = FastAPI(title="Embedding service", lifespan=lifespan)


class TextRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class ImageRequest(BaseModel):
    images: list[str] = Field(..., min_length=1, max_length=MAX_BATCH, description="base64-encoded images")


class EmbeddingResponse(BaseModel):
    model: str
    dim: int
    vectors: list[list[float]]


def _respond(features: torch.Tensor) -> EmbeddingResponse:
    features = features / features.norm(dim=-1, keepdim=True)
    return EmbeddingResponse(
        model=state["name"], dim=state["dim"], vectors=[[round(x, 6) for x in row] for row in features.tolist()]
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": state.get("name"), "dim": state.get("dim")}


@app.post("/embed/text", response_model=EmbeddingResponse)
def embed_text(body: TextRequest) -> EmbeddingResponse:
    with torch.inference_mode():
        return _respond(state["model"].encode_text(state["tokenizer"](body.texts)))


@app.post("/embed/image", response_model=EmbeddingResponse)
def embed_image(body: ImageRequest) -> EmbeddingResponse:
    try:
        images = [Image.open(io.BytesIO(base64.b64decode(b64, validate=True))).convert("RGB") for b64 in body.images]
    except (binascii.Error, UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    batch = torch.stack([state["preprocess"](image) for image in images])
    with torch.inference_mode():
        return _respond(state["model"].encode_image(batch))
