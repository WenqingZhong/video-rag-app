import base64
import io
import json

import httpx
import pytest
from PIL import Image

from src.services.captioning import Captioner, CaptionError, downscale_jpeg
from src.services.embeddings import EmbeddingClient, EmbeddingError


def jpeg(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_downscale_fits_longest_side():
    small = Image.open(io.BytesIO(downscale_jpeg(jpeg(640, 1138), 448)))
    assert max(small.size) == 448 and small.size[0] < small.size[1]  # portrait stays portrait


def embedder(handler, batch_size: int = 2) -> EmbeddingClient:
    client = EmbeddingClient("http://embedder", batch_size=batch_size)
    client.http = httpx.Client(base_url="http://embedder", transport=httpx.MockTransport(handler))
    return client


def test_embed_images_batches_and_records_model():
    calls = []

    def handler(request):
        batch = json.loads(request.content)["images"]
        calls.append(len(batch))
        base64.b64decode(batch[0], validate=True)  # images travel as base64
        return httpx.Response(200, json={"model": "ViT-B-32/test", "dim": 2, "vectors": [[1.0, 0.0]] * len(batch)})

    client = embedder(handler, batch_size=2)
    vectors = client.embed_images([b"a", b"b", b"c"])
    assert calls == [2, 1] and len(vectors) == 3 and client.model == "ViT-B-32/test"


def test_embedder_4xx_is_permanent_5xx_is_retryable():
    with pytest.raises(EmbeddingError):
        embedder(lambda r: httpx.Response(400, text="invalid image")).embed_images([b"x"])
    with pytest.raises(httpx.HTTPStatusError):
        embedder(lambda r: httpx.Response(503)).embed_texts(["dog"])


def captioner(handler) -> Captioner:
    c = Captioner("http://ollama", model="qwen2.5vl:3b", prompt="Describe.", max_side=448)
    c.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return c


def test_caption_sends_downscaled_image_and_normalises_text():
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen.update(model=body["model"], size=Image.open(io.BytesIO(base64.b64decode(body["images"][0]))).size)
        return httpx.Response(200, json={"response": "  Two dogs\n lying on grass. "})

    text = captioner(handler).caption(jpeg(1280, 720))
    assert text == "Two dogs lying on grass."
    assert seen == {"model": "qwen2.5vl:3b", "size": (448, 252)}


def test_empty_caption_is_an_error():
    with pytest.raises(CaptionError):
        captioner(lambda r: httpx.Response(200, json={"response": "   "})).caption(jpeg(10, 10))


def test_caption_health_requires_model_pulled():
    c = captioner(lambda r: httpx.Response(200, json={"models": [{"name": "llama3.2:1b"}]}))
    assert c.health_check()["status"] == "unhealthy"
