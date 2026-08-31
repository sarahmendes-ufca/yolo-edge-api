# tests/test_api.py

import base64
import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import _decode_image
from app.main import app


client = TestClient(app)


# ── Smoke test ─────────────────────────────────────────────
def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Unit test: decode de imagem ────────────────────────────
def test_decode_image_valid_base64():
    # Cria imagem 10x10 px branca e codifica em base64
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    b64 = base64.b64encode(buf.getvalue()).decode()

    result = _decode_image(b64)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10, 3)


# ── Integration test: inferência completa ──────────────────
def test_predict_returns_detections():
    with open("tests/assets/zidane.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = client.post(
        "/predict",
        json={
            "image_base64": b64,
            "confidence": 0.3,
            "model_name": "yolov8n.pt",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["detections"]) >= 1
    assert data["inference_ms"] > 0
