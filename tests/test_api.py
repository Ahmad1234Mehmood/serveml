import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Set SERVEML_SMOKE=1 when testing with a random-weight model (no real sentiment).
real_model = pytest.mark.skipif(
    os.environ.get("SERVEML_SMOKE") == "1", reason="needs the real fine-tuned model"
)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_schema():
    r = client.post("/predict", json={"text": "This film is wonderful"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"positive", "negative"}
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0


@real_model
def test_positive():
    r = client.post("/predict", json={"text": "This film is wonderful"})
    assert r.json()["label"] == "positive"


@real_model
def test_negative():
    r = client.post("/predict", json={"text": "This was terrible and boring"})
    assert r.json()["label"] == "negative"


def test_empty_text_rejected():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_whitespace_text_rejected():
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_too_long_text_rejected():
    assert client.post("/predict", json={"text": "a" * 2001}).status_code == 422


def test_metrics_endpoint():
    assert client.get("/metrics/").status_code == 200


def test_drift_needs_data():
    client.post("/drift/reset")
    assert client.get("/drift").json()["status"] == "not_enough_data"
