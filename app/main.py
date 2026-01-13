"""FastAPI service for the INT8 ONNX sentiment model."""
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from optimum.onnxruntime import ORTModelForSequenceClassification
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from app.monitoring import drift_report, log_prediction, reset_window

MODEL_DIR = os.environ.get("MODEL_DIR", "models/onnx_int8")
MODEL_FILE = os.environ.get("MODEL_FILE", "model_quantized.onnx")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR, file_name=MODEL_FILE)
LABELS = {0: "negative", 1: "positive"}

app = FastAPI(title="ServeML", description="INT8 ONNX sentiment model with monitoring")
app.mount("/metrics", make_asgi_app())


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text is empty")
    start = time.perf_counter()
    inputs = tok(req.text, return_tensors="pt", truncation=True, max_length=128)
    logits = model(**inputs).logits.detach().numpy()[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    idx = int(probs.argmax())
    latency_ms = (time.perf_counter() - start) * 1000
    confidence = float(probs[idx])  # log the unrounded value so it matches the reference
    log_prediction(req.text, LABELS[idx], confidence, latency_ms)
    return {
        "label": LABELS[idx],
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 1),
    }


@app.get("/drift")
def drift():
    return drift_report()


@app.post("/drift/reset")
def drift_reset():
    """Clear the in-memory window (useful between simulations)."""
    reset_window()
    return {"status": "reset"}
