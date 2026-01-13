"""Compare PyTorch FP32, ONNX FP32 and ONNX INT8: accuracy, size, latency."""
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import load_val  # noqa: E402

FP32 = os.environ.get("FP32_DIR", "models/fp32")
ONNX = os.environ.get("ONNX_DIR", "models/onnx")
INT8 = os.environ.get("INT8_DIR", "models/onnx_int8")
N_LAT = int(os.environ.get("N_LATENCY", "200"))

texts, labels = load_val()
labels = np.array(labels)
tok = AutoTokenizer.from_pretrained(FP32)


def predict(model, batch_texts):
    inputs = tok(batch_texts, return_tensors="pt", truncation=True,
                 max_length=128, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(-1).numpy()


def evaluate_model(name, model, size_mb):
    preds = []
    for i in range(0, len(texts), 32):
        preds.extend(predict(model, texts[i:i + 32]))
    accuracy = float((np.array(preds) == labels).mean())
    for t in texts[:10]:  # warmup
        predict(model, [t])
    times = []
    for t in texts[:N_LAT]:
        s = time.perf_counter()
        predict(model, [t])
        times.append((time.perf_counter() - s) * 1000)
    return {
        "variant": name,
        "accuracy": round(accuracy * 100, 2),
        "size_mb": round(size_mb, 1),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
    }


def mb(path):
    return os.path.getsize(path) / 1e6


def torch_weights():
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(FP32, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no weights file in " + FP32)


rows = []
pt = AutoModelForSequenceClassification.from_pretrained(FP32).eval()
rows.append(evaluate_model("PyTorch FP32", pt, mb(torch_weights())))

ox = ORTModelForSequenceClassification.from_pretrained(ONNX)
rows.append(evaluate_model("ONNX FP32", ox, mb(os.path.join(ONNX, "model.onnx"))))

q = ORTModelForSequenceClassification.from_pretrained(INT8, file_name="model_quantized.onnx")
rows.append(evaluate_model("ONNX INT8", q, mb(os.path.join(INT8, "model_quantized.onnx"))))

os.makedirs("results", exist_ok=True)
meta = {"machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(), "batch_size_latency": 1,
        "n_validation": len(texts), "n_latency_samples": N_LAT}
json.dump({"meta": meta, "rows": rows}, open("results/benchmark.json", "w"), indent=2)

lines = ["| Variant | Accuracy % | Size MB | p50 ms | p95 ms |",
         "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['variant']} | {r['accuracy']} | {r['size_mb']} | {r['p50_ms']} | {r['p95_ms']} |")
open("results/benchmark.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
