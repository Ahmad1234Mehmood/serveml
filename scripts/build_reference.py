"""Save what 'normal' traffic looks like (length, confidence, label share)."""
import json
import os
import sys

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import load_val  # noqa: E402

INT8 = os.environ.get("INT8_DIR", "models/onnx_int8")
texts, _ = load_val()
tok = AutoTokenizer.from_pretrained(INT8)
model = ORTModelForSequenceClassification.from_pretrained(INT8, file_name="model_quantized.onnx")

lengths, confs, labels = [], [], []
for t in texts:
    inp = tok(t, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inp).logits.numpy()[0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    lengths.append(len(t.split()))
    confs.append(float(p.max()))
    labels.append("positive" if p.argmax() == 1 else "negative")

os.makedirs("results", exist_ok=True)
json.dump({
    "length": lengths,
    "confidence": confs,
    "label_share": {"negative": labels.count("negative") / len(labels),
                    "positive": labels.count("positive") / len(labels)},
}, open("results/reference.json", "w"))
print("reference saved from", len(texts), "samples")
