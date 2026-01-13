"""Send normal traffic, then shifted (long) text, and print the drift status."""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(__file__))
from common import load_long_texts, load_val  # noqa: E402

URL = os.environ.get("SERVEML_URL", "http://127.0.0.1:8000")

normal, _ = load_val()
for t in normal[:150]:
    requests.post(f"{URL}/predict", json={"text": t}, timeout=30)
print("normal traffic:", requests.get(f"{URL}/drift", timeout=30).json()["status"])

for t in load_long_texts(150):
    requests.post(f"{URL}/predict", json={"text": t}, timeout=30)
report = requests.get(f"{URL}/drift", timeout=30).json()
print("after shift:", report["status"])
print(report)
