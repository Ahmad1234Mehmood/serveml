"""Request logging, Prometheus metrics and a statistical drift check."""
import json
import os
import time
from collections import deque

from prometheus_client import Counter, Histogram
from scipy.stats import chisquare, ks_2samp

LOG_PATH = os.environ.get("SERVEML_LOG", "logs/predictions.jsonl")
REFERENCE_PATH = os.environ.get("SERVEML_REFERENCE", "results/reference.json")
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

REQUESTS = Counter("serveml_requests_total", "Total predictions", ["label"])
LATENCY = Histogram("serveml_latency_seconds", "Inference latency")

recent = deque(maxlen=1000)


def log_prediction(text, label, confidence, latency_ms):
    row = {
        "ts": time.time(),
        "length": len(text.split()),
        "label": label,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }
    recent.append(row)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    REQUESTS.labels(label=label).inc()
    LATENCY.observe(latency_ms / 1000)


def reset_window():
    recent.clear()


def drift_report(reference_path=None, min_samples=100, alpha=0.01):
    if len(recent) < min_samples:
        return {"status": "not_enough_data", "samples": len(recent),
                "needed": min_samples}
    ref = json.load(open(reference_path or REFERENCE_PATH))
    cur_len = [r["length"] for r in recent]
    cur_conf = [r["confidence"] for r in recent]
    cur_labels = [r["label"] for r in recent]

    ks_len = ks_2samp(ref["length"], cur_len)
    ks_conf = ks_2samp(ref["confidence"], cur_conf)

    n = len(cur_labels)
    observed = [cur_labels.count("negative"), cur_labels.count("positive")]
    # smooth the reference shares so a class with share 0 does not divide by zero
    eps = 1e-3
    share_neg = (ref["label_share"]["negative"] + eps) / (1 + 2 * eps)
    share_pos = (ref["label_share"]["positive"] + eps) / (1 + 2 * eps)
    chi = chisquare(observed, f_exp=[share_neg * n, share_pos * n])

    def check(p_value):
        p = float(p_value)
        if p != p:  # NaN: test could not be computed, report it without flagging drift
            return {"p_value": None, "drift": False}
        return {"p_value": p, "drift": bool(p < alpha)}

    checks = {
        "input_length": check(ks_len.pvalue),
        "confidence": check(ks_conf.pvalue),
        "label_share": check(chi.pvalue),
    }
    status = "drift" if any(c["drift"] for c in checks.values()) else "ok"
    return {"status": status, "samples": n, "checks": checks}
