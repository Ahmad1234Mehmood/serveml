"""Compare two benchmark runs, for example CPU subset vs Kaggle full dataset.

Usage:
  python scripts/compare_runs.py results_cpu_subset results \
      --label-a "8000 sentences (CPU)" --label-b "67,349 sentences (Kaggle GPU)"
"""
import argparse
import json
import math
import os


def load(folder):
    data = json.load(open(os.path.join(folder, "benchmark.json"), encoding="utf-8"))
    return data["meta"], {r["variant"]: r for r in data["rows"]}


def margin_points(acc_pct, n):
    """95% margin of error of an accuracy measured on n samples, in points."""
    p = acc_pct / 100
    return 1.96 * math.sqrt(p * (1 - p) / n) * 100


parser = argparse.ArgumentParser()
parser.add_argument("dir_a")
parser.add_argument("dir_b")
parser.add_argument("--label-a", default="Run A")
parser.add_argument("--label-b", default="Run B")
args = parser.parse_args()

runs = [(args.label_a, *load(args.dir_a)), (args.label_b, *load(args.dir_b))]

lines = [
    "| Training data | PyTorch FP32 acc % | ONNX INT8 acc % | INT8 change (points) | INT8 size MB |",
    "|---|---|---|---|---|",
]
margins = []
for label, meta, rows in runs:
    fp32, int8 = rows["PyTorch FP32"], rows["ONNX INT8"]
    change = int8["accuracy"] - fp32["accuracy"]
    margins.append(margin_points(fp32["accuracy"], meta["n_validation"]))
    lines.append(
        f"| {label} | {fp32['accuracy']} | {int8['accuracy']} | {change:+.2f} "
        f"| {int8['size_mb']} |"
    )

n = runs[0][1]["n_validation"]
note = (
    f"\nAccuracy is measured on {n} validation sentences. The 95% margin of error is about "
    f"+-{max(margins):.1f} points, so differences smaller than that are within the noise "
    "of this validation set. Latency is not compared here: it does not depend on the training "
    "data and it varies with the load of the machine.\n"
)
text = "\n".join(lines) + "\n" + note
out = os.path.join(args.dir_b, "comparison.md")
open(out, "w", encoding="utf-8").write(text)
print(text)
print("saved to", out)
