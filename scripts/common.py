"""Shared data loading. Uses SST-2 / IMDB from Hugging Face by default.

For offline smoke tests set SERVEML_OFFLINE_DATA to a JSONL file with
{"sentence": ..., "label": 0|1} per line (used for both normal and long text).
"""
import json
import os


def _offline():
    path = os.environ.get("SERVEML_OFFLINE_DATA")
    if not path:
        return None
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_val():
    """Return (texts, labels) of the SST-2 validation set."""
    rows = _offline()
    if rows is not None:
        rows = [r for r in rows if len(r["sentence"].split()) < 60]
        return [r["sentence"] for r in rows], [r["label"] for r in rows]
    from datasets import load_dataset

    val = load_dataset("glue", "sst2")["validation"]
    return list(val["sentence"]), list(val["label"])


def load_long_texts(n=150):
    """Long movie reviews used to simulate drift."""
    rows = _offline()
    if rows is not None:
        return [r["sentence"] for r in rows if len(r["sentence"].split()) >= 60][:n]
    from datasets import load_dataset

    imdb = load_dataset("imdb", split="test").shuffle(seed=1).select(range(n))
    return [t[:1500] for t in imdb["text"]]


def load_train_val(train_size):
    """Return (train, validation) Hugging Face datasets with 'sentence' and 'label'."""
    rows = _offline()
    if rows is not None:
        from datasets import Dataset

        short = [r for r in rows if len(r["sentence"].split()) < 60]
        cut = int(len(short) * 0.8)
        return (Dataset.from_list(short[:cut]).select(range(min(train_size, cut))),
                Dataset.from_list(short[cut:]))
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2")
    return ds["train"].shuffle(seed=42).select(range(train_size)), ds["validation"]
