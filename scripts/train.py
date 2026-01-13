"""Fine-tune DistilBERT on SST-2 (sentiment) on CPU with a training subset."""
import json
import os
import sys

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(__file__))
from common import load_train_val  # noqa: E402

MODEL = os.environ.get("BASE_MODEL", "distilbert-base-uncased")
TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", "8000"))  # subset so it trains on CPU
EPOCHS = int(os.environ.get("EPOCHS", "2"))

train, val = load_train_val(TRAIN_SIZE)

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(batch):
    return tok(batch["sentence"], truncation=True, max_length=128)


train = train.map(encode, batched=True)
val = val.map(encode, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": float((preds == p.label_ids).mean())}


args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
)
trainer.train()
result = trainer.evaluate()
print(result)

os.makedirs("results", exist_ok=True)
model.save_pretrained("models/fp32")
tok.save_pretrained("models/fp32")
json.dump(result, open("results/train_eval.json", "w"), indent=2)
