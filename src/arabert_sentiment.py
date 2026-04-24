import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import ast
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score

from src.preprocessing import preprocess_text

MODEL_NAME = "aubmindlab/bert-base-arabertv2"

sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}

# -----------------------------
# Load
# -----------------------------
train = pd.read_excel("data/train_fixed.xlsx")
val = pd.read_excel("data/validation_fixed.xlsx")

# -----------------------------
# Preprocess
# -----------------------------
for df in [train, val]:
    df["clean_text"] = df["review_text"].apply(preprocess_text).str.lower()

# -----------------------------
# Parse
# -----------------------------
train["aspect_sentiments"] = train["aspect_sentiments"].apply(ast.literal_eval)
val["aspect_sentiments"] = val["aspect_sentiments"].apply(ast.literal_eval)

# -----------------------------
# Expand data
# -----------------------------
def expand(df):
    rows = []
    for _, row in df.iterrows():
        for aspect, sent in row["aspect_sentiments"].items():
            rows.append({
                "text": row["clean_text"] + " [SEP] " + aspect,
                "label": sentiment_map[sent]
            })
    return pd.DataFrame(rows)

train_exp = expand(train)
val_exp = expand(val)

# -----------------------------
# HF Dataset
# -----------------------------
train_ds = Dataset.from_pandas(train_exp)
val_ds = Dataset.from_pandas(val_exp)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------
# Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "f1_micro": f1_score(labels, preds, average="micro")
    }

# -----------------------------
# Training
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/arabert_sentiment",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate()
print("🔥 Sentiment F1:", metrics)

trainer.save_model("models/arabert_sentiment")
tokenizer.save_pretrained("models/arabert_sentiment")