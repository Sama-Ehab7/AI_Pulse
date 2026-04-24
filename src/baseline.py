import pandas as pd
import ast
import json
import joblib
import numpy as np



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.preprocessing import preprocess_text

# -----------------------------
# Config
# -----------------------------
ALL_ASPECTS = [
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience",
    "general" , "none",
]

# -----------------------------
# Load data
# -----------------------------
train = pd.read_excel("data/train_fixed.xlsx")
val = pd.read_excel("data/validation_fixed.xlsx")
test = pd.read_excel("data/unlabeled_fixed.xlsx")

# -----------------------------
# Preprocess
# -----------------------------
for df in [train, val, test]:
    df["clean_text"] = df["review_text"].apply(preprocess_text)
train["clean_text"] = train["clean_text"].str.lower()
# -----------------------------
# Parse labels
# -----------------------------
train["aspects"] = train["aspects"].apply(ast.literal_eval)
val["aspects"] = val["aspects"].apply(ast.literal_eval)

train["aspect_sentiments"] = train["aspect_sentiments"].apply(ast.literal_eval)
val["aspect_sentiments"] = val["aspect_sentiments"].apply(ast.literal_eval)

# -----------------------------
# Multi-label encoding
# -----------------------------
mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
y_train = mlb.fit_transform(train["aspects"])
y_val = mlb.transform(val["aspects"])

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
    
)
X_train = vectorizer.fit_transform(train["clean_text"])
X_val = vectorizer.transform(val["clean_text"])
X_test = vectorizer.transform(test["clean_text"])

# -----------------------------
# Model (One-vs-Rest Logistic Regression)
# -----------------------------
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced")
)

model.fit(X_train, y_train)
y_probs = model.predict_proba(X_val)

y_pred = (y_probs > 0.3).astype(int)
# -----------------------------
# Evaluate (F1)
# -----------------------------
from sklearn.metrics import f1_score

y_val_pred = model.predict(X_val)
f1 = f1_score(y_val, y_val_pred, average="micro")

print("Validation F1 (Micro):", f1)

# -----------------------------
# Predict on test
# -----------------------------
y_test_pred = model.predict(X_test)

# -----------------------------
# Helper: get aspects list
# -----------------------------
def decode_aspects(row):
    return [ALL_ASPECTS[i] for i, v in enumerate(row) if v == 1]

# -----------------------------
# Simple sentiment fallback
# -----------------------------
def assign_sentiment(text, aspect):
    """
    Very simple heuristic:
    you can improve later
    """
    if "سي" in text or "وحش" in text or "بطئ" in text:
        return "negative"
    elif "حلو" in text or "جميل" in text or "ممتاز" in text:
        return "positive"
    else:
        return "neutral"

# -----------------------------
# Build JSON output
# -----------------------------
results = []

for i, row in test.iterrows():
    pred_aspects = decode_aspects(y_test_pred[i])

    # 🔥 Fallback
    if len(pred_aspects) == 0:
        pred_aspects = ["general"]

    aspect_sentiments = {}

    for aspect in pred_aspects:
        sentiment = assign_sentiment(row["clean_text"], aspect)
        aspect_sentiments[aspect] = sentiment

    results.append({
        "review_id": int(row["review_id"]),
        "aspects": pred_aspects,
        "aspect_sentiments": aspect_sentiments
    })

# -----------------------------
# Save output
# -----------------------------



import os
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
with open("outputs/predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("✅ predictions.json generated!")
# -----------------------------
# Save model (optional)
# -----------------------------
joblib.dump(model, "models/baseline_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")    