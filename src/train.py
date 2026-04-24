import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# Load Data
# -----------------------------
train = pd.read_excel("data/train_fixed.xlsx")
val = pd.read_excel("data/validation_fixed.xlsx")

# -----------------------------
# Preprocessing
# -----------------------------
from src.preprocessing import preprocess_text

train["clean_text"] = train["review_text"].apply(preprocess_text)
val["clean_text"] = val["review_text"].apply(preprocess_text)

# -----------------------------
# Parse Labels (string → actual)
# -----------------------------
train["aspects"] = train["aspects"].apply(ast.literal_eval)
val["aspects"] = val["aspects"].apply(ast.literal_eval)

train["aspect_sentiments"] = train["aspect_sentiments"].apply(ast.literal_eval)
val["aspect_sentiments"] = val["aspect_sentiments"].apply(ast.literal_eval)

# -----------------------------
# Define Aspect Classes
# -----------------------------
all_aspects = [
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience",
    "general", "none"
]

# -----------------------------
# Multi-label Encoding (Aspects)
# -----------------------------
mlb = MultiLabelBinarizer(classes=all_aspects)

y_aspects_train = mlb.fit_transform(train["aspects"])
y_aspects_val = mlb.transform(val["aspects"])

# -----------------------------
# Sentiment Encoding
# -----------------------------
sentiment_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def encode_sentiments(sent_dict):
    return {k: sentiment_map[v] for k, v in sent_dict.items()}

train["encoded_sentiments"] = train["aspect_sentiments"].apply(encode_sentiments)
val["encoded_sentiments"] = val["aspect_sentiments"].apply(encode_sentiments)

# -----------------------------
# Debug / Check Output
# -----------------------------
print("===== CLEAN TEXT SAMPLE =====")
print(train[["review_text", "clean_text"]].head(), "\n")

print("===== ASPECT ENCODING SHAPE =====")
print(y_aspects_train.shape, "\n")

print("===== SAMPLE ENCODED SENTIMENT =====")
print(train["encoded_sentiments"].iloc[0])