import pandas as pd
from src.preprocessing import preprocess_text

# -----------------------------
# Load datasets
# -----------------------------
train_df = pd.read_excel("data/train_fixed.xlsx")
val_df = pd.read_excel("data/validation_fixed.xlsx")
unlabeled_df = pd.read_excel("data/unlabeled_fixed.xlsx")

# -----------------------------
# Apply preprocessing
# -----------------------------
train_df["clean_text"] = train_df["review_text"].apply(preprocess_text)
val_df["clean_text"] = val_df["review_text"].apply(preprocess_text)
unlabeled_df["clean_text"] = unlabeled_df["review_text"].apply(preprocess_text)

# -----------------------------
# Save cleaned datasets
# -----------------------------
train_df.to_excel("data/train_cleaned.xlsx", index=False)
val_df.to_excel("data/validation_cleaned.xlsx", index=False)
unlabeled_df.to_excel("data/unlabeled_cleaned.xlsx", index=False)

print("✅ Preprocessing completed and saved!")