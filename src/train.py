from src.preprocessing import preprocess_text
import pandas as pd

df = pd.read_excel("data/train_fixed.xlsx")

df["clean_text"] = df["review_text"].apply(preprocess_text)

print(df[["review_text", "clean_text"]].head())