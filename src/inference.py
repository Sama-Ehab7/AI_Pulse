from src.preprocessing import preprocess_text
import pandas as pd

df = pd.read_excel("data/unlabeled_fixed.xlsx")

df["clean_text"] = df["review_text"].apply(preprocess_text)