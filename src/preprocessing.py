"""
preprocessing.py — Person 2, Part 1
Arabic text cleaning pipeline for ABSA system.
Handles: normalization, noise removal, mixed-language text.

Run:
    python preprocessing.py

Outputs:
    train_clean.csv
    val_clean.csv
    unlabeled_clean.csv
"""

import re
import pandas as pd
import pyarabic.araby as araby

# ── Arabic Normalization ──────────────────────────────────────────────────────

def normalize_arabic(text):
    """
    Standardize Arabic character variations:
    - أ إ آ ٱ  →  ا
    - ة        →  ه
    - ى        →  ي
    - ؤ        →  و
    - ئ        →  ي
    """
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    return text


def remove_diacritics(text):
    """
    Remove Arabic tashkeel (harakat) — short vowel marks.
    e.g. مُحَمَّد → محمد
    Uses pyarabic library.
    """
    return araby.strip_tashkeel(text)


def remove_tatweel(text):
    """
    Remove Arabic tatweel (elongation character ـ).
    e.g. جمييييل → جميل
    """
    return araby.strip_tatweel(text)


# ── Noise Removal ─────────────────────────────────────────────────────────────

def remove_emojis(text):
    """Remove all emoji and special unicode symbols."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_punctuation(text):
    """
    Remove Arabic and English punctuation.
    Keeps Arabic letters, English letters/digits, and spaces.
    """
    # Remove Arabic punctuation
    text = re.sub(r'[،؛؟٪٫٬«»]', ' ', text)
    # Remove English punctuation
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    # Remove dots/ellipsis
    text = re.sub(r'\.{2,}', ' ', text)
    return text


def normalize_whitespace(text):
    """Collapse multiple spaces and strip edges."""
    return re.sub(r'\s+', ' ', text).strip()


def handle_mixed_language(text):
    """
    Keep both Arabic and English words — the model handles bilingual text.
    Only remove non-linguistic noise (numbers, symbols).
    Numbers are removed as they rarely carry sentiment meaning.
    """
    # Remove standalone numbers (keep if attached to words)
    text = re.sub(r'\b\d+\b', '', text)
    return text


def remove_repeated_chars(text):
    """
    Reduce elongated words: مرررررحبا → مرحبا
    Works for both Arabic and English.
    """
    # Reduce any character repeated 3+ times to 2
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def clean_text(text):
    """
    Apply full cleaning pipeline in order.
    Returns cleaned string.
    """
    if not isinstance(text, str) or text.strip() == '':
        return ''

    text = remove_emojis(text)
    text = remove_diacritics(text)
    text = remove_tatweel(text)
    text = normalize_arabic(text)
    text = remove_punctuation(text)
    text = handle_mixed_language(text)
    text = remove_repeated_chars(text)
    text = normalize_whitespace(text)

    return text


# ── Process Dataset ───────────────────────────────────────────────────────────

def process_file(input_path, output_path, label=None):
    """
    Load an Excel file, clean review_text, save as CSV.

    Args:
        input_path  : path to .xlsx file
        output_path : where to save cleaned .csv
        label       : label to print (e.g. 'TRAIN')
    """
    df = pd.read_excel(input_path)
    label = label or input_path

    print(f"\n{'─'*50}")
    print(f"Processing: {label}  ({df.shape[0]} rows)")

    # Keep original for reference
    df['review_text_original'] = df['review_text']

    # Apply cleaning
    df['review_text'] = df['review_text'].apply(clean_text)

    # Flag empty reviews after cleaning
    empty_after = (df['review_text'].str.strip() == '').sum()
    if empty_after > 0:
        print(f"  ⚠️  {empty_after} reviews became empty after cleaning → filling with 'غير محدد'")
        df['review_text'] = df['review_text'].replace('', 'غير محدد')

    # Stats
    avg_len_before = df['review_text_original'].str.len().mean()
    avg_len_after  = df['review_text'].str.len().mean()
    print(f"  Avg length before: {avg_len_before:.1f} chars")
    print(f"  Avg length after:  {avg_len_after:.1f} chars")
    print(f"  Sample before: {df['review_text_original'].iloc[1][:80]}")
    print(f"  Sample after:  {df['review_text'].iloc[1][:80]}")

    # Save
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Saved → {output_path}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df     = process_file('DeepX_train.xlsx',      'train_clean.csv',     'TRAIN')
    val_df       = process_file('DeepX_validation.xlsx', 'val_clean.csv',       'VALIDATION')
    unlabeled_df = process_file('DeepX_unlabeled.xlsx',  'unlabeled_clean.csv', 'UNLABELED')

    print("\n" + "="*50)
    print("✅ Preprocessing complete. Files saved:")
    print("   train_clean.csv")
    print("   val_clean.csv")
    print("   unlabeled_clean.csv")
    print("\n📌 Note for teammates:")
    print("   - Load cleaned files using: pd.read_csv('train_clean.csv')")
    print("   - Use 'review_text' column for model input (cleaned)")
    print("   - 'review_text_original' kept for debugging")