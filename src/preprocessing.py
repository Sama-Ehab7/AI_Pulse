import re

# -----------------------------
# Remove emojis
# -----------------------------
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff]", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', str(text))


# -----------------------------
# Remove punctuation
# -----------------------------
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', str(text))


# -----------------------------
# Normalize Arabic text
# -----------------------------
def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"گ", "ك", text)
    return text


# -----------------------------
# Remove extra spaces
# -----------------------------
def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


# -----------------------------
# Handle mixed Arabic/English text
# -----------------------------
def clean_mixed_text(text):
    # Keep English words (important for meaning)
    return text


# -----------------------------
# Main preprocessing function
# -----------------------------
def preprocess_text(text):
    text = str(text)

    text = remove_emojis(text)
    text = remove_punctuation(text)
    text = normalize_arabic(text)
    text = clean_mixed_text(text)
    text = remove_extra_spaces(text)

    return text