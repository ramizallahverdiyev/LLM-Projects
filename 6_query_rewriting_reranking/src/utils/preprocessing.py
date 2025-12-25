import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning: remove extra spaces, special characters, lowercasing.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;!?çğıöşüəəİ ]", "", text)
    return text.strip()

def tokenize(text: str) -> list:
    """
    Simple whitespace tokenizer.
    """
    return text.split()
