import re
from typing import List


def basic_text_cleaning(text: str) -> str:
    """
    Minimal and safe text cleaning.
    Used as a shared base for all models.
    """

    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)   # HTML line breaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_tfidf(text: str) -> str:
    """
    Preprocessing optimized for TF-IDF models.
    Slight normalization is applied.
    """

    text = basic_text_cleaning(text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    return text


def preprocess_for_transformer(text: str) -> str:
    """
    Preprocessing optimized for transformer models.
    Minimal intervention to preserve context.
    """

    text = basic_text_cleaning(text)

    return text


def batch_preprocess(
    texts: List[str],
    mode: str = "tfidf"
) -> List[str]:
    """
    Batch preprocessing wrapper.
    """

    if mode == "tfidf":
        return [preprocess_for_tfidf(t) for t in texts]

    elif mode == "transformer":
        return [preprocess_for_transformer(t) for t in texts]

    else:
        raise ValueError("mode must be 'tfidf' or 'transformer'")
