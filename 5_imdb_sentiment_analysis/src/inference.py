import joblib
import torch
from typing import Literal

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import batch_preprocess


BASELINE_MODEL_PATH = "models/baseline"
TRANSFORMER_MODEL_PATH = "models/transformer/imdb_sentiment_model"


def predict(
    text: str,
    model_type: Literal["baseline", "transformer"] = "baseline"
) -> dict:
    """
    Runs sentiment prediction on a single text input.

    Parameters
    ----------
    text : str
        Input text (movie review).
    model_type : {"baseline", "transformer"}
        Which model to use.

    Returns
    -------
    dict
        {
            "model": model_type,
            "label": int,
            "sentiment": str
        }
    """

    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    if model_type == "baseline":
        return _predict_baseline(text)

    elif model_type == "transformer":
        return _predict_transformer(text)

    else:
        raise ValueError("model_type must be 'baseline' or 'transformer'")


def _predict_baseline(text: str) -> dict:
    vectorizer = joblib.load(f"{BASELINE_MODEL_PATH}/vectorizer.pkl")
    clf = joblib.load(f"{BASELINE_MODEL_PATH}/classifier.pkl")

    clean_text = batch_preprocess([text], mode="tfidf")
    vec = vectorizer.transform(clean_text)

    label = int(clf.predict(vec)[0])

    return {
        "model": "baseline",
        "label": label,
        "sentiment": "positive" if label == 1 else "negative"
    }


def _predict_transformer(text: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_PATH
    )

    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="torch"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    label = int(torch.argmax(outputs.logits, dim=1).item())

    return {
        "model": "transformer",
        "label": label,
        "sentiment": "positive" if label == 1 else "negative"
    }


if __name__ == "__main__":
    sample_text = "The movie had great visuals but the story was disappointing."

    print(predict(sample_text, model_type="baseline"))
    print(predict(sample_text, model_type="transformer"))