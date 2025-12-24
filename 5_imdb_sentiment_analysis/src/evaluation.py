import numpy as np
import joblib
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import batch_preprocess


BASELINE_MODEL_PATH = "models/baseline"
TRANSFORMER_MODEL_PATH = "models/transformer/imdb_sentiment_model"


def evaluate_baseline():
    """
    Evaluates baseline model on a fixed test subset
    for fair comparison with transformer.
    """

    dataset = load_dataset("imdb")
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(5000))

    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    vectorizer = joblib.load(f"{BASELINE_MODEL_PATH}/vectorizer.pkl")
    clf = joblib.load(f"{BASELINE_MODEL_PATH}/classifier.pkl")

    X_test_clean = batch_preprocess(X_test, mode="tfidf")
    X_test_vec = vectorizer.transform(X_test_clean)

    y_pred = clf.predict(X_test_vec)

    return y_test, y_pred


def evaluate_transformer():
    """
    Loads transformer predictions using the inference logic.
    """

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    dataset = load_dataset("imdb")
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(5000))

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_PATH
    )

    model.eval()

    y_true = []
    y_pred = []

    for sample in dataset["test"]:
        inputs = tokenizer(
            sample["text"],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

        y_true.append(sample["label"])
        y_pred.append(pred)

    return y_true, y_pred


def compare_models():
    """
    Compares baseline and transformer models side by side.
    """

    y_true_b, y_pred_b = evaluate_baseline()
    y_true_t, y_pred_t = evaluate_transformer()

    print("\n=== BASELINE MODEL ===")
    print(f"Accuracy: {accuracy_score(y_true_b, y_pred_b):.4f}")
    print(classification_report(y_true_b, y_pred_b, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_b, y_pred_b))

    print("\n=== TRANSFORMER MODEL ===")
    print(f"Accuracy: {accuracy_score(y_true_t, y_pred_t):.4f}")
    print(classification_report(y_true_t, y_pred_t, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_t, y_pred_t))


if __name__ == "__main__":
    compare_models()