import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_DIR = "models/transformer/imdb_sentiment_model"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def evaluate_transformer_model():
    """
    Evaluates the fine-tuned transformer model on the IMDb test set.
    """

    # 1) Load dataset
    dataset = load_dataset("imdb")

    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(5000))

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized_test = dataset["test"].map(tokenize, batched=True)
    tokenized_test = tokenized_test.remove_columns(["text"])
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format("torch")

    # 3) Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # 4) Trainer (evaluation only)
    args = TrainingArguments(
        output_dir="models/transformer/eval_tmp",
        per_device_eval_batch_size=32,
        eval_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 5) Predict
    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # 6) Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Transformer Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate_transformer_model()
