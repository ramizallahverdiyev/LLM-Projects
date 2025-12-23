import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import load_imdb_dataset
from src.preprocessing import batch_preprocess


def evaluate_baseline_model(
    model_path: str = "models/baseline"
):
    """
    Evaluates the baseline TF-IDF + Logistic Regression model.
    """

    # 1) Load data
    _, _, X_test, y_test = load_imdb_dataset()

    # 2) Load artifacts
    vectorizer = joblib.load(f"{model_path}/vectorizer.pkl")
    clf = joblib.load(f"{model_path}/classifier.pkl")

    # 3) Preprocess
    X_test_clean = batch_preprocess(X_test, mode="tfidf")

    # 4) Vectorize
    X_test_vec = vectorizer.transform(X_test_clean)

    # 5) Predict
    y_pred = clf.predict(X_test_vec)

    # 6) Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_baseline_model()
