import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_imdb_dataset
from src.preprocessing import batch_preprocess


def train_baseline_model(
    max_features: int = 20000,
    ngram_range: tuple = (1, 2),
    model_path: str = "models/baseline"
):
    """
    Trains a TF-IDF + Logistic Regression baseline model
    and saves the vectorizer and classifier.
    """

    # 1) Load data
    X_train, y_train, _, _ = load_imdb_dataset()

    # 2) Preprocess for TF-IDF
    X_train_clean = batch_preprocess(X_train, mode="tfidf")

    # 3) Vectorize
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train_clean)

    # 4) Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    clf.fit(X_train_vec, y_train)

    # 5) Save artifacts
    joblib.dump(vectorizer, f"{model_path}/vectorizer.pkl")
    joblib.dump(clf, f"{model_path}/classifier.pkl")

    print("Baseline model trained and saved successfully.")


if __name__ == "__main__":
    train_baseline_model()