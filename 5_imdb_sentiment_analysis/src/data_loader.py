from datasets import load_dataset
from typing import Tuple, List


def load_imdb_dataset() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Loads the IMDb sentiment analysis dataset from Hugging Face.

    Returns
    -------
    X_train : List[str]
        Training text samples (raw movie reviews).
    y_train : List[int]
        Training labels (0 = negative, 1 = positive).
    X_test : List[str]
        Test text samples (raw movie reviews).
    y_test : List[int]
        Test labels (0 = negative, 1 = positive).
    """

    dataset = load_dataset("imdb")

    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]

    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_imdb_dataset()

    print("IMDb dataset loaded successfully.")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\nSample review:")
    print(X_train[0][:500])
    print(f"Label: {y_train[0]}")
