from pathlib import Path
from typing import List


def load_sentences(file_path: str) -> List[str]:
    """
    Reads a text file where each line is a sentence
    and returns a list of cleaned sentences.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    sentences = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()

            if sentence:
                sentences.append(sentence)

    return sentences
