# IMDb Sentiment Analysis: Baseline vs Transformer

This project demonstrates **sentiment classification** of movie reviews using both a **baseline TF-IDF + Logistic Regression model** and a **Transformer-based model** (`DistilBERT`). The goal is to compare classical ML techniques with modern deep learning approaches for NLP tasks.

---

## Project Motivation

Sentiment analysis is a core NLP task. Understanding differences between traditional ML and Transformer models helps in:

* Evaluating performance vs complexity trade-offs
* Designing pipelines for real-world text data
* Learning about preprocessing, feature extraction, and deep learning embeddings

This project provides a **hands-on, comparative analysis** using the IMDb movie reviews dataset.

---

## Key Concepts Covered

* Text preprocessing and tokenization
* TF-IDF vectorization
* Logistic Regression baseline
* Transformer models for sequence classification
* Evaluation metrics: accuracy, precision, recall, F1-score
* Confusion matrix analysis
* Trade-offs between classical ML and Transformer models

---

## Project Structure

```
5_imdb_sentiment_analysis/
│
├── data_loader.py         # Loads and processes IMDb dataset
├── preprocessing.py       # Text cleaning and batch preprocessing
├── src/
│   ├── baseline/
│   │   ├── train_baseline.py
│   │   └── evaluate_baseline.py
│   ├── transformer/
│   │   ├── train_transformer.py
│   │   └── evaluate_transformer.py
│   └── evaluation.py      # Side-by-side comparison of both models
├── models/
│   ├── baseline/
│   └── transformer/imdb_sentiment_model/
├── main.py                # Entry point for running the pipeline
└── README.md
```

All modules are designed with **single responsibility** principles, mirroring real ML engineering practices.

---

## Dataset

* Source: `datasets` library, IMDb movie reviews
* 50,000 samples: 25,000 train, 25,000 test
* Balanced classes (positive/negative sentiment)
* Test subset (5,000 reviews) used for evaluation

Each review is a single text unit with a corresponding label (0 = negative, 1 = positive).

---

## Baseline Model

* **Vectorization:** TF-IDF
* **Classifier:** Logistic Regression
* **Evaluation:** Accuracy, precision, recall, F1-score
* **Confusion Matrix:** Visualizes misclassifications

Results on 5,000 test samples:

```
Accuracy: 0.8928
Confusion Matrix: [[2206  288]
                   [ 248 2258]]
```

---

## Transformer Model

* **Model:** `DistilBERT` for sequence classification
* **Tokenizer:** HuggingFace `AutoTokenizer`
* **Training:** Fine-tuned on IMDb dataset
* **Evaluation:** Accuracy, precision, recall, F1-score

Results on 5,000 test samples:

```
Accuracy: 0.8762
Confusion Matrix: [[2155  339]
                   [ 280 2226]]
```

*Training Transformer models can take significantly longer than baseline models, especially on CPU.*

---

## Running the Project

### Baseline

```bash
python -m src.baseline.train_baseline
python -m src.baseline.evaluate_baseline
```

### Transformer

```bash
python -m src.transformer.train_transformer
python -m src.transformer.evaluate_transformer
```

### Evaluation & Comparison

```bash
python -m src.evaluation
```

This will generate metrics for both models and compare them side by side.

---

## Observations

* Baseline model is **fast and interpretable** with good accuracy.
* Transformer model captures **contextual semantics** but may not outperform baseline on small datasets.
* Evaluation time differs greatly: Transformer evaluation can be slow without batching or GPU acceleration.
* Preprocessing and tokenization are critical for both approaches.

---

## Summary

This project provides a hands-on comparison of **classical and modern NLP methods** for sentiment analysis, emphasizing **engineering best practices, model evaluation, and performance trade-offs**.