"""Compare TF-IDF vs BERT on the arXiv classification task.

Assumes both models have already been trained:
    - models/arxiv_clf_svm.pkl        (TF-IDF + SVM)
    - models/arxiv_clf_bert_svm.pkl   (BERT  + SVM)
    - models/tfidf.pkl                (TF-IDF vectorizer)

Usage:
    python scripts/compare_models.py
"""

from __future__ import annotations

import logging
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_engine.logging_config import setup_logging
from nlp_engine.modeling import TextClassifier
from nlp_engine.preprocessing import preprocess_batch
from nlp_engine.vectorization import EmbeddingModel, TfidfModel

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging(level="INFO")

    # 1. Load data + reproduce the same split
    df = pd.read_csv("data/raw/arxiv_labeled.csv")
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    labels = df["label"].tolist()

    _, X_test_text, _, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 2. Load models
    logger.info("Loading TF-IDF model + classifier")
    tfidf = TfidfModel.load("models/tfidf.pkl")
    clf_tfidf = TextClassifier.load("models/arxiv_clf_svm.pkl")

    logger.info("Loading BERT embeddings + classifier")
    embedder = EmbeddingModel()
    clf_bert = TextClassifier.load("models/arxiv_clf_bert_svm.pkl")

    # 3. Evaluate TF-IDF
    X_test_tfidf = tfidf.transform(preprocess_batch(X_test_text))
    metrics_tfidf = clf_tfidf.evaluate(X_test_tfidf, y_test)

    # 4. Evaluate BERT
    X_test_bert = embedder.encode(X_test_text)
    metrics_bert = clf_bert.evaluate(X_test_bert, y_test)

    # 5. Print comparison
    print("\n" + "=" * 60)
    print("TF-IDF vs BERT — Classification comparison")
    print("=" * 60)
    print(f"{'Metric':<15} {'TF-IDF':>10} {'BERT':>10} {'Δ':>10}")
    print("-" * 60)
    print(
        f"{'Accuracy':<15} {metrics_tfidf.accuracy:>10.4f} "
        f"{metrics_bert.accuracy:>10.4f} "
        f"{(metrics_bert.accuracy - metrics_tfidf.accuracy):>+10.4f}"
    )
    print(
        f"{'F1 (macro)':<15} {metrics_tfidf.f1_macro:>10.4f} "
        f"{metrics_bert.f1_macro:>10.4f} "
        f"{(metrics_bert.f1_macro - metrics_tfidf.f1_macro):>+10.4f}"
    )
    print(
        f"{'F1 (weighted)':<15} {metrics_tfidf.f1_weighted:>10.4f} "
        f"{metrics_bert.f1_weighted:>10.4f} "
        f"{(metrics_bert.f1_weighted - metrics_tfidf.f1_weighted):>+10.4f}"
    )
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
