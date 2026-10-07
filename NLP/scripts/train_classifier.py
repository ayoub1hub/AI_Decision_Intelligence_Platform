"""Train a text classifier on the labeled arXiv dataset.

Usage:
    python scripts/train_classifier.py \
        --data data/raw/arxiv_labeled.csv \
        --model logreg \
        --out models/arxiv_clf.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_engine.logging_config import setup_logging
from nlp_engine.modeling import TextClassifier
from nlp_engine.preprocessing import preprocess_batch
from nlp_engine.vectorization import TfidfModel

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/arxiv_labeled.csv")
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm", "rf"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", default="models/arxiv_clf.pkl")
    parser.add_argument("--vec-out", default="models/tfidf.pkl")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # 1. Load
    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows, %d classes", len(df), df["label"].nunique())

    # 2. Build text = title + abstract
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    # 3. Preprocess
    logger.info("Preprocessing %d texts", len(texts))
    processed = preprocess_batch(texts)

    # 4. Vectorize (TF-IDF)
    logger.info("Vectorizing with TF-IDF")
    vectorizer = TfidfModel(max_features=5000, min_df=2, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed)

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["label"].tolist(),
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )
    logger.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])

    # 6. Train
    logger.info("Training model: %s", args.model)
    clf = TextClassifier(model_type=args.model)
    clf.fit(X_train, y_train)

    # 7. Evaluate
    metrics = clf.evaluate(X_test, y_test)
    print("\n" + metrics.summary())
    print("\n" + metrics.report)

    # 8. Save
    clf.save(args.out)
    vectorizer.save(args.vec_out)
    print(f"\n✅ Model saved: {args.out}")
    print(f"✅ Vectorizer saved: {args.vec_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
