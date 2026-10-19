"""Train a text classifier using BERT embeddings (vs TF-IDF baseline).

Usage:
    python scripts/train_classifier_bert.py \
        --data data/raw/arxiv_labeled.csv \
        --model svm \
        --out models/arxiv_clf_bert_svm.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_engine.logging_config import setup_logging
from nlp_engine.modeling import TextClassifier
from nlp_engine.vectorization import EmbeddingModel

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/arxiv_labeled.csv")
    parser.add_argument("--model", default="svm", choices=["logreg", "svm", "rf"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", default="models/arxiv_clf_bert_svm.pkl")
    parser.add_argument(
        "--emb-cache",
        default="data/processed/arxiv_embeddings.npz",
        help="Where to cache computed embeddings",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    # 1. Load
    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows, %d classes", len(df), df["label"].nunique())

    # 2. Build text = title + abstract
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    # 3. Embeddings (with caching)
    cache_path = Path(args.emb_cache)
    if cache_path.exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        X, _ = EmbeddingModel.load_embeddings(cache_path)
    else:
        logger.info("Computing BERT embeddings (first time only)")
        embedder = EmbeddingModel()
        X = embedder.encode(texts, show_progress=True)
        embedder.save_embeddings(X, texts, cache_path)
        logger.info("Embeddings saved to %s", cache_path)

    logger.info("Embeddings shape: %s", X.shape)

    # 4. Split (same random_state as TF-IDF for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["label"].tolist(),
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )
    logger.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])

    # 5. Train
    logger.info("Training model: %s", args.model)
    clf = TextClassifier(model_type=args.model)
    clf.fit(X_train, y_train)

    # 6. Evaluate
    metrics = clf.evaluate(X_test, y_test)
    print("\n" + metrics.summary())
    print("\n" + metrics.report)

    # 7. Save
    clf.save(args.out)
    print(f"\n✅ Model saved: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
