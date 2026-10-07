"""Text classification models.

Wraps scikit-learn classifiers with a common API:

    clf = TextClassifier(model_type="logreg")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = clf.evaluate(X_test, y_test)
    clf.save("models/clf.pkl")

Supports:
- LogisticRegression (baseline, rapide)
- LinearSVC (SVM linéaire, souvent meilleur)
- RandomForest (non-linéaire, plus lent)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC

__all__ = ["ClassificationMetrics", "TextClassifier"]

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Metrics for a trained classifier."""

    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.array([]))
    report: str = ""
    labels: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Classification Metrics",
            "─" * 40,
            f"Accuracy    : {self.accuracy:.4f}",
            f"F1 (macro)  : {self.f1_macro:.4f}",
            f"F1 (weighted): {self.f1_weighted:.4f}",
            f"N classes   : {len(self.labels)}",
        ]
        return "\n".join(lines)


class TextClassifier:
    """Wrapper around scikit-learn classifiers for text.

    Parameters
    ----------
    model_type : {"logreg", "svm", "rf"}, default "logreg"
        Which classifier to use.
    **model_kwargs
        Passed to the underlying sklearn estimator.

    Examples
    --------
    >>> clf = TextClassifier(model_type="logreg")
    >>> # clf.fit(X_train, y_train)  # doctest: +SKIP
    """

    _MODELS: ClassVar[dict[str, type]] = {
        "logreg": LogisticRegression,
        "svm": LinearSVC,
        "rf": RandomForestClassifier,
    }

    def __init__(self, *, model_type: str = "logreg", **model_kwargs) -> None:
        if model_type not in self._MODELS:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Choose from {list(self._MODELS.keys())}."
            )
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self._model = self._build_model()
        self._classes: list[str] = []

    def _build_model(self):
        """Build the sklearn estimator with sensible defaults."""
        cls = self._MODELS[self.model_type]

        if self.model_type == "logreg":
            defaults = {"max_iter": 1000, "C": 1.0, "class_weight": "balanced"}
        elif self.model_type == "svm":
            defaults = {"max_iter": 5000, "C": 1.0, "class_weight": "balanced"}
        elif self.model_type == "rf":
            defaults = {
                "n_estimators": 100,
                "n_jobs": -1,
                "random_state": 42,
                "class_weight": "balanced",
            }
        else:
            defaults = {}

        defaults.update(self.model_kwargs)
        return cls(**defaults)

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------
    def fit(self, X: spmatrix, y: list[str]) -> TextClassifier:
        """Train the classifier."""
        logger.info("Fitting %s on %d samples", self.model_type, X.shape[0])
        self._model.fit(X, y)
        self._classes = list(self._model.classes_)
        logger.info("Trained on %d classes: %s", len(self._classes), self._classes)
        return self

    def predict(self, X: spmatrix) -> np.ndarray:
        """Predict labels."""
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X: spmatrix, y_true: list[str]) -> ClassificationMetrics:
        """Compute accuracy, F1, confusion matrix."""
        y_pred = self.predict(X)

        metrics = ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            confusion=confusion_matrix(y_true, y_pred, labels=self._classes),
            report=classification_report(y_true, y_pred, zero_division=0),
            labels=self._classes,
        )

        logger.info("Accuracy: %.4f | F1 macro: %.4f", metrics.accuracy, metrics.f1_macro)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> TextClassifier:
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected TextClassifier, got {type(model).__name__}")
        return model

    def __repr__(self) -> str:
        fitted = bool(self._classes)
        return (
            f"TextClassifier(model_type={self.model_type!r}, "
            f"fitted={fitted}, n_classes={len(self._classes)})"
        )
