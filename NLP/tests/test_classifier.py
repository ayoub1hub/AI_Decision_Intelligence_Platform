"""Tests for text classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nlp_engine.modeling import TextClassifier

# Small synthetic dataset: 3 classes, 12 samples, 5 features
# Dataset parfaitement séparable : chaque classe a sa feature unique
_X = csr_matrix(
    np.array(
        [
            [1, 0, 0, 0, 0],  # Classe A
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],  # Classe B
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],  # Classe C
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
)
_Y = ["A"] * 4 + ["B"] * 4 + ["C"] * 4


class TestTextClassifierFit:
    @pytest.mark.parametrize("model_type", ["logreg", "svm", "rf"])
    def test_fit_returns_self(self, model_type: str) -> None:
        clf = TextClassifier(model_type=model_type)
        assert clf.fit(_X, _Y) is clf

    def test_unknown_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            TextClassifier(model_type="foo")

    def test_predict_after_fit(self) -> None:
        clf = TextClassifier(model_type="logreg")
        clf.fit(_X, _Y)
        preds = clf.predict(_X)
        assert len(preds) == len(_Y)


class TestTextClassifierEvaluate:
    def test_perfect_classification(self) -> None:
        clf = TextClassifier(model_type="logreg")
        clf.fit(_X, _Y)
        metrics = clf.evaluate(_X, _Y)
        assert metrics.accuracy == 1.0
        assert metrics.f1_macro == 1.0

    def test_metrics_labels(self) -> None:
        clf = TextClassifier(model_type="logreg")
        clf.fit(_X, _Y)
        metrics = clf.evaluate(_X, _Y)
        assert set(metrics.labels) == {"A", "B", "C"}

    def test_confusion_matrix_shape(self) -> None:
        clf = TextClassifier(model_type="logreg")
        clf.fit(_X, _Y)
        metrics = clf.evaluate(_X, _Y)
        assert metrics.confusion.shape == (3, 3)
        # Perfect classification → diagonal only
        assert np.trace(metrics.confusion) == 12

    def test_summary_string(self) -> None:
        clf = TextClassifier(model_type="logreg")
        clf.fit(_X, _Y)
        metrics = clf.evaluate(_X, _Y)
        s = metrics.summary()
        assert "Accuracy" in s
        assert "F1" in s


class TestTextClassifierPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        clf = TextClassifier(model_type="logreg")
        clf.fit(_X, _Y)

        path = tmp_path / "clf.pkl"
        clf.save(path)
        assert path.exists()

        loaded = TextClassifier.load(path)
        assert loaded.model_type == clf.model_type
        # Same predictions
        assert (loaded.predict(_X) == clf.predict(_X)).all()

    def test_load_wrong_type(self, tmp_path: Path) -> None:
        import pickle

        path = tmp_path / "wrong.pkl"
        with path.open("wb") as f:
            pickle.dump({"not": "a classifier"}, f)
        with pytest.raises(TypeError):
            TextClassifier.load(path)
