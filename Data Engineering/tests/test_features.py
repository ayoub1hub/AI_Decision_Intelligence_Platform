"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd

from data_engine.transformation.features import engineer_features


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["http://arxiv.org/abs/2401.00001v1", "http://arxiv.org/abs/2401.00002v1"],
            "title": ["Paper A", "A much longer title for paper B"],
            "abstract": ["Short abstract.", "A longer abstract with more words in it."],
            "authors": ["Alice;Bob", "Carol"],
            "categories": ["cs.LG; cs.AI", "cs.CL"],
            "published_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "updated_date": pd.to_datetime(["2024-01-05", "2024-01-02"]),
        }
    )


class TestEngineerFeatures:
    def test_arxiv_id_extracted(self) -> None:
        df = engineer_features(_sample_df())
        assert df["arxiv_id"].iloc[0] == "2401.00001v1"
        assert df["arxiv_id"].iloc[1] == "2401.00002v1"

    def test_title_length(self) -> None:
        df = engineer_features(_sample_df())
        assert df["title_length"].iloc[0] == len("Paper A")
        assert df["title_length"].iloc[1] == len("A much longer title for paper B")

    def test_n_authors(self) -> None:
        df = engineer_features(_sample_df())
        assert df["n_authors"].iloc[0] == 2  # Alice;Bob
        assert df["n_authors"].iloc[1] == 1  # Carol

    def test_n_categories(self) -> None:
        df = engineer_features(_sample_df())
        assert df["n_categories"].iloc[0] == 2  # cs.LG; cs.AI
        assert df["n_categories"].iloc[1] == 1  # cs.CL

    def test_primary_category(self) -> None:
        df = engineer_features(_sample_df())
        assert df["primary_category"].iloc[0] == "cs.LG"
        assert df["primary_category"].iloc[1] == "cs.CL"

    def test_days_to_update(self) -> None:
        df = engineer_features(_sample_df())
        assert df["days_to_update"].iloc[0] == 4  # 01-01 → 01-05
        assert df["days_to_update"].iloc[1] == 0

    def test_published_year(self) -> None:
        df = engineer_features(_sample_df())
        assert df["published_year"].iloc[0] == 2024

    def test_missing_authors(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["http://arxiv.org/abs/2401.00001v1"],
                "title": ["X"],
                "abstract": ["Y"],
                "authors": [None],
                "categories": [None],
                "published_date": pd.to_datetime(["2024-01-01"]),
                "updated_date": pd.to_datetime(["2024-01-01"]),
            }
        )
        out = engineer_features(df)
        assert out["n_authors"].iloc[0] == 0
        assert out["n_categories"].iloc[0] == 0
