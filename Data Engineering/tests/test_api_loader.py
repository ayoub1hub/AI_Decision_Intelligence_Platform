"""Tests for the API loader (uses mocking — no network calls)."""

from __future__ import annotations

from unittest.mock import patch

from data_engine.ingestion.api_loader import fetch_arxiv

# Minimal valid arXiv Atom response
_FAKE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Deep Learning for NLP</title>
    <summary>A survey.</summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <category term="cs.CL"/>
    <published>2024-01-01T00:00:00Z</published>
    <updated>2024-01-05T00:00:00Z</updated>
  </entry>
</feed>
"""


class TestFetchArxiv:
    @patch("data_engine.ingestion.api_loader.requests.get")
    def test_parses_response(self, mock_get) -> None:
        mock_resp = mock_get.return_value
        mock_resp.content = _FAKE_XML
        mock_resp.raise_for_status = lambda: None

        df = fetch_arxiv(query="cat:cs.CL", max_results=10)

        assert len(df) == 1
        assert df.iloc[0]["title"] == "Deep Learning for NLP"
        assert "Alice" in df.iloc[0]["authors"]
        assert "cs.CL" in df.iloc[0]["categories"]

    @patch("data_engine.ingestion.api_loader.requests.get")
    def test_empty_response(self, mock_get) -> None:
        mock_resp = mock_get.return_value
        mock_resp.content = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        mock_resp.raise_for_status = lambda: None

        df = fetch_arxiv()
        assert df.empty
