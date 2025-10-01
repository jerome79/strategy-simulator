import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strategy_simulator import llm_sentiment
from strategy_simulator.llm_sentiment import score_headlines


def test_scoring_schema() -> None:
    """
    Recruiter-facing guarantee:
    - Output includes sentiment columns and signed sentiment in [-1,1]
    - Downstream backtest requires only [date, ticker, sentiment] after daily aggregation
    """
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "ticker": ["AAPL"], "headline": ["Apple releases new AI features"]})
    out = score_headlines(df, batch_size=8)
    assert {"sentiment_label", "sentiment_score", "sentiment"}.issubset(set(out.columns))
    assert -1.0 <= out["sentiment"].iloc[0] <= 1.0


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Sample DataFrame fixture with headlines, dates, and tickers for testing.
    """
    return pd.DataFrame(
        {"headline": ["good news", "bad news", "neutral news"], "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]), "ticker": ["AAPL", "AAPL", "GOOG"]}
    )


def test_score_headlines_empty() -> None:
    """
    Test that score_headlines returns an empty DataFrame with the sentiment column
    when given an empty input DataFrame.
    """
    df = pd.DataFrame(columns=["headline"])
    result = llm_sentiment.score_headlines(df)
    assert result.empty
    assert "sentiment" in result.columns


@patch("strategy_simulator.llm_sentiment._load_pipeline")
def test_score_headlines_basic(mock_load: MagicMock, sample_df: pd.DataFrame) -> None:
    """
    Test that score_headlines returns correct sentiment columns and values
    for a basic set of headlines, using a mocked pipeline.
    """
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {"label": "POSITIVE", "score": 0.9},
        {"label": "NEGATIVE", "score": 0.8},
        {"label": "NEUTRAL", "score": 0.5},
    ]
    mock_load.return_value = mock_pipe
    # Patch tqdm to avoid progress bar in test output
    with patch("strategy_simulator.llm_sentiment.tqdm", lambda x, **k: x):
        result = llm_sentiment.score_headlines(sample_df, batch_size=3)
    assert set(result.columns) >= {"sentiment_label", "sentiment_score", "sentiment"}
    assert result["sentiment"].iloc[0] > 0
    assert result["sentiment"].iloc[1] < 0
    assert result["sentiment"].iloc[2] == 0


@patch("transformers.pipeline")
def test__load_pipeline_fallback(mock_pipeline: MagicMock) -> None:
    """
    Test that _load_pipeline falls back to the default model if the primary model fails.
    """

    # Simulate exception for first model, success for fallback
    def side_effect(*args: Any, **kwargs: Any) -> str:
        """
        Simulates pipeline loading: raises an exception for the primary model,
        returns 'fallback' for any other model.
        """
        if kwargs.get("model") == "ProsusAI/finbert":
            raise Exception("Model not found")
        return "fallback"

    mock_pipeline.side_effect = side_effect
    result = llm_sentiment._load_pipeline("ProsusAI/finbert")
    assert result == "fallback"


@patch("strategy_simulator.llm_sentiment.score_headlines")
@patch("strategy_simulator.llm_sentiment.score_headlines")
def test_to_parquet_panel(mock_score: MagicMock, tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """
    Test that to_parquet_panel writes a parquet file with the expected columns and data.
    """
    df = sample_df.copy()
    df["sentiment"] = [1.0, -1.0, 0.0]
    mock_score.return_value = df
    csv_path = tmp_path / "input.csv"
    parquet_path = tmp_path / "out.parquet"
    df.to_csv(str(csv_path), index=False)
    out_path = llm_sentiment.to_parquet_panel(str(csv_path), str(parquet_path), text_col="headline", date_col="date", ticker_col="ticker")
    assert os.path.exists(out_path)
    panel = pd.read_parquet(out_path)
    assert set(panel.columns) >= {"date", "ticker", "sentiment", "source_count"}
    assert len(panel) > 0


def test_map_signed_branches() -> None:
    """
    Test the map_signed function within score_headlines to ensure it handles
    :return:
    """
    df = pd.DataFrame({"headline": ["a", "b", "c"]})
    with patch("strategy_simulator.llm_sentiment._load_pipeline") as mock_load:
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "NEGATIVE", "score": 0.7},
            {"label": "POSITIVE", "score": 0.8},
            {"label": "OTHER", "score": 0.5},
        ]
        mock_load.return_value = mock_pipe
        with patch("strategy_simulator.llm_sentiment.tqdm", lambda x, **k: x):
            result = llm_sentiment.score_headlines(df, batch_size=3)
    assert result["sentiment"].tolist() == [-0.7, 0.8, 0.0]
