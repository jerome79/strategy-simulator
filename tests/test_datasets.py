from pathlib import Path

import numpy as np
import pandas as pd
from pytest import MonkeyPatch

from strategy_simulator.datasets import load_prices, load_sentiment


def test_load_sentiment_converts_dates(tmp_path: "Path") -> None:
    """
    Test that `load_sentiment` converts the 'date' column to datetime and preserves order.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for file operations.
    """
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "BBB"],
            "avg_sentiment": [0.1, 0.2],
        }
    )
    path = tmp_path / "sentiment.parquet"
    df.to_parquet(path)

    loaded = load_sentiment(str(path))
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert loaded.equals(loaded.sort_values("date"))  # still intact


def test_load_prices_single_ticker(monkeypatch: "MonkeyPatch") -> None:
    """
    Test that `load_prices` correctly handles a single ticker by mocking yfinance.download.

    Args:
        monkeypatch (MonkeyPatch): pytest fixture to replace yfinance with a mock.
    """
    dates = pd.date_range("2024-01-01", periods=3)
    mock_df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1.1, 2.1, 3.1],
            "Low": [0.9, 1.9, 2.9],
            "Close": [1.0, 2.0, 3.0],
            "Volume": [100, 110, 120],
        },
        index=dates,
    )

    class MockYF:
        @staticmethod
        def download(tickers: list[str], start: str, end: str, progress: bool, auto_adjust: bool) -> pd.DataFrame:
            """
            Mock download function for yfinance.

            Args:
                tickers (list[str]): List of ticker symbols.
                start (str): Start date for data retrieval.
                end (str): End date for data retrieval.
                progress (bool): Whether to show progress bar.
                auto_adjust (bool): Whether to auto-adjust prices.

            Returns:
                pd.DataFrame: Mocked price data.
            """
            if len(tickers) == 1:
                assert tickers == ["AAA"]
            else:
                assert tickers == "AAA"
            return mock_df

    monkeypatch.setattr("strategy_simulator.datasets.yf", MockYF)

    out = load_prices(["AAA"], "2024-01-01", "2024-01-05")
    # Selecting ["Close"] should return a Series or DataFrame with close prices
    # In our mock, indexing ["Close"] yields a Series
    assert isinstance(out, (pd.Series, pd.DataFrame))
    if isinstance(out, pd.Series):
        assert list(out.values) == [1.0, 2.0, 3.0]


def test_load_prices_multi_ticker(monkeypatch: "MonkeyPatch") -> None:
    """
    Test that `load_prices` correctly handles multiple tickers by mocking yfinance.download.

    Args:
        monkeypatch (MonkeyPatch): pytest fixture to replace yfinance with a mock.
    """
    dates = pd.date_range("2024-01-01", periods=3)
    arrays = [
        ["Open", "Open", "Close", "Close"],
        ["AAA", "BBB", "AAA", "BBB"],
    ]
    cols = pd.MultiIndex.from_arrays(arrays)
    mock_df = pd.DataFrame(
        np.array(
            [
                [1.0, 2.0, 1.1, 2.1],
                [1.1, 2.1, 1.2, 2.2],
                [1.2, 2.2, 1.3, 2.3],
            ]
        ),
        index=dates,
        columns=cols,
    )

    class MockYF:
        @staticmethod
        def download(tickers: list[str], start: str, end: str, progress: bool, auto_adjust: bool) -> pd.DataFrame:
            """
            Mock download function for yfinance.

            Args:
                tickers (list[str]): List of ticker symbols.
                start (str): Start date for data retrieval.
                end (str): End date for data retrieval.
                progress (bool): Whether to show progress bar.
                auto_adjust (bool): Whether to auto-adjust prices.

            Returns:
                pd.DataFrame: Mocked price data.
            """
            assert tickers == ["AAA", "BBB"]
            return mock_df

    monkeypatch.setattr("strategy_simulator.datasets.yf", MockYF)

    out = load_prices(["AAA", "BBB"], "2024-01-01", "2024-01-05")
    # After ["Close"] selection you expect a DataFrame of shape (3,2)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["AAA", "BBB"]
    assert out.shape == (3, 2)
