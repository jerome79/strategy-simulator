from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def date_index() -> pd.DatetimeIndex:
    """10 sequential business-like days (no weekends logic needed here)."""
    start = datetime(2024, 1, 1)
    return pd.DatetimeIndex([start + timedelta(days=i) for i in range(10)])


@pytest.fixture
def tickers() -> list[str]:
    """
    Returns a list of example ticker symbols.

    Returns:
        list[str]: List of ticker symbols.
    """
    return ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


@pytest.fixture
def sentiment_panel(date_index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """
    Builds a synthetic sentiment panel DataFrame with monotonic drift for each ticker.
    Each row contains: date, ticker, sentiment.
    Sentiment values drift over time with added noise for signal generation.
    Args:
        date_index (pd.DatetimeIndex): Sequence of dates.
        tickers (list[str]): List of ticker symbols.
    Returns:
        pd.DataFrame: DataFrame with columns [date, ticker, sentiment].
    """
    rows = []
    rng = np.random.default_rng(42)
    for t in tickers:
        base = rng.normal(loc=0, scale=0.05)
        for i, d in enumerate(date_index):
            # Sentiment drift + small noise
            val = base + 0.01 * i + rng.normal(scale=0.02)
            rows.append({"date": d, "ticker": t, "sentiment": float(val)})
    df = pd.DataFrame(rows)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture
def prices(date_index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """
    Generates a synthetic price panel DataFrame for the given tickers and dates.
    Each ticker starts near 100 and follows a mild upward trend with small random increments.
    Returns are stable, ensuring deterministic forward return tests.

    Args:
        date_index (pd.DatetimeIndex): Sequence of dates.
        tickers (list[str]): List of ticker symbols.

    Returns:
        pd.DataFrame: DataFrame with tickers as columns and dates as index.
    """
    rng = np.random.default_rng(123)
    data = {}
    for t in tickers:
        # Start price ~100 +/- small jitter
        start_price = 100 + rng.normal(scale=1.0)
        # Controlled increments
        increments = 1 + 0.002 + rng.normal(scale=0.001, size=len(date_index))
        series = [start_price]
        for inc in increments[1:]:
            series.append(series[-1] * inc)
        data[t] = series
    df = pd.DataFrame(data, index=date_index)
    return df


@pytest.fixture
def minimal_joined(sentiment_panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a joined panel DataFrame with columns:
    date, ticker, SENT_L1, SENT_SHOCK, fwd_return.

    Args:
        sentiment_panel (pd.DataFrame): DataFrame with sentiment data for each ticker and date.
        prices (pd.DataFrame): DataFrame with price data for each ticker and date.

    Returns:
        pd.DataFrame: DataFrame containing joined factors and forward returns for each ticker and date.
    """
    # Import lazily so tests don't fail if modules missing
    from src.pipeline import compute_forward_returns

    from src.factors import compute_factors

    factors = compute_factors(sentiment_panel)  # adds SENT_L1, SENT_SHOCK
    fwd = compute_forward_returns(prices, horizon_days=1)
    fwd_stacked = fwd.stack().rename("fwd_return").to_frame().rename_axis(["date", "ticker"]).reset_index()
    joined = factors.merge(fwd_stacked, on=["date", "ticker"], how="inner").sort_values(["date", "ticker"]).reset_index(drop=True)
    return joined
