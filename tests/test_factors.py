import numpy as np
import pandas as pd
import pytest

from src.factors import compute_factors


@pytest.fixture
def sentiment_panel() -> pd.DataFrame:
    """
    Deterministic sentiment panel:
    - 2 tickers
    - 6 business days
    - Varying avg_sentiment so rolling std != 0
    Shuffled to verify internal sorting inside compute_factors.
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    values = {
        "AAA": [0.10, 0.12, 0.15, 0.11, 0.13, 0.18],
        "BBB": [0.05, 0.07, 0.06, 0.09, 0.08, 0.04],
    }
    rows = []
    for ticker, vals in values.items():
        for d, v in zip(dates, vals, strict=True):
            rows.append({"date": d, "ticker": ticker, "avg_sentiment": v})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def test_compute_factors_basic(sentiment_panel: pd.DataFrame) -> None:
    """
    Test that compute_factors produces expected columns, correct sorting, and accurate calculations
    for a shuffled deterministic sentiment panel with two tickers and six business days.
    """
    out = compute_factors(sentiment_panel)

    expected_cols = {
        "date",
        "ticker",
        "avg_sentiment",
        "SENT_L1",
        "SENT_MEAN3",
        "SENT_STD3",
        "SENT_SHOCK",
    }
    assert expected_cols.issubset(out.columns)

    # Sorted within each ticker
    for _, g in out.groupby("ticker"):
        assert g["date"].is_monotonic_increasing

    # Expect 4 rows per ticker (6 original - 2 lost due to lag + rolling window)
    rows_ticker = 8
    assert len(out) == rows_ticker

    # Validate SENT_L1 via merge
    sorted_src = sentiment_panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    sorted_src["SENT_L1_EXPECTED"] = sorted_src.groupby("ticker")["avg_sentiment"].shift(1)
    merged = out.merge(
        sorted_src[["date", "ticker", "SENT_L1_EXPECTED"]],
        on=["date", "ticker"],
        how="left",
    )
    assert np.allclose(
        merged["SENT_L1"].to_numpy(),
        merged["SENT_L1_EXPECTED"].to_numpy(),
        atol=1e-12,
        rtol=1e-9,
    )

    # Recompute rolling features exactly like the function (on a clean, sorted copy)
    recompute = sentiment_panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    recompute["SENT_L1"] = recompute.groupby("ticker")["avg_sentiment"].shift(1)
    recompute["SENT_MEAN3"] = recompute.groupby("ticker")["avg_sentiment"].rolling(3).mean().reset_index(0, drop=True)
    recompute["SENT_STD3"] = recompute.groupby("ticker")["avg_sentiment"].rolling(3).std().reset_index(0, drop=True)
    recompute["SENT_SHOCK_EXPECTED"] = (recompute["avg_sentiment"] - recompute["SENT_MEAN3"]) / recompute["SENT_STD3"]

    # Apply same dropna rule the function uses
    recompute_final = recompute.dropna(subset=["SENT_L1", "SENT_SHOCK_EXPECTED"])

    # Join expected shock back to output
    check = out.merge(
        recompute_final[["date", "ticker", "SENT_SHOCK_EXPECTED"]],
        on=["date", "ticker"],
        how="left",
    )

    # Ensure no NaNs in required outputs
    assert not out[["SENT_L1", "SENT_SHOCK"]].isna().any().any()

    # Compare computed shock values (allow tiny floating noise)
    assert np.allclose(
        check["SENT_SHOCK"].to_numpy(),
        check["SENT_SHOCK_EXPECTED"].to_numpy(),
        rtol=1e-9,
        atol=1e-12,
    )


def test_compute_factors_constant_window_produces_nan_shock_filtered() -> None:
    """
    If a ticker has constant sentiment for 3+ days, rolling std becomes 0 -> shock = inf or NaN.
    The function drops rows with NaN shock. We craft a minimal panel to ensure those rows are excluded.
    """
    dates = pd.date_range("2024-02-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["CCC"] * 5 + ["DDD"] * 5,
            # CCC constant, DDD varying
            "avg_sentiment": [0.2] * 5 + [0.1, 0.11, 0.09, 0.12, 0.13],
        }
    ).sample(frac=1.0, random_state=1)
    nb_ticker = 2
    out = compute_factors(df)
    # All CCC rows should be dropped (since SENT_STD3 == 0 => shock NaN or inf; std==0 gives NaN shock with division)
    assert "CCC" not in out["ticker"].unique()
    assert "DDD" in out["ticker"].unique()
    # DDD should still have rows (5 original -> 3 usable after lag+rolling)
    assert len(out[out["ticker"] == "DDD"]) >= nb_ticker
