import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ample_panel() -> pd.DataFrame:
    """
    Panel with >=10 rows per date, some NaNs injected.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    tickers = [f"T{i:02d}" for i in range(12)]  # 12 ≥ threshold (10)
    rows = []
    for d_i, d in enumerate(dates):
        for t_i, t in enumerate(tickers):
            factor = (d_i + 1) * 0.1 + t_i * 0.01 + rng.normal(scale=0.002)
            fwd = factor * 0.05 + rng.normal(scale=0.004)
            rows.append({"date": d, "ticker": t, "factor": factor, "fwd_return": fwd})
    df = pd.DataFrame(rows)
    # Inject NaNs on one date
    mask = (df["date"] == dates[1]) & (df["ticker"].isin(tickers[:4]))
    df.loc[mask, ["factor", "fwd_return"]] = np.nan
    return df


@pytest.fixture
def insufficient_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-02-01", periods=2)
    tickers = ["A", "B", "C", "D", "E"]  # 5 < 10 threshold
    rows = []
    for d_i, d in enumerate(dates):
        for t_i, t in enumerate(tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "factor": (d_i + 1) * 0.2 + t_i * 0.01,
                    "fwd_return": 0.01 * (t_i + 1),
                }
            )
    return pd.DataFrame(rows)
