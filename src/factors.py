import pandas as pd

def compute_factors(sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple factors from sentiment panel.
    Returns panel: date, ticker, factor (SENT_L1, SHOCK).
    """
    df = sentiment.copy()
    df = df.sort_values(["ticker", "date"])

    # lag sentiment (yesterday)
    df["SENT_L1"] = df.groupby("ticker")["avg_sentiment"].shift(1)

    # 3-day z-score "shock"
    df["SENT_MEAN3"] = df.groupby("ticker")["avg_sentiment"].rolling(3).mean().reset_index(0, drop=True)
    df["SENT_STD3"] = df.groupby("ticker")["avg_sentiment"].rolling(3).std().reset_index(0, drop=True)
    df["SENT_SHOCK"] = (df["avg_sentiment"] - df["SENT_MEAN3"]) / df["SENT_STD3"]

    return df.dropna(subset=["SENT_L1", "SENT_SHOCK"])
