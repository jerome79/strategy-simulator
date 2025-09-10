import pandas as pd
import yfinance as yf

def load_sentiment(path: str) -> pd.DataFrame:
    """Load sentiment panel parquet."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo."""
    #data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    data = yf.download(tickers, start=start, end=end, progress=False,auto_adjust=True)['Close']
    return data
