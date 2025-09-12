# scripts/make_fake_panel.py  (run: python scripts/make_fake_panel.py)
import numpy as np
import pandas as pd

rng = pd.date_range("2023-01-01", "2023-06-30", freq="B")
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "XOM", "UNH"]
rows = []
rs = np.random.RandomState(42)
for t in tickers:
    s = rs.normal(0, 0.4, size=len(rng)).cumsum()  # latent trend
    noise = rs.normal(0, 0.2, size=len(rng))
    avg_sent = np.tanh(0.4 * s + noise)  # bounded [-1,1]
    for d, a in zip(rng, avg_sent, strict=False):
        rows.append({"date": d, "ticker": t, "avg_sentiment": float(a)})
df = pd.DataFrame(rows)
df.to_parquet("data/sentiment_panel.parquet", index=False)
print("Wrote data/sentiment_panel.parquet", df.shape)
