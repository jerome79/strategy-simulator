# run_test.py (drop in repo root and run: python run_test.py)


from src.backtest import run_longshort
from src.datasets import load_prices, load_sentiment
from src.factors import compute_factors
from src.plots import plot_equity_curve

panel = load_sentiment("data/sentiment_panel.parquet")
tickers = sorted(panel["ticker"].dropna().unique().tolist())[:50]  # cap for speed
start, end = str(panel["date"].min().date()), str(panel["date"].max().date())
prices = load_prices(tickers, start, end).ffill().dropna(how="all", axis=1)

# forward 1D returns (close-to-close)
rets = prices.pct_change().shift(-1)  # returns aligned to today signal predicting tomorrow

# compute factors
factors = compute_factors(panel)  # adds SENT_L1, SENT_SHOCK

# join factors with returns
factors = factors.set_index(["date", "ticker"]).sort_index()
rets_stacked = rets.stack().rename("fwd_return").to_frame()
rets_stacked.index.set_names(["date", "ticker"], inplace=True)
joined = factors.join(rets_stacked, how="inner").reset_index()

# run backtest
strat, metrics = run_longshort(joined, factor_col="SENT_L1", fwd_return_col="fwd_return")
print("Metrics:", metrics)

# plot
fig = plot_equity_curve(strat, title="Sentiment Long/Short — SENT_L1")
fig.savefig("reports/equity_curve.png", dpi=150)
print("Saved: reports/equity_curve.png")
