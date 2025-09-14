import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import json
import os

from src.backtest import run_longshort
from src.datasets import load_prices, load_sentiment
from src.factors import compute_factors
from src.plots import plot_equity_curve


def last_metrics(metrics_path: str = "reports/metrics.json", curve_path: str = "reports/equity_curve.png") -> dict:
    """
    Loads the latest backtest metrics and equity curve path.

    Args:
        metrics_path (str): Path to the metrics JSON file.
        curve_path (str): Path to the equity curve image.

    Returns:
        dict: Dictionary with keys 'metrics' (dict) and 'equity_curve_path' (str).
    """
    metrics = {"IC": None, "Sharpe": None, "MaxDD": None, "Turnover": None}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            pass
    return {"metrics": metrics, "equity_curve_path": curve_path}


def run_backtest_from_panel(panel_path: str, factor: str = "SENT_L1", horizon: int = 1) -> dict:
    """
    Runs a long-short backtest using sentiment panel data.

    Args:
        panel_path (str): Path to the sentiment panel file.
        factor (str): Name of the factor column to use for ranking.
        horizon (int): Forward return horizon in days.

    Returns:
        dict: Dictionary with keys 'metrics' (dict) and 'equity_curve_path' (str).
    """
    panel = load_sentiment(panel_path)
    tickers = sorted(panel["ticker"].dropna().unique().tolist())[:100]
    start, end = str(panel["date"].min().date()), str(panel["date"].max().date())
    prices = load_prices(tickers, start, end).ffill().dropna(how="all", axis=1)
    rets = prices.pct_change(periods=horizon).shift(-horizon)

    fac = compute_factors(panel).set_index(["date", "ticker"]).sort_index()
    joined = fac.join(rets.stack().rename("fwd_return").to_frame(), how="inner").reset_index()

    strat, metrics = run_longshort(joined, factor_col=factor, fwd_return_col="fwd_return")
    fig = plot_equity_curve(strat, title=f"Sentiment L/S — {factor}")
    os.makedirs("reports", exist_ok=True)
    fig.savefig("reports/equity_curve.png", dpi=150)

    # save metrics for future `last_metrics()`
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)

    return {"metrics": metrics, "equity_curve_path": "reports/equity_curve.png"}
