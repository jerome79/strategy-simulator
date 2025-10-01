#!/usr/bin/env python3
"""
Recruiter-facing runner (Backwards Compatible)

WHAT CHANGED:
- Optionally builds the sentiment parquet from raw headlines if config includes an `llm` section.
- If `llm` is absent, behavior is unchanged: we assume a precomputed sentiment parquet.

WHY IT MATTERS:
- Enables a one-command, end-to-end demo for recruiters (LLM -> factor -> backtest).
- Keeps Research Copilot integration intact (no API renames; parquet schema unchanged).
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yfinance as yf


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download and format daily close prices for given tickers and date range.

    Args:
        tickers (list[str]): List of ticker symbols.
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).

    Returns:
           pd.DataFrame: DataFrame with columns [date, ticker, close].
    """

    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
    return px.rename(columns={"Date": "date"}).sort_values(["date", "ticker"])


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["fwd_ret_1d"] = prices.groupby("ticker")["close"].pct_change().shift(-1)
    return prices


def compute_factor(sentiment: pd.DataFrame, name: str, shock_window: int = 5) -> pd.DataFrame:
    df = sentiment.sort_values(["ticker", "date"]).copy()
    if name == "SENT_L1":
        df["factor"] = df.groupby("ticker")["sentiment"].shift(1)
    elif name == "SENT_SHOCK":
        roll = df.groupby("ticker")["sentiment"].transform(lambda s: s.rolling(shock_window, min_periods=1).mean())
        df["factor"] = df["sentiment"] - roll
    else:
        raise ValueError(f"Unknown factor {name}")
    return df.dropna(subset=["factor"])


def rank_and_portfolio(factor_panel: pd.DataFrame, long_p: float = 0.2, short_p: float = 0.2) -> pd.DataFrame:
    """
    Assigns long and short portfolio weights based on factor ranks for each date.

    Args:
        factor_panel (pd.DataFrame): DataFrame with columns including 'date', 'ticker', and 'factor'.
        long_p (float, optional): Top percentile to go long. Defaults to 0.2.
        short_p (float, optional): Bottom percentile to go short. Defaults to 0.2.

    Returns:
        pd.DataFrame: DataFrame with an added 'weight' column indicating portfolio positions.
    """
    df = factor_panel.copy()

    def assign_bucket(x: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns long and short portfolio weights to each row in the input DataFrame based on factor ranks.

        Args:
            x (pd.DataFrame): DataFrame for a single date with columns including 'factor'.

        Returns:
            pd.DataFrame: DataFrame with an added 'weight' column indicating portfolio positions for that date.
        """
        q = x["factor"].rank(pct=True)
        x["weight"] = 0.0
        x.loc[q >= 1 - long_p, "weight"] = 1.0
        x.loc[q <= short_p, "weight"] = -1.0
        if x["weight"].abs().sum() > 0:
            x["weight"] = x["weight"] / x["weight"].abs().sum()
        return x

    return df.groupby("date", group_keys=False).apply(assign_bucket)


def evaluate(portfolio: pd.DataFrame, fwd: pd.DataFrame, out_dir: str) -> dict:
    core = portfolio.merge(fwd[["date", "ticker", "fwd_ret_1d"]], on=["date", "ticker"], how="left").dropna()
    core["pnl"] = core["weight"] * core["fwd_ret_1d"]
    daily = core.groupby("date")["pnl"].sum().to_frame("ret").reset_index()
    daily["equity"] = (1 + daily["ret"]).cumprod()
    sr = daily["ret"].mean() / (daily["ret"].std() + 1e-12) * np.sqrt(252)
    mdd = (daily["equity"].cummax() - daily["equity"]).max()
    ic = core.groupby("date").apply(lambda x: x["factor"].rank().corr(x["fwd_ret_1d"].rank(), method="spearman")).mean()
    metrics = {"sharpe": float(sr), "max_drawdown": float(mdd), "ic": float(ic)}

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "equity_curve.png")
    plt.figure()
    plt.plot(daily["date"], daily["equity"])
    plt.title("Equity Curve (Long/Short)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    return metrics


def maybe_build_llm_sentiment(cfg: dict) -> None:
    """
    OPTIONAL step (non-breaking): build the sentiment parquet only if config requests it.
    This keeps external integrations (e.g., Research Copilot) unchanged when they pass an
    already-built parquet.
    """
    llm_cfg = cfg.get("llm", None)
    raw_csv = cfg["data"].get("raw_headlines_csv")
    parquet_out = cfg["data"]["sentiment_parquet"]

    if llm_cfg and raw_csv:
        # Local import keeps transformers/torch optional (only needed when invoked).
        from strategy_simulator.llm_sentiment import to_parquet_panel

        try:
            to_parquet_panel(
                raw_csv,
                parquet_out,
                model_name=llm_cfg.get("model_name", "ProsusAI/finbert"),
                batch_size=llm_cfg.get("batch_size", 16),
                text_col=llm_cfg.get("text_column", "headline"),
                date_col=llm_cfg.get("date_column", "date"),
                ticker_col=llm_cfg.get("ticker_column", "ticker"),
            )
        except Exception as e:
            print(f"[warn] LLM build skipped (will use existing parquet if present): {e}")


def main() -> None:
    """
    Main entry point for running the backtest pipeline.

    Steps:
    1. Optionally builds the sentiment parquet from raw headlines if requested in config.
    2. Loads the sentiment panel (either just built or precomputed).
    3. Loads price data and computes forward returns.
    4. Computes the factor based on sentiment.
    5. Constructs the long/short portfolio.
    6. Evaluates performance and saves the equity curve plot.

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backtest.llm.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1) Optionally build sentiment parquet from raw headlines (config-gated)
    maybe_build_llm_sentiment(cfg)

    # 2) Load sentiment panel (either just built or precomputed)
    sent = pd.read_parquet(cfg["data"]["sentiment_parquet"])

    # 3) Prices + forward returns
    px = load_prices(cfg["data"]["universe"], cfg["data"]["start"], cfg["data"]["end"])
    fwd = forward_returns(px)

    # 4) Factor -> 5) Portfolio -> 6) Evaluate + save equity curve
    fac = compute_factor(sent, cfg["factor"]["name"], cfg["factor"].get("shock_window", 5))
    port = rank_and_portfolio(fac, cfg["portfolio"]["long_percentile"], cfg["portfolio"]["short_percentile"])
    metrics = evaluate(port, fwd, cfg["reports"]["out_dir"])
    print(metrics)


if __name__ == "__main__":
    main()
