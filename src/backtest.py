import pandas as pd

from src.metrics import compute_ic, max_drawdown, sharpe_ratio


def run_longshort(
    df: pd.DataFrame,
    factor_col: str,
    fwd_return_col: str = "fwd_return",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Runs a simple long/short backtest.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing factor and forward return columns.
                           Must contain a 'date' column (daily cross section).
        factor_col (str): Name of the column containing factor values.
        fwd_return_col (str, optional): Name of the column containing forward returns. Defaults to "fwd_return".

    Returns:
        tuple[pd.DataFrame, dict]:
            - DataFrame with strategy returns and cumulative returns indexed by date.
              If no date meets the minimum cross section size, an empty DataFrame with the
              expected columns ('strategy_return', 'cum_return') is returned.
            - Dictionary with IC, Sharpe ratio, and Max Drawdown metrics. Sharpe / MaxDD are
              NaN if no strategy returns were generated.
    """
    df = df.copy()
    results: list[dict] = []

    # Parameters controlling the selection
    max_day = 10  # minimum number of instruments required for the day to be considered
    low = 0.3
    high = 0.7

    for date, daily in df.groupby("date"):
        # Drop rows where either factor or forward return is NaN
        daily_val = daily.dropna(subset=[factor_col, fwd_return_col])
        if len(daily_val) < max_day:
            continue

        q_low = daily_val[factor_col].quantile(low)
        q_high = daily_val[factor_col].quantile(high)

        longs = daily_val[daily[factor_col] >= q_high]
        shorts = daily_val[daily[factor_col] <= q_low]

        # If either side is empty, skip the day (alternatively could treat missing side as 0 exposure)
        if longs.empty or shorts.empty:
            continue

        ret = longs[fwd_return_col].mean() - shorts[fwd_return_col].mean()
        results.append({"date": date, "strategy_return": ret})

    strat = pd.DataFrame(results)

    if strat.empty:
        # Construct an empty, well formed result frame
        strat = pd.DataFrame({"strategy_return": pd.Series(dtype="float64"), "cum_return": pd.Series(dtype="float64")})
        strat.index.name = "date"
        sharpe = float("nan")
        mdd = float("nan")
    else:
        strat = strat.set_index("date").sort_index()
        strat["cum_return"] = (1 + strat["strategy_return"]).cumprod()
        sharpe = sharpe_ratio(strat["strategy_return"])
        mdd = max_drawdown(strat["cum_return"])

    ic = compute_ic(df[factor_col], df[fwd_return_col])

    return strat, {"IC": ic, "Sharpe": sharpe, "MaxDD": mdd}
