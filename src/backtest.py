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
        factor_col (str): Name of the column containing factor values.
        fwd_return_col (str, optional): Name of the column containing forward returns. Defaults to "fwd_return".

    Returns:
        tuple[pd.DataFrame, dict]:
            - DataFrame with strategy returns and cumulative returns indexed by date.
            - Dictionary with IC, Sharpe ratio, and Max Drawdown metrics.
    """
    df = df.copy()
    results = []

    for date, daily in df.groupby("date"):
        daily.dropna(subset=[factor_col, fwd_return_col], inplace=True)
        max_day = 10
        low = 0.3
        high = 0.7
        if len(daily) < max_day:
            continue
        q_low = daily[factor_col].quantile(low)
        q_high = daily[factor_col].quantile(high)
        longs = daily[daily[factor_col] >= q_high]
        shorts = daily[daily[factor_col] <= q_low]

        ret = longs[fwd_return_col].mean() - shorts[fwd_return_col].mean()
        results.append({"date": date, "strategy_return": ret})

    strat = pd.DataFrame(results).set_index("date")
    strat["cum_return"] = (1 + strat["strategy_return"]).cumprod()
    ic = compute_ic(df[factor_col], df[fwd_return_col])
    sharpe = sharpe_ratio(strat["strategy_return"])
    mdd = max_drawdown(strat["cum_return"])

    return strat, {"IC": ic, "Sharpe": sharpe, "MaxDD": mdd}
