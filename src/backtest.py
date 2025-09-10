import pandas as pd
import numpy as np
from src.metrics import compute_ic, sharpe_ratio, max_drawdown

def run_longshort(df: pd.DataFrame, factor_col: str, fwd_return_col: str = "fwd_return"):
    """
    Simple long/short backtest:
    - Sort by factor
    - Long top 30%, short bottom 30%
    """
    df = df.copy()
    results = []

    for date, daily in df.groupby("date"):
        daily = daily.dropna(subset=[factor_col, fwd_return_col])
        if len(daily) < 10:
            continue
        q_low = daily[factor_col].quantile(0.3)
        q_high = daily[factor_col].quantile(0.7)
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
